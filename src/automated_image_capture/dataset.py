from __future__ import annotations

import csv
import json
import os
import shutil
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

TRAIN_POSES = frozenset({155, 160, 180, 190, 210, 1155, 1200, 2155, 2200})
VALIDATION_POSES = frozenset({170, 1170, 2170})
TEST_POSES = frozenset({200, 1185, 2185})
ALL_POSES = TRAIN_POSES | VALIDATION_POSES | TEST_POSES

Split = Literal["train", "val", "test"]
RecordKind = Literal["positive", "empty"]
ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


class DatasetError(RuntimeError):
    """The curated dataset could not be collected, built, or verified."""


class DatasetBuildCancelled(DatasetError):
    """Dataset generation was cancelled by the caller."""


@dataclass(frozen=True, slots=True)
class DatasetBuildConfig:
    source_dataset: Path
    output_root: Path
    curation_path: Path | None = None
    prefer_hardlinks: bool = True
    version_name: str | None = None

    def validated(self) -> DatasetBuildConfig:
        source = self.source_dataset.expanduser().resolve()
        output = self.output_root.expanduser().resolve()
        if not source.is_dir():
            raise DatasetError(f"OBB-Datensatzordner nicht gefunden: {source}")
        for required in ("data.yaml", "label_report.csv", "label_summary.json", "images", "labels"):
            if not (source / required).exists():
                raise DatasetError(f"{required} fehlt in {source}")
        curation = self.curation_path
        if curation is not None:
            curation = curation.expanduser().resolve()
        return replace(
            self,
            source_dataset=source,
            output_root=output,
            curation_path=curation,
        )


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    record_id: str
    kind: RecordKind
    class_id: int | None
    class_name: str
    pose_id: int
    panel_1: int
    panel_2: int
    exposure: str
    quality: str
    consensus_iou: float | None
    source_image: Path
    source_label: Path
    target_name: str
    split: Split
    conveyor_station_id: int | None = None
    conveyor_direction: str = "fixed"
    conveyor_nominal_position_mm: float | None = None
    conveyor_measured_position_mm: float | None = None
    ramp_sample_id: int | None = None
    conveyor_track_used: bool = False
    track_correction_applied: bool = False
    excluded: bool = False

    @property
    def label_name(self) -> str:
        return f"{Path(self.target_name).stem}.txt"


@dataclass(frozen=True, slots=True)
class DatasetBuildResult:
    dataset_directory: Path
    manifest_path: Path
    data_yaml_path: Path
    included_images: int
    excluded_images: int
    split_counts: dict[str, int]
    class_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class DatasetIntegrityResult:
    image_count: int
    label_count: int
    split_counts: dict[str, int]
    class_counts: dict[str, int]
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class ResizedDatasetResult:
    dataset_directory: Path
    image_count: int
    reused: bool


def default_build_config() -> DatasetBuildConfig:
    pictures = Path.home() / "Pictures"
    return DatasetBuildConfig(
        source_dataset=pictures / "Kl1i" / "OBB",
        output_root=pictures / "YOLO_Training",
        curation_path=pictures / "YOLO_Training" / "curation.json",
    )


def split_for_pose(pose_id: int) -> Split:
    if pose_id in TRAIN_POSES:
        return "train"
    if pose_id in VALIDATION_POSES:
        return "val"
    if pose_id in TEST_POSES:
        return "test"
    raise DatasetError(
        f"UR-Pose {pose_id} ist keinem Split zugeordnet. Erwartet: {sorted(ALL_POSES)}"
    )


def _index_files(root: Path, directory: str, suffix: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in (root / directory).rglob(f"*{suffix}"):
        if path.name in result:
            raise DatasetError(f"Mehrdeutiger Dateiname in {root / directory}: {path.name}")
        result[path.name] = path
    return result


def _integer(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetError(f"Ungültiger Wert für {key!r}: {row.get(key)!r}") from exc


def _float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _integer_or_none(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _csv_bool(value: str | None) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "ja"}


def _load_class_names(root: Path) -> dict[int, str]:
    try:
        payload = json.loads((root / "label_summary.json").read_text(encoding="utf-8"))
        raw_classes = payload["classes"]
        names = {
            int(class_id): str(value["name"] if isinstance(value, dict) else value)
            for class_id, value in raw_classes.items()
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DatasetError(
            f"Klassen können nicht aus label_summary.json gelesen werden: {exc}"
        ) from exc
    if not names or sorted(names) != list(range(len(names))):
        raise DatasetError("Klassen-IDs müssen lückenlos bei 0 beginnen.")
    return names


def _collect_positive_records(root: Path, class_names: dict[int, str]) -> list[DatasetRecord]:
    image_index = _index_files(root, "images", ".png")
    label_index = _index_files(root, "labels", ".txt")
    records: list[DatasetRecord] = []
    with (root / "label_report.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            pose_id = _integer(row, "pose_id")
            class_id = _integer(row, "class_id")
            if class_id not in class_names:
                raise DatasetError(f"Unbekannte Klasse {class_id} im Label-Bericht.")
            dataset_image = row.get("dataset_image", "")
            label_file = row.get("label_file", "")
            if dataset_image not in image_index:
                raise DatasetError(f"Bild aus Label-Bericht fehlt: {dataset_image}")
            if label_file not in label_index:
                raise DatasetError(f"Label aus Label-Bericht fehlt: {label_file}")
            parsed = _parse_obb(label_index[label_file])
            if parsed is None or parsed[0] != class_id:
                raise DatasetError(
                    f"Klasse in Bericht und Label stimmt nicht überein: {label_file}"
                )
            records.append(
                DatasetRecord(
                    record_id=f"positive:{dataset_image}",
                    kind="positive",
                    class_id=class_id,
                    class_name=class_names[class_id],
                    pose_id=pose_id,
                    panel_1=_integer(row, "panel_1"),
                    panel_2=_integer(row, "panel_2"),
                    exposure=row.get("exposure", "auto"),
                    quality=row.get("quality", "PASS").upper(),
                    consensus_iou=_float_or_none(row.get("consensus_iou")),
                    source_image=image_index[dataset_image],
                    source_label=label_index[label_file],
                    target_name=dataset_image,
                    split=split_for_pose(pose_id),
                    conveyor_station_id=_integer_or_none(row.get("conveyor_station_id")),
                    conveyor_direction=row.get("conveyor_direction", "fixed"),
                    conveyor_nominal_position_mm=_float_or_none(
                        row.get("conveyor_nominal_metadata_position_mm")
                        or row.get("conveyor_position_mm")
                    ),
                    conveyor_measured_position_mm=_float_or_none(
                        row.get("conveyor_measured_position_mm")
                    ),
                    ramp_sample_id=_integer_or_none(row.get("ramp_sample_id")),
                    conveyor_track_used=_csv_bool(row.get("conveyor_track_used")),
                    track_correction_applied=_csv_bool(row.get("track_correction_applied")),
                )
            )
    return records


def _collect_empty_records(root: Path) -> list[DatasetRecord]:
    image_index = _index_files(root, "images", ".png")
    label_index = _index_files(root, "labels", ".txt")
    records: list[DatasetRecord] = []
    seen: set[str] = set()
    with (root / "label_report.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            background_file = row.get("background_file", "")
            if background_file in seen:
                continue
            seen.add(background_file)
            pose_id = _integer(row, "pose_id")
            old_image_name = f"empty_{background_file}"
            old_label_name = f"{Path(old_image_name).stem}.txt"
            if old_image_name not in image_index:
                raise DatasetError(f"Leerbild fehlt: {old_image_name}")
            if old_label_name not in label_index:
                raise DatasetError(f"Leeres Label fehlt: {old_label_name}")
            records.append(
                DatasetRecord(
                    record_id=f"empty:{background_file}",
                    kind="empty",
                    class_id=None,
                    class_name="Leere Rutsche",
                    pose_id=pose_id,
                    panel_1=_integer(row, "panel_1"),
                    panel_2=_integer(row, "panel_2"),
                    exposure=row.get("exposure", "auto"),
                    quality="PASS",
                    consensus_iou=None,
                    source_image=image_index[old_image_name],
                    source_label=label_index[old_label_name],
                    target_name=old_image_name,
                    split=split_for_pose(pose_id),
                    conveyor_station_id=_integer_or_none(row.get("conveyor_station_id")),
                    conveyor_direction=row.get("conveyor_direction", "fixed"),
                    conveyor_nominal_position_mm=_float_or_none(
                        row.get("conveyor_nominal_metadata_position_mm")
                        or row.get("conveyor_position_mm")
                    ),
                    conveyor_measured_position_mm=_float_or_none(
                        row.get("conveyor_measured_position_mm")
                    ),
                    ramp_sample_id=_integer_or_none(row.get("ramp_sample_id")),
                )
            )
    return records


def load_curation(path: Path | None, source_dataset: Path | None = None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        saved_source = str(payload.get("source_dataset", "")).strip()
        if source_dataset is not None and (
            not saved_source or Path(saved_source).resolve() != source_dataset.resolve()
        ):
            return set()
        values = payload.get("excluded_ids", [])
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError("excluded_ids muss eine Liste von Zeichenketten sein")
        return set(values)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Curation-Datei kann nicht gelesen werden: {path}: {exc}") from exc


def save_curation(
    path: Path,
    excluded_ids: Iterable[str],
    source_dataset: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_dataset": "" if source_dataset is None else str(source_dataset.resolve()),
        "excluded_ids": sorted(set(excluded_ids)),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_dataset_records(config: DatasetBuildConfig) -> list[DatasetRecord]:
    config = config.validated()
    class_names = _load_class_names(config.source_dataset)
    records = [
        *_collect_positive_records(config.source_dataset, class_names),
        *_collect_empty_records(config.source_dataset),
    ]
    identifiers = [record.record_id for record in records]
    targets = [record.target_name for record in records]
    duplicate_ids = [value for value, count in Counter(identifiers).items() if count > 1]
    duplicate_targets = [value for value, count in Counter(targets).items() if count > 1]
    if duplicate_ids:
        raise DatasetError(f"Doppelte Datensatz-IDs: {duplicate_ids[:3]}")
    if duplicate_targets:
        raise DatasetError(f"Doppelte Zieldateinamen: {duplicate_targets[:3]}")
    exclusions = load_curation(config.curation_path, config.source_dataset)
    known = set(identifiers)
    unknown = exclusions - known
    if unknown:
        raise DatasetError(f"Curation enthält unbekannte Bilder: {sorted(unknown)[:3]}")
    return [replace(record, excluded=record.record_id in exclusions) for record in records]


def _parse_obb(path: Path, class_id: int | None = None) -> tuple[int, np.ndarray] | None:
    text = path.read_text(encoding="ascii").strip()
    if not text:
        return None
    tokens = text.split()
    if len(tokens) != 9:
        raise DatasetError(f"OBB-Label benötigt 9 Werte: {path}")
    try:
        original_class = int(tokens[0])
        coordinates = np.asarray([float(value) for value in tokens[1:]], dtype=np.float32)
    except ValueError as exc:
        raise DatasetError(f"Ungültiges OBB-Label: {path}") from exc
    if not np.all(np.isfinite(coordinates)) or np.any(coordinates < 0) or np.any(coordinates > 1):
        raise DatasetError(f"OBB-Koordinaten außerhalb [0, 1]: {path}")
    return (original_class if class_id is None else class_id, coordinates.reshape(4, 2))


def render_record_preview(record: DatasetRecord, max_width: int = 1100) -> np.ndarray:
    image = cv2.imread(str(record.source_image), cv2.IMREAD_COLOR)
    if image is None:
        raise DatasetError(f"Bild kann nicht gelesen werden: {record.source_image}")
    parsed = _parse_obb(record.source_label, record.class_id)
    if parsed is not None:
        _, normalized = parsed
        height, width = image.shape[:2]
        points = normalized * np.asarray([width, height], dtype=np.float32)
        cv2.polylines(image, [np.rint(points).astype(np.int32)], True, (0, 255, 0), 5)
    label = (
        f"{record.class_name} | UR {record.pose_id} | "
        f"P1 {record.panel_1}% | P2 {record.panel_2}% | {record.quality}"
    )
    cv2.putText(image, label, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    if image.shape[1] > max_width:
        factor = max_width / image.shape[1]
        image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _link_or_copy(source: Path, target: Path, prefer_hardlinks: bool) -> None:
    if prefer_hardlinks:
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def _version_directory(config: DatasetBuildConfig) -> Path:
    base_name = config.version_name or datetime.now().strftime("dataset_%Y%m%d_%H%M%S")
    candidate = config.output_root / base_name
    suffix = 2
    while candidate.exists():
        candidate = config.output_root / f"{base_name}_{suffix}"
        suffix += 1
    return candidate


def _manifest_record(record: DatasetRecord) -> dict[str, object]:
    payload = asdict(record)
    payload["source_image"] = str(record.source_image)
    payload["source_label"] = str(record.source_label)
    return payload


def _data_yaml_text(root: Path, class_names: dict[int, str]) -> str:
    names = "".join(
        f"  {class_id}: {json.dumps(name, ensure_ascii=False)}\n"
        for class_id, name in sorted(class_names.items())
    )
    return (
        f"path: {json.dumps(root.as_posix())}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        + names
    )


def build_curated_dataset(
    config: DatasetBuildConfig,
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
) -> DatasetBuildResult:
    config = config.validated()
    class_names = _load_class_names(config.source_dataset)
    records = collect_dataset_records(config)
    included = [record for record in records if not record.excluded]
    if not included:
        raise DatasetError("Alle Bilder wurden ausgeschlossen; der Datensatz wäre leer.")
    output = _version_directory(config)
    output.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)

    total = len(included)
    for index, record in enumerate(included, 1):
        if cancelled is not None and cancelled():
            raise DatasetBuildCancelled("Datensatzaufbereitung abgebrochen.")
        image_target = output / "images" / record.split / record.target_name
        label_target = output / "labels" / record.split / record.label_name
        _link_or_copy(record.source_image, image_target, config.prefer_hardlinks)
        if record.kind == "empty":
            label_target.write_text("", encoding="ascii")
        else:
            parsed = _parse_obb(record.source_label, record.class_id)
            if parsed is None:
                raise DatasetError(f"Positives Bild besitzt kein Label: {record.source_image}")
            class_id, coordinates = parsed
            flat = " ".join(f"{float(value):.6f}" for value in coordinates.ravel())
            label_target.write_text(f"{class_id} {flat}\n", encoding="ascii")
        if progress is not None:
            progress(index, total, f"{record.class_name}: {record.target_name}")

    split_counts = Counter(record.split for record in included)
    class_counts = Counter(record.class_name for record in included)
    manifest = {
        "format_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "classes": {str(class_id): name for class_id, name in class_names.items()},
        "splits": {
            "train_poses": sorted(TRAIN_POSES),
            "validation_poses": sorted(VALIDATION_POSES),
            "test_poses": sorted(TEST_POSES),
        },
        "sources": {
            "source_dataset": str(config.source_dataset),
        },
        "included_images": len(included),
        "excluded_images": len(records) - len(included),
        "split_counts": dict(split_counts),
        "class_counts": dict(class_counts),
        "records": [_manifest_record(record) for record in records],
    }
    manifest_path = output / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if config.curation_path is not None and config.curation_path.is_file():
        shutil.copy2(config.curation_path, output / "curation.json")
    data_yaml = output / "data.yaml"
    data_yaml.write_text(_data_yaml_text(output, class_names), encoding="utf-8")
    integrity = verify_curated_dataset(output)
    if not integrity.valid:
        raise DatasetError("Datensatzprüfung fehlgeschlagen: " + "; ".join(integrity.errors[:5]))
    return DatasetBuildResult(
        dataset_directory=output,
        manifest_path=manifest_path,
        data_yaml_path=data_yaml,
        included_images=len(included),
        excluded_images=len(records) - len(included),
        split_counts=dict(split_counts),
        class_counts=dict(class_counts),
    )


def verify_curated_dataset(directory: Path) -> DatasetIntegrityResult:
    directory = directory.expanduser().resolve()
    errors: list[str] = []
    split_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    images: dict[tuple[str, str], Path] = {}
    labels: dict[tuple[str, str], Path] = {}
    source_splits: dict[str, str] = {}
    class_names: dict[int, str] = {}
    manifest_path = directory / "dataset_manifest.json"
    if not manifest_path.is_file():
        errors.append("dataset_manifest.json fehlt")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            class_names = {
                int(class_id): str(name)
                for class_id, name in manifest.get("classes", {}).items()
            }
            if not class_names:
                errors.append("Manifest enthält keine Klassen")
            for record in manifest.get("records", []):
                if record.get("excluded"):
                    continue
                source = str(record.get("source_image", ""))
                split = str(record.get("split", ""))
                old_split = source_splits.setdefault(source, split)
                if old_split != split:
                    errors.append(f"Quelldatei liegt in mehreren Splits: {source}")
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            errors.append(f"Manifest ungültig: {exc}")
    for split in ("train", "val", "test"):
        image_dir = directory / "images" / split
        label_dir = directory / "labels" / split
        for path in image_dir.glob("*.png"):
            key = (split, path.stem)
            images[key] = path
            split_counts[split] += 1
            if cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) is None:
                errors.append(f"Bild nicht lesbar: {path}")
        for path in label_dir.glob("*.txt"):
            key = (split, path.stem)
            labels[key] = path
            try:
                parsed = _parse_obb(path)
                if parsed is None:
                    class_counts["Leere Rutsche"] += 1
                elif parsed[0] in class_names:
                    class_counts[class_names[parsed[0]]] += 1
                else:
                    errors.append(f"Ungültige Klasse {parsed[0]}: {path}")
            except DatasetError as exc:
                errors.append(str(exc))
    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))
    if missing_labels:
        errors.append(f"{len(missing_labels)} Bilder ohne Label")
    if missing_images:
        errors.append(f"{len(missing_images)} Labels ohne Bild")
    return DatasetIntegrityResult(
        image_count=len(images),
        label_count=len(labels),
        split_counts=dict(split_counts),
        class_counts=dict(class_counts),
        errors=tuple(errors),
    )


def prepare_resized_training_dataset(
    source_directory: Path,
    max_edge: int,
    progress: ProgressCallback | None = None,
) -> ResizedDatasetResult:
    source = source_directory.expanduser().resolve()
    if max_edge < 128:
        raise DatasetError("Die Trainingskantenlänge muss mindestens 128 Pixel betragen.")
    source_integrity = verify_curated_dataset(source)
    if not source_integrity.valid:
        raise DatasetError(
            "Quelldatensatz ist ungültig: " + "; ".join(source_integrity.errors[:5])
        )
    output = source.parent / f"{source.name}_imgsz{max_edge}"
    marker = output / "resize_manifest.json"
    if marker.is_file():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if (
                Path(str(payload["source_directory"])).resolve() == source
                and int(payload["max_edge"]) == max_edge
            ):
                integrity = verify_curated_dataset(output)
                if integrity.valid and integrity.image_count == source_integrity.image_count:
                    return ResizedDatasetResult(output, integrity.image_count, True)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
        raise DatasetError(
            f"Vorhandener Trainingscache ist unvollständig oder veraltet: {output}"
        )
    if output.exists():
        raise DatasetError(f"Trainingscache existiert ohne gültiges Manifest: {output}")
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)

    image_paths = [
        path
        for split in ("train", "val", "test")
        for path in sorted((source / "images" / split).glob("*.png"))
    ]
    total = len(image_paths)
    for index, image_path in enumerate(image_paths, 1):
        split = image_path.parent.name
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise DatasetError(f"Bild kann nicht gelesen werden: {image_path}")
        height, width = image.shape[:2]
        factor = min(1.0, max_edge / max(height, width))
        if factor < 1.0:
            image = cv2.resize(
                image,
                (round(width * factor), round(height * factor)),
                interpolation=cv2.INTER_AREA,
            )
        target = output / "images" / split / image_path.name
        if not cv2.imwrite(str(target), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise DatasetError(f"Trainingsbild kann nicht geschrieben werden: {target}")
        label_source = source / "labels" / split / f"{image_path.stem}.txt"
        label_target = output / "labels" / split / label_source.name
        _link_or_copy(label_source, label_target, True)
        if progress is not None:
            progress(index, total, f"Trainingscache {index}/{total}")

    source_manifest = json.loads(
        (source / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    source_manifest["derived_from"] = str(source)
    source_manifest["image_max_edge"] = max_edge
    source_manifest["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    (output / "dataset_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    curation = source / "curation.json"
    if curation.is_file():
        shutil.copy2(curation, output / "curation.json")
    class_names = {
        int(class_id): str(name)
        for class_id, name in source_manifest.get("classes", {}).items()
    }
    (output / "data.yaml").write_text(
        _data_yaml_text(output, class_names), encoding="utf-8"
    )
    integrity = verify_curated_dataset(output)
    if not integrity.valid or integrity.image_count != source_integrity.image_count:
        details = "; ".join(integrity.errors[:5]) or "abweichende Bildanzahl"
        raise DatasetError(f"Trainingscache-Prüfung fehlgeschlagen: {details}")
    marker.write_text(
        json.dumps(
            {
                "format_version": 1,
                "source_directory": str(source),
                "max_edge": max_edge,
                "image_count": integrity.image_count,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return ResizedDatasetResult(output, integrity.image_count, False)
