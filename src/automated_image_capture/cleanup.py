from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import yaml

from automated_image_capture.image_files import CAPTURE_NAME, is_supported_image

CleanupFormat = Literal["png", "webp_lossless", "webp", "jpeg"]
ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]

_RESIZED_CACHE = re.compile(r"^(?P<source>.+)_imgsz\d+$", re.IGNORECASE)


class CleanupError(RuntimeError):
    """A cleanup plan could not be created or executed safely."""


@dataclass(frozen=True, slots=True)
class CleanupSettings:
    root_directory: Path
    output_format: CleanupFormat = "png"
    max_edge: int = 0
    png_compression: int = 3
    quality: int = 90
    remove_caches: bool = True
    deduplicate: bool = True

    def validated(self) -> CleanupSettings:
        root = self.root_directory.expanduser().resolve()
        if not root.is_dir():
            raise CleanupError(f"Ordner nicht gefunden: {root}")
        if self.output_format not in {"png", "webp_lossless", "webp", "jpeg"}:
            raise CleanupError("Unbekanntes Ausgabeformat.")
        if self.max_edge and not 128 <= self.max_edge <= 8192:
            raise CleanupError("Die maximale Kantenlänge muss 0 oder 128–8192 sein.")
        if not 0 <= self.png_compression <= 9:
            raise CleanupError("Die PNG-Kompressionsstufe muss zwischen 0 und 9 liegen.")
        if not 1 <= self.quality <= 100:
            raise CleanupError("Die Bildqualität muss zwischen 1 und 100 liegen.")
        return replace(self, root_directory=root)


@dataclass(frozen=True, slots=True)
class ImageAction:
    source_paths: tuple[Path, ...]
    source_physical_bytes: int
    target_suffix: str
    estimated_target_bytes: int
    external_links: int = 0


@dataclass(frozen=True, slots=True)
class CleanupPlan:
    settings: CleanupSettings
    image_actions: tuple[ImageAction, ...]
    cache_directories: tuple[Path, ...]
    cache_files: tuple[Path, ...]
    capture_directories: tuple[Path, ...]
    obb_directories: tuple[Path, ...]
    dataset_directories: tuple[Path, ...]
    logical_bytes_before: int
    physical_bytes_before: int
    estimated_physical_bytes_after: int
    managed_image_count: int
    skipped_image_count: int
    warnings: tuple[str, ...]

    @property
    def estimated_savings(self) -> int:
        return max(0, self.physical_bytes_before - self.estimated_physical_bytes_after)


@dataclass(frozen=True, slots=True)
class CleanupResult:
    processed_images: int
    deleted_cache_files: int
    physical_bytes_before: int
    physical_bytes_after: int
    report_path: Path
    cancelled: bool = False

    @property
    def freed_bytes(self) -> int:
        return max(0, self.physical_bytes_before - self.physical_bytes_after)


@dataclass(slots=True)
class _PhysicalImage:
    paths: list[Path]
    physical_bytes: int
    link_count: int
    shape: tuple[int, ...] = ()
    dtype: str = ""
    quick_digest: str = ""


def _walk_files(root: Path) -> Iterator[Path]:
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            path = Path(entry.path)
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if getattr(path, "is_junction", lambda: False)():
                        continue
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    yield path
            except OSError:
                continue


def _tree_sizes(root: Path) -> tuple[int, int]:
    logical = 0
    inodes: dict[tuple[int, int], int] = {}
    for path in _walk_files(root):
        try:
            stat = path.stat()
        except OSError:
            continue
        logical += stat.st_size
        inodes.setdefault((stat.st_dev, stat.st_ino), stat.st_size)
    return logical, sum(inodes.values())


def _completed_capture(directory: Path) -> bool:
    state_path = directory / "capture_session.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        expected = int(state["total"])
        if state.get("status") != "completed" or int(state.get("next_index", -1)) != expected:
            return False
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    images = [
        path
        for path in _walk_files(directory)
        if path.parent == directory and is_supported_image(path) and CAPTURE_NAME.match(path.name)
    ]
    return len(images) == expected and all(path.with_suffix(".yaml").is_file() for path in images)


def _resized_cache_directories(root: Path) -> tuple[Path, ...]:
    caches: list[Path] = []
    for marker in _walk_files(root):
        if marker.name != "resize_manifest.json":
            continue
        directory = marker.parent
        match = _RESIZED_CACHE.match(directory.name)
        if match is None:
            continue
        source = directory.with_name(match.group("source"))
        required = ("data.yaml", "dataset_manifest.json", "images")
        if all((source / name).exists() for name in required):
            caches.append(directory)
    return tuple(sorted(set(caches)))


def _inside_any(path: Path, directories: tuple[Path, ...]) -> bool:
    for directory in directories:
        try:
            path.relative_to(directory)
            return True
        except ValueError:
            pass
    return False


def _managed_layout(
    root: Path,
) -> tuple[
    set[Path],
    tuple[Path, ...],
    tuple[Path, ...],
    tuple[Path, ...],
    tuple[Path, ...],
    tuple[Path, ...],
]:
    cache_directories = _resized_cache_directories(root)
    captures: list[Path] = []
    obb: list[Path] = []
    datasets: list[Path] = []
    for path in _walk_files(root):
        if path.name == "capture_session.json" and _completed_capture(path.parent):
            captures.append(path.parent)
        elif path.name == "label_report.csv" and all(
            (path.parent / name).exists()
            for name in ("data.yaml", "label_summary.json", "images", "labels")
        ):
            obb.append(path.parent)
        elif path.name == "dataset_manifest.json" and all(
            (path.parent / name).exists() for name in ("data.yaml", "images", "labels")
        ) and not _inside_any(path.parent, cache_directories):
            datasets.append(path.parent)
    managed: set[Path] = set()
    for directory in captures:
        managed.update(
            path
            for path in _walk_files(directory)
            if path.parent == directory
            and is_supported_image(path)
            and CAPTURE_NAME.match(path.name)
        )
    for directory in [*obb, *datasets]:
        image_root = directory / "images"
        managed.update(path for path in _walk_files(image_root) if is_supported_image(path))
    roots_for_cache_files = tuple(sorted(set([*obb, *datasets, *cache_directories])))
    cache_files = tuple(
        sorted(
            path
            for directory in roots_for_cache_files
            for path in _walk_files(directory)
            if path.suffix.casefold() == ".cache"
        )
    )
    return (
        managed,
        tuple(sorted(set(captures))),
        tuple(sorted(set(obb))),
        tuple(sorted(set(datasets))),
        cache_directories,
        cache_files,
    )


def _target_suffix(output_format: CleanupFormat) -> str:
    return {"png": ".png", "webp_lossless": ".webp", "webp": ".webp", "jpeg": ".jpg"}[
        output_format
    ]


def _transform(image: np.ndarray, settings: CleanupSettings) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] in {3, 4}:
        channels = image[..., :3]
        if np.array_equal(channels[..., 0], channels[..., 1]) and np.array_equal(
            channels[..., 0], channels[..., 2]
        ):
            image = channels[..., 0]
    if settings.max_edge:
        height, width = image.shape[:2]
        factor = min(1.0, settings.max_edge / max(height, width))
        if factor < 1.0:
            image = cv2.resize(
                image,
                (max(1, round(width * factor)), max(1, round(height * factor))),
                interpolation=cv2.INTER_AREA,
            )
    return np.ascontiguousarray(image)


def _encode(image: np.ndarray, settings: CleanupSettings) -> bytes:
    if settings.output_format == "png":
        extension = ".png"
        parameters = [cv2.IMWRITE_PNG_COMPRESSION, settings.png_compression]
    elif settings.output_format == "jpeg":
        extension = ".jpg"
        parameters = [cv2.IMWRITE_JPEG_QUALITY, settings.quality]
    else:
        extension = ".webp"
        quality = 101 if settings.output_format == "webp_lossless" else settings.quality
        parameters = [cv2.IMWRITE_WEBP_QUALITY, quality]
    success, encoded = cv2.imencode(extension, image, parameters)
    if not success:
        raise CleanupError(f"{extension}-Kodierung ist fehlgeschlagen.")
    return encoded.tobytes()


def _quick_digest(path: Path) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as stream:
        digest.update(stream.read(65536))
        if size > 65536:
            stream.seek(max(0, size - 65536))
            digest.update(stream.read(65536))
    return digest.hexdigest()


def _full_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _merge_exact_duplicates(groups: list[_PhysicalImage]) -> list[_PhysicalImage]:
    quick_buckets: dict[tuple[int, str], list[_PhysicalImage]] = defaultdict(list)
    for group in groups:
        group.quick_digest = _quick_digest(group.paths[0])
        quick_buckets[(group.physical_bytes, group.quick_digest)].append(group)
    merged: list[_PhysicalImage] = []
    for bucket in quick_buckets.values():
        if len(bucket) == 1:
            merged.extend(bucket)
            continue
        full_buckets: dict[str, list[_PhysicalImage]] = defaultdict(list)
        for group in bucket:
            full_buckets[_full_digest(group.paths[0])].append(group)
        for duplicates in full_buckets.values():
            merged.append(
                _PhysicalImage(
                    paths=[path for item in duplicates for path in item.paths],
                    physical_bytes=sum(item.physical_bytes for item in duplicates),
                    link_count=sum(item.link_count for item in duplicates),
                )
            )
    return merged


def analyze_cleanup(
    settings: CleanupSettings,
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
) -> CleanupPlan:
    settings = settings.validated()
    root = settings.root_directory
    all_images = {path for path in _walk_files(root) if is_supported_image(path)}
    managed, captures, obb, datasets, cache_directories, cache_files = _managed_layout(root)
    managed.difference_update(
        path for path in managed if _inside_any(path, cache_directories)
    )
    inode_groups: dict[tuple[int, int], _PhysicalImage] = {}
    for path in sorted(managed):
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino)
        group = inode_groups.setdefault(
            key,
            _PhysicalImage([], stat.st_size, int(getattr(stat, "st_nlink", 1))),
        )
        group.paths.append(path)
    groups = list(inode_groups.values())
    if settings.deduplicate and groups:
        groups = _merge_exact_duplicates(groups)
    target_suffix = _target_suffix(settings.output_format)
    signature_groups: dict[tuple[tuple[int, ...], str], list[_PhysicalImage]] = defaultdict(list)
    valid_groups: list[_PhysicalImage] = []
    warnings: list[str] = []
    total = len(groups)
    for index, group in enumerate(groups, 1):
        if cancelled is not None and cancelled():
            raise CleanupError("Analyse abgebrochen.")
        image = cv2.imread(str(group.paths[0]), cv2.IMREAD_UNCHANGED)
        if image is None:
            warnings.append(f"Beschädigtes Bild übersprungen: {group.paths[0]}")
            continue
        transformed = _transform(image, settings)
        group.shape = tuple(transformed.shape)
        group.dtype = str(transformed.dtype)
        signature_groups[(group.shape, group.dtype)].append(group)
        valid_groups.append(group)
        if progress is not None:
            progress(index, total, f"Analysiere Bild {index}/{total}")
    ratios: dict[tuple[tuple[int, ...], str], float] = {}
    for signature, matching_groups in signature_groups.items():
        sample_count = min(24, len(matching_groups))
        if sample_count == 1:
            sample_indices = (0,)
        else:
            sample_indices = tuple(
                round(index * (len(matching_groups) - 1) / (sample_count - 1))
                for index in range(sample_count)
            )
        samples: list[tuple[int, int]] = []
        for sample_index in sample_indices:
            sample_path = matching_groups[sample_index].paths[0]
            sample_image = cv2.imread(str(sample_path), cv2.IMREAD_UNCHANGED)
            if sample_image is None:
                continue
            encoded = _encode(_transform(sample_image, settings), settings)
            samples.append((sample_path.stat().st_size, len(encoded)))
        ratios[signature] = (
            sum(target / source for source, target in samples) / len(samples)
        )
    actions: list[ImageAction] = []
    for group in valid_groups:
        ratio = ratios[(group.shape, group.dtype)]
        estimated = max(1, round(group.paths[0].stat().st_size * ratio))
        internal_links = len(group.paths)
        external = max(0, group.link_count - internal_links)
        actions.append(
            ImageAction(
                tuple(sorted(group.paths)),
                group.physical_bytes,
                target_suffix,
                estimated,
                external,
            )
        )
        if external:
            warnings.append(
                f"{group.paths[0].name}: {external} Hardlink(s) liegen außerhalb des Ordners; "
                "deren alter Speicher bleibt bestehen."
            )
    logical_before, physical_before = _tree_sizes(root)
    estimated_after = physical_before
    for action in actions:
        if action.external_links:
            estimated_after += action.estimated_target_bytes
        else:
            estimated_after += action.estimated_target_bytes - action.source_physical_bytes
    if settings.remove_caches:
        cache_targets = [*cache_files]
        for directory in cache_directories:
            cache_targets.extend(_walk_files(directory))
        cache_inodes: dict[tuple[int, int], int] = {}
        for path in cache_targets:
            try:
                stat = path.stat()
            except OSError:
                continue
            if int(getattr(stat, "st_nlink", 1)) == 1:
                cache_inodes.setdefault((stat.st_dev, stat.st_ino), stat.st_size)
        estimated_after -= sum(cache_inodes.values())
    if not managed:
        warnings.append("Keine abgeschlossenen oder gültigen Projektbildstrukturen gefunden.")
    if (root / ".aic_cleanup_journal.json").is_file():
        warnings.append(
            "Ein Journal eines unterbrochenen Laufs wurde gefunden. Der neue Analyseplan "
            "setzt auf dem konsistent abgeschlossenen Zwischenstand auf."
        )
    return CleanupPlan(
        settings=settings,
        image_actions=tuple(actions),
        cache_directories=cache_directories if settings.remove_caches else (),
        cache_files=cache_files if settings.remove_caches else (),
        capture_directories=captures,
        obb_directories=obb,
        dataset_directories=datasets,
        logical_bytes_before=logical_before,
        physical_bytes_before=physical_before,
        estimated_physical_bytes_after=max(0, estimated_after),
        managed_image_count=len(managed),
        skipped_image_count=len(all_images - managed),
        warnings=tuple(warnings),
    )


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.cleanup-part")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _replace_strings(value: object, names: dict[str, str]) -> object:
    if isinstance(value, str):
        for old, new in names.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace_strings(item, names) for item in value]
    if isinstance(value, dict):
        return {key: _replace_strings(item, names) for key, item in value.items()}
    return value


def _rewrite_references(root: Path, names: dict[str, str]) -> None:
    if not names or all(old == new for old, new in names.items()):
        return
    for path in _walk_files(root):
        if path.name in {"dataset_manifest.json", "label_summary.json", "curation.json"}:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                updated = _replace_strings(payload, names)
                _atomic_write(
                    path,
                    json.dumps(updated, ensure_ascii=False, indent=2).encode("utf-8"),
                )
            except (OSError, json.JSONDecodeError):
                continue
        elif path.name == "label_report.csv":
            try:
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.DictReader(stream)
                    fieldnames = reader.fieldnames
                    rows = [
                        {key: _replace_strings(value, names) for key, value in row.items()}
                        for row in reader
                    ]
                if not fieldnames:
                    continue
                temporary = path.with_name(f".{path.name}.cleanup-part")
                with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                os.replace(temporary, path)
            except OSError:
                continue
    for sidecar in _walk_files(root):
        if sidecar.suffix.casefold() != ".yaml" or not sidecar.name.startswith("img_"):
            continue
        try:
            payload = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("image"), dict):
                old_name = str(payload["image"].get("file", ""))
                new_name = names.get(old_name)
                if new_name is None:
                    continue
                payload["image"]["file"] = new_name
                _atomic_write(
                    sidecar,
                    yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).encode("utf-8"),
                )
        except (OSError, yaml.YAMLError):
            continue


def _convert_action(action: ImageAction, settings: CleanupSettings) -> dict[str, str]:
    source = action.source_paths[0]
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise CleanupError(f"Bild kann nicht gelesen werden: {source}")
    transformed = _transform(image, settings)
    encoded = _encode(transformed, settings)
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if transformed.ndim == 2 and decoded is not None and decoded.ndim == 3:
        if np.array_equal(decoded[..., 0], decoded[..., 1]) and np.array_equal(
            decoded[..., 0], decoded[..., 2]
        ):
            decoded = decoded[..., 0]
    if decoded is None or decoded.shape != transformed.shape:
        raise CleanupError(f"Kontrolllesen der Zieldatei ist fehlgeschlagen: {source}")
    lossless = settings.output_format in {"png", "webp_lossless"}
    if lossless and not np.array_equal(decoded, transformed):
        raise CleanupError(f"Verlustfreie Pixelprüfung ist fehlgeschlagen: {source}")
    targets = [path.with_suffix(action.target_suffix) for path in action.source_paths]
    source_set = set(action.source_paths)
    for target in targets:
        if target.exists() and target not in source_set:
            raise CleanupError(f"Zieldatei existiert bereits: {target}")
    canonical = targets[0]
    _atomic_write(canonical, encoded)
    for target in targets[1:]:
        temporary = target.with_name(f".{target.name}.cleanup-link")
        temporary.unlink(missing_ok=True)
        try:
            os.link(canonical, temporary)
            os.replace(temporary, target)
        except OSError:
            temporary.unlink(missing_ok=True)
            _atomic_write(target, encoded)
    for old in action.source_paths:
        if old not in targets:
            old.unlink(missing_ok=True)
    names: dict[str, str] = {}
    for old, new in zip(action.source_paths, targets, strict=True):
        names[old.name] = new.name
        old_image_index = old.name.find("img_")
        new_image_index = new.name.find("img_")
        if old_image_index >= 0 and new_image_index >= 0:
            names[old.name[old_image_index:]] = new.name[new_image_index:]
    return names


def _remove_cache_targets(plan: CleanupPlan) -> int:
    deleted = 0
    root = plan.settings.root_directory
    currently_valid_caches = set(_resized_cache_directories(root))
    for path in plan.cache_files:
        try:
            path.resolve().relative_to(root)
            path.unlink(missing_ok=True)
            deleted += 1
        except (OSError, ValueError):
            continue
    for directory in sorted(plan.cache_directories, key=lambda path: len(path.parts), reverse=True):
        if directory not in currently_valid_caches:
            continue
        try:
            directory.resolve().relative_to(root)
        except ValueError:
            continue
        if directory.is_dir():
            shutil.rmtree(directory)
            deleted += 1
    return deleted


def _validate_result(plan: CleanupPlan) -> None:
    from automated_image_capture.dataset import (
        DatasetBuildConfig,
        collect_dataset_records,
        verify_curated_dataset,
    )
    from automated_image_capture.labeling import scan_capture

    for directory in plan.capture_directories:
        records = scan_capture(directory)
        if len(records) != json.loads(
            (directory / "capture_session.json").read_text(encoding="utf-8")
        )["total"]:
            raise CleanupError(f"Capture-Prüfung fehlgeschlagen: {directory}")
    for directory in plan.obb_directories:
        collect_dataset_records(DatasetBuildConfig(directory, directory.parent))
    for directory in plan.dataset_directories:
        if directory.exists():
            integrity = verify_curated_dataset(directory)
            if not integrity.valid:
                raise CleanupError(
                    f"Datensatzprüfung fehlgeschlagen: {directory}: "
                    + "; ".join(integrity.errors[:3])
                )


def execute_cleanup(
    plan: CleanupPlan,
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
) -> CleanupResult:
    settings = plan.settings.validated()
    root = settings.root_directory
    lock = root / ".aic_cleanup.lock"
    journal = root / ".aic_cleanup_journal.json"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(descriptor)
    except FileExistsError as exc:
        raise CleanupError(
            "Im Ordner liegt bereits eine Bereinigungssperre. Prüfe ein eventuell "
            "unterbrochenes cleanup_journal."
        ) from exc
    converted_names: dict[str, str] = {}
    processed = 0
    was_cancelled = False
    deleted = 0
    try:
        journal_payload = {
            "format_version": 1,
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "settings": {**asdict(settings), "root_directory": str(root)},
            "completed_actions": 0,
        }
        _atomic_write(journal, json.dumps(journal_payload, indent=2).encode("utf-8"))
        total = len(plan.image_actions)
        try:
            for index, action in enumerate(plan.image_actions, 1):
                if cancelled is not None and cancelled():
                    was_cancelled = True
                    break
                converted_names.update(_convert_action(action, settings))
                processed += len(action.source_paths)
                journal_payload["completed_actions"] = index
                _atomic_write(journal, json.dumps(journal_payload, indent=2).encode("utf-8"))
                if progress is not None:
                    progress(index, total, f"Optimiere Bildgruppe {index}/{total}")
        finally:
            _rewrite_references(root, converted_names)
        if not was_cancelled:
            deleted = _remove_cache_targets(plan)
        _validate_result(plan)
        journal.unlink(missing_ok=True)
        _, physical_after = _tree_sizes(root)
        report_path = root / datetime.now().strftime("cleanup_report_%Y%m%d_%H%M%S.json")
        report = {
            "format_version": 1,
            "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "cancelled": was_cancelled,
            "settings": {**asdict(settings), "root_directory": str(root)},
            "processed_image_paths": processed,
            "deleted_cache_targets": deleted,
            "physical_bytes_before": plan.physical_bytes_before,
            "physical_bytes_after": physical_after,
            "freed_bytes": max(0, plan.physical_bytes_before - physical_after),
        }
        _atomic_write(report_path, json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"))
        return CleanupResult(
            processed,
            deleted,
            plan.physical_bytes_before,
            physical_after,
            report_path,
            was_cancelled,
        )
    finally:
        lock.unlink(missing_ok=True)
