from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

CAPTURE_NAME = re.compile(
    r"^img_(?P<index>\d+)_(?:ur(?P<pose>\d+)|ura-(?P<angle>\d+))_"
    r"(?:belt-(?P<belt>\d+)_pos-(?P<position>\d+)_(?P<direction>out|back)_)?"
    r"(?:ramp-(?P<ramp>\d+)_)?"
    r"p1-(?P<p1>\d+)_p2-(?P<p2>\d+)_"
    r"(?P<exposure>auto|e\d+us)\.png$",
    re.IGNORECASE,
)


class LabelingError(RuntimeError):
    pass


class LabelingCancelled(LabelingError):
    pass


@dataclass(slots=True, frozen=True, order=True)
class CaptureKey:
    pose_id: int
    panel_2: int
    panel_1: int
    exposure: str
    ramp_sample_id: int | None = None
    robot_mode: str = "pose_id"
    conveyor_station_id: int = 0
    conveyor_position_tenths_mm: int = 0
    conveyor_direction: str = "fixed"

    @property
    def view_id(self) -> tuple[int, int]:
        return (self.pose_id, self.conveyor_station_id)


def _capture_sort_key(key: CaptureKey) -> tuple[int, int, str, int, int, int]:
    return (
        key.pose_id,
        key.conveyor_station_id,
        key.exposure,
        -1 if key.ramp_sample_id is None else key.ramp_sample_id,
        key.panel_2,
        key.panel_1,
    )


@dataclass(slots=True, frozen=True)
class CaptureRecord:
    path: Path
    key: CaptureKey
    sequence_index: int


@dataclass(slots=True, frozen=True)
class MatchedPair:
    foreground: CaptureRecord
    background: CaptureRecord


@dataclass(slots=True, frozen=True)
class LabelSource:
    name: str
    directory: Path
    is_empty: bool = False


@dataclass(slots=True, frozen=True)
class LabelingConfig:
    sources: tuple[LabelSource, ...]
    output_directory: Path
    validation_fraction: float = 0.2
    minimum_difference: int = 80
    consensus_fraction: float = 0.55
    box_margin_pixels: int = 8
    include_background_negatives: bool = True
    prefer_hardlinks: bool = True

    @property
    def pose_sources(self) -> tuple[LabelSource, ...]:
        return tuple(source for source in self.sources if not source.is_empty)

    @property
    def empty_source(self) -> LabelSource:
        empty = tuple(source for source in self.sources if source.is_empty)
        if len(empty) != 1:
            raise LabelingError("Es muss genau eine Quelle 'Leere Rutsche' geben.")
        return empty[0]

    def validated(self) -> LabelingConfig:
        output = self.output_directory.expanduser().resolve()
        if not self.pose_sources:
            raise LabelingError("Mindestens eine Pose muss angelegt sein.")
        _ = self.empty_source
        normalized_sources: list[LabelSource] = []
        names: set[str] = set()
        directories: set[Path] = set()
        for source in self.sources:
            name = source.name.strip()
            directory = source.directory.expanduser().resolve()
            if not name:
                raise LabelingError("Jeder Listeneintrag benötigt einen Namen.")
            if re.search(r'[<>:"/\\|?*]', name):
                raise LabelingError(f"Der Name '{name}' enthält ein unzulässiges Dateizeichen.")
            if name.casefold() in names:
                raise LabelingError(f"Der Name '{name}' ist doppelt vergeben.")
            if not directory.is_dir():
                raise LabelingError(f"Aufnahmeordner nicht gefunden: {directory}")
            if directory in directories:
                raise LabelingError(f"Der Aufnahmeordner ist doppelt vergeben: {directory}")
            if output.is_relative_to(directory):
                raise LabelingError("Der Ausgabeordner darf nicht in einem Eingabeordner liegen.")
            names.add(name.casefold())
            directories.add(directory)
            normalized_sources.append(LabelSource(name, directory, source.is_empty))
        if not 0.0 <= self.validation_fraction < 1.0:
            raise LabelingError("Der Validierungsanteil muss zwischen 0 und unter 1 liegen.")
        if not 1 <= self.minimum_difference <= 255:
            raise LabelingError("Die Mindestdifferenz muss zwischen 1 und 255 liegen.")
        if not 0.05 <= self.consensus_fraction <= 0.95:
            raise LabelingError("Der Konsensanteil muss zwischen 0,05 und 0,95 liegen.")
        if self.box_margin_pixels < 0:
            raise LabelingError("Der OBB-Rand darf nicht negativ sein.")
        if output.exists() and any(output.iterdir()):
            raise LabelingError(f"Der Ausgabeordner ist nicht leer: {output}")
        return LabelingConfig(
            tuple(normalized_sources),
            output,
            self.validation_fraction,
            self.minimum_difference,
            self.consensus_fraction,
            self.box_margin_pixels,
            self.include_background_negatives,
            self.prefer_hardlinks,
        )


@dataclass(slots=True, frozen=True)
class SegmentationMeasurement:
    threshold: float
    foreground_area: int
    mask: np.ndarray


@dataclass(slots=True, frozen=True)
class PoseConsensus:
    pose_id: int
    box: np.ndarray
    mask: np.ndarray
    pair_count: int
    median_iou: float
    minimum_iou: float
    flagged_count: int


@dataclass(slots=True, frozen=True)
class LabelingResult:
    output_directory: Path
    positive_images: int
    negative_images: int
    classes: int
    poses: int
    flagged_images: int
    review_directory: Path
    report_path: Path


ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


def scan_capture(directory: Path) -> dict[CaptureKey, CaptureRecord]:
    records: dict[CaptureKey, CaptureRecord] = {}
    for path in sorted(directory.glob("*.png")):
        match = CAPTURE_NAME.match(path.name)
        if match is None:
            continue
        key = CaptureKey(
            pose_id=int(match.group("pose") or match.group("angle")),
            panel_2=int(match.group("p2")),
            panel_1=int(match.group("p1")),
            exposure=match.group("exposure").lower(),
            ramp_sample_id=(None if match.group("ramp") is None else int(match.group("ramp"))),
            robot_mode="angle" if match.group("angle") is not None else "pose_id",
            conveyor_station_id=(0 if match.group("belt") is None else int(match.group("belt"))),
            conveyor_position_tenths_mm=(
                0 if match.group("position") is None else int(match.group("position"))
            ),
            conveyor_direction=match.group("direction") or "fixed",
        )
        if key in records:
            raise LabelingError(
                f"Doppelte Parameterkombination für Pose {key.pose_id}: {path.name}"
            )
        records[key] = CaptureRecord(path, key, int(match.group("index")))
    if not records:
        raise LabelingError(f"Keine passenden Aufnahmebilder in {directory} gefunden.")
    return records


def match_captures(
    foreground: dict[CaptureKey, CaptureRecord],
    background: dict[CaptureKey, CaptureRecord],
) -> list[MatchedPair]:
    missing_background = sorted(set(foreground) - set(background), key=_capture_sort_key)
    extra_background = sorted(set(background) - set(foreground), key=_capture_sort_key)
    if missing_background or extra_background:
        details: list[str] = []
        if missing_background:
            details.append(f"{len(missing_background)} Leerbilder fehlen")
        if extra_background:
            details.append(f"{len(extra_background)} Leerbilder haben kein Bauteilbild")
        modes = {
            "ramp" if key.ramp_sample_id is not None else "grid"
            for key in set(foreground) | set(background)
        }
        profile_hint = (
            " Raster- und Rampenserie wurden gemischt oder die Rampenprofile weichen ab."
            if len(modes) > 1
            else " Prüfe Pose, Exposure, Sample-IDs und das verwendete Rampenprofil."
        )
        raise LabelingError(
            "Die Aufnahmeserien sind nicht vollständig paarbar ("
            + ", ".join(details)
            + ")."
            + profile_hint
        )
    return [
        MatchedPair(foreground[key], background[key])
        for key in sorted(foreground, key=_capture_sort_key)
    ]


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise LabelingError(f"Bild konnte nicht gelesen werden: {path}")
    return image


def _largest_component(mask: np.ndarray, minimum_area: int = 250) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask)
    candidates = [
        index for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= minimum_area
    ]
    if not candidates:
        return np.zeros_like(mask)
    largest = max(candidates, key=lambda index: int(stats[index, cv2.CC_STAT_AREA]))
    component = np.where(labels == largest, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return filled


def _align_background(foreground: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Register the empty scene to the object image with a rigid 2-D transform."""
    scale = min(1.0, 640.0 / foreground.shape[1])
    size = (round(foreground.shape[1] * scale), round(foreground.shape[0] * scale))
    foreground_small = cv2.resize(foreground, size, interpolation=cv2.INTER_AREA)
    background_small = cv2.resize(background, size, interpolation=cv2.INTER_AREA)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        80,
        1e-5,
    )
    try:
        _, warp = cv2.findTransformECC(
            foreground_small,
            background_small,
            warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None,
            5,
        )
        warp[0, 2] /= scale
        warp[1, 2] /= scale
        return cv2.warpAffine(
            background,
            warp,
            (foreground.shape[1], foreground.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
    except cv2.error:
        return background


def segment_pair(
    foreground: np.ndarray,
    background: np.ndarray,
    minimum_difference: int = 8,
) -> SegmentationMeasurement:
    if foreground.shape != background.shape:
        raise LabelingError(
            f"Bildgrößen stimmen nicht überein: {foreground.shape} / {background.shape}"
        )
    background = _align_background(foreground, background)
    foreground_blurred = cv2.GaussianBlur(foreground, (5, 5), 0)
    background_blurred = cv2.GaussianBlur(background, (5, 5), 0)
    signed = foreground_blurred.astype(np.int16) - background_blurred.astype(np.int16)
    offset = float(np.median(signed))
    difference = np.abs(signed.astype(np.float32) - offset)
    median = float(np.median(difference))
    mad = float(np.median(np.abs(difference - median)))
    threshold = max(float(minimum_difference), median + 7.0 * max(1.0, 1.4826 * mad))
    mask = np.where(difference >= threshold, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    )
    mask[:12, :] = 0
    mask[-12:, :] = 0
    mask[:, :12] = 0
    mask[:, -12:] = 0
    mask = _largest_component(mask)
    return SegmentationMeasurement(threshold, int(np.count_nonzero(mask)), mask)


def _intersection_over_union(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero((left > 0) & (right > 0)))
    union = int(np.count_nonzero((left > 0) | (right > 0)))
    return intersection / union if union else 0.0


def _ordered_box(rect: tuple[tuple[float, float], tuple[float, float], float]) -> np.ndarray:
    points = cv2.boxPoints(rect).astype(np.float32)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    start = int(np.argmin(points[:, 0] + points[:, 1]))
    return np.roll(points, -start, axis=0)


def _box_from_mask(mask: np.ndarray, margin: int) -> np.ndarray:
    points = cv2.findNonZero(mask)
    if points is None or len(points) < 4:
        raise LabelingError("Aus dem Konsens konnte keine OBB erzeugt werden.")
    center, size, angle = cv2.minAreaRect(points)
    expanded = (max(1.0, size[0] + 2 * margin), max(1.0, size[1] + 2 * margin))
    box = _ordered_box((center, expanded, angle))
    height, width = mask.shape
    box[:, 0] = np.clip(box[:, 0], 0, width - 1)
    box[:, 1] = np.clip(box[:, 1], 0, height - 1)
    return box


def build_pose_consensus(
    pose_id: int,
    pairs: list[MatchedPair],
    config: LabelingConfig,
    progress: ProgressCallback | None = None,
    progress_offset: int = 0,
    progress_total: int = 1,
    cancelled: CancelCallback | None = None,
) -> tuple[PoseConsensus, list[dict[str, object]]]:
    votes: np.ndarray | None = None
    measurements: list[tuple[MatchedPair, SegmentationMeasurement]] = []
    for item_index, pair in enumerate(pairs):
        if cancelled is not None and cancelled():
            raise LabelingCancelled("Label-Erzeugung abgebrochen.")
        measurement = segment_pair(
            _read_gray(pair.foreground.path),
            _read_gray(pair.background.path),
            config.minimum_difference,
        )
        if votes is None:
            votes = np.zeros(measurement.mask.shape, dtype=np.uint16)
        votes += (measurement.mask > 0).astype(np.uint16)
        measurements.append((pair, measurement))
        if progress is not None:
            progress(
                progress_offset + item_index + 1,
                progress_total,
                f"Segmentiere Pose {pose_id}",
            )
    assert votes is not None
    required = max(1, math.ceil(len(pairs) * config.consensus_fraction))
    consensus = np.where(votes >= required, 255, 0).astype(np.uint8)
    consensus = cv2.morphologyEx(
        consensus,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
    )
    consensus = _largest_component(consensus, minimum_area=500)
    if np.count_nonzero(consensus) < 500:
        raise LabelingError(f"Pose {pose_id}: kein stabiler Bauteilkonsens gefunden.")
    box = _box_from_mask(consensus, config.box_margin_pixels)
    ious = np.asarray(
        [_intersection_over_union(item.mask, consensus) for _, item in measurements],
        dtype=np.float32,
    )
    median_iou = float(np.median(ious))
    mad = float(np.median(np.abs(ious - median_iou)))
    lower_limit = max(0.30, median_iou - max(0.12, 3.5 * mad))
    rows: list[dict[str, object]] = []
    for (pair, measurement), iou in zip(measurements, ious, strict=True):
        rows.append(
            {
                "pose_id": pose_id,
                "panel_1": pair.foreground.key.panel_1,
                "panel_2": pair.foreground.key.panel_2,
                "exposure": pair.foreground.key.exposure,
                "ramp_sample_id": pair.foreground.key.ramp_sample_id,
                "foreground_file": pair.foreground.path.name,
                "background_file": pair.background.path.name,
                "difference_threshold": round(measurement.threshold, 3),
                "foreground_area_pixels": measurement.foreground_area,
                "consensus_iou": round(float(iou), 6),
                "quality": "REVIEW" if iou < lower_limit else "PASS",
            }
        )
    flagged = sum(row["quality"] == "REVIEW" for row in rows)
    return (
        PoseConsensus(
            pose_id,
            box,
            consensus,
            len(pairs),
            median_iou,
            float(np.min(ious)),
            flagged,
        ),
        rows,
    )


def _validation_poses(pose_ids: list[int], fraction: float) -> set[int]:
    if fraction <= 0 or len(pose_ids) < 2:
        return set()
    count = max(1, min(len(pose_ids) - 1, round(len(pose_ids) * fraction)))
    positions = np.linspace(0, len(pose_ids) - 1, count + 2)[1:-1]
    return {pose_ids[int(round(position))] for position in positions}


def _normalized_obb(box: np.ndarray, width: int, height: int, class_id: int) -> str:
    normalized = box.copy().astype(np.float64)
    normalized[:, 0] /= width
    normalized[:, 1] /= height
    normalized = np.clip(normalized, 0.0, 1.0)
    coordinates = " ".join(f"{value:.6f}" for value in normalized.reshape(-1))
    return f"{class_id} {coordinates}\n"


def _link_or_copy(source: Path, destination: Path, prefer_hardlink: bool) -> None:
    if prefer_hardlink:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def _make_review_sheet(
    pose: PoseConsensus,
    pairs: list[MatchedPair],
    destination: Path,
) -> None:
    selection = np.linspace(0, len(pairs) - 1, min(6, len(pairs)), dtype=int)
    tiles: list[np.ndarray] = []
    for index in selection:
        pair = pairs[int(index)]
        image = cv2.imread(str(pair.foreground.path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        cv2.polylines(image, [np.rint(pose.box).astype(np.int32)], True, (0, 255, 0), 5)
        sample = pair.foreground.key.ramp_sample_id
        cv2.putText(
            image,
            f"Pose {pose.pose_id}  P1={pair.foreground.key.panel_1} "
            f"P2={pair.foreground.key.panel_2}" + ("" if sample is None else f"  Ramp={sample}"),
            (25, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )
        scale = 480 / image.shape[1]
        tiles.append(cv2.resize(image, (480, round(image.shape[0] * scale))))
    if not tiles:
        return
    tile_height = tiles[0].shape[0]
    blank = np.zeros((tile_height, 480, 3), dtype=np.uint8)
    while len(tiles) < 6:
        tiles.append(blank.copy())
    sheet = np.vstack((np.hstack(tiles[:3]), np.hstack(tiles[3:6])))
    cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def _make_pose_overview(
    consensuses: dict[int | tuple[int, int], PoseConsensus],
    grouped: dict[int | tuple[int, int], list[MatchedPair]],
    destination: Path,
) -> None:
    tiles: list[np.ndarray] = []
    for view_id, consensus in consensuses.items():
        pairs = grouped[view_id]
        pair = pairs[len(pairs) // 2]
        image = cv2.imread(str(pair.foreground.path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        cv2.polylines(image, [np.rint(consensus.box).astype(np.int32)], True, (0, 255, 0), 6)
        cv2.putText(
            image,
            (
                f"Pose {view_id[0]} / Band {view_id[1]}"
                if isinstance(view_id, tuple)
                else f"Pose {view_id}"
            ),
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (0, 255, 0),
            5,
            cv2.LINE_AA,
        )
        scale = 360 / image.shape[1]
        tiles.append(cv2.resize(image, (360, round(image.shape[0] * scale))))
    if not tiles:
        return
    tile_height = tiles[0].shape[0]
    blank = np.zeros((tile_height, 360, 3), dtype=np.uint8)
    columns = 5
    while len(tiles) % columns:
        tiles.append(blank.copy())
    rows = [np.hstack(tiles[index : index + columns]) for index in range(0, len(tiles), columns)]
    cv2.imwrite(
        str(destination),
        np.vstack(rows),
        [cv2.IMWRITE_JPEG_QUALITY, 94],
    )


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _filename_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-_")
    return slug or "pose"


def generate_obb_dataset(
    config: LabelingConfig,
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
) -> LabelingResult:
    config = config.validated()
    background = scan_capture(config.empty_source.directory)
    source_data: list[
        tuple[
            int,
            LabelSource,
            list[MatchedPair],
            dict[tuple[int, int], list[MatchedPair]],
        ]
    ] = []
    all_pose_ids: set[int] = set()
    for class_id, source in enumerate(config.pose_sources):
        foreground = scan_capture(source.directory)
        pairs = match_captures(foreground, background)
        grouped: dict[tuple[int, int], list[MatchedPair]] = defaultdict(list)
        for pair in pairs:
            grouped[pair.foreground.key.view_id].append(pair)
            all_pose_ids.add(pair.foreground.key.pose_id)
        source_data.append((class_id, source, pairs, grouped))

    pose_ids = sorted(all_pose_ids)
    positive_count = sum(len(pairs) for _, _, pairs, _ in source_data)
    total_steps = positive_count * 2
    if config.include_background_negatives:
        total_steps += len(background)
    completed = 0
    all_consensuses: dict[tuple[int, int, int], PoseConsensus] = {}
    report_rows: list[dict[str, object]] = []
    write_records: list[tuple[dict[str, object], MatchedPair, int, LabelSource]] = []
    for class_id, source, _, grouped in source_data:
        for view_id in sorted(grouped):
            pose_id, station_id = view_id
            consensus, rows = build_pose_consensus(
                pose_id,
                grouped[view_id],
                config,
                progress,
                completed,
                total_steps,
                cancelled,
            )
            completed += len(grouped[view_id])
            all_consensuses[(class_id, pose_id, station_id)] = consensus
            for row, pair in zip(rows, grouped[view_id], strict=True):
                row["class_id"] = class_id
                row["class_name"] = source.name
                row["source_directory"] = str(source.directory)
                row["conveyor_station_id"] = station_id
                row["conveyor_position_mm"] = (
                    pair.foreground.key.conveyor_position_tenths_mm / 10.0
                )
                row["conveyor_direction"] = pair.foreground.key.conveyor_direction
                report_rows.append(row)
                write_records.append((row, pair, class_id, source))

    output = config.output_directory
    output.mkdir(parents=True, exist_ok=True)
    review = output / "review"
    review.mkdir()
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    validation_poses = _validation_poses(pose_ids, config.validation_fraction)

    for row, pair, class_id, source in write_records:
        if cancelled is not None and cancelled():
            raise LabelingCancelled("Label-Erzeugung abgebrochen.")
        pose_id = pair.foreground.key.pose_id
        station_id = pair.foreground.key.conveyor_station_id
        split = "val" if pose_id in validation_poses else "train"
        image = _read_gray(pair.foreground.path)
        height, width = image.shape
        positive_name = (
            f"class_{class_id:03d}_{_filename_slug(source.name)}_{pair.foreground.path.name}"
        )
        positive_image = output / "images" / split / positive_name
        positive_label = output / "labels" / split / f"{Path(positive_name).stem}.txt"
        _link_or_copy(pair.foreground.path, positive_image, config.prefer_hardlinks)
        positive_label.write_text(
            _normalized_obb(
                all_consensuses[(class_id, pose_id, station_id)].box,
                width,
                height,
                class_id,
            ),
            encoding="ascii",
        )
        row["split"] = split
        row["dataset_image"] = positive_name
        row["label_file"] = positive_label.name
        completed += 1
        if progress is not None:
            progress(
                completed,
                total_steps,
                f"Schreibe {source.name} · UR-Pose {pose_id}",
            )

    negative_count = 0
    if config.include_background_negatives:
        for background_record in background.values():
            if cancelled is not None and cancelled():
                raise LabelingCancelled("Label-Erzeugung abgebrochen.")
            pose_id = background_record.key.pose_id
            split = "val" if pose_id in validation_poses else "train"
            negative_name = f"empty_{background_record.path.name}"
            negative_image = output / "images" / split / negative_name
            negative_label = output / "labels" / split / f"{Path(negative_name).stem}.txt"
            _link_or_copy(background_record.path, negative_image, config.prefer_hardlinks)
            negative_label.write_text("", encoding="ascii")
            negative_count += 1
            completed += 1
            if progress is not None:
                progress(completed, total_steps, f"Übernehme Leerbild · UR-Pose {pose_id}")

    for class_id, source, _, grouped in source_data:
        class_consensuses: dict[tuple[int, int], PoseConsensus] = {}
        prefix = f"class_{class_id:03d}_{_filename_slug(source.name)}"
        for view_id in sorted(grouped):
            pose_id, station_id = view_id
            consensus = all_consensuses[(class_id, pose_id, station_id)]
            class_consensuses[view_id] = consensus
            cv2.imwrite(
                str(review / f"{prefix}_ur_{pose_id}_belt_{station_id}_consensus_mask.png"),
                consensus.mask,
            )
            _make_review_sheet(
                consensus,
                grouped[view_id],
                review / f"{prefix}_ur_{pose_id}_belt_{station_id}_obb.jpg",
            )
        _make_pose_overview(
            class_consensuses,
            grouped,
            review / f"{prefix}_all_poses_obb.jpg",
        )

    _write_csv(output / "label_report.csv", report_rows)
    summary = {
        "format": "YOLO OBB",
        "positive_images": positive_count,
        "negative_images": negative_count,
        "validation_poses": sorted(validation_poses),
        "train_poses": sorted(set(pose_ids) - validation_poses),
        "flagged_images": sum(item.flagged_count for item in all_consensuses.values()),
        "classes": {
            str(class_id): {
                "name": source.name,
                "source_directory": str(source.directory),
                "poses": {
                    f"{pose_id}/belt-{station_id}": {
                        "images": all_consensuses[(class_id, pose_id, station_id)].pair_count,
                        "median_consensus_iou": round(
                            all_consensuses[(class_id, pose_id, station_id)].median_iou,
                            6,
                        ),
                        "minimum_consensus_iou": round(
                            all_consensuses[(class_id, pose_id, station_id)].minimum_iou,
                            6,
                        ),
                        "flagged_images": all_consensuses[
                            (class_id, pose_id, station_id)
                        ].flagged_count,
                        "obb_pixels": [
                            [round(float(x), 2), round(float(y), 2)]
                            for x, y in all_consensuses[(class_id, pose_id, station_id)].box
                        ],
                    }
                    for pose_id, station_id in sorted(grouped)
                },
            }
            for class_id, source, _, grouped in source_data
        },
        "empty_source_directory": str(config.empty_source.directory),
    }
    (output / "label_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    names_yaml = "".join(
        f"  {class_id}: {json.dumps(source.name, ensure_ascii=False)}\n"
        for class_id, source in enumerate(config.pose_sources)
    )
    (output / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n" + names_yaml,
        encoding="utf-8",
    )
    flagged = sum(item.flagged_count for item in all_consensuses.values())
    return LabelingResult(
        output,
        positive_count,
        negative_count,
        len(config.pose_sources),
        len(pose_ids),
        flagged,
        review,
        output / "label_report.csv",
    )
