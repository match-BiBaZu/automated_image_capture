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
    r"^img_(?P<index>\d+)_ur(?P<pose>\d+)_p1-(?P<p1>\d+)_p2-(?P<p2>\d+)_"
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
class LabelingConfig:
    foreground_directory: Path
    background_directory: Path
    output_directory: Path
    class_name: str = "Kk1"
    class_id: int = 0
    validation_fraction: float = 0.2
    minimum_difference: int = 80
    consensus_fraction: float = 0.55
    box_margin_pixels: int = 8
    include_background_negatives: bool = True
    prefer_hardlinks: bool = True

    def validated(self) -> LabelingConfig:
        foreground = self.foreground_directory.expanduser().resolve()
        background = self.background_directory.expanduser().resolve()
        output = self.output_directory.expanduser().resolve()
        if not foreground.is_dir():
            raise LabelingError(f"Bauteilordner nicht gefunden: {foreground}")
        if not background.is_dir():
            raise LabelingError(f"Leerbildordner nicht gefunden: {background}")
        if foreground == background:
            raise LabelingError("Bauteil- und Leerbildordner müssen verschieden sein.")
        if output.is_relative_to(foreground) or output.is_relative_to(background):
            raise LabelingError("Der Ausgabeordner darf nicht in einem Eingabeordner liegen.")
        if not self.class_name.strip():
            raise LabelingError("Der Klassenname darf nicht leer sein.")
        if re.search(r'[<>:"/\\|?*]', self.class_name):
            raise LabelingError("Der Klassenname enthält ein unzulässiges Dateizeichen.")
        if self.class_id < 0:
            raise LabelingError("Die Klassen-ID darf nicht negativ sein.")
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
            foreground,
            background,
            output,
            self.class_name.strip(),
            self.class_id,
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
            pose_id=int(match.group("pose")),
            panel_2=int(match.group("p2")),
            panel_1=int(match.group("p1")),
            exposure=match.group("exposure").lower(),
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
    missing_background = sorted(set(foreground) - set(background))
    extra_background = sorted(set(background) - set(foreground))
    if missing_background or extra_background:
        details: list[str] = []
        if missing_background:
            details.append(f"{len(missing_background)} Leerbilder fehlen")
        if extra_background:
            details.append(f"{len(extra_background)} Leerbilder haben kein Bauteilbild")
        raise LabelingError(
            "Die Aufnahmeserien sind nicht vollständig paarbar (" + ", ".join(details) + ")."
        )
    return [MatchedPair(foreground[key], background[key]) for key in sorted(foreground)]


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
        index
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= minimum_area
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
        cv2.putText(
            image,
            f"Pose {pose.pose_id}  P1={pair.foreground.key.panel_1} "
            f"P2={pair.foreground.key.panel_2}",
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
    consensuses: dict[int, PoseConsensus],
    grouped: dict[int, list[MatchedPair]],
    destination: Path,
) -> None:
    tiles: list[np.ndarray] = []
    for pose_id, consensus in consensuses.items():
        pairs = grouped[pose_id]
        pair = pairs[len(pairs) // 2]
        image = cv2.imread(str(pair.foreground.path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        cv2.polylines(image, [np.rint(consensus.box).astype(np.int32)], True, (0, 255, 0), 6)
        cv2.putText(
            image,
            f"Pose {pose_id}",
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


def generate_obb_dataset(
    config: LabelingConfig,
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
) -> LabelingResult:
    config = config.validated()
    foreground = scan_capture(config.foreground_directory)
    background = scan_capture(config.background_directory)
    pairs = match_captures(foreground, background)
    grouped: dict[int, list[MatchedPair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.foreground.key.pose_id].append(pair)
    pose_ids = sorted(grouped)
    total_steps = len(pairs) + len(pairs) * (2 if config.include_background_negatives else 1)
    completed = 0
    consensuses: dict[int, PoseConsensus] = {}
    report_rows: list[dict[str, object]] = []
    for pose_id in pose_ids:
        consensus, rows = build_pose_consensus(
            pose_id,
            grouped[pose_id],
            config,
            progress,
            completed,
            total_steps,
            cancelled,
        )
        completed += len(grouped[pose_id])
        consensuses[pose_id] = consensus
        report_rows.extend(rows)

    output = config.output_directory
    output.mkdir(parents=True, exist_ok=True)
    review = output / "review"
    review.mkdir()
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    validation_poses = _validation_poses(pose_ids, config.validation_fraction)

    negative_count = 0
    for row, pair in zip(report_rows, pairs, strict=True):
        if cancelled is not None and cancelled():
            raise LabelingCancelled("Label-Erzeugung abgebrochen.")
        pose_id = pair.foreground.key.pose_id
        split = "val" if pose_id in validation_poses else "train"
        image = _read_gray(pair.foreground.path)
        height, width = image.shape
        positive_name = f"{config.class_name}_{pair.foreground.path.name}"
        positive_image = output / "images" / split / positive_name
        positive_label = output / "labels" / split / f"{Path(positive_name).stem}.txt"
        _link_or_copy(pair.foreground.path, positive_image, config.prefer_hardlinks)
        positive_label.write_text(
            _normalized_obb(consensuses[pose_id].box, width, height, config.class_id),
            encoding="ascii",
        )
        row["split"] = split
        row["dataset_image"] = positive_name
        row["label_file"] = positive_label.name
        completed += 1
        if progress is not None:
            progress(completed, total_steps, f"Schreibe Labels für Pose {pose_id}")
        if config.include_background_negatives:
            negative_name = f"background_{pair.background.path.name}"
            negative_image = output / "images" / split / negative_name
            negative_label = output / "labels" / split / f"{Path(negative_name).stem}.txt"
            _link_or_copy(pair.background.path, negative_image, config.prefer_hardlinks)
            negative_label.write_text("", encoding="ascii")
            negative_count += 1
            completed += 1
            if progress is not None:
                progress(completed, total_steps, f"Übernehme Leerbilder für Pose {pose_id}")

    for pose_id in pose_ids:
        consensus = consensuses[pose_id]
        cv2.imwrite(str(review / f"pose_{pose_id}_consensus_mask.png"), consensus.mask)
        _make_review_sheet(consensus, grouped[pose_id], review / f"pose_{pose_id}_obb.jpg")
    _make_pose_overview(consensuses, grouped, review / "all_poses_obb.jpg")

    _write_csv(output / "label_report.csv", report_rows)
    summary = {
        "format": "YOLO OBB",
        "class_id": config.class_id,
        "class_name": config.class_name,
        "positive_images": len(pairs),
        "negative_images": negative_count,
        "validation_poses": sorted(validation_poses),
        "train_poses": sorted(set(pose_ids) - validation_poses),
        "flagged_images": sum(item.flagged_count for item in consensuses.values()),
        "poses": {
            str(pose_id): {
                "images": item.pair_count,
                "median_consensus_iou": round(item.median_iou, 6),
                "minimum_consensus_iou": round(item.minimum_iou, 6),
                "flagged_images": item.flagged_count,
                "obb_pixels": [[round(float(x), 2), round(float(y), 2)] for x, y in item.box],
            }
            for pose_id, item in consensuses.items()
        },
    }
    (output / "label_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\n"
        f"names:\n  {config.class_id}: {json.dumps(config.class_name, ensure_ascii=False)}\n",
        encoding="utf-8",
    )
    flagged = sum(item.flagged_count for item in consensuses.values())
    return LabelingResult(
        output,
        len(pairs),
        negative_count,
        len(pose_ids),
        flagged,
        review,
        output / "label_report.csv",
    )
