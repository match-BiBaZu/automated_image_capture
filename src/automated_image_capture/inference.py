from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


@dataclass(slots=True, frozen=True)
class LiveInferenceConfig:
    model_path: Path
    confidence: float = 0.25
    image_size: int = 640
    max_fps: float = 5.0
    device: str = "0"

    def validated(self, *, require_model: bool = True) -> LiveInferenceConfig:
        model_path = Path(self.model_path).expanduser()
        if require_model and (not model_path.is_file() or model_path.suffix.lower() != ".pt"):
            raise ValueError(f"YOLO-Modell nicht gefunden: {model_path}")
        if not 0.01 <= self.confidence <= 1.0:
            raise ValueError("Die Konfidenz muss zwischen 0,01 und 1,00 liegen.")
        if self.image_size < 32 or self.image_size % 32:
            raise ValueError("Die Inferenz-Bildgröße muss ein Vielfaches von 32 sein.")
        if not 0.2 <= self.max_fps <= 60:
            raise ValueError("Die Inferenz-Bildrate muss zwischen 0,2 und 60 FPS liegen.")
        return replace(self, model_path=model_path)


@dataclass(slots=True, frozen=True)
class InferenceDetection:
    class_id: int
    class_name: str
    confidence: float
    corners: tuple[tuple[float, float], ...]


@dataclass(slots=True, frozen=True)
class InferenceFrame:
    image: np.ndarray
    detections: tuple[InferenceDetection, ...]
    inference_ms: float
    timestamp: float


def find_latest_model(search_root: Path | None = None) -> Path:
    """Find the newest trained best.pt without making its location mandatory."""
    root = search_root or Path.home() / "Pictures" / "Kk1_pose12_yolo26_obb" / "runs"
    if not root.exists():
        return Path()
    candidates = list(root.glob("*/weights/best.pt"))
    if not candidates:
        candidates = list(root.rglob("best.pt"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else Path()


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, f"Klasse {class_id}"))
    try:
        return str(names[class_id])
    except (IndexError, KeyError, TypeError):
        return f"Klasse {class_id}"


def extract_obb_detections(result: Any) -> tuple[InferenceDetection, ...]:
    obb = getattr(result, "obb", None)
    if obb is None or getattr(obb, "xyxyxyxy", None) is None:
        return ()
    corners = _as_numpy(obb.xyxyxyxy)
    classes = _as_numpy(obb.cls).reshape(-1)
    confidences = _as_numpy(obb.conf).reshape(-1)
    names = getattr(result, "names", {})
    detections: list[InferenceDetection] = []
    for points, class_value, confidence in zip(corners, classes, confidences, strict=False):
        flattened = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if flattened.shape != (4, 2) or not np.isfinite(flattened).all():
            continue
        class_id = int(class_value)
        detections.append(
            InferenceDetection(
                class_id=class_id,
                class_name=_class_name(names, class_id),
                confidence=float(confidence),
                corners=tuple((float(x), float(y)) for x, y in flattened),
            )
        )
    return tuple(detections)


def draw_obb_overlay(
    image: np.ndarray,
    detections: tuple[InferenceDetection, ...],
) -> np.ndarray:
    """Draw OBBs onto an RGB image and return an independent RGB array."""
    annotated = np.ascontiguousarray(image).copy()
    if annotated.ndim != 3 or annotated.shape[2] != 3:
        raise ValueError("Für das OBB-Overlay wird ein RGB-Bild erwartet.")
    height, width = annotated.shape[:2]
    thickness = max(2, round(min(height, width) / 500))
    font_scale = max(0.55, min(1.1, min(height, width) / 900))
    palette = ((34, 197, 94), (249, 115, 22), (56, 189, 248), (232, 121, 249))

    for detection in detections:
        color = palette[detection.class_id % len(palette)]
        points = np.rint(np.asarray(detection.corners)).astype(np.int32)
        cv2.polylines(annotated, [points], True, color, thickness, cv2.LINE_AA)
        label = f"{detection.class_name} {detection.confidence:.0%}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        x = int(np.clip(points[:, 0].min(), 0, max(0, width - text_width - 8)))
        y = int(np.clip(points[:, 1].min() - 7, text_height + 8, height - baseline - 2))
        cv2.rectangle(
            annotated,
            (x, y - text_height - 7),
            (x + text_width + 8, y + baseline + 3),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x + 4, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (15, 23, 42),
            thickness,
            cv2.LINE_AA,
        )
    return annotated


class LiveInferenceWorker(QThread):
    """Run YOLO OBB inference off the GUI thread and keep only the newest frame."""

    frame_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(
        self,
        config: LiveInferenceConfig,
        parent: Any = None,
        *,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config.validated()
        self._model_factory = model_factory
        self._condition = threading.Condition()
        self._latest_frame: tuple[np.ndarray, float] | None = None
        self._stopping = False

    def submit(self, image: np.ndarray, timestamp: float | None = None) -> None:
        if image.ndim != 3 or image.shape[2] != 3:
            return
        with self._condition:
            self._latest_frame = (image, time.time() if timestamp is None else timestamp)
            self._condition.notify()

    def update_runtime_settings(self, confidence: float, max_fps: float) -> None:
        candidate = replace(self._config, confidence=confidence, max_fps=max_fps).validated()
        with self._condition:
            self._config = candidate
            self._condition.notify()

    def stop(self, wait_ms: int = 10_000) -> bool:
        with self._condition:
            self._stopping = True
            self._latest_frame = None
            self._condition.notify_all()
        if QThread.currentThread() is not self:
            return self.wait(wait_ms)
        return True

    def _create_model(self) -> Any:
        if self._model_factory is not None:
            return self._model_factory(str(self._config.model_path))
        from ultralytics import YOLO

        return YOLO(str(self._config.model_path))

    def run(self) -> None:
        try:
            self.status_changed.emit("Modell wird geladen …")
            model = self._create_model()
            self.status_changed.emit(f"Bereit · {self._config.model_path.name}")
            next_allowed = 0.0
            while True:
                with self._condition:
                    while not self._stopping:
                        delay = max(0.0, next_allowed - time.monotonic())
                        if self._latest_frame is not None and delay <= 0:
                            break
                        self._condition.wait(timeout=delay if delay > 0 else 0.5)
                    if self._stopping:
                        return
                    image, timestamp = self._latest_frame
                    self._latest_frame = None
                    config = self._config

                started = time.perf_counter()
                bgr = np.ascontiguousarray(image[:, :, ::-1])
                results = model.predict(
                    source=bgr,
                    imgsz=config.image_size,
                    conf=config.confidence,
                    device=config.device,
                    verbose=False,
                )
                inference_ms = (time.perf_counter() - started) * 1000
                result = results[0] if results else None
                detections = () if result is None else extract_obb_detections(result)
                annotated = draw_obb_overlay(image, detections)
                self.frame_ready.emit(
                    InferenceFrame(annotated, detections, inference_ms, timestamp)
                )
                next_allowed = started + 1.0 / config.max_fps
        except Exception as exc:  # hardware/model/runtime errors must not take down the GUI
            message = f"Live-Erkennung fehlgeschlagen: {exc}"
            self.status_changed.emit("Fehler")
            self.error.emit(message)
