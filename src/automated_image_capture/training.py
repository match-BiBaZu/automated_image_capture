from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from automated_image_capture.dataset import (
    DatasetBuildConfig,
    DatasetBuildResult,
    DatasetError,
    build_curated_dataset,
    default_build_config,
    prepare_resized_training_dataset,
    verify_curated_dataset,
)

EventCallback = Callable[[dict[str, object]], None]
EVENT_PREFIX = "AIC_EVENT "


class TrainingError(RuntimeError):
    """Training cannot start or did not produce a usable checkpoint."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    dataset_directory: Path
    output_root: Path
    model: str = "yolo26n-obb.pt"
    epochs: int = 200
    patience: int = 40
    image_size: int = 640
    batch: int = 16
    device: str = "0"
    workers: int = 8
    seed: int = 42
    confidence: float = 0.25
    run_name: str | None = None

    def validated(self) -> TrainingConfig:
        dataset = self.dataset_directory.expanduser().resolve()
        output = self.output_root.expanduser().resolve()
        if not (dataset / "data.yaml").is_file():
            raise TrainingError(f"data.yaml fehlt im Datensatz: {dataset}")
        integrity = verify_curated_dataset(dataset)
        if not integrity.valid:
            raise TrainingError(
                "Datensatzprüfung fehlgeschlagen: " + "; ".join(integrity.errors[:5])
            )
        missing_splits = [
            split
            for split in ("train", "val", "test")
            if not integrity.split_counts.get(split, 0)
        ]
        if missing_splits:
            raise TrainingError(
                "Für Training und Auswertung fehlen Bilder in folgenden Splits: "
                + ", ".join(missing_splits)
            )
        if self.epochs < 1 or self.patience < 0:
            raise TrainingError("Epochen müssen positiv und Patience darf nicht negativ sein.")
        if self.image_size < 128:
            raise TrainingError("Die Eingangsgröße muss mindestens 128 Pixel betragen.")
        if self.batch < 1:
            raise TrainingError("Die Batchgröße muss mindestens 1 betragen.")
        if self.workers < 0:
            raise TrainingError("Die Anzahl der Loader-Prozesse darf nicht negativ sein.")
        if not 0 < self.confidence < 1:
            raise TrainingError("Der Konfidenzschwellwert muss zwischen 0 und 1 liegen.")
        return replace(self, dataset_directory=dataset, output_root=output)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    run_directory: Path
    best_checkpoint: Path
    summary_path: Path
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    empty_false_positive_rate: float


def default_training_config(dataset_directory: Path) -> TrainingConfig:
    return TrainingConfig(
        dataset_directory=dataset_directory,
        output_root=dataset_directory.parent / "runs",
    )


def _emit(callback: EventCallback | None, event: str, **values: object) -> None:
    if callback is not None:
        callback({"event": event, **values})


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metrics_dict(metrics: object) -> dict[str, float]:
    source = getattr(metrics, "results_dict", {})
    if not isinstance(source, dict):
        return {}
    result: dict[str, float] = {}
    for key, value in source.items():
        number = _number(value)
        if number is not None:
            result[str(key)] = number
    return result


def _per_class_metrics(metrics: object, names: dict[int, str]) -> dict[str, float]:
    component = getattr(metrics, "obb", None)
    if component is None:
        component = getattr(metrics, "box", None)
    maps = getattr(component, "maps", None)
    if maps is None:
        return {}
    result: dict[str, float] = {}
    for class_id, value in enumerate(maps):
        number = _number(value)
        if number is not None:
            result[names.get(class_id, str(class_id))] = number
    return result


def gpu_diagnostics() -> dict[str, object]:
    try:
        import torch
    except ImportError as exc:
        raise TrainingError(
            "PyTorch ist nicht installiert. Bitte zuerst 'uv sync --extra dev' ausführen."
        ) from exc
    available = bool(torch.cuda.is_available())
    devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "torch_version": torch.__version__,
        "cuda_available": available,
        "torch_cuda_version": torch.version.cuda,
        "devices": devices,
    }


def _load_manifest(dataset: Path) -> dict[str, object]:
    try:
        return json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError(f"Datensatzmanifest kann nicht gelesen werden: {exc}") from exc


def _empty_test_images(dataset: Path) -> list[Path]:
    manifest = _load_manifest(dataset)
    result: list[Path] = []
    for record in manifest.get("records", []):
        if not isinstance(record, dict):
            continue
        if record.get("excluded") or record.get("kind") != "empty" or record.get("split") != "test":
            continue
        result.append(dataset / "images" / "test" / str(record["target_name"]))
    return result


def _count_detections(result: object) -> int:
    detections = getattr(result, "obb", None)
    if detections is None:
        detections = getattr(result, "boxes", None)
    return 0 if detections is None else len(detections)


def _evaluate_empty_images(
    model: object,
    config: TrainingConfig,
    run_directory: Path,
    event: EventCallback | None,
) -> tuple[int, int, float]:
    images = _empty_test_images(config.dataset_directory)
    if not images:
        return 0, 0, 0.0
    false_positives = 0
    predictions = model.predict(  # type: ignore[attr-defined]
        source=[str(path) for path in images],
        imgsz=config.image_size,
        conf=config.confidence,
        device=config.device,
        stream=True,
        save=True,
        project=str(run_directory),
        name="empty_test_predictions",
        exist_ok=True,
        verbose=False,
    )
    for index, result in enumerate(predictions, 1):
        if _count_detections(result):
            false_positives += 1
        if index % 10 == 0 or index == len(images):
            _emit(event, "evaluation_progress", done=index, total=len(images))
    return false_positives, len(images), false_positives / len(images)


def run_training(
    config: TrainingConfig,
    event: EventCallback | None = None,
) -> TrainingResult:
    config = config.validated()
    _emit(event, "stage", name="cache", message="Bereite verlustfreien Trainingscache vor")
    resized = prepare_resized_training_dataset(
        config.dataset_directory,
        config.image_size,
        lambda done, total, _message: _emit(
            event, "cache_progress", done=done, total=total
        ),
    )
    training_dataset = resized.dataset_directory
    _emit(
        event,
        "cache_ready",
        dataset_directory=str(training_dataset),
        image_count=resized.image_count,
        reused=resized.reused,
    )
    diagnostics = gpu_diagnostics()
    _emit(event, "diagnostics", **diagnostics)
    if config.device != "cpu" and not diagnostics["cuda_available"]:
        raise TrainingError(
            "CUDA ist für PyTorch nicht verfügbar. Das Training wurde nicht versehentlich "
            "auf der CPU gestartet."
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise TrainingError(
            "Ultralytics ist nicht installiert. Bitte 'uv sync --extra dev' ausführen."
        ) from exc

    run_name = config.run_name or datetime.now().strftime("pose12_yolo26n_obb_%Y%m%d_%H%M%S")
    config.output_root.mkdir(parents=True, exist_ok=True)
    _emit(event, "stage", name="model", message=f"Lade {config.model}")
    model = YOLO(config.model)

    def epoch_finished(trainer: object) -> None:
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        epochs = int(getattr(trainer, "epochs", config.epochs))
        metrics = getattr(trainer, "metrics", {})
        serializable: dict[str, float] = {}
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                number = _number(value)
                if number is not None:
                    serializable[str(key)] = number
        _emit(event, "epoch", epoch=epoch, total=epochs, metrics=serializable)

    model.add_callback("on_train_epoch_end", epoch_finished)
    _emit(event, "stage", name="training", message="Training gestartet")
    model.train(
        data=str(training_dataset / "data.yaml"),
        epochs=config.epochs,
        patience=config.patience,
        imgsz=config.image_size,
        batch=config.batch,
        device=config.device,
        workers=config.workers,
        seed=config.seed,
        deterministic=True,
        amp=True,
        cache=False,
        project=str(config.output_root),
        name=run_name,
        exist_ok=False,
        plots=True,
        save=True,
        save_period=10,
        degrees=0.0,
        translate=0.05,
        scale=0.10,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.10,
        verbose=True,
    )
    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", config.output_root / run_name)
    run_directory = Path(save_dir).resolve()
    best_checkpoint = run_directory / "weights" / "best.pt"
    if not best_checkpoint.is_file():
        raise TrainingError(f"Training beendet, aber best.pt fehlt: {best_checkpoint}")

    _emit(event, "stage", name="validation", message="Validiere best.pt")
    best_model = YOLO(str(best_checkpoint))
    validation = best_model.val(
        data=str(training_dataset / "data.yaml"),
        split="val",
        imgsz=config.image_size,
        batch=max(1, config.batch),
        device=config.device,
        workers=config.workers,
        plots=True,
        project=str(run_directory),
        name="validation",
        exist_ok=True,
    )
    _emit(event, "stage", name="test", message="Werte unabhängigen Testsatz aus")
    test = best_model.val(
        data=str(training_dataset / "data.yaml"),
        split="test",
        imgsz=config.image_size,
        batch=max(1, config.batch),
        device=config.device,
        workers=config.workers,
        plots=True,
        project=str(run_directory),
        name="test",
        exist_ok=True,
    )
    validation_metrics = _metrics_dict(validation)
    test_metrics = _metrics_dict(test)
    false_positives, empty_count, false_positive_rate = _evaluate_empty_images(
        best_model, replace(config, dataset_directory=training_dataset), run_directory, event
    )
    names = {0: "Pose 1", 1: "Pose 2"}
    summary = {
        "format_version": 1,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": {
            **asdict(config),
            "dataset_directory": str(config.dataset_directory),
            "output_root": str(config.output_root),
        },
        "run_directory": str(run_directory),
        "training_dataset": str(training_dataset),
        "best_checkpoint": str(best_checkpoint),
        "gpu": diagnostics,
        "validation": {
            "metrics": validation_metrics,
            "per_class_map50_95": _per_class_metrics(validation, names),
        },
        "test": {
            "metrics": test_metrics,
            "per_class_map50_95": _per_class_metrics(test, names),
            "empty_images": empty_count,
            "empty_images_with_detection": false_positives,
            "empty_false_positive_rate": false_positive_rate,
            "confidence_threshold": config.confidence,
        },
    }
    summary_path = run_directory / "training_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _emit(
        event,
        "completed",
        run_directory=str(run_directory),
        best_checkpoint=str(best_checkpoint),
        summary_path=str(summary_path),
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        empty_false_positive_rate=false_positive_rate,
    )
    return TrainingResult(
        run_directory=run_directory,
        best_checkpoint=best_checkpoint,
        summary_path=summary_path,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        empty_false_positive_rate=false_positive_rate,
    )


def evaluate_checkpoint(
    config: TrainingConfig,
    event: EventCallback | None = None,
) -> TrainingResult:
    config = config.validated()
    checkpoint = Path(config.model).expanduser().resolve()
    if not checkpoint.is_file():
        raise TrainingError(f"Checkpoint nicht gefunden: {checkpoint}")
    resized = prepare_resized_training_dataset(config.dataset_directory, config.image_size)
    evaluation_dataset = resized.dataset_directory
    diagnostics = gpu_diagnostics()
    _emit(event, "diagnostics", **diagnostics)
    if config.device != "cpu" and not diagnostics["cuda_available"]:
        raise TrainingError("CUDA ist für die Modellauswertung nicht verfügbar.")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise TrainingError("Ultralytics ist nicht installiert.") from exc

    run_directory = checkpoint.parent.parent
    model = YOLO(str(checkpoint))
    evaluation_batch = max(1, config.batch)
    _emit(event, "stage", name="validation", message="Validiere vorhandenen Checkpoint")
    validation = model.val(
        data=str(evaluation_dataset / "data.yaml"),
        split="val",
        imgsz=config.image_size,
        batch=evaluation_batch,
        device=config.device,
        workers=0,
        plots=True,
        project=str(run_directory),
        name="validation",
        exist_ok=True,
    )
    _emit(event, "stage", name="test", message="Werte unabhängigen Testsatz aus")
    test = model.val(
        data=str(evaluation_dataset / "data.yaml"),
        split="test",
        imgsz=config.image_size,
        batch=evaluation_batch,
        device=config.device,
        workers=0,
        plots=True,
        project=str(run_directory),
        name="test",
        exist_ok=True,
    )
    validation_metrics = _metrics_dict(validation)
    test_metrics = _metrics_dict(test)
    false_positives, empty_count, false_positive_rate = _evaluate_empty_images(
        model,
        replace(config, dataset_directory=evaluation_dataset),
        run_directory,
        event,
    )
    names = {0: "Pose 1", 1: "Pose 2"}
    summary = {
        "format_version": 1,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "checkpoint_evaluation",
        "dataset_directory": str(config.dataset_directory),
        "evaluation_dataset": str(evaluation_dataset),
        "checkpoint": str(checkpoint),
        "image_size": config.image_size,
        "batch": evaluation_batch,
        "gpu": diagnostics,
        "validation": {
            "metrics": validation_metrics,
            "per_class_map50_95": _per_class_metrics(validation, names),
        },
        "test": {
            "metrics": test_metrics,
            "per_class_map50_95": _per_class_metrics(test, names),
            "empty_images": empty_count,
            "empty_images_with_detection": false_positives,
            "empty_false_positive_rate": false_positive_rate,
            "confidence_threshold": config.confidence,
        },
    }
    summary_path = run_directory / "training_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _emit(
        event,
        "completed",
        run_directory=str(run_directory),
        best_checkpoint=str(checkpoint),
        summary_path=str(summary_path),
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        empty_false_positive_rate=false_positive_rate,
    )
    return TrainingResult(
        run_directory=run_directory,
        best_checkpoint=checkpoint,
        summary_path=summary_path,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        empty_false_positive_rate=false_positive_rate,
    )


def _print_event(payload: dict[str, object]) -> None:
    print(EVENT_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def _result_payload(result: DatasetBuildResult) -> dict[str, object]:
    return {
        "dataset_directory": str(result.dataset_directory),
        "manifest_path": str(result.manifest_path),
        "data_yaml_path": str(result.data_yaml_path),
        "included_images": result.included_images,
        "excluded_images": result.excluded_images,
        "split_counts": result.split_counts,
        "class_counts": result.class_counts,
    }


def _parser() -> argparse.ArgumentParser:
    defaults = default_build_config()
    parser = argparse.ArgumentParser(description="Kuratierung und YOLO26-OBB-Training")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Kuratierten YOLO-OBB-Datensatz bauen")
    prepare.add_argument("--source", type=Path, default=defaults.source_dataset)
    prepare.add_argument("--output", type=Path, default=defaults.output_root)
    prepare.add_argument("--curation", type=Path, default=defaults.curation_path)
    prepare.add_argument("--copy", action="store_true", help="Kopieren statt Hardlinks verwenden")

    train = subparsers.add_parser("train", help="YOLO26n-OBB trainieren und auswerten")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--output", type=Path)
    train.add_argument("--model", default="yolo26n-obb.pt")
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--patience", type=int, default=40)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--device", default="0")
    train.add_argument("--workers", type=int, default=8)
    train.add_argument("--name")

    evaluate = subparsers.add_parser("evaluate", help="Vorhandenen Checkpoint auswerten")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--imgsz", type=int, default=640)
    evaluate.add_argument("--batch", type=int, default=2)
    evaluate.add_argument("--device", default="0")
    evaluate.add_argument("--confidence", type=float, default=0.25)

    subparsers.add_parser("diagnose", help="PyTorch- und CUDA-Status anzeigen")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = build_curated_dataset(
                DatasetBuildConfig(
                    source_dataset=args.source,
                    output_root=args.output,
                    curation_path=args.curation,
                    prefer_hardlinks=not args.copy,
                ),
                lambda done, total, message: _print_event(
                    {"event": "prepare_progress", "done": done, "total": total, "message": message}
                ),
            )
            _print_event({"event": "prepared", **_result_payload(result)})
        elif args.command == "train":
            output = args.output or args.dataset.parent / "runs"
            run_training(
                TrainingConfig(
                    dataset_directory=args.dataset,
                    output_root=output,
                    model=args.model,
                    epochs=args.epochs,
                    patience=args.patience,
                    image_size=args.imgsz,
                    batch=args.batch,
                    device=args.device,
                    workers=args.workers,
                    run_name=args.name,
                ),
                _print_event,
            )
        elif args.command == "evaluate":
            evaluate_checkpoint(
                TrainingConfig(
                    dataset_directory=args.dataset,
                    output_root=args.checkpoint.parent.parent,
                    model=str(args.checkpoint),
                    image_size=args.imgsz,
                    batch=args.batch,
                    device=args.device,
                    confidence=args.confidence,
                ),
                _print_event,
            )
        else:
            _print_event({"event": "diagnostics", **gpu_diagnostics()})
    except (DatasetError, TrainingError, OSError, ValueError, RuntimeError) as exc:
        _print_event({"event": "error", "message": str(exc) or type(exc).__name__})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
