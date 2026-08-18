from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from automated_image_capture.labeling import (
    AnchorReviewItem,
    LabelingConfig,
    LabelSource,
    VisibilityReviewItem,
    generate_obb_dataset,
)


def _class_source(value: str) -> LabelSource:
    try:
        name, directory = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Erwartet NAME=ORDNER") from exc
    return LabelSource(name.strip(), Path(directory.strip()))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduzierbare OBB-Erzeugung ohne GUI")
    parser.add_argument("--class-source", action="append", type=_class_source, required=True)
    parser.add_argument("--empty-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-snapshot", type=Path)
    parser.add_argument("--minimum-difference", type=int, default=80)
    parser.add_argument("--consensus-fraction", type=float, default=0.55)
    parser.add_argument("--box-margin", type=int, default=8)
    parser.add_argument(
        "--trim-cast-shadows",
        action="store_true",
        help="Schattenauslaeufer vor der OBB-Anpassung geometrisch abschneiden",
    )
    parser.add_argument(
        "--exclude-recommended",
        action="store_true",
        help="Von der Sichtbarkeitsprüfung eindeutig beanstandete Bilder ausschließen",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    snapshot = args.review_snapshot
    if snapshot is not None:
        snapshot.mkdir(parents=True, exist_ok=False)
    last_progress = -1

    def report_progress(done: int, total: int, message: str) -> None:
        nonlocal last_progress
        if (done % 100 == 0 or done == total) and done != last_progress:
            last_progress = done
            print(f"FORTSCHRITT {done}/{total} {message}", flush=True)

    def review_anchors(items: tuple[AnchorReviewItem, ...]) -> bool:
        for item in items:
            if snapshot is not None:
                target = snapshot / (
                    f"anchor_class-{item.class_id:03d}_pose-{item.pose_id}_"
                    f"{item.anchor_count}-of-{item.image_count}.jpg"
                )
                shutil.copy2(item.preview_path, target)
            print(
                f"ANKER class={item.class_id} pose={item.pose_id} "
                f"count={item.anchor_count}/{item.image_count}",
                flush=True,
            )
        return True

    def review_visibility(
        items: tuple[VisibilityReviewItem, ...],
    ) -> frozenset[Path]:
        selected: set[Path] = set()
        for index, item in enumerate(items):
            if snapshot is not None:
                target = snapshot / (
                    f"visibility_{index:04d}_class-{item.class_id:03d}_"
                    f"pose-{item.pose_id}_{item.source_path.stem}.jpg"
                )
                shutil.copy2(item.preview_path, target)
            if args.exclude_recommended and item.recommended_exclude:
                selected.add(item.source_path)
        print(
            f"SICHTBARKEIT flagged={len(items)} excluded={len(selected)}",
            flush=True,
        )
        return frozenset(selected)

    sources = tuple(args.class_source) + (
        LabelSource("Leere Rutsche", args.empty_source, True),
    )
    result = generate_obb_dataset(
        LabelingConfig(
            sources=sources,
            output_directory=args.output,
            minimum_difference=args.minimum_difference,
            consensus_fraction=args.consensus_fraction,
            box_margin_pixels=args.box_margin,
            trim_cast_shadows=args.trim_cast_shadows,
        ),
        report_progress,
        anchor_review=review_anchors,
        visibility_review=review_visibility,
    )
    print(f"ERGEBNIS {result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
