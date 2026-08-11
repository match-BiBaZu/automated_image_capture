from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QSettings

from automated_image_capture.cleanup import CleanupSettings, analyze_cleanup, execute_cleanup
from automated_image_capture.labeling import scan_capture
from automated_image_capture.settings import SettingsStore
from automated_image_capture.ui.cleanup_dialog import CleanupDialog


def _capture_name(index: int, panel: int, suffix: str = ".png") -> str:
    return f"img_{index:06d}_ura-0155_p1-{panel:03d}_p2-010_auto{suffix}"


def _write_capture(root: Path, images: list[np.ndarray]) -> Path:
    capture = root / "capture_20260811_120000"
    capture.mkdir(parents=True)
    for index, image in enumerate(images, 1):
        name = _capture_name(index, index * 10)
        assert cv2.imwrite(
            str(capture / name), image, [cv2.IMWRITE_PNG_COMPRESSION, 0]
        )
        (capture / Path(name).with_suffix(".yaml")).write_text(
            f"image:\n  file: {name}\n",
            encoding="utf-8",
        )
    (capture / "capture_session.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "completed",
                "next_index": len(images),
                "total": len(images),
            }
        ),
        encoding="utf-8",
    )
    return capture


def test_analysis_is_read_only_and_counts_hardlinks_physically(tmp_path: Path) -> None:
    image = np.tile(np.arange(240, dtype=np.uint8), (160, 1))
    capture = _write_capture(tmp_path, [image, image])
    first = capture / _capture_name(1, 10)
    second = capture / _capture_name(2, 20)
    second.unlink()
    os.link(first, second)
    before = {path: path.stat().st_mtime_ns for path in capture.iterdir()}

    plan = analyze_cleanup(CleanupSettings(tmp_path, deduplicate=True))

    assert plan.managed_image_count == 2
    assert len(plan.image_actions) == 1
    assert plan.logical_bytes_before > plan.physical_bytes_before
    assert plan.estimated_savings > 0
    assert before == {path: path.stat().st_mtime_ns for path in capture.iterdir()}


def test_lossless_cleanup_preserves_pixels_and_hardlinks(tmp_path: Path) -> None:
    gray = np.tile(np.arange(240, dtype=np.uint8), (160, 1))
    rgb_gray = np.repeat(gray[..., None], 3, axis=2)
    capture = _write_capture(tmp_path, [rgb_gray, rgb_gray])
    plan = analyze_cleanup(CleanupSettings(tmp_path, output_format="png"))

    result = execute_cleanup(plan)

    paths = sorted(capture.glob("*.png"))
    assert not result.cancelled
    assert result.report_path.is_file()
    assert len(paths) == 2
    assert paths[0].stat().st_ino == paths[1].stat().st_ino
    decoded = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
    assert decoded is not None
    assert decoded.ndim == 2
    assert np.array_equal(decoded, gray)
    assert len(scan_capture(capture)) == 2
    assert not (tmp_path / ".aic_cleanup.lock").exists()
    assert not (tmp_path / ".aic_cleanup_journal.json").exists()


def test_webp_conversion_updates_capture_yaml_and_remains_scannable(tmp_path: Path) -> None:
    gray = np.full((80, 120), 73, dtype=np.uint8)
    capture = _write_capture(tmp_path, [gray])
    plan = analyze_cleanup(
        CleanupSettings(tmp_path, output_format="webp_lossless", max_edge=128)
    )

    execute_cleanup(plan)

    webp = next(capture.glob("*.webp"))
    assert not list(capture.glob("*.png"))
    decoded = cv2.imread(str(webp), cv2.IMREAD_GRAYSCALE)
    assert decoded is not None
    assert decoded.shape == gray.shape
    yaml_text = webp.with_suffix(".yaml").read_text(encoding="utf-8")
    assert webp.name in yaml_text
    assert len(scan_capture(capture)) == 1


def test_hardlink_failure_falls_back_to_valid_copies(
    monkeypatch, tmp_path: Path
) -> None:
    image = np.full((80, 120), 44, dtype=np.uint8)
    capture = _write_capture(tmp_path, [image, image])
    plan = analyze_cleanup(CleanupSettings(tmp_path))

    def fail_link(_source: Path, _target: Path) -> None:
        raise OSError("hardlinks unavailable")

    monkeypatch.setattr(os, "link", fail_link)
    execute_cleanup(plan)

    paths = sorted(capture.glob("*.png"))
    assert len(paths) == 2
    assert all(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) is not None for path in paths)
    assert len(scan_capture(capture)) == 2


def test_only_validated_resized_cache_is_scheduled_for_deletion(tmp_path: Path) -> None:
    source = tmp_path / "dataset_v1"
    for directory in (source / "images", source / "labels"):
        directory.mkdir(parents=True)
    for name in ("data.yaml", "dataset_manifest.json"):
        (source / name).write_text("{}", encoding="utf-8")
    cache = tmp_path / "dataset_v1_imgsz640"
    cache.mkdir()
    (cache / "resize_manifest.json").write_text("{}", encoding="utf-8")
    invalid = tmp_path / "orphan_imgsz640"
    invalid.mkdir()
    (invalid / "resize_manifest.json").write_text("{}", encoding="utf-8")

    plan = analyze_cleanup(CleanupSettings(tmp_path))

    assert plan.cache_directories == (cache,)
    assert invalid not in plan.cache_directories


def test_cleanup_dialog_uses_safe_defaults_and_invalidates_plan(qtbot, tmp_path: Path) -> None:
    backend = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    store = SettingsStore(backend)
    store.save_cleanup(CleanupSettings(tmp_path))
    dialog = CleanupDialog(store)
    qtbot.addWidget(dialog)

    assert dialog.output_format.currentData() == "png"
    assert dialog.max_edge.value() == 0
    assert dialog.png_compression.value() == 3
    assert not dialog.execute_button.isEnabled()

    dialog._plan = analyze_cleanup(CleanupSettings(tmp_path))
    dialog.execute_button.setEnabled(True)
    dialog.quality.setValue(91)
    assert dialog._plan is None
    assert not dialog.execute_button.isEnabled()
