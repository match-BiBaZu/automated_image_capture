from __future__ import annotations

import pytest
from PyQt6.QtCore import QSettings

from automated_image_capture.settings import AppSettings, SettingsStore


def test_settings_round_trip(tmp_path) -> None:
    backend = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    store = SettingsStore(backend)
    expected = AppSettings(
        camera_ip="169.254.117.70",
        robot_ip="10.10.10.10",
        camera_cti_path=str(tmp_path / "producer.cti"),
        camera_serial="1234",
        light_address="AA:BB:CC:DD:EE:FF",
        light_name="RGB660",
        preview_max_fps=12,
        auto_reconnect=False,
    )

    store.save(expected)

    assert store.load() == expected


@pytest.mark.parametrize("field", ["camera_ip", "robot_ip"])
def test_invalid_ip_is_rejected(field: str) -> None:
    values = {field: "not-an-ip"}
    with pytest.raises(ValueError):
        AppSettings(**values).validated()


def test_invalid_preview_rate_is_rejected() -> None:
    with pytest.raises(ValueError, match="Bildrate"):
        AppSettings(preview_max_fps=0).validated()

