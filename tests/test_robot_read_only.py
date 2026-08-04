from __future__ import annotations

import inspect

import pytest

import automated_image_capture.hardware.robot as robot_module
from automated_image_capture.hardware.robot import DashboardReadClient


def test_dashboard_client_rejects_write_commands() -> None:
    client = DashboardReadClient("127.0.0.1")

    for command in ("power on", "brake release", "play", "stop", "unlock protective stop"):
        with pytest.raises(ValueError, match="Nicht freigegebener"):
            client.query(command)


def test_robot_module_has_no_control_interface() -> None:
    source = inspect.getsource(robot_module)

    assert "rtde_control" not in source
    assert "RTDEControlInterface" not in source
    assert DashboardReadClient.ALLOWED_COMMANDS == {
        "robotmode",
        "safetymode",
        "programState",
        "is in remote control",
        "PolyscopeVersion",
    }

