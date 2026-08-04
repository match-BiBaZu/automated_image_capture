from __future__ import annotations

import inspect

import pytest

import automated_image_capture.hardware.robot as robot_module
from automated_image_capture.hardware.robot import (
    ALLOWED_POSES,
    CURRENT_POSE_OUTPUT_REGISTER,
    POSE_INPUT_REGISTER,
    SEQUENCE_INPUT_REGISTER,
    DashboardReadClient,
    _create_rtde_receive,
)


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
        "get loaded program",
        "is in remote control",
        "PolyscopeVersion",
    }


def test_pose_channel_is_limited_to_whitelist_and_external_registers() -> None:
    assert ALLOWED_POSES == (155, 160, 170, 180, 190, 200, 210)
    assert POSE_INPUT_REGISTER == 42
    assert SEQUENCE_INPUT_REGISTER == 43
    assert CURRENT_POSE_OUTPUT_REGISTER == 41
    assert "moveJ" not in inspect.getsource(robot_module)
    assert "moveL" not in inspect.getsource(robot_module)
    assert "sendall" in inspect.getsource(DashboardReadClient)


def test_receive_interface_uses_upper_register_range() -> None:
    captured: tuple[object, ...] = ()

    def fake_interface(*args: object) -> object:
        nonlocal captured
        captured = args
        return object()

    result = _create_rtde_receive(fake_interface, "10.10.10.10")

    assert result is not None
    assert captured == ("10.10.10.10", 10.0, [], False, True)
