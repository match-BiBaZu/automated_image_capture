from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


class ConnectionState(StrEnum):
    DISCONNECTED = "Getrennt"
    DISCOVERING = "Suche"
    CONNECTING = "Verbinde"
    CONNECTED = "Verbunden"
    DEGRADED = "Eingeschränkt"
    ERROR = "Fehler"


@dataclass(slots=True)
class CameraStatus:
    model: str = "–"
    serial_number: str = "–"
    ip_address: str = "–"
    width: int = 0
    height: int = 0
    pixel_format: str = "–"
    camera_fps: float | None = None
    preview_fps: float = 0.0


@dataclass(slots=True)
class CameraFrame:
    image: np.ndarray
    pixel_format: str
    timestamp: float


@dataclass(slots=True)
class RobotStatus:
    rtde_connected: bool = False
    dashboard_connected: bool = False
    robot_mode: str = "–"
    safety_mode: str = "–"
    remote_control: str = "–"
    program_state: str = "–"
    loaded_program: str = "–"
    polyscope_version: str = "–"
    speed_scaling: float | None = None
    joint_positions: tuple[float, ...] = field(default_factory=tuple)
    tcp_pose: tuple[float, ...] = field(default_factory=tuple)
    command_channel_connected: bool = False
    command_state_code: int | None = None
    command_state: str = "–"
    requested_pose: int | None = None
    requested_sequence: int | None = None
    acknowledged_pose: int | None = None
    acknowledged_sequence: int | None = None
    command_pending: bool = False


@dataclass(slots=True, frozen=True)
class LightCapabilities:
    power: bool = True
    brightness: bool = True
    cct: bool = True
    hsi: bool = True
    min_cct_kelvin: int = 3200
    max_cct_kelvin: int = 5600


@dataclass(slots=True)
class LightStatus:
    name: str = "–"
    address: str = "–"
    rssi: int | None = None
    connected: bool = False
    power: bool | None = None
    mode: str = "CCT"
    brightness: int = 50
    cct_kelvin: int = 5600
    hue: int = 0
    saturation: int = 100
    capabilities: LightCapabilities = field(default_factory=LightCapabilities)
    values_are_confirmed_commands: bool = False


@dataclass(slots=True, frozen=True)
class DiscoveredLight:
    name: str
    address: str
    rssi: int | None = None
    raw: Any = field(default=None, compare=False, repr=False)
