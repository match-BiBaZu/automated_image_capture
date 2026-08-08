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


class RobotCommandMode(StrEnum):
    POSE_ID = "pose_id"
    ANGLE = "angle"


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
    exposure_time_us: float | None = None
    exposure_min_us: float | None = None
    exposure_max_us: float | None = None
    exposure_writable: bool = False
    exposure_auto: str = "–"
    gain: float | None = None


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
    command_mode: RobotCommandMode = RobotCommandMode.POSE_ID
    requested_raw_value: int | None = None
    acknowledged_raw_value: int | None = None
    requested_angle_deg: float | None = None
    acknowledged_angle_deg: float | None = None


@dataclass(slots=True, frozen=True)
class ConveyorMove:
    sequence: int
    requested_offset_mm: float
    actual_offset_mm: float
    target_full_steps: int
    delta_full_steps: int
    speed_mm_per_s: float
    speed_full_steps_per_s: float
    plc_direction: str
    start_internal_position: int | None = None
    commanded_at: float | None = None


@dataclass(slots=True)
class ConveyorStatus:
    connected: bool = False
    calibration_valid: bool = False
    mm_per_full_step: float = 0.0
    ready_to_execute: bool = False
    busy: bool = False
    warning: bool = False
    error: bool = False
    status_code: int = 0
    internal_position: int | None = None
    conveyor_reverse: bool = False
    control_enabled: bool = False
    preparing_drive: bool = False
    wc_state: bool = False
    info_data_state: int = 0
    forward_direction: str = ""
    origin_position: int | None = None
    logical_offset_mm: float | None = None
    requested_offset_mm: float | None = None
    actual_target_offset_mm: float | None = None
    requested_sequence: int | None = None
    completed_sequence: int | None = None
    completed_internal_position: int | None = None
    completed_at: float | None = None
    movement_started_at: float | None = None
    sampled_at: float | None = None
    position_feedback_verified: bool = False
    last_move: ConveyorMove | None = None


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
    last_command_confirmed_at: float | None = None
    last_command_duration_ms: float | None = None


@dataclass(slots=True, frozen=True)
class DiscoveredLight:
    name: str
    address: str
    rssi: int | None = None
    raw: Any = field(default=None, compare=False, repr=False)
