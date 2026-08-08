from __future__ import annotations

import pytest

from automated_image_capture.acquisition import angle_values, conveyor_positions
from automated_image_capture.hardware.conveyor import (
    UINT32_MODULUS,
    ConveyorAdapter,
    ConveyorWorker,
    effective_direction_sign,
    full_steps_for_offset,
    signed_u32_delta,
    speed_in_full_steps,
)
from automated_image_capture.models import ConnectionState, ConveyorMove, ConveyorStatus
from automated_image_capture.settings import AppSettings


def test_default_angle_range_has_twelve_targets() -> None:
    assert angle_values(15.5, 21.0, 0.5) == tuple(range(155, 211, 5))


def test_conveyor_path_records_outward_and_return_passes() -> None:
    assert conveyor_positions(50.0, 10.0) == (
        (0, 0.0, "out"),
        (1, 10.0, "out"),
        (2, 20.0, "out"),
        (3, 30.0, "out"),
        (4, 40.0, "out"),
        (5, 50.0, "out"),
        (6, 40.0, "back"),
        (7, 30.0, "back"),
        (8, 20.0, "back"),
        (9, 10.0, "back"),
        (10, 0.0, "back"),
    )


def test_absolute_step_targets_do_not_accumulate_rounding() -> None:
    calibration = 0.32960026
    targets = [full_steps_for_offset(offset, calibration) for offset in (0, 10, 20, 30)]

    assert targets == [0, 30, 61, 91]
    assert targets[-1] * calibration == pytest.approx(29.99362366)


def test_signed_plc_position_delta_handles_udint_wrap() -> None:
    assert signed_u32_delta(5, UINT32_MODULUS - 5) == 10
    assert signed_u32_delta(UINT32_MODULUS - 5, 5) == -10


def test_direction_sign_respects_plc_reverse_setting() -> None:
    assert effective_direction_sign("right", False) == 1
    assert effective_direction_sign("left", False) == -1
    assert effective_direction_sign("right", True) == -1
    assert effective_direction_sign("left", True) == 1


def test_conveyor_speed_is_clamped_to_plc_range() -> None:
    assert speed_in_full_steps(10.0, 0.5) == 20.0
    assert speed_in_full_steps(1000.0, 0.5) == 500.0


class FakePlc:
    def __init__(self) -> None:
        self.writes: list[dict[str, object]] = []

    def write_list_by_name(self, values, *, cache_symbol_info: bool):
        assert cache_symbol_info
        self.writes.append(dict(values))
        return {name: "no error" for name in values}


def test_worker_sends_one_calibrated_relative_move_batch() -> None:
    plc = FakePlc()
    worker = ConveyorWorker(AppSettings())
    move = ConveyorMove(7, 10.0, 9.888, 30, 30, 10.0, 30.34, "right")

    worker._handle_command(plc, "move", move, object())

    assert plc.writes == [
        {
            "MAIN.GuiConveyorEnabled": False,
            "MAIN.GuiConveyorCalibrationMode": True,
            "MAIN.GuiCalibrationJogSteps": 30,
            "MAIN.GuiCalibrationJogSpeedFullStepsPerSec": 30.34,
            "MAIN.GuiCalibrationMoveRight": True,
        }
    ]


def test_adapter_computes_each_target_from_origin_not_previous_rounding() -> None:
    class DummyThread:
        @staticmethod
        def isRunning() -> bool:
            return True

    class DummyWorker:
        def __init__(self) -> None:
            self.moves: list[ConveyorMove] = []

        def enqueue_move(self, move: ConveyorMove) -> None:
            self.moves.append(move)

    adapter = ConveyorAdapter(AppSettings(conveyor_forward_direction="right"))
    worker = DummyWorker()
    adapter._worker = worker
    adapter._thread = DummyThread()
    adapter._state = ConnectionState.CONNECTED
    adapter._origin_position = 1000
    adapter._status = ConveyorStatus(
        connected=True,
        calibration_valid=True,
        mm_per_full_step=0.32960026,
        ready_to_execute=True,
        internal_position=1000,
        forward_direction="right",
        origin_position=1000,
    )

    adapter.request_offset(10.0, 10.0)
    first = worker.moves[-1]
    adapter._on_status(
        ConveyorStatus(
            connected=True,
            calibration_valid=True,
            mm_per_full_step=0.32960026,
            ready_to_execute=True,
            internal_position=1000 + first.target_full_steps * 64,
        )
    )
    adapter.request_offset(20.0, 10.0)
    second = worker.moves[-1]

    assert first.target_full_steps == 30
    assert first.delta_full_steps == 30
    assert second.target_full_steps == 61
    assert second.delta_full_steps == 31


def test_adapter_treats_plc_position_zero_as_a_real_position() -> None:
    class DummyThread:
        @staticmethod
        def isRunning() -> bool:
            return True

    class DummyWorker:
        move: ConveyorMove | None = None

        def enqueue_move(self, move: ConveyorMove) -> None:
            self.move = move

    adapter = ConveyorAdapter(AppSettings(conveyor_forward_direction="right"))
    worker = DummyWorker()
    adapter._worker = worker
    adapter._thread = DummyThread()
    adapter._state = ConnectionState.CONNECTED
    adapter._origin_position = 64
    adapter._status = ConveyorStatus(
        connected=True,
        calibration_valid=True,
        mm_per_full_step=1.0,
        ready_to_execute=True,
        internal_position=0,
    )

    adapter.request_offset(0.0, 10.0)

    assert worker.move is not None
    assert worker.move.delta_full_steps == 1
