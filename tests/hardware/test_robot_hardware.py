from __future__ import annotations

import os
import time

import pytest

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        os.environ.get("RUN_HARDWARE_TESTS") != "1",
        reason="Set RUN_HARDWARE_TESTS=1 for physical hardware tests",
    ),
]


def test_ur_receive_only_for_30_seconds() -> None:
    from rtde_receive import RTDEReceiveInterface

    robot = RTDEReceiveInterface(os.environ.get("UR_IP", "10.10.10.10"), 10.0)
    try:
        deadline = time.monotonic() + 30.0
        samples = 0
        while time.monotonic() < deadline:
            assert len(robot.getActualQ()) == 6
            assert len(robot.getActualTCPPose()) == 6
            robot.getRobotMode()
            robot.getSafetyMode()
            samples += 1
            time.sleep(0.1)
        assert samples >= 250
    finally:
        robot.disconnect()

