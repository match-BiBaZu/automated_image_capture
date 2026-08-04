from __future__ import annotations

import os

import pytest

from automated_image_capture.hardware.camera import convert_to_rgb

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        os.environ.get("RUN_HARDWARE_TESTS") != "1",
        reason="Set RUN_HARDWARE_TESTS=1 for physical hardware tests",
    ),
]


def test_baumer_acquires_100_frames() -> None:
    from harvesters.core import Harvester

    producer = os.environ.get(
        "BAUMER_CTI", r"C:\Program Files\Baumer Camera Explorer\bgapi2_gige.cti"
    )
    harvester = Harvester()
    acquirer = None
    try:
        harvester.add_file(producer, check_existence=True, check_validity=True)
        harvester.update()
        assert harvester.device_info_list, "No camera discovered"
        acquirer = harvester.create(0)
        acquirer.start()
        for _ in range(100):
            with acquirer.fetch(timeout=2.0) as buffer:
                component = buffer.payload.components[0]
                assert component.width > 0
                assert component.height > 0
                assert component.data.size > 0
                preview = convert_to_rgb(
                    component.data,
                    int(component.width),
                    int(component.height),
                    str(component.data_format),
                )
                assert preview.shape == (component.height, component.width, 3)
    finally:
        if acquirer is not None:
            acquirer.stop()
            acquirer.destroy()
        harvester.reset()
