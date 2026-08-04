from __future__ import annotations

import os

import pytest

from automated_image_capture.hardware.light import _as_discovered, _looks_like_neewer

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.environ.get("RUN_HARDWARE_TESTS") != "1",
        reason="Set RUN_HARDWARE_TESTS=1 for physical hardware tests",
    ),
]


async def test_rgb660_scan_and_connect_without_changing_output() -> None:
    from neewerlite import NeewerLight, NeewerScanner

    try:
        raw_devices = await NeewerScanner.scan(timeout=5.0)
    except TypeError:
        raw_devices = await NeewerScanner.scan()
    devices = [_as_discovered(device) for device in raw_devices]
    candidates = [device for device in devices if _looks_like_neewer(device)]
    assert candidates, "No RGB660/NEEWER light discovered"
    selected = max(candidates, key=lambda item: item.rssi or -999)
    light = NeewerLight(selected.address, name="RGB660")
    try:
        await light.connect()
    finally:
        await light.disconnect()

