from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from automated_image_capture.hardware.light import LightAdapter
from automated_image_capture.models import ConnectionState, DiscoveredLight, LightStatus
from automated_image_capture.settings import AppSettings


async def wait_for(predicate, timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


class FakeScanner:
    @staticmethod
    async def scan(timeout: float = 5.0):
        return [SimpleNamespace(name="NEEWER-RGB660", address="AA:BB")]


class FakeLight:
    instances: list[FakeLight] = []

    def __init__(self, address: str, name: str = "") -> None:
        self.address = address
        self.name = name
        self.client = SimpleNamespace(is_connected=False)
        self.is_on = True
        self.commands: list[tuple[object, ...]] = []
        self.instances.append(self)

    async def connect(self) -> None:
        self.client.is_connected = True

    async def disconnect(self) -> None:
        self.client.is_connected = False

    async def turn_on(self) -> None:
        self.commands.append(("power", True))

    async def turn_off(self) -> None:
        self.commands.append(("power", False))

    async def set_cct(self, kelvin: int, brightness: int, gm: int = 50) -> None:
        self.commands.append(("cct", kelvin, brightness, gm))

    async def set_rgb(self, hue: int, saturation: int, brightness: int) -> None:
        self.commands.append(("hsi", hue, saturation, brightness))


@pytest.mark.asyncio
async def test_light_connect_does_not_change_output_and_commands_are_explicit(
    qtbot, monkeypatch
) -> None:
    FakeLight.instances.clear()
    fake_module = SimpleNamespace(NeewerScanner=FakeScanner, NeewerLight=FakeLight)
    monkeypatch.setitem(sys.modules, "neewerlite", fake_module)
    adapter = LightAdapter(AppSettings(auto_reconnect=False))
    statuses: list[LightStatus] = []
    adapter.status_changed.connect(statuses.append)

    adapter.connect()
    await wait_for(lambda: adapter.state is ConnectionState.CONNECTED)
    light = FakeLight.instances[-1]

    assert light.commands == []
    assert statuses[-1].power is True
    assert statuses[-1].values_are_confirmed_commands is False

    adapter.set_cct(4200, 35)
    await wait_for(lambda: bool(light.commands))
    assert light.commands == [("cct", 4200, 35, 50)]
    assert statuses[-1].values_are_confirmed_commands is True

    await adapter.shutdown()
    assert light.commands == [] or all(command[0] != "power" for command in light.commands)
    assert adapter.state is ConnectionState.DISCONNECTED


@pytest.mark.asyncio
async def test_light_hsi_values_are_clamped(qtbot, monkeypatch) -> None:
    FakeLight.instances.clear()
    fake_module = SimpleNamespace(NeewerScanner=FakeScanner, NeewerLight=FakeLight)
    monkeypatch.setitem(sys.modules, "neewerlite", fake_module)
    adapter = LightAdapter(AppSettings(auto_reconnect=False))

    adapter.connect()
    await wait_for(lambda: adapter.state is ConnectionState.CONNECTED)
    light = FakeLight.instances[-1]
    adapter.set_hsi(999, -10, 150)
    await wait_for(lambda: bool(light.commands))

    assert light.commands == [("hsi", 360, 0, 100)]
    await adapter.shutdown()


def test_second_light_selection_excludes_first_address() -> None:
    config = AppSettings(light_address="AA:01", light_2_address="AA:02")
    second = LightAdapter(
        config,
        display_name="Neewer-Licht 2",
        address_attribute="light_2_address",
        excluded_addresses=lambda: {config.light_address},
    )
    devices = [
        DiscoveredLight("NEEWER-RGB660 PRO", "AA:01", -30),
        DiscoveredLight("NEEWER-RGB660 PRO", "AA:02", -60),
    ]

    assert second._select(devices).address == "AA:02"


def test_unconfigured_second_light_does_not_select_first_address() -> None:
    config = AppSettings(light_address="AA:01")
    second = LightAdapter(
        config,
        address_attribute="light_2_address",
        excluded_addresses=lambda: {config.light_address},
    )
    devices = [
        DiscoveredLight("NEEWER-RGB660 PRO", "AA:01", -20),
        DiscoveredLight("NEEWER-RGB660 PRO", "AA:02", -50),
    ]

    assert second._select(devices).address == "AA:02"
