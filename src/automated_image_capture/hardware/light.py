from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from automated_image_capture.hardware.base import DeviceAdapter
from automated_image_capture.models import (
    ConnectionState,
    DiscoveredLight,
    LightCapabilities,
    LightStatus,
)
from automated_image_capture.settings import AppSettings


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    result = getattr(value, name, default)
    return result() if callable(result) else result


def _as_discovered(device: Any) -> DiscoveredLight:
    name = str(_attribute(device, "name", "") or "Unbenanntes BLE-Gerät")
    address = str(_attribute(device, "address", "") or "")
    rssi_raw = _attribute(device, "rssi", None)
    if rssi_raw is None:
        advertisement = _attribute(device, "advertisement_data", None)
        rssi_raw = _attribute(advertisement, "rssi", None)
    try:
        rssi = int(rssi_raw) if rssi_raw is not None else None
    except (TypeError, ValueError):
        rssi = None
    return DiscoveredLight(name=name, address=address, rssi=rssi, raw=device)


def _looks_like_neewer(light: DiscoveredLight) -> bool:
    name = light.name.upper()
    return any(token in name for token in ("NEEWER", "RGB660", "RGB 660", "NW-", "ZN-"))


class LightAdapter(DeviceAdapter):
    devices_discovered = pyqtSignal(object)

    def __init__(self, config: AppSettings, parent: QObject | None = None) -> None:
        super().__init__("Neewer-Licht", parent)
        self.config = config
        self._light: Any = None
        self._status = LightStatus()
        self._desired_connection = False
        self._operation_task: asyncio.Task[Any] | None = None
        self._reconnect_task: asyncio.Task[Any] | None = None
        self._monitor = QTimer(self)
        self._monitor.setInterval(2000)
        self._monitor.timeout.connect(self._check_connection)
        self._monitor.start()

    def _start_task(self, coroutine: Awaitable[Any]) -> asyncio.Task[Any] | None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            if inspect.iscoroutine(coroutine):
                coroutine.close()
            self._emit_error("Kein laufender Qt/Asyncio-Eventloop für Bluetooth verfügbar.")
            return None
        return loop.create_task(coroutine)

    def connect(self) -> None:
        if self.state in {
            ConnectionState.DISCOVERING,
            ConnectionState.CONNECTING,
            ConnectionState.CONNECTED,
        }:
            return
        self._desired_connection = True
        self._operation_task = self._start_task(self._discover_and_connect())

    async def _scan(self) -> list[DiscoveredLight]:
        from neewerlite import NeewerScanner

        try:
            raw_devices = await NeewerScanner.scan(timeout=5.0)
        except TypeError:
            raw_devices = await NeewerScanner.scan()
        return [_as_discovered(device) for device in raw_devices]

    def _select(self, devices: list[DiscoveredLight]) -> DiscoveredLight:
        if self.config.light_address:
            for device in devices:
                if device.address.casefold() == self.config.light_address.casefold():
                    return device
        supported = [device for device in devices if _looks_like_neewer(device)]
        if not supported:
            raise RuntimeError(
                "Kein RGB660/NEEWER-BLE-Gerät gefunden. Panel einschalten, Bluetooth-Symbol "
                "aktivieren und die Smartphone-App trennen."
            )
        return sorted(supported, key=lambda item: item.rssi or -999, reverse=True)[0]

    async def _discover_and_connect(self) -> None:
        try:
            self._set_state(ConnectionState.DISCOVERING)
            self._emit_event("Suche fünf Sekunden lang nach BLE-Leuchten …")
            devices = await self._scan()
            self.devices_discovered.emit(devices)
            selected = self._select(devices)
            await self._connect_selected(selected)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._set_state(ConnectionState.ERROR)
            self._emit_error(str(exc) or type(exc).__name__)
            self._schedule_reconnect()

    async def _connect_selected(self, selected: DiscoveredLight) -> None:
        from neewerlite import NeewerLight

        self._set_state(ConnectionState.CONNECTING)
        self._emit_event(f"Verbinde {selected.name} ({selected.address}) …")
        profile_name = "RGB660" if "660" in selected.name.upper() else selected.name
        light = NeewerLight(selected.address, name=profile_name)
        await light.connect()
        self._light = light
        self._status.name = selected.name
        self._status.address = selected.address
        self._status.rssi = selected.rssi
        self._status.connected = True
        self._status.power = _attribute(light, "is_on", None)
        self._status.capabilities = LightCapabilities()
        self._status.values_are_confirmed_commands = False
        self.status_changed.emit(self._status)
        self._set_state(ConnectionState.CONNECTED)
        self._emit_event("Verbunden; beim Verbindungsaufbau wurden keine Lichtwerte verändert.")

    def disconnect(self) -> None:
        self._desired_connection = False
        if self._reconnect_task is not None:
            self._reconnect_task.cancel()
            self._reconnect_task = None
        self._operation_task = self._start_task(self.disconnect_async())

    async def disconnect_async(self) -> None:
        light, self._light = self._light, None
        if light is not None:
            with contextlib.suppress(Exception):
                await light.disconnect()
        self._status.connected = False
        self._status.values_are_confirmed_commands = False
        self.status_changed.emit(self._status)
        self._set_state(ConnectionState.DISCONNECTED)
        self._emit_event("Bluetooth getrennt; der Lichtzustand wurde nicht verändert.")

    async def shutdown(self) -> None:
        self._desired_connection = False
        for task in (self._operation_task, self._reconnect_task):
            if task is not None and not task.done():
                task.cancel()
        self._operation_task = None
        self._reconnect_task = None
        await self.disconnect_async()

    def _check_connection(self) -> None:
        if self._light is None or self.state is not ConnectionState.CONNECTED:
            return
        client = _attribute(self._light, "client", None)
        connected = bool(client is not None and _attribute(client, "is_connected", False))
        if connected is False:
            self._status.connected = False
            self.status_changed.emit(self._status)
            self._set_state(ConnectionState.ERROR)
            self._emit_error("Bluetooth-Verbindung zum Panel wurde unterbrochen.")
            self._light = None
            self._schedule_reconnect()

    def _schedule_reconnect(self) -> None:
        if not self._desired_connection or not self.config.auto_reconnect:
            return
        if self._reconnect_task is not None and not self._reconnect_task.done():
            return
        self._reconnect_task = self._start_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        delay = 1.0
        while self._desired_connection and self._light is None:
            await asyncio.sleep(delay)
            try:
                devices = await self._scan()
                selected = self._select(devices)
                await self._connect_selected(selected)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._emit_event(f"Wiederverbindung fehlgeschlagen: {exc}")
                delay = min(delay * 2.0, 30.0)

    def set_power(self, enabled: bool) -> None:
        method = "turn_on" if enabled else "turn_off"

        async def command() -> None:
            await getattr(self._light, method)()
            self._status.power = enabled

        self._run_command(command, f"Licht {'ein' if enabled else 'aus'}")

    def set_brightness(self, brightness: int) -> None:
        brightness = max(0, min(100, int(brightness)))

        async def command() -> None:
            if self._status.mode == "HSI":
                await self._light.set_rgb(
                    self._status.hue, self._status.saturation, brightness
                )
            else:
                await self._light.set_cct(self._status.cct_kelvin, brightness, gm=50)
            self._status.brightness = brightness

        self._run_command(command, f"Helligkeit {brightness} %")

    def set_cct(self, kelvin: int, brightness: int) -> None:
        kelvin = max(3200, min(5600, int(kelvin)))
        brightness = max(0, min(100, int(brightness)))

        async def command() -> None:
            await self._light.set_cct(kelvin, brightness, gm=50)
            self._status.mode = "CCT"
            self._status.cct_kelvin = kelvin
            self._status.brightness = brightness

        self._run_command(command, f"CCT {kelvin} K bei {brightness} %")

    def set_hsi(self, hue: int, saturation: int, brightness: int) -> None:
        hue = max(0, min(360, int(hue)))
        saturation = max(0, min(100, int(saturation)))
        brightness = max(0, min(100, int(brightness)))

        async def command() -> None:
            await self._light.set_rgb(hue, saturation, brightness)
            self._status.mode = "HSI"
            self._status.hue = hue
            self._status.saturation = saturation
            self._status.brightness = brightness

        self._run_command(command, f"HSI {hue}°/{saturation} % bei {brightness} %")

    def _run_command(
        self,
        operation: Callable[[], Awaitable[None]],
        description: str,
    ) -> None:
        if self._light is None or self.state is not ConnectionState.CONNECTED:
            self._emit_error("Lichtbefehl verworfen: Das Panel ist nicht verbunden.")
            return

        async def execute() -> None:
            try:
                await operation()
                self._status.values_are_confirmed_commands = True
                self.status_changed.emit(self._status)
                self._emit_event(f"Bestätigter Befehl: {description}.")
            except Exception as exc:
                self._set_state(ConnectionState.ERROR)
                self._emit_error(f"Lichtbefehl fehlgeschlagen: {exc}")
                self._status.connected = False
                self.status_changed.emit(self._status)
                self._light = None
                self._schedule_reconnect()

        self._operation_task = self._start_task(execute())
