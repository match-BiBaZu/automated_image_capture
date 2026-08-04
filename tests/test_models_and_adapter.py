from __future__ import annotations

from PyQt6.QtTest import QSignalSpy

from automated_image_capture.hardware.base import DeviceAdapter
from automated_image_capture.models import ConnectionState


class FakeAdapter(DeviceAdapter):
    def __init__(self) -> None:
        super().__init__("Fake")
        self.connect_calls = 0
        self.disconnect_calls = 0

    def connect(self) -> None:
        self.connect_calls += 1
        self._set_state(ConnectionState.CONNECTING)

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._set_state(ConnectionState.DISCONNECTED)


def test_adapter_emits_only_real_state_changes(qtbot) -> None:
    adapter = FakeAdapter()
    qtbot.addWidget(adapter) if hasattr(adapter, "show") else None
    spy = QSignalSpy(adapter.state_changed)

    adapter.connect()
    adapter._set_state(ConnectionState.CONNECTING)
    adapter._set_state(ConnectionState.CONNECTED)
    adapter.disconnect()

    assert [args[0] for args in spy] == [
        ConnectionState.CONNECTING,
        ConnectionState.CONNECTED,
        ConnectionState.DISCONNECTED,
    ]


def test_errors_are_device_local() -> None:
    first = FakeAdapter()
    second = FakeAdapter()
    first._set_state(ConnectionState.ERROR)

    assert first.state is ConnectionState.ERROR
    assert second.state is ConnectionState.DISCONNECTED

