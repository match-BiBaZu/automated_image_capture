from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from automated_image_capture.models import ConnectionState


class DeviceAdapter(QObject):
    state_changed = pyqtSignal(object)
    status_changed = pyqtSignal(object)
    error = pyqtSignal(str)
    event_message = pyqtSignal(str)

    def __init__(self, display_name: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.display_name = display_name
        self._state = ConnectionState.DISCONNECTED
        self._logger = logging.getLogger(f"hardware.{display_name.lower()}")

    @property
    def state(self) -> ConnectionState:
        return self._state

    def _set_state(self, state: ConnectionState) -> None:
        if state == self._state:
            return
        self._state = state
        self._logger.info("state=%s", state.name)
        self.state_changed.emit(state)

    def _emit_error(self, message: str) -> None:
        self._logger.error(message)
        self.error.emit(message)
        self.event_message.emit(f"{self.display_name}: {message}")

    def _emit_event(self, message: str) -> None:
        self._logger.info(message)
        self.event_message.emit(f"{self.display_name}: {message}")

    @pyqtSlot(object)
    def _forward_status(self, status: object) -> None:
        self.status_changed.emit(status)

    def connect(self) -> None:
        """Start connecting without blocking the GUI thread."""
        raise NotImplementedError

    def disconnect(self) -> None:
        """Disconnect and release resources."""
        raise NotImplementedError
