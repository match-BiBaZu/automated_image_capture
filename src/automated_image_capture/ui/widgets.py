from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from automated_image_capture.models import ConnectionState
from automated_image_capture.settings import AppSettings

STATE_COLORS = {
    ConnectionState.DISCONNECTED: "#6b7280",
    ConnectionState.DISCOVERING: "#2563eb",
    ConnectionState.CONNECTING: "#2563eb",
    ConnectionState.CONNECTED: "#15803d",
    ConnectionState.DEGRADED: "#b45309",
    ConnectionState.ERROR: "#b91c1c",
}


class DeviceCard(QGroupBox):
    action_requested = pyqtSignal()

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.state = ConnectionState.DISCONNECTED
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        self.badge = QLabel()
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.action_button = QPushButton("Verbinden")
        self.action_button.clicked.connect(self.action_requested.emit)
        header.addWidget(self.badge)
        header.addStretch(1)
        header.addWidget(self.action_button)
        layout.addLayout(header)
        self.details = QLabel("Noch keine Statusdaten.")
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        self.content_layout = QVBoxLayout()
        layout.addLayout(self.content_layout)
        self.set_state(ConnectionState.DISCONNECTED)

    def set_state(self, state: ConnectionState) -> None:
        self.state = state
        color = STATE_COLORS[state]
        self.badge.setText(state.value)
        self.badge.setStyleSheet(
            f"background:{color}; color:white; border-radius:9px; padding:3px 9px; font-weight:600;"
        )
        self.action_button.setText(
            "Verbinden" if state is ConnectionState.DISCONNECTED else "Trennen"
        )


class LabeledSlider(QWidget):
    value_changed = pyqtSignal(int)

    def __init__(
        self,
        label: str,
        minimum: int,
        maximum: int,
        value: int,
        suffix: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._suffix = suffix
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        header.addWidget(QLabel(label))
        header.addStretch(1)
        self.value_label = QLabel()
        header.addWidget(self.value_label)
        layout.addLayout(header)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.slider.valueChanged.connect(self._changed)
        layout.addWidget(self.slider)
        self._changed(value)

    def _changed(self, value: int) -> None:
        self.value_label.setText(f"{value}{self._suffix}")
        self.value_changed.emit(value)

    def value(self) -> int:
        return self.slider.value()

    def set_value(self, value: int) -> None:
        blocked = self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(blocked)
        self.value_label.setText(f"{value}{self._suffix}")


class SettingsDialog(QDialog):
    def __init__(self, config: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Hardware-Einstellungen")
        self.setMinimumWidth(620)
        self.result_config: AppSettings | None = None
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.camera_ip = QLineEdit(config.camera_ip)
        self.robot_ip = QLineEdit(config.robot_ip)
        self.cti_path = QLineEdit(config.camera_cti_path)
        browse = QPushButton("Durchsuchen …")
        browse.clicked.connect(self._browse_cti)
        cti_row = QWidget()
        cti_layout = QHBoxLayout(cti_row)
        cti_layout.setContentsMargins(0, 0, 0, 0)
        cti_layout.addWidget(self.cti_path, 1)
        cti_layout.addWidget(browse)

        self.preview_fps = QSpinBox()
        self.preview_fps.setRange(1, 60)
        self.preview_fps.setValue(config.preview_max_fps)
        self.auto_reconnect = QCheckBox("Verlorene Verbindungen erneut aufbauen")
        self.auto_reconnect.setChecked(config.auto_reconnect)

        form.addRow("Baumer-IP", self.camera_ip)
        form.addRow("UR16e-IP", self.robot_ip)
        form.addRow("Baumer GenTL (.cti)", cti_row)
        form.addRow("Maximale Vorschau-FPS", self.preview_fps)
        form.addRow("Wiederverbindung", self.auto_reconnect)
        layout.addLayout(form)

        note = QLabel(
            "Änderungen werden bei der nächsten Verbindung wirksam. Seriennummer und "
            "BLE-Adresse werden nach erfolgreicher Geräteauswahl automatisch gespeichert."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_cti(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "GenTL-Producer auswählen",
            self.cti_path.text(),
            "GenTL Producer (*.cti);;Alle Dateien (*)",
        )
        if path:
            self.cti_path.setText(path)

    def _validate_and_accept(self) -> None:
        try:
            config = AppSettings(
                camera_ip=self.camera_ip.text(),
                robot_ip=self.robot_ip.text(),
                camera_cti_path=self.cti_path.text(),
                preview_max_fps=self.preview_fps.value(),
                auto_reconnect=self.auto_reconnect.isChecked(),
            ).validated()
            if not Path(config.camera_cti_path).is_file():
                raise ValueError("Der angegebene GenTL-Producer existiert nicht.")
        except ValueError as exc:
            QMessageBox.warning(self, "Ungültige Einstellung", str(exc))
            return
        self.result_config = config
        self.accept()
