from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
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

from automated_image_capture.models import ConnectionState, LightStatus, RobotStatus
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


class RobotPoseControlCard(DeviceCard):
    pose_requested = pyqtSignal(int)

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self._status = RobotStatus()

        warning = QLabel(
            "Die Auswahl kann eine reale Roboterbewegung auslösen. "
            "Arbeitsraum und Sicherheitsfreigaben vorher prüfen."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color:#b45309; font-weight:600;")
        self.content_layout.addWidget(warning)

        row = QHBoxLayout()
        row.addWidget(QLabel("Freigegebene Ansicht"))
        self.pose_selector = QComboBox()
        for pose in (155, 160, 170, 180, 190, 200, 210):
            self.pose_selector.addItem(f"{pose}°", pose)
        row.addWidget(self.pose_selector)
        self.content_layout.addLayout(row)

        self.motion_consent = QCheckBox(
            "Arbeitsraum ist frei; Bewegung für den nächsten Befehl freigeben"
        )
        self.motion_consent.toggled.connect(self._update_controls)
        self.content_layout.addWidget(self.motion_consent)

        self.move_button = QPushButton("Ausgewählte Pose anfordern")
        self.move_button.clicked.connect(self._request_pose)
        self.move_button.setEnabled(False)
        self.content_layout.addWidget(self.move_button)

        self.command_details = QLabel("Pose-Auswahlkanal nicht verbunden.")
        self.command_details.setWordWrap(True)
        self.content_layout.addWidget(self.command_details)

    def set_status(self, status: RobotStatus) -> None:
        self._status = status
        pose = "–" if status.acknowledged_pose is None else f"{status.acknowledged_pose}°"
        sequence = (
            "–" if status.acknowledged_sequence is None else str(status.acknowledged_sequence)
        )
        pending = "ja" if status.command_pending else "nein"
        self.command_details.setText(
            f"Pose-Kanal: {'verbunden' if status.command_channel_connected else 'getrennt'} · "
            f"UR-Handshake: {status.command_state}\n"
            f"Bestätigte Pose: {pose} · Quittung: {sequence} · Ausstehend: {pending}"
        )
        self._update_controls()

    def set_state(self, state: ConnectionState) -> None:
        super().set_state(state)
        self._update_controls()

    def _update_controls(self) -> None:
        if not hasattr(self, "move_button"):
            return
        program_running = "PLAYING" in self._status.program_state.upper()
        correct_program = "BIBAZU" in self._status.loaded_program.upper()
        handshake_ready = self._status.command_state_code in {1, 3, -1}
        safe_mode = self._status.safety_mode.upper() in {"NORMAL", "REDUCED"}
        can_request = (
            self._status.command_channel_connected
            and self._status.rtde_connected
            and self._status.robot_mode.upper() == "RUNNING"
            and program_running
            and correct_program
            and handshake_ready
            and safe_mode
            and not self._status.command_pending
        )
        self.pose_selector.setEnabled(can_request)
        self.motion_consent.setEnabled(can_request)
        self.move_button.setEnabled(can_request and self.motion_consent.isChecked())

    def _request_pose(self) -> None:
        if not self.move_button.isEnabled():
            return
        pose = int(self.pose_selector.currentData())
        self.motion_consent.setChecked(False)
        self.pose_requested.emit(pose)


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


class LightControlCard(DeviceCard):
    power_requested = pyqtSignal(bool)
    cct_requested = pyqtSignal(int, int)
    hsi_requested = pyqtSignal(int, int, int)

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self._updating = False

        row = QHBoxLayout()
        self.power_button = QPushButton("Licht einschalten")
        self.power_button.setCheckable(True)
        self.power_button.setEnabled(False)
        self.power_button.toggled.connect(self._power_toggled)
        self.mode = QComboBox()
        self.mode.addItems(["CCT", "HSI"])
        self.mode.setEnabled(False)
        self.mode.currentTextChanged.connect(self._values_changed)
        row.addWidget(self.power_button)
        row.addWidget(QLabel("Modus"))
        row.addWidget(self.mode)
        self.content_layout.addLayout(row)

        self.brightness = LabeledSlider("Helligkeit", 0, 100, 50, " %")
        self.cct = LabeledSlider("Farbtemperatur", 3200, 5600, 5600, " K")
        self.hue = LabeledSlider("Farbton", 0, 360, 0, "°")
        self.saturation = LabeledSlider("Sättigung", 0, 100, 100, " %")
        for control in (self.brightness, self.cct, self.hue, self.saturation):
            control.setEnabled(False)
            control.value_changed.connect(self._values_changed)
            self.content_layout.addWidget(control)

        self.command_timer = QTimer(self)
        self.command_timer.setSingleShot(True)
        self.command_timer.setInterval(150)
        self.command_timer.timeout.connect(self._send_values)
        self._update_mode_visibility("CCT")

    def set_connection_state(self, state: ConnectionState) -> None:
        self.set_state(state)
        enabled = state is ConnectionState.CONNECTED
        for control in (
            self.power_button,
            self.mode,
            self.brightness,
            self.cct,
            self.hue,
            self.saturation,
        ):
            control.setEnabled(enabled)

    def set_status(self, status: LightStatus) -> None:
        confirmed = (
            "bestätigter letzter Befehl"
            if status.values_are_confirmed_commands
            else "kein Istwert"
        )
        rssi = "–" if status.rssi is None else f"{status.rssi} dBm"
        power = "–" if status.power is None else ("Ein" if status.power else "Aus")
        self.details.setText(
            f"Gerät: {status.name}\nAdresse: {status.address} · RSSI: {rssi}\n"
            f"Leistung: {power} · Modus: {status.mode}\n"
            f"Helligkeit: {status.brightness} % · CCT: {status.cct_kelvin} K\n"
            f"HSI: {status.hue}° / {status.saturation} % · Werte: {confirmed}"
        )
        self._updating = True
        try:
            self.mode.setCurrentText(status.mode)
            self.brightness.set_value(status.brightness)
            self.cct.set_value(status.cct_kelvin)
            self.hue.set_value(status.hue)
            self.saturation.set_value(status.saturation)
            if status.power is not None:
                blocked = self.power_button.blockSignals(True)
                self.power_button.setChecked(status.power)
                self.power_button.blockSignals(blocked)
                self.power_button.setText(
                    "Licht ausschalten" if status.power else "Licht einschalten"
                )
        finally:
            self._updating = False
        self._update_mode_visibility(status.mode)

    def _power_toggled(self, enabled: bool) -> None:
        self.power_button.setText(
            "Licht ausschalten" if enabled else "Licht einschalten"
        )
        if not self._updating:
            self.power_requested.emit(enabled)

    def _values_changed(self, *_: object) -> None:
        mode = self.mode.currentText()
        self._update_mode_visibility(mode)
        if not self._updating and self.state is ConnectionState.CONNECTED:
            self.command_timer.start()

    def _update_mode_visibility(self, mode: str) -> None:
        is_cct = mode == "CCT"
        self.cct.setVisible(is_cct)
        self.hue.setVisible(not is_cct)
        self.saturation.setVisible(not is_cct)

    def _send_values(self) -> None:
        brightness = self.brightness.value()
        if self.mode.currentText() == "CCT":
            self.cct_requested.emit(self.cct.value(), brightness)
        else:
            self.hsi_requested.emit(
                self.hue.value(),
                self.saturation.value(),
                brightness,
            )
class SettingsDialog(QDialog):
    def __init__(self, config: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source_config = config
        self.setWindowTitle("Hardware-Einstellungen")
        self.setMinimumWidth(620)
        self.result_config: AppSettings | None = None
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.camera_ip = QLineEdit(config.camera_ip)
        self.robot_ip = QLineEdit(config.robot_ip)
        self.cti_path = QLineEdit(config.camera_cti_path)
        self.light_1_address = QLineEdit(config.light_address)
        self.light_2_address = QLineEdit(config.light_2_address)
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
        form.addRow("Licht 1 BLE-Adresse", self.light_1_address)
        form.addRow("Licht 2 BLE-Adresse", self.light_2_address)
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
                camera_serial=self._source_config.camera_serial,
                light_address=self.light_1_address.text().strip(),
                light_name=self._source_config.light_name,
                light_2_address=self.light_2_address.text().strip(),
                light_2_name=self._source_config.light_2_name,
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
