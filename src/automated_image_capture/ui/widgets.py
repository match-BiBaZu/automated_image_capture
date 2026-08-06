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
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from automated_image_capture.acquisition import (
    AcquisitionSettings,
    build_capture_points,
)
from automated_image_capture.hardware.robot import ALLOWED_POSES
from automated_image_capture.models import CameraStatus, ConnectionState, LightStatus, RobotStatus
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


class CameraControlCard(DeviceCard):
    exposure_requested = pyqtSignal(float)

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self._status = CameraStatus()

        row = QHBoxLayout()
        row.addWidget(QLabel("Belichtungszeit"))
        self.exposure_time = QSpinBox()
        self.exposure_time.setRange(1, 2_000_000_000)
        self.exposure_time.setSuffix(" µs")
        self.exposure_time.setSingleStep(100)
        self.exposure_time.setEnabled(False)
        row.addWidget(self.exposure_time, 1)
        self.apply_exposure_button = QPushButton("Übernehmen")
        self.apply_exposure_button.setEnabled(False)
        self.apply_exposure_button.clicked.connect(self._apply_exposure)
        row.addWidget(self.apply_exposure_button)
        self.content_layout.addLayout(row)

        self.exposure_hint = QLabel(
            "Die Einstellung gilt bis zum Trennen der Kamera; danach wird der "
            "ursprüngliche Wert wiederhergestellt."
        )
        self.exposure_hint.setWordWrap(True)
        self.content_layout.addWidget(self.exposure_hint)

    def set_status(self, status: CameraStatus) -> None:
        self._status = status
        minimum = max(1, round(status.exposure_min_us or 1))
        maximum = max(minimum, round(status.exposure_max_us or 2_000_000_000))
        self.exposure_time.setRange(minimum, maximum)
        if status.exposure_time_us is not None and not self.exposure_time.hasFocus():
            self.exposure_time.setValue(round(status.exposure_time_us))
        self._update_exposure_controls()

    def set_state(self, state: ConnectionState) -> None:
        super().set_state(state)
        self._update_exposure_controls()

    def _update_exposure_controls(self) -> None:
        if not hasattr(self, "exposure_time"):
            return
        enabled = self.state is ConnectionState.CONNECTED and self._status.exposure_writable
        self.exposure_time.setEnabled(enabled)
        self.apply_exposure_button.setEnabled(enabled)
        if (
            self._status.exposure_auto.lower() not in {"off", "–", "none"}
            and enabled
        ):
            self.exposure_hint.setText(
                "Übernehmen deaktiviert ExposureAuto für diese Verbindung. Beim Trennen "
                "wird der ursprüngliche Automatikmodus wiederhergestellt."
            )
        elif self._status.exposure_auto.lower() not in {"off", "–", "none"}:
            self.exposure_hint.setText(
                "ExposureAuto ist aktiv und kann von der Kamera aktuell nicht "
                "deaktiviert werden."
            )
        else:
            self.exposure_hint.setText(
                "Die Einstellung gilt bis zum Trennen der Kamera; danach wird der "
                "ursprüngliche Wert wiederhergestellt."
            )

    def _apply_exposure(self) -> None:
        if self.apply_exposure_button.isEnabled():
            self.exposure_requested.emit(float(self.exposure_time.value()))


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
        for pose in ALLOWED_POSES:
            self.pose_selector.addItem(str(pose), pose)
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
        pose = "–" if status.acknowledged_pose is None else str(status.acknowledged_pose)
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
            "bestätigter letzter Befehl" if status.values_are_confirmed_commands else "kein Istwert"
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
        self.power_button.setText("Licht ausschalten" if enabled else "Licht einschalten")
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


class AcquisitionCard(QGroupBox):
    configure_requested = pyqtSignal()
    start_requested = pyqtSignal()
    resume_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Automatisierte Bildaufnahme", parent)
        layout = QVBoxLayout(self)
        self.summary = QLabel("Noch keine Aufnahmeeinstellungen geladen.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self._running = False
        self._resume_available = False
        buttons = QGridLayout()
        self.configure_button = QPushButton("Aufnahme konfigurieren …")
        self.start_button = QPushButton("Aufnahme starten")
        self.resume_button = QPushButton("Aufnahme fortsetzen")
        self.stop_button = QPushButton("Aufnahme stoppen")
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.configure_button.clicked.connect(self.configure_requested.emit)
        self.start_button.clicked.connect(self.start_requested.emit)
        self.resume_button.clicked.connect(self.resume_requested.emit)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        buttons.addWidget(self.configure_button, 0, 0)
        buttons.addWidget(self.start_button, 0, 1)
        buttons.addWidget(self.resume_button, 1, 0)
        buttons.addWidget(self.stop_button, 1, 1)
        layout.addLayout(buttons)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.status = QLabel("Bereit.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def set_settings(self, settings: AcquisitionSettings) -> None:
        count = len(build_capture_points(settings))
        exposure = (
            f" · Belichtung {settings.exposure_start_us}–{settings.exposure_end_us} µs"
            if settings.exposure_enabled
            else " · Belichtung unverändert"
        )
        self.summary.setText(
            f"UR-Pose {settings.pose_start} bis {settings.pose_end} · "
            f"Panel 1 {settings.light_1_start}–{settings.light_1_end} % / "
            f"Panel 2 {settings.light_2_start}–{settings.light_2_end} %"
            f"{exposure}\n{count} Bilder · Ziel: {settings.output_directory}"
        )

    def set_running(self, running: bool) -> None:
        self._running = running
        self.configure_button.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.resume_button.setEnabled(not running and self._resume_available)
        self.stop_button.setEnabled(running)

    def set_resume_available(self, available: bool) -> None:
        self._resume_available = available
        self.resume_button.setEnabled(available and not self._running)

    def set_progress(self, current: int, total: int) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(current)
        self.progress.setFormat(f"{current} / {total} Bilder")


class AcquisitionDialog(QDialog):
    def __init__(
        self,
        config: AcquisitionSettings,
        camera_status: object | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Automatisierte Bildaufnahme konfigurieren")
        self.setMinimumWidth(700)
        self.result_config: AcquisitionSettings | None = None
        layout = QVBoxLayout(self)

        output_form = QFormLayout()
        self.output_directory = QLineEdit(str(config.output_directory))
        browse = QPushButton("Durchsuchen …")
        browse.clicked.connect(self._browse_output)
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_directory, 1)
        output_layout.addWidget(browse)
        output_form.addRow("Speicherort", output_row)
        layout.addLayout(output_form)

        ranges = QGroupBox("Variationen")
        ranges_form = QFormLayout(ranges)
        self.pose_start = self._pose_combo(config.pose_start)
        self.pose_end = self._pose_combo(config.pose_end)
        ranges_form.addRow("UR Startpose", self.pose_start)
        ranges_form.addRow("UR Endpose", self.pose_end)

        self.light_1_start, self.light_1_end, self.light_1_step = self._range_row(
            0, 100, config.light_1_start, config.light_1_end, config.light_1_step, "%"
        )
        ranges_form.addRow(
            "Panel 1 Start / Ende / Schritt",
            self._row_widget(self.light_1_start, self.light_1_end, self.light_1_step),
        )
        self.light_2_start, self.light_2_end, self.light_2_step = self._range_row(
            0, 100, config.light_2_start, config.light_2_end, config.light_2_step, "%"
        )
        ranges_form.addRow(
            "Panel 2 Start / Ende / Schritt",
            self._row_widget(self.light_2_start, self.light_2_end, self.light_2_step),
        )

        self.exposure_enabled = QCheckBox("Belichtungszeit zusätzlich variieren")
        self.exposure_enabled.setChecked(config.exposure_enabled)
        exposure_writable = bool(getattr(camera_status, "exposure_writable", False))
        self.exposure_enabled.setEnabled(exposure_writable)
        self.exposure_start, self.exposure_end, self.exposure_step = self._range_row(
            1,
            10_000_000,
            config.exposure_start_us,
            config.exposure_end_us,
            config.exposure_step_us,
            " µs",
        )
        exposure_row = self._row_widget(
            self.exposure_start,
            self.exposure_end,
            self.exposure_step,
        )
        ranges_form.addRow(self.exposure_enabled)
        ranges_form.addRow("Belichtung Start / Ende / Schritt", exposure_row)
        if exposure_writable:
            minimum = getattr(camera_status, "exposure_min_us", None)
            maximum = getattr(camera_status, "exposure_max_us", None)
            current = getattr(camera_status, "exposure_time_us", None)
            range_text = (
                f"{minimum:.0f}–{maximum:.0f} µs"
                if minimum is not None and maximum is not None
                else "Bereich unbekannt"
            )
            current_text = "–" if current is None else f"{current:.0f} µs"
            ranges_form.addRow(
                "Kamerabereich",
                QLabel(f"{range_text} · aktuell {current_text}"),
            )
        else:
            ranges_form.addRow(
                "Kamerabelichtung",
                QLabel("Nicht verbunden, ExposureAuto aktiv oder nicht beschreibbar."),
            )
        self.exposure_enabled.toggled.connect(self._update_exposure_controls)
        layout.addWidget(ranges)

        timing = QGroupBox("Stabilisierungszeiten")
        timing_form = QFormLayout(timing)
        self.light_settle = self._spin(0, 10_000, config.light_settle_ms, " ms")
        self.robot_settle = self._spin(0, 10_000, config.robot_settle_ms, " ms")
        self.camera_settle = self._spin(0, 10_000, config.camera_settle_ms, " ms")
        timing_form.addRow("Nach Lichtänderung", self.light_settle)
        timing_form.addRow("Nach Roboterfahrt", self.robot_settle)
        timing_form.addRow("Nach Belichtungsänderung", self.camera_settle)
        layout.addWidget(timing)

        self.estimate = QLabel()
        self.estimate.setStyleSheet("font-weight:600;")
        layout.addWidget(self.estimate)
        for control in (
            self.pose_start,
            self.pose_end,
            self.light_1_start,
            self.light_1_end,
            self.light_1_step,
            self.light_2_start,
            self.light_2_end,
            self.light_2_step,
            self.exposure_start,
            self.exposure_end,
            self.exposure_step,
        ):
            signal = getattr(control, "valueChanged", None) or control.currentIndexChanged
            signal.connect(self._update_estimate)
        self.exposure_enabled.toggled.connect(self._update_estimate)
        self._update_exposure_controls()
        self._update_estimate()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int, suffix: str = "") -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSuffix(suffix)
        return spin

    @classmethod
    def _range_row(
        cls,
        minimum: int,
        maximum: int,
        start: int,
        end: int,
        step: int,
        suffix: str,
    ) -> tuple[QSpinBox, QSpinBox, QSpinBox]:
        return (
            cls._spin(minimum, maximum, start, suffix),
            cls._spin(minimum, maximum, end, suffix),
            cls._spin(1, maximum, step, suffix),
        )

    @staticmethod
    def _row_widget(*widgets: QWidget) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            row_layout.addWidget(widget)
        return row

    @staticmethod
    def _pose_combo(value: int) -> QComboBox:
        combo = QComboBox()
        for pose in ALLOWED_POSES:
            combo.addItem(str(pose), pose)
        combo.setCurrentIndex(max(0, combo.findData(value)))
        return combo

    def _browse_output(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Speicherort für Aufnahmen wählen",
            self.output_directory.text(),
        )
        if directory:
            self.output_directory.setText(directory)

    def _update_exposure_controls(self) -> None:
        enabled = self.exposure_enabled.isChecked() and self.exposure_enabled.isEnabled()
        for control in (self.exposure_start, self.exposure_end, self.exposure_step):
            control.setEnabled(enabled)

    def _current_config(self) -> AcquisitionSettings:
        return AcquisitionSettings(
            output_directory=Path(self.output_directory.text().strip()),
            pose_start=int(self.pose_start.currentData()),
            pose_end=int(self.pose_end.currentData()),
            light_1_start=self.light_1_start.value(),
            light_1_end=self.light_1_end.value(),
            light_1_step=self.light_1_step.value(),
            light_2_start=self.light_2_start.value(),
            light_2_end=self.light_2_end.value(),
            light_2_step=self.light_2_step.value(),
            exposure_enabled=self.exposure_enabled.isChecked(),
            exposure_start_us=self.exposure_start.value(),
            exposure_end_us=self.exposure_end.value(),
            exposure_step_us=self.exposure_step.value(),
            light_settle_ms=self.light_settle.value(),
            robot_settle_ms=self.robot_settle.value(),
            camera_settle_ms=self.camera_settle.value(),
        )

    def _update_estimate(self, *_: object) -> None:
        try:
            count = len(build_capture_points(self._current_config()))
            self.estimate.setText(f"Geplante Aufnahmen: {count}")
        except ValueError as exc:
            self.estimate.setText(str(exc))

    def _validate_and_accept(self) -> None:
        if not self.output_directory.text().strip():
            QMessageBox.warning(
                self,
                "Ungültige Aufnahmeeinstellung",
                "Bitte einen Speicherort auswählen.",
            )
            return
        try:
            config = self._current_config().validated()
            count = len(build_capture_points(config))
            if count > 100_000:
                raise ValueError("Mehr als 100.000 Aufnahmen sind in einer Sitzung nicht erlaubt.")
        except ValueError as exc:
            QMessageBox.warning(self, "Ungültige Aufnahmeeinstellung", str(exc))
            return
        self.result_config = config
        self.accept()


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
