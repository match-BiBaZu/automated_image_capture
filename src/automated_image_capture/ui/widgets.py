from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
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
    PreflightCheck,
    build_capture_points,
    synchronized_sweep_profile,
)
from automated_image_capture.hardware.robot import ALLOWED_POSES
from automated_image_capture.models import (
    CameraStatus,
    ConnectionState,
    ConveyorStatus,
    LightStatus,
    RobotCommandMode,
    RobotStatus,
)
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
        if self._status.exposure_auto.lower() not in {"off", "–", "none"} and enabled:
            self.exposure_hint.setText(
                "Übernehmen deaktiviert ExposureAuto für diese Verbindung. Beim Trennen "
                "wird der ursprüngliche Automatikmodus wiederhergestellt."
            )
        elif self._status.exposure_auto.lower() not in {"off", "–", "none"}:
            self.exposure_hint.setText(
                "ExposureAuto ist aktiv und kann von der Kamera aktuell nicht deaktiviert werden."
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
        loaded_program = self._status.loaded_program.upper()
        correct_program = "BIBAZU" in loaded_program and "CONTINUOUS" not in loaded_program
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


class ConveyorControlCard(DeviceCard):
    jog_requested = pyqtSignal(str, float)
    stop_requested = pyqtSignal()
    origin_requested = pyqtSignal()
    forward_direction_changed = pyqtSignal(str)

    def __init__(
        self,
        title: str,
        forward_direction: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(title, parent)
        self._status = ConveyorStatus(forward_direction=forward_direction)
        distance_row = QHBoxLayout()
        distance_row.addWidget(QLabel("Manueller Fahrweg"))
        self.jog_distance = QDoubleSpinBox()
        self.jog_distance.setRange(0.1, 5000.0)
        self.jog_distance.setDecimals(1)
        self.jog_distance.setSingleStep(1.0)
        self.jog_distance.setValue(1.0)
        self.jog_distance.setSuffix(" mm")
        self.jog_distance.setToolTip("Gewünschte relative Fahrstrecke")
        distance_row.addWidget(self.jog_distance, 1)
        self.content_layout.addLayout(distance_row)

        direction_row = QHBoxLayout()
        self.left_button = QPushButton("← Links")
        self.stop_button = QPushButton("Stop")
        self.right_button = QPushButton("Rechts →")
        self.left_button.clicked.connect(lambda: self._request_jog("left"))
        self.right_button.clicked.connect(lambda: self._request_jog("right"))
        self.stop_button.clicked.connect(self.stop_requested.emit)
        direction_row.addWidget(self.left_button)
        direction_row.addWidget(self.stop_button)
        direction_row.addWidget(self.right_button)
        self.content_layout.addLayout(direction_row)

        mapping_row = QHBoxLayout()
        mapping_row.addWidget(QLabel("Vorwärtsrichtung"))
        self.forward_direction = QComboBox()
        self.forward_direction.addItem("Bitte testen …", "")
        self.forward_direction.addItem("Links ist vorwärts", "left")
        self.forward_direction.addItem("Rechts ist vorwärts", "right")
        self.forward_direction.setCurrentIndex(
            max(0, self.forward_direction.findData(forward_direction))
        )
        self.forward_direction.currentIndexChanged.connect(
            lambda: self.forward_direction_changed.emit(str(self.forward_direction.currentData()))
        )
        mapping_row.addWidget(self.forward_direction, 1)
        self.content_layout.addLayout(mapping_row)

        self.origin_button = QPushButton("Aktuelle Position = 0 mm")
        self.origin_button.clicked.connect(self.origin_requested.emit)
        self.content_layout.addWidget(self.origin_button)
        self._update_controls()

    def _request_jog(self, direction: str) -> None:
        self.jog_requested.emit(direction, self.jog_distance.value())

    def set_status(self, status: ConveyorStatus) -> None:
        self._status = status
        offset = "–" if status.logical_offset_mm is None else f"{status.logical_offset_mm:.3f} mm"
        calibration = (
            f"gültig · {status.mm_per_full_step:.8f} mm/Vollschritt"
            if status.calibration_valid
            else "ungültig"
        )
        internal = "–" if status.internal_position is None else str(status.internal_position)
        drive_state = (
            "Positioniermodus wird aktiviert"
            if status.preparing_drive
            else "bereit"
            if status.ready_to_execute
            else "Standby"
        )
        feedback = "geprüft" if status.position_feedback_verified else "noch ungeprüft"
        self.details.setText(
            f"Kalibrierung: {calibration}\n"
            f"Antrieb: {drive_state} · beschäftigt/Warning/Fehler: "
            f"{'ja' if status.busy else 'nein'} / "
            f"{'ja' if status.warning else 'nein'} / {'ja' if status.error else 'nein'}\n"
            f"SPS-Position: {internal} · Rückmeldung: {feedback}\n"
            f"Logischer Offset: {offset} · Status {status.status_code}"
        )
        self._update_controls()

    def set_state(self, state: ConnectionState) -> None:
        super().set_state(state)
        self._update_controls()

    def _update_controls(self) -> None:
        if not hasattr(self, "left_button"):
            return
        controls_available = (
            self.state is ConnectionState.CONNECTED
            and self._status.calibration_valid
            and not self._status.busy
            and not self._status.preparing_drive
            and not self._status.error
            and self._status.status_code not in {4, 5}
        )
        self.left_button.setEnabled(controls_available)
        self.right_button.setEnabled(controls_available)
        self.jog_distance.setEnabled(controls_available)
        self.origin_button.setEnabled(
            controls_available and self._status.internal_position is not None
        )
        self.forward_direction.setEnabled(not self._status.busy)
        self.stop_button.setEnabled(self.state is ConnectionState.CONNECTED)


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
    align_conveyor_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Automatisierte Bildaufnahme", parent)
        layout = QVBoxLayout(self)
        self.summary = QLabel("Noch keine Aufnahmeeinstellungen geladen.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self._running = False
        self._resume_available = False
        self._preflight_ready = False
        self._last_preflight: tuple[PreflightCheck, ...] | None = None
        self.preflight = QLabel("Startfreigabe wird geprüft …")
        self.preflight.setWordWrap(True)
        self.preflight.setStyleSheet(
            "background:#fef2f2; color:#991b1b; padding:7px; border-radius:4px;"
        )
        layout.addWidget(self.preflight)
        buttons = QGridLayout()
        self.configure_button = QPushButton("Aufnahme konfigurieren …")
        self.start_button = QPushButton("Aufnahme starten")
        self.resume_button = QPushButton("Aufnahme fortsetzen")
        self.stop_button = QPushButton("Aufnahme stoppen")
        self.align_button = QPushButton("Förderband auf Checkpoint ausrichten")
        self.align_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.configure_button.clicked.connect(self.configure_requested.emit)
        self.start_button.clicked.connect(self.start_requested.emit)
        self.resume_button.clicked.connect(self.resume_requested.emit)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.align_button.clicked.connect(self.align_conveyor_requested.emit)
        buttons.addWidget(self.configure_button, 0, 0)
        buttons.addWidget(self.start_button, 0, 1)
        buttons.addWidget(self.resume_button, 1, 0)
        buttons.addWidget(self.stop_button, 1, 1)
        buttons.addWidget(self.align_button, 2, 0, 1, 2)
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
        robot = (
            f"UR-Winkel {settings.angle_start_deg:g}–{settings.angle_end_deg:g}° "
            f"in {settings.angle_step_deg:g}°-Schritten"
            if settings.robot_control_mode == RobotCommandMode.ANGLE.value
            else f"UR-Pose {settings.pose_start} bis {settings.pose_end}"
        )
        if settings.conveyor_enabled:
            belt = (
                f" · Förderband synchron 0–{settings.conveyor_max_offset_mm:g}–0 mm"
                if settings.conveyor_motion_mode == "synchronized"
                else f" · Förderband 0–{settings.conveyor_max_offset_mm:g}–0 mm / "
                f"{settings.conveyor_step_mm:g} mm"
            )
        else:
            belt = " · Förderband aus"
        exposure = (
            f" · Belichtung {settings.exposure_start_us}–{settings.exposure_end_us} µs"
            if settings.exposure_enabled
            else " · Belichtung unverändert"
        )
        if settings.capture_mode == "ramp":
            if settings.conveyor_enabled and settings.conveyor_motion_mode == "synchronized":
                duration, speed, samples = synchronized_sweep_profile(settings)
                timing = f"Bandlauf {duration:.1f} s bei {speed:.2f} mm/s"
            else:
                samples = round(settings.ramp_duration_s * settings.ramp_image_rate_fps)
                timing = f"{settings.ramp_duration_s:g} s"
            mode = (
                f"Schnelle Rampe · {samples} Bilder pro Pose/Belichtung · "
                f"{timing} bei {settings.ramp_image_rate_fps} Bildern/s · "
                f"Perioden {settings.ramp_light_1_period_s:g}/{settings.ramp_light_2_period_s:g} s"
            )
        else:
            label = (
                "Diskrete Lichtfolge während Bandfahrt"
                if settings.conveyor_enabled
                and settings.conveyor_motion_mode == "synchronized"
                else "Exaktes Raster"
            )
            mode = (
                f"{label} · Panel 1 {settings.light_1_start}–{settings.light_1_end} % / "
                f"Panel 2 {settings.light_2_start}–{settings.light_2_end} %"
            )
        self.summary.setText(
            f"{mode}\n{robot}{belt}{exposure}\n"
            f"{count} Bilder · Ziel: {settings.output_directory}"
        )

    def set_running(self, running: bool) -> None:
        self._running = running
        self.configure_button.setEnabled(not running)
        self.start_button.setEnabled(not running and self._preflight_ready)
        self.resume_button.setEnabled(
            not running and self._resume_available and self._preflight_ready
        )
        self.stop_button.setEnabled(running)

    def set_resume_available(self, available: bool) -> None:
        self._resume_available = available
        self.resume_button.setEnabled(
            available and not self._running and self._preflight_ready
        )

    def set_preflight(self, checks: tuple[PreflightCheck, ...]) -> None:
        if checks == self._last_preflight:
            return
        self._last_preflight = checks
        failed = [check for check in checks if not check.ready]
        self._preflight_ready = bool(checks) and not failed
        if self._preflight_ready:
            self.preflight.setText(
                f"✓ Startbereit – alle {len(checks)} Prüfungen sind erfüllt."
            )
            self.preflight.setStyleSheet(
                "background:#ecfdf5; color:#166534; padding:7px; border-radius:4px;"
            )
            self.start_button.setToolTip("Alle Startbedingungen sind erfüllt.")
        else:
            problems = "\n".join(
                f"• {check.label}: {check.detail}" for check in failed
            )
            self.preflight.setText(
                f"✗ Start blockiert – {len(failed)} Punkt(e) fehlen:\n{problems}"
            )
            self.preflight.setStyleSheet(
                "background:#fef2f2; color:#991b1b; padding:7px; border-radius:4px;"
            )
            first = failed[0].detail if failed else "Startprüfung noch nicht verfügbar."
            self.start_button.setToolTip(first)
        self.start_button.setEnabled(not self._running and self._preflight_ready)
        self.resume_button.setEnabled(
            not self._running and self._resume_available and self._preflight_ready
        )

    def set_alignment_required(self, required: bool, expected: float, actual: float) -> None:
        self.align_button.setEnabled(required and not self._running)
        if required:
            self.align_button.setText(
                f"Förderband ausrichten: Ist {actual:.3f} mm → Soll {expected:g} mm"
            )
        else:
            self.align_button.setText("Förderband auf Checkpoint ausrichten")

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
        self.capture_mode = QComboBox()
        self.capture_mode.addItem("Exaktes Raster", "grid")
        self.capture_mode.addItem("Schnelle Rampe", "ramp")
        self.capture_mode.setCurrentIndex(max(0, self.capture_mode.findData(config.capture_mode)))
        output_form.addRow("Aufnahmemodus", self.capture_mode)
        layout.addLayout(output_form)

        ranges = QGroupBox("Variationen")
        ranges_form = QFormLayout(ranges)
        self.ranges_form = ranges_form
        self.pose_start = self._pose_combo(config.pose_start)
        self.pose_end = self._pose_combo(config.pose_end)
        self.robot_control_mode = QComboBox()
        self.robot_control_mode.addItem("Feste Pose-IDs", RobotCommandMode.POSE_ID.value)
        self.robot_control_mode.addItem(
            "Kontinuierlicher Winkel (BiBaZu_Continuous)", RobotCommandMode.ANGLE.value
        )
        self.robot_control_mode.setCurrentIndex(
            max(0, self.robot_control_mode.findData(config.robot_control_mode))
        )
        ranges_form.addRow("UR-Steuerung", self.robot_control_mode)
        self.pose_row = self._row_widget(self.pose_start, self.pose_end)
        ranges_form.addRow("UR Startpose / Endpose", self.pose_row)
        self.angle_start = self._double_spin(15.5, 21.0, config.angle_start_deg, "°")
        self.angle_end = self._double_spin(15.5, 21.0, config.angle_end_deg, "°")
        self.angle_step = self._double_spin(0.1, 5.5, config.angle_step_deg, "°")
        self.angle_row = self._row_widget(self.angle_start, self.angle_end, self.angle_step)
        ranges_form.addRow("UR-Winkel Start / Ende / Schritt", self.angle_row)

        self.light_1_start, self.light_1_end, self.light_1_step = self._range_row(
            0, 100, config.light_1_start, config.light_1_end, config.light_1_step, "%"
        )
        self.light_1_row = self._row_widget(self.light_1_start, self.light_1_end, self.light_1_step)
        ranges_form.addRow(
            "Panel 1 Start / Ende / Schritt",
            self.light_1_row,
        )
        self.light_2_start, self.light_2_end, self.light_2_step = self._range_row(
            0, 100, config.light_2_start, config.light_2_end, config.light_2_step, "%"
        )
        self.light_2_row = self._row_widget(self.light_2_start, self.light_2_end, self.light_2_step)
        ranges_form.addRow(
            "Panel 2 Start / Ende / Schritt",
            self.light_2_row,
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

        self.conveyor_group = QGroupBox("Förderbandvariation")
        conveyor_form = QFormLayout(self.conveyor_group)
        self.conveyor_enabled = QCheckBox("Bauteilposition über das Förderband variieren")
        self.conveyor_enabled.setChecked(config.conveyor_enabled)
        self.conveyor_motion_mode = QComboBox()
        self.conveyor_motion_mode.addItem("Diskrete Stationen", "stations")
        self.conveyor_motion_mode.addItem(
            "Kontinuierliche synchronisierte Fahrt", "synchronized"
        )
        self.conveyor_motion_mode.setCurrentIndex(
            max(0, self.conveyor_motion_mode.findData(config.conveyor_motion_mode))
        )
        self.conveyor_max_offset = self._double_spin(
            0.0, 5000.0, config.conveyor_max_offset_mm, " mm"
        )
        self.conveyor_step = self._double_spin(0.1, 5000.0, config.conveyor_step_mm, " mm")
        self.conveyor_speed = self._double_spin(
            0.1, 5000.0, config.conveyor_speed_mm_per_s, " mm/s"
        )
        self.conveyor_settle = self._spin(0, 10_000, config.conveyor_settle_ms, " ms")
        conveyor_form.addRow(self.conveyor_enabled)
        conveyor_form.addRow("Fahrmodus", self.conveyor_motion_mode)
        conveyor_form.addRow("Maximaler Offset", self.conveyor_max_offset)
        conveyor_form.addRow("Schrittweite", self.conveyor_step)
        conveyor_form.addRow("Geschwindigkeit", self.conveyor_speed)
        conveyor_form.addRow("Beruhigungszeit", self.conveyor_settle)
        conveyor_note = QLabel(
            "Stationen: Das Band hält an jeder Position. Synchronisiert: Pro UR-Winkel "
            "und Belichtung fährt das Band einmal 0 → Maximum → 0; Kamera und Licht "
            "laufen gleichzeitig. Die SPS-Istposition wird zu jedem Bild gespeichert."
        )
        conveyor_note.setWordWrap(True)
        conveyor_form.addRow(conveyor_note)
        layout.addWidget(self.conveyor_group)

        self.ramp_group = QGroupBox("Zeitsteuerung für schnelle Aufnahme")
        ramp_form = QFormLayout(self.ramp_group)
        self.ramp_duration = self._double_spin(2.0, 120.0, config.ramp_duration_s, " s")
        self.ramp_duration.setToolTip(
            "Bei synchronisierter Bandfahrt wird die Dauer automatisch aus Weg und "
            "Bandgeschwindigkeit berechnet."
        )
        self.ramp_rate = self._spin(1, 240, config.ramp_image_rate_fps, " Bilder/s")
        self.ramp_light_1_period = self._double_spin(0.8, 120.0, config.ramp_light_1_period_s, " s")
        self.ramp_light_2_period = self._double_spin(0.8, 120.0, config.ramp_light_2_period_s, " s")
        ramp_form.addRow("Dauer pro Pose/Belichtung", self.ramp_duration)
        ramp_form.addRow("Bildrate", self.ramp_rate)
        ramp_form.addRow("Periode Panel 1", self.ramp_light_1_period)
        ramp_form.addRow("Periode Panel 2", self.ramp_light_2_period)
        ramp_note = QLabel(
            "Beide Panels folgen 0–100–0-Dreieckskurven. Gespeichert werden nominelle "
            "Sollwerte und der jeweils letzte bestätigte BLE-Befehl. Bei synchronisierter "
            "Fahrt bestimmt das Band die Dauer; im Rastermodus werden diskrete Sollwerte "
            "ohne Befehlsstau über den Bandlauf verteilt."
        )
        ramp_note.setWordWrap(True)
        ramp_form.addRow(ramp_note)
        layout.addWidget(self.ramp_group)

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
            self.robot_control_mode,
            self.angle_start,
            self.angle_end,
            self.angle_step,
            self.conveyor_motion_mode,
            self.conveyor_max_offset,
            self.conveyor_step,
            self.conveyor_speed,
            self.conveyor_settle,
            self.light_1_start,
            self.light_1_end,
            self.light_1_step,
            self.light_2_start,
            self.light_2_end,
            self.light_2_step,
            self.exposure_start,
            self.exposure_end,
            self.exposure_step,
            self.ramp_duration,
            self.ramp_rate,
            self.ramp_light_1_period,
            self.ramp_light_2_period,
        ):
            signal = getattr(control, "valueChanged", None) or control.currentIndexChanged
            signal.connect(self._update_estimate)
        self.exposure_enabled.toggled.connect(self._update_estimate)
        self.conveyor_enabled.toggled.connect(self._update_conveyor_controls)
        self.conveyor_enabled.toggled.connect(self._update_estimate)
        self.conveyor_enabled.toggled.connect(self._update_mode_controls)
        self.conveyor_motion_mode.currentIndexChanged.connect(self._update_conveyor_controls)
        self.conveyor_motion_mode.currentIndexChanged.connect(self._update_mode_controls)
        self.robot_control_mode.currentIndexChanged.connect(self._update_robot_controls)
        self.robot_control_mode.currentIndexChanged.connect(self._update_estimate)
        self.capture_mode.currentIndexChanged.connect(self._update_mode_controls)
        self.capture_mode.currentIndexChanged.connect(self._update_estimate)
        self._update_exposure_controls()
        self._update_conveyor_controls()
        self._update_robot_controls()
        self._update_mode_controls()
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

    @staticmethod
    def _double_spin(
        minimum: float, maximum: float, value: float, suffix: str = ""
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(1)
        spin.setSingleStep(0.1)
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

    def _update_conveyor_controls(self, *_: object) -> None:
        enabled = self.conveyor_enabled.isChecked()
        synchronized = self.conveyor_motion_mode.currentData() == "synchronized"
        for control in (self.conveyor_max_offset, self.conveyor_speed):
            control.setEnabled(enabled)
        self.conveyor_motion_mode.setEnabled(enabled)
        self.conveyor_step.setEnabled(enabled and not synchronized)
        self.conveyor_settle.setEnabled(enabled and not synchronized)

    def _update_robot_controls(self, *_: object) -> None:
        angle_mode = self.robot_control_mode.currentData() == RobotCommandMode.ANGLE.value
        self.ranges_form.setRowVisible(self.pose_row, not angle_mode)
        self.ranges_form.setRowVisible(self.angle_row, angle_mode)

    def _update_mode_controls(self, *_: object) -> None:
        ramp = self.capture_mode.currentData() == "ramp"
        synchronized = (
            self.conveyor_enabled.isChecked()
            and self.conveyor_motion_mode.currentData() == "synchronized"
        )
        self.ramp_group.setVisible(ramp or synchronized)
        self.ramp_duration.setEnabled(ramp and not synchronized)
        self.ramp_rate.setEnabled(ramp or synchronized)
        self.ramp_light_1_period.setEnabled(ramp)
        self.ramp_light_2_period.setEnabled(ramp)
        self.ranges_form.setRowVisible(self.light_1_row, not ramp)
        self.ranges_form.setRowVisible(self.light_2_row, not ramp)

    def _current_config(self) -> AcquisitionSettings:
        return AcquisitionSettings(
            output_directory=Path(self.output_directory.text().strip()),
            capture_mode=str(self.capture_mode.currentData()),
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
            ramp_duration_s=self.ramp_duration.value(),
            ramp_image_rate_fps=self.ramp_rate.value(),
            ramp_light_1_period_s=self.ramp_light_1_period.value(),
            ramp_light_2_period_s=self.ramp_light_2_period.value(),
            robot_control_mode=str(self.robot_control_mode.currentData()),
            angle_start_deg=self.angle_start.value(),
            angle_end_deg=self.angle_end.value(),
            angle_step_deg=self.angle_step.value(),
            conveyor_enabled=self.conveyor_enabled.isChecked(),
            conveyor_motion_mode=str(self.conveyor_motion_mode.currentData()),
            conveyor_max_offset_mm=self.conveyor_max_offset.value(),
            conveyor_step_mm=self.conveyor_step.value(),
            conveyor_speed_mm_per_s=self.conveyor_speed.value(),
            conveyor_settle_ms=self.conveyor_settle.value(),
        )

    def _update_estimate(self, *_: object) -> None:
        try:
            config = self._current_config()
            points = build_capture_points(config)
            count = len(points)
            if config.conveyor_enabled and config.conveyor_motion_mode == "synchronized":
                duration, speed, samples = synchronized_sweep_profile(config)
                passes = len({(point.robot_key, point.exposure_time_us) for point in points})
                self.estimate.setText(
                    f"Geplante Aufnahmen: {count} · {samples} pro Bandlauf · "
                    f"0–{config.conveyor_max_offset_mm:g}–0 mm in {duration:.1f} s "
                    f"bei {speed:.2f} mm/s · reine Fahrzeit {passes * duration:.1f} s"
                )
                return
            if config.capture_mode == "ramp":
                samples = round(config.ramp_duration_s * config.ramp_image_rate_fps)
                passes = count // max(1, samples)
                self.estimate.setText(
                    f"Geplante Aufnahmen: {count} · {samples} pro Rampe · "
                    f"geschätzte Rampenzeit {passes * config.ramp_duration_s:.1f} s "
                    f"zzgl. UR- und Belichtungswechsel"
                )
            else:
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
        self.plc_ip = QLineEdit(config.plc_ip)
        self.plc_ams_net_id = QLineEdit(config.plc_ams_net_id)
        self.plc_port = QSpinBox()
        self.plc_port.setRange(1, 65535)
        self.plc_port.setValue(config.plc_port)
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
        self.maximize_camera_fps = QCheckBox(
            "Kameraseitiges Bildratenlimit abschalten (maximale Rohbildrate)"
        )
        self.maximize_camera_fps.setChecked(config.maximize_camera_frame_rate)
        self.auto_reconnect = QCheckBox("Verlorene Verbindungen erneut aufbauen")
        self.auto_reconnect.setChecked(config.auto_reconnect)

        form.addRow("Baumer-IP", self.camera_ip)
        form.addRow("UR16e-IP", self.robot_ip)
        form.addRow("TwinCAT-SPS-IP", self.plc_ip)
        form.addRow("SPS AMS-Net-ID", self.plc_ams_net_id)
        form.addRow("TwinCAT-PLC-Port", self.plc_port)
        form.addRow("Baumer GenTL (.cti)", cti_row)
        form.addRow("Licht 1 BLE-Adresse", self.light_1_address)
        form.addRow("Licht 2 BLE-Adresse", self.light_2_address)
        form.addRow("Maximale Vorschau-FPS", self.preview_fps)
        form.addRow("Kamera-Rohdatenstrom", self.maximize_camera_fps)
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
                plc_ip=self.plc_ip.text(),
                plc_ams_net_id=self.plc_ams_net_id.text(),
                plc_port=self.plc_port.value(),
                conveyor_forward_direction=self._source_config.conveyor_forward_direction,
                camera_cti_path=self.cti_path.text(),
                camera_serial=self._source_config.camera_serial,
                light_address=self.light_1_address.text().strip(),
                light_name=self._source_config.light_name,
                light_2_address=self.light_2_address.text().strip(),
                light_2_name=self._source_config.light_2_name,
                preview_max_fps=self.preview_fps.value(),
                maximize_camera_frame_rate=self.maximize_camera_fps.isChecked(),
                auto_reconnect=self.auto_reconnect.isChecked(),
            ).validated()
            if not Path(config.camera_cti_path).is_file():
                raise ValueError("Der angegebene GenTL-Producer existiert nicht.")
        except ValueError as exc:
            QMessageBox.warning(self, "Ungültige Einstellung", str(exc))
            return
        self.result_config = config
        self.accept()
