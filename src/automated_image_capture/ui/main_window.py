from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from automated_image_capture.hardware import CameraAdapter, LightAdapter, RobotAdapter
from automated_image_capture.models import (
    CameraFrame,
    CameraStatus,
    ConnectionState,
    LightStatus,
    RobotStatus,
)
from automated_image_capture.settings import SettingsStore
from automated_image_capture.ui.widgets import DeviceCard, LabeledSlider, SettingsDialog


class MainWindow(QMainWindow):
    def __init__(
        self,
        settings_store: SettingsStore | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("YOLO Trainingsbild-Aufnahme – Hardware-Dashboard")
        self.resize(1420, 900)
        self._logger = logging.getLogger("ui")
        self.settings_store = settings_store or SettingsStore()
        self.config = self.settings_store.load()
        self._last_image: QImage | None = None
        self._closing = False
        self._updating_light_ui = False

        self.camera = CameraAdapter(self.config, self)
        self.robot = RobotAdapter(self.config, self)
        self.light = LightAdapter(self.config, self)

        self._build_ui()
        self._wire_adapters()
        self._append_event("Dashboard bereit. Es werden noch keine Geräte automatisch verbunden.")

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)

        toolbar = QHBoxLayout()
        connect_all = QPushButton("Alle verbinden")
        disconnect_all = QPushButton("Alle trennen")
        settings_button = QPushButton("Einstellungen …")
        connect_all.clicked.connect(self.connect_all)
        disconnect_all.clicked.connect(self.disconnect_all)
        settings_button.clicked.connect(self.open_settings)
        toolbar.addWidget(connect_all)
        toolbar.addWidget(disconnect_all)
        toolbar.addStretch(1)
        safety = QLabel("UR16e: nur Status – keine Bewegung oder Schreibbefehle")
        safety.setStyleSheet("color:#166534; font-weight:600;")
        toolbar.addWidget(safety)
        toolbar.addWidget(settings_button)
        root.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        cards_container = QWidget()
        cards_layout = QVBoxLayout(cards_container)
        self.camera_card = DeviceCard("Baumer Industriekamera")
        self.robot_card = DeviceCard("Universal Robots UR16e")
        self.light_card = DeviceCard("Neewer RGB660 Pro II")
        self.camera_card.action_requested.connect(lambda: self._toggle(self.camera))
        self.robot_card.action_requested.connect(lambda: self._toggle(self.robot))
        self.light_card.action_requested.connect(lambda: self._toggle(self.light))
        cards_layout.addWidget(self.camera_card)
        cards_layout.addWidget(self.robot_card)
        cards_layout.addWidget(self.light_card)
        self._build_light_controls()
        cards_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(cards_container)
        scroll.setMinimumWidth(430)
        splitter.addWidget(scroll)

        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_title = QLabel("Kamera-Livebild")
        preview_title.setStyleSheet("font-size:16px; font-weight:600;")
        preview_layout.addWidget(preview_title)
        self.preview = QLabel("Noch kein Kamerabild")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(640, 480)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview.setStyleSheet("background:#111827; color:#d1d5db; border-radius:5px;")
        preview_layout.addWidget(self.preview, 1)
        splitter.addWidget(preview_container)
        splitter.setSizes([480, 940])
        root.addWidget(splitter, 1)

        log_header = QHBoxLayout()
        log_header.addWidget(QLabel("Ereignisprotokoll"))
        clear_log = QPushButton("Leeren")
        clear_log.clicked.connect(lambda: self.event_log.clear())
        log_header.addStretch(1)
        log_header.addWidget(clear_log)
        root.addLayout(log_header)
        self.event_log = QPlainTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setMaximumBlockCount(1000)
        self.event_log.setMaximumHeight(180)
        root.addWidget(self.event_log)

        self.setCentralWidget(central)
        self.statusBar().showMessage(
            f"Kamera {self.config.camera_ip} · UR {self.config.robot_ip}"
        )

    def _build_light_controls(self) -> None:
        row = QHBoxLayout()
        self.light_power = QPushButton("Licht einschalten")
        self.light_power.setCheckable(True)
        self.light_power.setEnabled(False)
        self.light_power.toggled.connect(self._power_toggled)
        self.light_mode = QComboBox()
        self.light_mode.addItems(["CCT", "HSI"])
        self.light_mode.setEnabled(False)
        self.light_mode.currentTextChanged.connect(self._light_values_changed)
        row.addWidget(self.light_power)
        row.addWidget(QLabel("Modus"))
        row.addWidget(self.light_mode)
        self.light_card.content_layout.addLayout(row)

        self.light_brightness = LabeledSlider("Helligkeit", 0, 100, 50, " %")
        self.light_cct = LabeledSlider("Farbtemperatur", 3200, 5600, 5600, " K")
        self.light_hue = LabeledSlider("Farbton", 0, 360, 0, "°")
        self.light_saturation = LabeledSlider("Sättigung", 0, 100, 100, " %")
        for control in (
            self.light_brightness,
            self.light_cct,
            self.light_hue,
            self.light_saturation,
        ):
            control.setEnabled(False)
            control.value_changed.connect(self._light_values_changed)
            self.light_card.content_layout.addWidget(control)

        self.light_command_timer = QTimer(self)
        self.light_command_timer.setSingleShot(True)
        self.light_command_timer.setInterval(150)
        self.light_command_timer.timeout.connect(self._send_light_values)
        self._update_mode_visibility("CCT")

    def _wire_adapters(self) -> None:
        for adapter, card in (
            (self.camera, self.camera_card),
            (self.robot, self.robot_card),
            (self.light, self.light_card),
        ):
            adapter.state_changed.connect(card.set_state)
            adapter.event_message.connect(self._append_event)
            adapter.error.connect(
                lambda message, name=adapter.display_name: self._show_error(name, message)
            )
        self.camera.status_changed.connect(self._camera_status)
        self.camera.frame_ready.connect(self._camera_frame)
        self.robot.status_changed.connect(self._robot_status)
        self.light.status_changed.connect(self._light_status)
        self.light.state_changed.connect(self._light_state)

    def _toggle(self, adapter: CameraAdapter | RobotAdapter | LightAdapter) -> None:
        if adapter.state is ConnectionState.DISCONNECTED:
            adapter.connect()
        else:
            adapter.disconnect()

    def connect_all(self) -> None:
        self.camera.connect()
        self.robot.connect()
        self.light.connect()

    def disconnect_all(self) -> None:
        self.camera.disconnect()
        self.robot.disconnect()
        self.light.disconnect()

    def _camera_status(self, status: CameraStatus) -> None:
        fps = "–" if status.camera_fps is None else f"{status.camera_fps:.1f}"
        self.camera_card.details.setText(
            f"Modell: {status.model}\n"
            f"Seriennummer: {status.serial_number}\n"
            f"IP: {status.ip_address}\n"
            f"Bild: {status.width} × {status.height} · {status.pixel_format}\n"
            f"Kamera/Vorschau: {fps} / {status.preview_fps:.1f} FPS"
        )
        if (
            status.serial_number not in ("", "–")
            and status.serial_number != self.config.camera_serial
        ):
            self.config.camera_serial = status.serial_number
            self.settings_store.save(self.config)

    def _camera_frame(self, frame: CameraFrame) -> None:
        image = frame.image
        height, width, channels = image.shape
        if channels != 3:
            return
        self._last_image = QImage(
            image.data,
            width,
            height,
            int(image.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        self._render_image()

    def _render_image(self) -> None:
        if self._last_image is None:
            return
        pixmap = QPixmap.fromImage(self._last_image).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(pixmap)

    def resizeEvent(self, event: object) -> None:  # noqa: N802
        super().resizeEvent(event)  # type: ignore[arg-type]
        self._render_image()

    def _robot_status(self, status: RobotStatus) -> None:
        scaling = "–" if status.speed_scaling is None else f"{status.speed_scaling * 100:.0f} %"
        joints = "–" if not status.joint_positions else ", ".join(
            f"{value:.3f}" for value in status.joint_positions
        )
        tcp = "–" if not status.tcp_pose else ", ".join(f"{value:.4f}" for value in status.tcp_pose)
        self.robot_card.details.setText(
            f"RTDE/Dashboard: {'ja' if status.rtde_connected else 'nein'} / "
            f"{'ja' if status.dashboard_connected else 'nein'}\n"
            f"Robot/Safety Mode: {status.robot_mode} / {status.safety_mode}\n"
            f"Remote: {status.remote_control} · Programm: {status.program_state}\n"
            f"Speed Scaling: {scaling}\n"
            f"Gelenke [rad]: {joints}\n"
            f"TCP [m, rotvec]: {tcp}\n"
            f"PolyScope: {status.polyscope_version}"
        )

    def _light_state(self, state: ConnectionState) -> None:
        enabled = state is ConnectionState.CONNECTED
        for control in (
            self.light_power,
            self.light_mode,
            self.light_brightness,
            self.light_cct,
            self.light_hue,
            self.light_saturation,
        ):
            control.setEnabled(enabled)

    def _light_status(self, status: LightStatus) -> None:
        confirmed = (
            "bestätigter letzter Befehl"
            if status.values_are_confirmed_commands
            else "kein Istwert"
        )
        rssi = "–" if status.rssi is None else f"{status.rssi} dBm"
        power = "–" if status.power is None else ("Ein" if status.power else "Aus")
        self.light_card.details.setText(
            f"Gerät: {status.name}\nAdresse: {status.address} · RSSI: {rssi}\n"
            f"Leistung: {power} · Modus: {status.mode}\n"
            f"Helligkeit: {status.brightness} % · CCT: {status.cct_kelvin} K\n"
            f"HSI: {status.hue}° / {status.saturation} % · Werte: {confirmed}"
        )
        self._updating_light_ui = True
        try:
            self.light_mode.setCurrentText(status.mode)
            self.light_brightness.set_value(status.brightness)
            self.light_cct.set_value(status.cct_kelvin)
            self.light_hue.set_value(status.hue)
            self.light_saturation.set_value(status.saturation)
            if status.power is not None:
                blocked = self.light_power.blockSignals(True)
                self.light_power.setChecked(status.power)
                self.light_power.blockSignals(blocked)
                self.light_power.setText(
                    "Licht ausschalten" if status.power else "Licht einschalten"
                )
        finally:
            self._updating_light_ui = False
        self._update_mode_visibility(status.mode)
        if status.address not in ("", "–") and status.address != self.config.light_address:
            self.config.light_address = status.address
            self.config.light_name = status.name
            self.settings_store.save(self.config)

    def _power_toggled(self, enabled: bool) -> None:
        self.light_power.setText("Licht ausschalten" if enabled else "Licht einschalten")
        if not self._updating_light_ui:
            self.light.set_power(enabled)

    def _light_values_changed(self, *_: object) -> None:
        mode = self.light_mode.currentText()
        self._update_mode_visibility(mode)
        if not self._updating_light_ui and self.light.state is ConnectionState.CONNECTED:
            self.light_command_timer.start()

    def _update_mode_visibility(self, mode: str) -> None:
        is_cct = mode == "CCT"
        self.light_cct.setVisible(is_cct)
        self.light_hue.setVisible(not is_cct)
        self.light_saturation.setVisible(not is_cct)

    def _send_light_values(self) -> None:
        brightness = self.light_brightness.value()
        if self.light_mode.currentText() == "CCT":
            self.light.set_cct(self.light_cct.value(), brightness)
        else:
            self.light.set_hsi(
                self.light_hue.value(),
                self.light_saturation.value(),
                brightness,
            )

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.config, self)
        if dialog.exec() != SettingsDialog.DialogCode.Accepted or dialog.result_config is None:
            return
        new_config = replace(
            dialog.result_config,
            camera_serial=self.config.camera_serial,
            light_address=self.config.light_address,
            light_name=self.config.light_name,
        )
        self.config = new_config
        self.settings_store.save(self.config)
        self.camera.config = self.config
        self.robot.config = self.config
        self.light.config = self.config
        self.statusBar().showMessage(
            f"Kamera {self.config.camera_ip} · UR {self.config.robot_ip}"
        )
        self._append_event("Einstellungen gespeichert; sie gelten ab der nächsten Verbindung.")

    def _show_error(self, device: str, message: str) -> None:
        self.statusBar().showMessage(f"{device}: {message}", 10000)
        self._logger.error("%s: %s", device, message)

    def _append_event(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.event_log.appendPlainText(f"[{timestamp}] {message}")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._closing:
            event.accept()
            return
        self._closing = True
        self.light_command_timer.stop()
        self.camera.disconnect()
        self.robot.disconnect()
        event.accept()

    async def shutdown_async(self) -> None:
        self.camera.disconnect()
        self.robot.disconnect()
        await self.light.shutdown()
