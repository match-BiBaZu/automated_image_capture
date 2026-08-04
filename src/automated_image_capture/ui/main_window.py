from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from automated_image_capture.acquisition import AcquisitionController, build_capture_points
from automated_image_capture.hardware import CameraAdapter, LightAdapter, RobotAdapter
from automated_image_capture.models import (
    CameraFrame,
    CameraStatus,
    ConnectionState,
    LightStatus,
    RobotStatus,
)
from automated_image_capture.settings import SettingsStore
from automated_image_capture.ui.widgets import (
    AcquisitionCard,
    AcquisitionDialog,
    DeviceCard,
    LightControlCard,
    RobotPoseControlCard,
    SettingsDialog,
)


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
        self.acquisition_config = self.settings_store.load_acquisition()
        self._last_image: QImage | None = None
        self._camera_status_data = CameraStatus()
        self._closing = False
        self.camera = CameraAdapter(self.config, self)
        self.robot = RobotAdapter(self.config, self)
        self.light = LightAdapter(
            self.config,
            self,
            display_name="Neewer-Licht 1",
            address_attribute="light_address",
            excluded_addresses=lambda: {self.config.light_2_address},
        )
        self.light_2 = LightAdapter(
            self.config,
            self,
            display_name="Neewer-Licht 2",
            address_attribute="light_2_address",
            excluded_addresses=lambda: {self.config.light_address},
        )
        self.acquisition = AcquisitionController(
            self.camera,
            self.robot,
            self.light,
            self.light_2,
            self,
        )

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
        safety = QLabel("UR16e: nur freigegebene Posen über RTDE-Handshake")
        safety.setStyleSheet("color:#b45309; font-weight:600;")
        toolbar.addWidget(safety)
        toolbar.addWidget(settings_button)
        root.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        cards_container = QWidget()
        cards_layout = QVBoxLayout(cards_container)
        self.camera_card = DeviceCard("Baumer Industriekamera")
        self.robot_card = RobotPoseControlCard("Universal Robots UR16e")
        self.light_card = LightControlCard("Neewer RGB660 Pro II · Licht 1")
        self.light_2_card = LightControlCard("Neewer RGB660 Pro II · Licht 2")
        self.acquisition_card = AcquisitionCard()
        self.camera_card.action_requested.connect(lambda: self._toggle(self.camera))
        self.robot_card.action_requested.connect(lambda: self._toggle(self.robot))
        self.light_card.action_requested.connect(lambda: self._toggle(self.light))
        self.light_2_card.action_requested.connect(lambda: self._toggle(self.light_2))
        cards_layout.addWidget(self.acquisition_card)
        cards_layout.addWidget(self.camera_card)
        cards_layout.addWidget(self.robot_card)
        cards_layout.addWidget(self.light_card)
        cards_layout.addWidget(self.light_2_card)
        self.light_power = self.light_card.power_button
        self.light_mode = self.light_card.mode
        self.light_brightness = self.light_card.brightness
        self.light_cct = self.light_card.cct
        self.light_hue = self.light_card.hue
        self.light_saturation = self.light_card.saturation
        self.light_command_timer = self.light_card.command_timer
        self.acquisition_card.set_settings(self.acquisition_config)
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
        self.statusBar().showMessage(f"Kamera {self.config.camera_ip} · UR {self.config.robot_ip}")

    def _wire_adapters(self) -> None:
        for adapter, card in (
            (self.camera, self.camera_card),
            (self.robot, self.robot_card),
            (self.light, self.light_card),
            (self.light_2, self.light_2_card),
        ):
            adapter.state_changed.connect(card.set_state)
            adapter.event_message.connect(self._append_event)
            adapter.error.connect(
                lambda message, name=adapter.display_name: self._show_error(name, message)
            )
        self.camera.status_changed.connect(self._camera_status)
        self.camera.frame_ready.connect(self._camera_frame)
        self.robot.status_changed.connect(self._robot_status)
        self.robot_card.pose_requested.connect(self.robot.request_pose)
        self.light.status_changed.connect(self._light_status)
        self.light.state_changed.connect(self._light_state)
        self.light_2.status_changed.connect(self._light_2_status)
        self.light_2.state_changed.connect(self._light_2_state)
        for adapter, card in (
            (self.light, self.light_card),
            (self.light_2, self.light_2_card),
        ):
            card.power_requested.connect(adapter.set_power)
            card.cct_requested.connect(adapter.set_cct)
            card.hsi_requested.connect(adapter.set_hsi)
        self.acquisition_card.configure_requested.connect(self.open_acquisition_settings)
        self.acquisition_card.start_requested.connect(self.start_acquisition)
        self.acquisition_card.stop_requested.connect(self.acquisition.stop)
        self.acquisition.running_changed.connect(self.acquisition_card.set_running)
        self.acquisition.running_changed.connect(self._acquisition_running)
        self.acquisition.progress_changed.connect(self.acquisition_card.set_progress)
        self.acquisition.status_changed.connect(self._acquisition_status)
        self.acquisition.error.connect(
            lambda message: self._show_error("Automatische Aufnahme", message)
        )

    def _toggle(self, adapter: CameraAdapter | RobotAdapter | LightAdapter) -> None:
        if adapter.state is ConnectionState.DISCONNECTED:
            adapter.connect()
        else:
            adapter.disconnect()

    def connect_all(self) -> None:
        self.camera.connect()
        self.robot.connect()
        self.light.connect()
        self.light_2.connect()

    def disconnect_all(self) -> None:
        self.acquisition.stop()
        self.camera.disconnect()
        self.robot.disconnect()
        self.light.disconnect()
        self.light_2.disconnect()

    def _camera_status(self, status: CameraStatus) -> None:
        self._camera_status_data = status
        fps = "–" if status.camera_fps is None else f"{status.camera_fps:.1f}"
        self.camera_card.details.setText(
            f"Modell: {status.model}\n"
            f"Seriennummer: {status.serial_number}\n"
            f"IP: {status.ip_address}\n"
            f"Bild: {status.width} × {status.height} · {status.pixel_format}\n"
            f"Kamera/Vorschau: {fps} / {status.preview_fps:.1f} FPS"
            f"\nBelichtung: "
            f"{'–' if status.exposure_time_us is None else f'{status.exposure_time_us:.0f} µs'}"
            f" · Auto: {status.exposure_auto}"
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
        self.robot_card.set_status(status)
        scaling = "–" if status.speed_scaling is None else f"{status.speed_scaling * 100:.0f} %"
        joints = (
            "–"
            if not status.joint_positions
            else ", ".join(f"{value:.3f}" for value in status.joint_positions)
        )
        tcp = "–" if not status.tcp_pose else ", ".join(f"{value:.4f}" for value in status.tcp_pose)
        self.robot_card.details.setText(
            f"RTDE/Dashboard: {'ja' if status.rtde_connected else 'nein'} / "
            f"{'ja' if status.dashboard_connected else 'nein'}\n"
            f"Robot/Safety Mode: {status.robot_mode} / {status.safety_mode}\n"
            f"Remote: {status.remote_control} · Programm: {status.program_state}\n"
            f"Geladen: {status.loaded_program}\n"
            f"Speed Scaling: {scaling}\n"
            f"Gelenke [rad]: {joints}\n"
            f"TCP [m, rotvec]: {tcp}\n"
            f"PolyScope: {status.polyscope_version}"
        )

    def _light_state(self, state: ConnectionState) -> None:
        self.light_card.set_connection_state(state)

    def _light_2_state(self, state: ConnectionState) -> None:
        self.light_2_card.set_connection_state(state)

    def _light_status(self, status: LightStatus) -> None:
        self._set_light_status(1, status)

    def _light_2_status(self, status: LightStatus) -> None:
        self._set_light_status(2, status)

    def _set_light_status(self, number: int, status: LightStatus) -> None:
        card = self.light_card if number == 1 else self.light_2_card
        card.set_status(status)
        address_attribute = "light_address" if number == 1 else "light_2_address"
        name_attribute = "light_name" if number == 1 else "light_2_name"
        if status.address not in ("", "–") and status.address != getattr(
            self.config, address_attribute
        ):
            setattr(self.config, address_attribute, status.address)
            setattr(self.config, name_attribute, status.name)
            self.settings_store.save(self.config)

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.config, self)
        if dialog.exec() != SettingsDialog.DialogCode.Accepted or dialog.result_config is None:
            return
        new_config = replace(
            dialog.result_config,
            camera_serial=self.config.camera_serial,
        )
        self.config = new_config
        self.settings_store.save(self.config)
        self.camera.config = self.config
        self.robot.config = self.config
        self.light.config = self.config
        self.light_2.config = self.config
        self.statusBar().showMessage(f"Kamera {self.config.camera_ip} · UR {self.config.robot_ip}")
        self._append_event("Einstellungen gespeichert; sie gelten ab der nächsten Verbindung.")

    def open_acquisition_settings(self) -> None:
        if self.acquisition.running:
            return
        dialog = AcquisitionDialog(
            self.acquisition_config,
            self._camera_status_data,
            self,
        )
        if dialog.exec() != AcquisitionDialog.DialogCode.Accepted:
            return
        if dialog.result_config is None:
            return
        self.acquisition_config = dialog.result_config
        self.settings_store.save_acquisition(self.acquisition_config)
        self.acquisition_card.set_settings(self.acquisition_config)
        self._append_event("Einstellungen für die automatische Aufnahme gespeichert.")

    def start_acquisition(self) -> None:
        count = len(build_capture_points(self.acquisition_config))
        answer = QMessageBox.question(
            self,
            "Automatische Roboterbewegung starten",
            f"Die Sitzung umfasst {count} Aufnahmen und verfährt den UR automatisch.\n\n"
            "Ist der Arbeitsraum frei und darf die Sequenz gestartet werden?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer is QMessageBox.StandardButton.Yes:
            self.acquisition.start(self.acquisition_config)

    def _acquisition_status(self, message: str) -> None:
        self.acquisition_card.status.setText(message)
        self._append_event(f"Aufnahme: {message}")

    def _acquisition_running(self, running: bool) -> None:
        for card in (
            self.camera_card,
            self.robot_card,
            self.light_card,
            self.light_2_card,
        ):
            card.setEnabled(not running)

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
        self.light_card.command_timer.stop()
        self.light_2_card.command_timer.stop()
        self.acquisition.stop()
        self.camera.disconnect()
        self.robot.disconnect()
        event.accept()

    async def shutdown_async(self) -> None:
        self.acquisition.close()
        self.camera.disconnect()
        self.robot.disconnect()
        await self.light.shutdown()
        await self.light_2.shutdown()
