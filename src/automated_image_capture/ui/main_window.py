from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from automated_image_capture.acquisition import AcquisitionController, build_capture_points
from automated_image_capture.hardware import (
    CameraAdapter,
    ConveyorAdapter,
    LightAdapter,
    RobotAdapter,
)
from automated_image_capture.inference import (
    InferenceFrame,
    LiveInferenceConfig,
    LiveInferenceWorker,
)
from automated_image_capture.models import (
    CameraFrame,
    CameraStatus,
    ConnectionState,
    ConveyorStatus,
    LightStatus,
    RobotStatus,
)
from automated_image_capture.settings import SettingsStore
from automated_image_capture.ui.labeling_dialog import LabelingDialog
from automated_image_capture.ui.training_dialog import TrainingDialog
from automated_image_capture.ui.widgets import (
    AcquisitionCard,
    AcquisitionDialog,
    CameraControlCard,
    ConveyorControlCard,
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
        self.inference_config = self.settings_store.load_live_inference()
        self._last_image: QImage | None = None
        self._last_raw_image: QImage | None = None
        self._inference_worker: LiveInferenceWorker | None = None
        self._inference_has_result = False
        self._camera_status_data = CameraStatus()
        self._closing = False
        self._training_dialog: TrainingDialog | None = None
        self.camera = CameraAdapter(self.config, self)
        self.robot = RobotAdapter(self.config, self)
        self.conveyor = ConveyorAdapter(self.config, self)
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
            conveyor=self.conveyor,
        )

        self._build_ui()
        self._wire_adapters()
        if self.acquisition.restore_interrupted(self.acquisition_config):
            if self.acquisition.session_settings is not None:
                self.acquisition_config = self.acquisition.session_settings
                self.acquisition_card.set_settings(self.acquisition_config)
        self._refresh_acquisition_preflight()
        self._append_event("Dashboard bereit. Es werden noch keine Geräte automatisch verbunden.")

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)

        toolbar = QHBoxLayout()
        connect_all = QPushButton("Alle verbinden")
        disconnect_all = QPushButton("Alle trennen")
        labeling_button = QPushButton("OBB-Labels …")
        settings_button = QPushButton("Einstellungen …")
        training_button = QPushButton("YOLO-Training …")
        connect_all.clicked.connect(self.connect_all)
        disconnect_all.clicked.connect(self.disconnect_all)
        labeling_button.clicked.connect(self.open_labeling)
        training_button.clicked.connect(self.open_training)
        settings_button.clicked.connect(self.open_settings)
        toolbar.addWidget(connect_all)
        toolbar.addWidget(disconnect_all)
        toolbar.addWidget(labeling_button)
        toolbar.addWidget(training_button)
        toolbar.addStretch(1)
        safety = QLabel("UR16e: Bewegungsziele nur über den geprüften RTDE-Handshake")
        safety.setStyleSheet("color:#b45309; font-weight:600;")
        toolbar.addWidget(safety)
        toolbar.addWidget(settings_button)
        root.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        cards_container = QWidget()
        cards_layout = QVBoxLayout(cards_container)
        self.camera_card = CameraControlCard("Baumer Industriekamera")
        self.robot_card = RobotPoseControlCard("Universal Robots UR16e")
        self.conveyor_card = ConveyorControlCard(
            "TwinCAT-Förderband", self.config.conveyor_forward_direction
        )
        self.light_card = LightControlCard("Neewer RGB660 Pro II · Licht 1")
        self.light_2_card = LightControlCard("Neewer RGB660 Pro II · Licht 2")
        self.acquisition_card = AcquisitionCard()
        self.camera_card.action_requested.connect(lambda: self._toggle(self.camera))
        self.robot_card.action_requested.connect(lambda: self._toggle(self.robot))
        self.conveyor_card.action_requested.connect(lambda: self._toggle(self.conveyor))
        self.light_card.action_requested.connect(lambda: self._toggle(self.light))
        self.light_2_card.action_requested.connect(lambda: self._toggle(self.light_2))
        cards_layout.addWidget(self.acquisition_card)
        cards_layout.addWidget(self.camera_card)
        cards_layout.addWidget(self.robot_card)
        cards_layout.addWidget(self.conveyor_card)
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

        inference_controls = QHBoxLayout()
        self.inference_toggle = QCheckBox("YOLO Live-Erkennung")
        self.inference_toggle.setToolTip(
            "Führt das trainierte OBB-Modell auf dem neuesten Kamerabild aus."
        )
        self.inference_model = QLineEdit()
        self.inference_model.setReadOnly(True)
        self.inference_model.setMinimumWidth(220)
        self.inference_model.setText(str(self.inference_config.model_path))
        self.inference_model.setToolTip(str(self.inference_config.model_path))
        choose_model = QPushButton("Modell …")
        choose_model.clicked.connect(self._choose_inference_model)
        self.inference_confidence = QDoubleSpinBox()
        self.inference_confidence.setRange(0.01, 1.0)
        self.inference_confidence.setSingleStep(0.05)
        self.inference_confidence.setDecimals(2)
        self.inference_confidence.setValue(self.inference_config.confidence)
        self.inference_confidence.setPrefix("Konf. ")
        self.inference_confidence.setToolTip("Minimale Konfidenz für eine angezeigte OBB")
        self.inference_max_fps = QSpinBox()
        self.inference_max_fps.setRange(1, 15)
        self.inference_max_fps.setValue(round(self.inference_config.max_fps))
        self.inference_max_fps.setSuffix(" FPS")
        self.inference_max_fps.setToolTip("Maximale Bildrate der YOLO-Inferenz")
        self.inference_status = QLabel("Aus")
        self.inference_status.setMinimumWidth(150)
        self.inference_status.setStyleSheet("color:#64748b;")
        self.inference_toggle.toggled.connect(self._toggle_live_inference)
        self.inference_confidence.editingFinished.connect(
            self._apply_inference_runtime_settings
        )
        self.inference_max_fps.editingFinished.connect(
            self._apply_inference_runtime_settings
        )
        inference_controls.addWidget(self.inference_toggle)
        inference_controls.addWidget(self.inference_model, 1)
        inference_controls.addWidget(choose_model)
        inference_controls.addWidget(self.inference_confidence)
        inference_controls.addWidget(self.inference_max_fps)
        inference_controls.addWidget(self.inference_status)
        preview_layout.addLayout(inference_controls)

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
            (self.conveyor, self.conveyor_card),
            (self.light, self.light_card),
            (self.light_2, self.light_2_card),
        ):
            adapter.state_changed.connect(card.set_state)
            adapter.state_changed.connect(self._refresh_acquisition_preflight)
            adapter.event_message.connect(self._append_event)
            adapter.error.connect(
                lambda message, name=adapter.display_name: self._show_error(name, message)
            )
        self.camera.status_changed.connect(self._camera_status)
        self.camera.frame_ready.connect(self._camera_frame)
        self.camera_card.exposure_requested.connect(self.camera.set_exposure_time)
        self.robot.status_changed.connect(self._robot_status)
        self.robot_card.pose_requested.connect(self.robot.request_pose)
        self.conveyor.status_changed.connect(self._conveyor_status)
        self.conveyor_card.jog_requested.connect(self._jog_conveyor)
        self.conveyor_card.stop_requested.connect(self.conveyor.stop_motion)
        self.conveyor_card.origin_requested.connect(self.conveyor.set_current_as_origin)
        self.conveyor_card.forward_direction_changed.connect(
            self._set_conveyor_forward_direction
        )
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
        self.acquisition_card.resume_requested.connect(self.resume_acquisition)
        self.acquisition_card.stop_requested.connect(self.acquisition.stop)
        self.acquisition_card.align_conveyor_requested.connect(
            self._align_conveyor_for_resume
        )
        self.acquisition.running_changed.connect(self.acquisition_card.set_running)
        self.acquisition.resume_available_changed.connect(
            self.acquisition_card.set_resume_available
        )
        self.acquisition.running_changed.connect(self._acquisition_running)
        self.acquisition.progress_changed.connect(self.acquisition_card.set_progress)
        self.acquisition.status_changed.connect(self._acquisition_status)
        self.acquisition.error.connect(
            lambda message: self._show_error("Automatische Aufnahme", message)
        )
        self.acquisition.alignment_required_changed.connect(
            self.acquisition_card.set_alignment_required
        )

    def _toggle(
        self, adapter: CameraAdapter | ConveyorAdapter | RobotAdapter | LightAdapter
    ) -> None:
        if adapter.state is ConnectionState.DISCONNECTED:
            adapter.connect()
        else:
            adapter.disconnect()

    def connect_all(self) -> None:
        self.camera.connect()
        self.robot.connect()
        self.conveyor.connect()
        self.light.connect()
        self.light_2.connect()

    def disconnect_all(self) -> None:
        self.acquisition.stop()
        self.camera.disconnect()
        self.robot.disconnect()
        self.conveyor.disconnect()
        self.light.disconnect()
        self.light_2.disconnect()

    def _camera_status(self, status: CameraStatus) -> None:
        self._camera_status_data = status
        self.camera_card.set_status(status)
        self._refresh_acquisition_preflight()
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
        raw_image = QImage(
            image.data,
            width,
            height,
            int(image.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        self._last_raw_image = raw_image
        worker = self._inference_worker
        if worker is not None and worker.isRunning():
            worker.submit(image, frame.timestamp)
            if not self._inference_has_result:
                self._last_image = raw_image
                self._render_image()
        else:
            self._last_image = raw_image
            self._render_image()

    def _current_inference_config(self) -> LiveInferenceConfig:
        return LiveInferenceConfig(
            model_path=self.inference_config.model_path,
            confidence=self.inference_confidence.value(),
            image_size=self.inference_config.image_size,
            max_fps=float(self.inference_max_fps.value()),
            device=self.inference_config.device,
        )

    def _choose_inference_model(self) -> None:
        current = self.inference_config.model_path
        start_directory = str(current.parent if current.is_file() else current)
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "YOLO-OBB-Modell auswählen",
            start_directory,
            "PyTorch-Modell (*.pt)",
        )
        if not filename:
            return
        was_running = self.inference_toggle.isChecked()
        if was_running:
            self.inference_toggle.setChecked(False)
        self.inference_config = replace(
            self._current_inference_config(), model_path=Path(filename)
        )
        self.inference_model.setText(filename)
        self.inference_model.setToolTip(filename)
        self.settings_store.save_live_inference(self.inference_config)
        self._append_event(f"Live-Modell ausgewählt: {filename}")
        if was_running:
            self.inference_toggle.setChecked(True)

    def _toggle_live_inference(self, enabled: bool) -> None:
        if not enabled:
            self._stop_live_inference()
            self.inference_status.setText("Aus")
            self._inference_has_result = False
            if self._last_raw_image is not None:
                self._last_image = self._last_raw_image
                self._render_image()
            return
        try:
            config = self._current_inference_config().validated()
        except ValueError as exc:
            self.inference_toggle.blockSignals(True)
            self.inference_toggle.setChecked(False)
            self.inference_toggle.blockSignals(False)
            self.inference_status.setText("Kein Modell")
            QMessageBox.warning(self, "Live-Erkennung", str(exc))
            return

        self.inference_config = config
        self.settings_store.save_live_inference(config)
        self._inference_has_result = False
        worker = LiveInferenceWorker(config, self)
        worker.frame_ready.connect(
            lambda frame, source=worker: self._inference_frame_from(source, frame)
        )
        worker.status_changed.connect(
            lambda message, source=worker: self._inference_status_changed(source, message)
        )
        worker.error.connect(
            lambda message, source=worker: self._inference_error_from(source, message)
        )
        worker.finished.connect(lambda source=worker: self._inference_finished(source))
        self._inference_worker = worker
        worker.start()
        self._append_event(
            f"YOLO-Live-Erkennung gestartet ({config.model_path.name}, "
            f"Konfidenz {config.confidence:.2f}, max. {config.max_fps:g} FPS)."
        )

    def _apply_inference_runtime_settings(self) -> None:
        try:
            config = self._current_inference_config().validated(require_model=False)
        except ValueError as exc:
            self._show_error("Live-Erkennung", str(exc))
            return
        self.inference_config = config
        self.settings_store.save_live_inference(config)
        if self._inference_worker is not None:
            self._inference_worker.update_runtime_settings(
                config.confidence,
                config.max_fps,
            )

    def _inference_frame(self, frame: InferenceFrame) -> None:
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
        self._inference_has_result = True
        count = len(frame.detections)
        detection_text = "1 Objekt" if count == 1 else f"{count} Objekte"
        self.inference_status.setText(f"{detection_text} · {frame.inference_ms:.0f} ms")
        self._render_image()

    def _inference_frame_from(
        self,
        worker: LiveInferenceWorker,
        frame: InferenceFrame,
    ) -> None:
        if worker is self._inference_worker:
            self._inference_frame(frame)

    def _inference_status_changed(
        self,
        worker: LiveInferenceWorker,
        message: str,
    ) -> None:
        if worker is self._inference_worker:
            self.inference_status.setText(message)

    def _inference_error(self, message: str) -> None:
        self._show_error("Live-Erkennung", message)
        self._append_event(message)

    def _inference_error_from(self, worker: LiveInferenceWorker, message: str) -> None:
        if worker is self._inference_worker:
            self._inference_error(message)

    def _inference_finished(self, worker: LiveInferenceWorker) -> None:
        if worker is not self._inference_worker:
            return
        self._inference_worker = None
        if self.inference_toggle.isChecked():
            self.inference_toggle.blockSignals(True)
            self.inference_toggle.setChecked(False)
            self.inference_toggle.blockSignals(False)
        self._inference_has_result = False
        if self._last_raw_image is not None:
            self._last_image = self._last_raw_image
            self._render_image()

    def _stop_live_inference(self) -> None:
        worker = self._inference_worker
        if worker is None:
            return
        self._inference_worker = None
        if not worker.stop():
            self._logger.warning("YOLO-Inferenz-Thread reagierte nicht innerhalb des Timeouts.")
        worker.deleteLater()

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
        self._refresh_acquisition_preflight()
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

    def _conveyor_status(self, status: ConveyorStatus) -> None:
        self.conveyor_card.set_status(status)
        self._refresh_acquisition_preflight()

    def _jog_conveyor(self, direction: str) -> None:
        answer = QMessageBox.question(
            self,
            "Förderband-Testfahrt",
            f"Das Förderband fährt 1 mm nach {direction}. Ist der Arbeitsraum frei?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer is QMessageBox.StandardButton.Yes:
            self.conveyor.jog(direction, 1.0, 10.0)

    def _set_conveyor_forward_direction(self, direction: str) -> None:
        if not direction:
            self.config.conveyor_forward_direction = ""
        else:
            self.conveyor.set_forward_direction(direction)
            self.config.conveyor_forward_direction = direction
        self.settings_store.save(self.config)
        self._refresh_acquisition_preflight()
        if direction:
            self._append_event(
                f"Förderband-Vorwärtsrichtung gespeichert: "
                f"{'links' if direction == 'left' else 'rechts'}."
            )

    def _align_conveyor_for_resume(self) -> None:
        expected = self.acquisition.expected_resume_offset_mm
        if expected is None:
            return
        current = self.conveyor.status.logical_offset_mm
        answer = QMessageBox.question(
            self,
            "Förderband auf Checkpoint ausrichten",
            f"Das Förderband fährt von "
            f"{'unbekannt' if current is None else f'{current:.3f} mm'} auf "
            f"{expected:g} mm. Ist der Arbeitsraum frei?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer is QMessageBox.StandardButton.Yes:
            self.acquisition.align_for_resume()

    def _light_state(self, state: ConnectionState) -> None:
        self.light_card.set_connection_state(state)

    def _light_2_state(self, state: ConnectionState) -> None:
        self.light_2_card.set_connection_state(state)

    def _light_status(self, status: LightStatus) -> None:
        self._set_light_status(1, status)
        self._refresh_acquisition_preflight()

    def _light_2_status(self, status: LightStatus) -> None:
        self._set_light_status(2, status)
        self._refresh_acquisition_preflight()

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
        self.conveyor.config = self.config
        self.light.config = self.config
        self.light_2.config = self.config
        self.statusBar().showMessage(f"Kamera {self.config.camera_ip} · UR {self.config.robot_ip}")
        self._append_event("Einstellungen gespeichert; sie gelten ab der nächsten Verbindung.")
        self._refresh_acquisition_preflight()

    def open_labeling(self) -> None:
        dialog = LabelingDialog(self.settings_store, self)
        dialog.exec()

    def open_training(self) -> None:
        if self._training_dialog is None:
            self._training_dialog = TrainingDialog(self.settings_store, self)
            self._training_dialog.finished.connect(self._training_dialog_closed)
        self._training_dialog.show()
        self._training_dialog.raise_()
        self._training_dialog.activateWindow()

    def _training_dialog_closed(self) -> None:
        if self._training_dialog is not None:
            self._training_dialog.deleteLater()
        self._training_dialog = None

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
        self._refresh_acquisition_preflight()
        self._append_event("Einstellungen für die automatische Aufnahme gespeichert.")

    def start_acquisition(self) -> None:
        checks = self.acquisition.preflight_checks(self.acquisition_config)
        self.acquisition_card.set_preflight(checks)
        failed = [check for check in checks if not check.ready]
        if failed:
            self._acquisition_status(
                "Start blockiert: "
                + "; ".join(f"{check.label}: {check.detail}" for check in failed)
            )
            return
        count = len(build_capture_points(self.acquisition_config))
        moving_devices = "UR und Förderband" if self.acquisition_config.conveyor_enabled else "UR"
        interrupted = (
            "\n\nEine unterbrochene Sitzung ist vorhanden. Beim Start einer neuen Sitzung "
            "wird deren Fortsetzung verworfen."
            if self.acquisition.resume_available
            else ""
        )
        answer = QMessageBox.question(
            self,
            "Automatische Roboterbewegung starten",
            f"Die Sitzung umfasst {count} Aufnahmen und verfährt {moving_devices} automatisch.\n\n"
            f"Ist der Arbeitsraum frei und darf die Sequenz gestartet werden?{interrupted}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer is QMessageBox.StandardButton.Yes:
            self.acquisition.start(self.acquisition_config)

    def resume_acquisition(self) -> None:
        if not self.acquisition.resume_available:
            return
        remaining = self.acquisition.remaining_count
        session = self.acquisition.session_directory or "–"
        answer = QMessageBox.question(
            self,
            "Automatische Aufnahme fortsetzen",
            f"Es fehlen noch {remaining} Aufnahmen in:\n{session}\n\n"
            "Die Hardware wird erneut geprüft; UR und gegebenenfalls Förderband fahren auf "
            "das nächste benötigte Ziel. "
            "Ist der Arbeitsraum frei und wurde die Fehlerursache behoben?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer is QMessageBox.StandardButton.Yes:
            self.acquisition.resume()

    def _acquisition_status(self, message: str) -> None:
        self.acquisition_card.status.setText(message)
        self._append_event(f"Aufnahme: {message}")

    def _acquisition_running(self, running: bool) -> None:
        for card in (
            self.camera_card,
            self.robot_card,
            self.conveyor_card,
            self.light_card,
            self.light_2_card,
        ):
            card.setEnabled(not running)
        if not running:
            self._refresh_acquisition_preflight()

    def _refresh_acquisition_preflight(self, *_: object) -> None:
        if self.acquisition.running:
            return
        checks = self.acquisition.preflight_checks(self.acquisition_config)
        self.acquisition_card.set_preflight(checks)

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
        self._stop_live_inference()
        if self._training_dialog is not None:
            self._training_dialog.shutdown()
        self.camera.disconnect()
        self.robot.disconnect()
        self.conveyor.disconnect()
        event.accept()

    async def shutdown_async(self) -> None:
        self._stop_live_inference()
        if self._training_dialog is not None:
            self._training_dialog.shutdown()
        self.acquisition.close()
        self.camera.disconnect()
        self.robot.disconnect()
        self.conveyor.disconnect()
        await self.light.shutdown()
        await self.light_2.shutdown()
