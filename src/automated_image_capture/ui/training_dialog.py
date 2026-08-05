from __future__ import annotations

import json
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, Qt, QThread, QTimer, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCloseEvent, QDesktopServices, QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from automated_image_capture.dataset import (
    DatasetBuildCancelled,
    DatasetBuildConfig,
    DatasetBuildResult,
    DatasetError,
    DatasetRecord,
    build_curated_dataset,
    collect_dataset_records,
    default_build_config,
    render_record_preview,
    save_curation,
)
from automated_image_capture.settings import SettingsStore
from automated_image_capture.training import EVENT_PREFIX


class DatasetBuildWorker(QObject):
    progress = pyqtSignal(int, int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, config: DatasetBuildConfig) -> None:
        super().__init__()
        self.config = config
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = build_curated_dataset(
                self.config,
                lambda done, total, message: self.progress.emit(done, total, message),
                self._cancelled.is_set,
            )
        except DatasetBuildCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc) or type(exc).__name__)
        else:
            self.completed.emit(result)


class TrainingDialog(QDialog):
    def __init__(self, settings_store: SettingsStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("YOLO26-OBB-Training")
        self.resize(1220, 850)
        self.settings_store = settings_store
        defaults = default_build_config()
        paths = settings_store.load_training_paths(defaults)
        self._records: list[DatasetRecord] = []
        self._excluded_ids: set[str] = set()
        self._dataset_directory: Path | None = paths.get("dataset_directory")
        if self._dataset_directory is None:
            candidates = sorted(
                (
                    path.parent
                    for path in Path(paths["output_root"]).glob("dataset_*/data.yaml")
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            self._dataset_directory = candidates[0] if candidates else None
        self._result_directory: Path | None = None
        self._build_thread: QThread | None = None
        self._build_worker: DatasetBuildWorker | None = None
        self._process: QProcess | None = None
        self._process_buffer = ""
        self._build_ui(paths)
        self._set_busy(False)
        if self._dataset_directory is not None and (
            self._dataset_directory / "data.yaml"
        ).is_file():
            self.dataset_status.setText(f"Vorhandener Datensatz: {self._dataset_directory}")
            self.train_button.setEnabled(True)
            self.open_dataset_button.setEnabled(True)

    def _build_ui(self, paths: dict[str, Path | None]) -> None:
        root = QVBoxLayout(self)
        path_form = QFormLayout()
        self.pose1_path = QLineEdit(str(paths["pose1_dataset"]))
        self.pose2_path = QLineEdit(str(paths["pose2_dataset"]))
        self.output_path = QLineEdit(str(paths["output_root"]))
        path_form.addRow("Pose-1-Labels", self._path_row(self.pose1_path, self._browse_pose1))
        path_form.addRow("Pose-2-Labels", self._path_row(self.pose2_path, self._browse_pose2))
        path_form.addRow("Datensatz-Ausgabe", self._path_row(self.output_path, self._browse_output))
        root.addLayout(path_form)

        review_toolbar = QHBoxLayout()
        self.load_button = QPushButton("Bilder laden / aktualisieren")
        self.load_button.clicked.connect(self.load_records)
        self.filter = QComboBox()
        self.filter.addItems(["Auffällige zuerst", "Nur REVIEW", "Alle", "Ausgeschlossen"])
        self.filter.currentIndexChanged.connect(self._populate_table)
        self.review_summary = QLabel("Noch keine Bilder geladen")
        review_toolbar.addWidget(self.load_button)
        review_toolbar.addWidget(QLabel("Anzeige:"))
        review_toolbar.addWidget(self.filter)
        review_toolbar.addWidget(self.review_summary, 1)
        root.addLayout(review_toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Verwenden", "Qualität", "Klasse", "UR-Pose", "Panel 1", "Panel 2", "Datei"]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._show_selection)
        self.table.itemChanged.connect(self._include_changed)
        splitter.addWidget(self.table)
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        self.preview = QLabel("Bild auswählen")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(520, 390)
        self.preview.setStyleSheet("background:#111827; color:#d1d5db; padding:8px;")
        preview_layout.addWidget(self.preview, 1)
        self.preview_details = QLabel("")
        self.preview_details.setWordWrap(True)
        preview_layout.addWidget(self.preview_details)
        splitter.addWidget(preview_container)
        splitter.setSizes([650, 550])
        root.addWidget(splitter, 1)

        build_row = QHBoxLayout()
        self.build_button = QPushButton("Kuratieren und Datensatz erzeugen")
        self.build_button.clicked.connect(self.build_dataset)
        self.open_dataset_button = QPushButton("Datensatz öffnen")
        self.open_dataset_button.clicked.connect(self._open_dataset)
        self.open_dataset_button.setEnabled(False)
        self.dataset_status = QLabel("Noch kein Datensatz erzeugt")
        self.dataset_status.setWordWrap(True)
        build_row.addWidget(self.build_button)
        build_row.addWidget(self.open_dataset_button)
        build_row.addWidget(self.dataset_status, 1)
        root.addLayout(build_row)

        training_form = QHBoxLayout()
        self.model = QLineEdit("yolo26n-obb.pt")
        self.model.setMaximumWidth(180)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 5000)
        self.epochs.setValue(200)
        self.patience = QSpinBox()
        self.patience.setRange(0, 1000)
        self.patience.setValue(40)
        self.image_size = QSpinBox()
        self.image_size.setRange(128, 4096)
        self.image_size.setSingleStep(32)
        self.image_size.setValue(640)
        training_form.addWidget(QLabel("Modell:"))
        training_form.addWidget(self.model)
        training_form.addWidget(QLabel("Epochen:"))
        training_form.addWidget(self.epochs)
        training_form.addWidget(QLabel("Patience:"))
        training_form.addWidget(self.patience)
        training_form.addWidget(QLabel("Bildgröße:"))
        training_form.addWidget(self.image_size)
        training_form.addStretch(1)
        root.addLayout(training_form)

        action_row = QHBoxLayout()
        self.train_button = QPushButton("Training starten")
        self.train_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton("Stoppen")
        self.stop_button.clicked.connect(self.stop)
        self.open_result_button = QPushButton("Ergebnisse öffnen")
        self.open_result_button.clicked.connect(self._open_results)
        self.open_result_button.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        action_row.addWidget(self.train_button)
        action_row.addWidget(self.stop_button)
        action_row.addWidget(self.open_result_button)
        action_row.addWidget(self.progress, 1)
        root.addLayout(action_row)

        self.status = QLabel("Bereit")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        self.log.setMaximumHeight(180)
        root.addWidget(self.log)

    @staticmethod
    def _path_row(field: QLineEdit, callback: object) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("…")
        button.setMaximumWidth(42)
        button.clicked.connect(callback)  # type: ignore[arg-type]
        layout.addWidget(field, 1)
        layout.addWidget(button)
        return widget

    def _choose_directory(self, field: QLineEdit, title: str) -> None:
        chosen = QFileDialog.getExistingDirectory(self, title, field.text())
        if chosen:
            field.setText(chosen)

    def _browse_pose1(self) -> None:
        self._choose_directory(self.pose1_path, "Pose-1-Labeldatensatz auswählen")

    def _browse_pose2(self) -> None:
        self._choose_directory(self.pose2_path, "Pose-2-Labeldatensatz auswählen")

    def _browse_output(self) -> None:
        self._choose_directory(self.output_path, "Ausgabeordner auswählen")

    def _config(self) -> DatasetBuildConfig:
        output = Path(self.output_path.text().strip())
        return DatasetBuildConfig(
            pose1_dataset=Path(self.pose1_path.text().strip()),
            pose2_dataset=Path(self.pose2_path.text().strip()),
            output_root=output,
            curation_path=output / "curation.json",
        ).validated()

    def load_records(self) -> None:
        try:
            config = self._config()
            self._records = collect_dataset_records(config)
        except DatasetError as exc:
            QMessageBox.critical(self, "Datensatz kann nicht geladen werden", str(exc))
            return
        self._excluded_ids = {record.record_id for record in self._records if record.excluded}
        self.settings_store.save_training_paths(
            config.pose1_dataset,
            config.pose2_dataset,
            config.output_root,
            self._dataset_directory,
        )
        self._populate_table()
        self._update_review_summary()
        self.status.setText("Review geladen. Deaktiviere nur sichtbar fehlerhafte Bilder.")

    def _filtered_records(self) -> list[DatasetRecord]:
        records = list(self._records)
        mode = self.filter.currentText()
        if mode == "Nur REVIEW":
            records = [record for record in records if record.quality == "REVIEW"]
        elif mode == "Ausgeschlossen":
            records = [record for record in records if record.record_id in self._excluded_ids]
        if mode == "Auffällige zuerst":
            records.sort(
                key=lambda record: (
                    record.quality != "REVIEW",
                    record.class_id is None,
                    record.class_id if record.class_id is not None else 9,
                    record.pose_id,
                    record.panel_2,
                    record.panel_1,
                )
            )
        else:
            records.sort(
                key=lambda record: (
                    record.class_id is None,
                    record.class_id if record.class_id is not None else 9,
                    record.pose_id,
                    record.panel_2,
                    record.panel_1,
                )
            )
        return records

    @pyqtSlot()
    def _populate_table(self) -> None:
        records = self._filtered_records()
        self.table.blockSignals(True)
        self.table.setRowCount(len(records))
        for row, record in enumerate(records):
            include = QTableWidgetItem("")
            include.setFlags(include.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            include.setCheckState(
                Qt.CheckState.Unchecked
                if record.record_id in self._excluded_ids
                else Qt.CheckState.Checked
            )
            include.setData(Qt.ItemDataRole.UserRole, record.record_id)
            self.table.setItem(row, 0, include)
            values = (
                record.quality,
                record.class_name,
                str(record.pose_id),
                f"{record.panel_1} %",
                f"{record.panel_2} %",
                record.target_name,
            )
            for column, value in enumerate(values, 1):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, record.record_id)
                self.table.setItem(row, column, item)
        self.table.blockSignals(False)
        if records:
            self.table.selectRow(0)

    def _record_by_id(self, record_id: str) -> DatasetRecord | None:
        return next((record for record in self._records if record.record_id == record_id), None)

    @pyqtSlot()
    def _show_selection(self) -> None:
        row = self.table.currentRow()
        if row < 0 or self.table.item(row, 0) is None:
            return
        record_id = str(self.table.item(row, 0).data(Qt.ItemDataRole.UserRole))
        record = self._record_by_id(record_id)
        if record is None:
            return
        try:
            image = render_record_preview(record)
        except DatasetError as exc:
            self.preview.setText(str(exc))
            return
        height, width, channels = image.shape
        qimage = QImage(
            image.data,
            width,
            height,
            int(image.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        self.preview.setPixmap(
            QPixmap.fromImage(qimage).scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        iou = "–" if record.consensus_iou is None else f"{record.consensus_iou:.3f}"
        self.preview_details.setText(
            f"{record.class_name} · Split {record.split} · UR {record.pose_id} · "
            f"P1/P2 {record.panel_1}/{record.panel_2} · Konsens-IoU {iou}\n"
            f"{record.source_image}"
        )

    @pyqtSlot(QTableWidgetItem)
    def _include_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        record_id = str(item.data(Qt.ItemDataRole.UserRole))
        if item.checkState() == Qt.CheckState.Checked:
            self._excluded_ids.discard(record_id)
        else:
            self._excluded_ids.add(record_id)
        self._update_review_summary()

    def _update_review_summary(self) -> None:
        flagged = sum(record.quality == "REVIEW" for record in self._records)
        self.review_summary.setText(
            f"{len(self._records)} Bilder · {flagged} automatisch auffällig · "
            f"{len(self._excluded_ids)} ausgeschlossen"
        )

    def build_dataset(self) -> None:
        if not self._records:
            self.load_records()
            if not self._records:
                return
        config = self._config()
        curation_path = config.curation_path or config.output_root / "curation.json"
        save_curation(curation_path, self._excluded_ids)
        config = replace(config, curation_path=curation_path)
        self._build_thread = QThread(self)
        self._build_worker = DatasetBuildWorker(config)
        self._build_worker.moveToThread(self._build_thread)
        self._build_thread.started.connect(self._build_worker.run)
        self._build_worker.progress.connect(self._build_progress)
        self._build_worker.completed.connect(self._build_completed)
        self._build_worker.failed.connect(self._build_failed)
        self._build_worker.cancelled.connect(self._build_cancelled)
        self._build_worker.completed.connect(self._build_thread.quit)
        self._build_worker.failed.connect(self._build_thread.quit)
        self._build_worker.cancelled.connect(self._build_thread.quit)
        self._build_thread.finished.connect(self._build_finished)
        self._set_busy(True)
        self.progress.setRange(0, len(self._records) - len(self._excluded_ids))
        self.progress.setValue(0)
        self.status.setText("Erzeuge versionierten Datensatz …")
        self._build_thread.start()

    @pyqtSlot(int, int, str)
    def _build_progress(self, done: int, total: int, message: str) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.status.setText(f"{message} · {done}/{total}")

    @pyqtSlot(object)
    def _build_completed(self, result: object) -> None:
        if not isinstance(result, DatasetBuildResult):
            self._build_failed("Unbekanntes Ergebnis der Datensatzaufbereitung.")
            return
        self._dataset_directory = result.dataset_directory
        self.dataset_status.setText(
            f"{result.dataset_directory} · {result.included_images} Bilder, "
            f"{result.excluded_images} ausgeschlossen"
        )
        self.status.setText(
            "Datensatz geprüft: "
            + ", ".join(f"{key} {value}" for key, value in result.split_counts.items())
        )
        self.open_dataset_button.setEnabled(True)
        self.train_button.setEnabled(True)
        config = self._config()
        self.settings_store.save_training_paths(
            config.pose1_dataset,
            config.pose2_dataset,
            config.output_root,
            self._dataset_directory,
        )

    @pyqtSlot(str)
    def _build_failed(self, message: str) -> None:
        self.status.setText(f"Datensatzfehler: {message}")
        QMessageBox.critical(self, "Datensatzaufbereitung fehlgeschlagen", message)

    @pyqtSlot()
    def _build_cancelled(self) -> None:
        self.status.setText("Datensatzaufbereitung abgebrochen; Quelldaten blieben unverändert.")

    @pyqtSlot()
    def _build_finished(self) -> None:
        if self._build_worker is not None:
            self._build_worker.deleteLater()
        if self._build_thread is not None:
            self._build_thread.deleteLater()
        self._build_worker = None
        self._build_thread = None
        self._set_busy(False)

    def start_training(self) -> None:
        if self._dataset_directory is None or not (self._dataset_directory / "data.yaml").is_file():
            QMessageBox.warning(self, "Kein Datensatz", "Bitte zuerst einen Datensatz erzeugen.")
            return
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_training_output)
        self._process.finished.connect(self._training_finished)
        arguments = [
            "-m",
            "automated_image_capture.training",
            "train",
            "--dataset",
            str(self._dataset_directory),
            "--model",
            self.model.text().strip(),
            "--epochs",
            str(self.epochs.value()),
            "--patience",
            str(self.patience.value()),
            "--imgsz",
            str(self.image_size.value()),
        ]
        self._process_buffer = ""
        self.progress.setRange(0, self.epochs.value())
        self.progress.setValue(0)
        self._set_busy(True)
        self.status.setText("Starte Training und CUDA-Prüfung …")
        self.log.clear()
        self._process.start(sys.executable, arguments)

    @pyqtSlot()
    def _read_training_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._process_buffer += data
        lines = self._process_buffer.splitlines(keepends=True)
        self._process_buffer = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._process_buffer = lines.pop()
        for line in lines:
            text = line.rstrip()
            if text.startswith(EVENT_PREFIX):
                try:
                    self._handle_training_event(json.loads(text[len(EVENT_PREFIX) :]))
                    continue
                except json.JSONDecodeError:
                    pass
            self.log.appendPlainText(text)

    def _handle_training_event(self, event: dict[str, object]) -> None:
        kind = event.get("event")
        if kind == "epoch":
            epoch = int(event.get("epoch", 0))
            total = int(event.get("total", self.epochs.value()))
            self.progress.setRange(0, total)
            self.progress.setValue(epoch)
            self.status.setText(f"Training: Epoche {epoch}/{total}")
        elif kind == "stage":
            self.status.setText(str(event.get("message", event.get("name", "Training"))))
        elif kind == "diagnostics":
            devices = event.get("devices", [])
            self.log.appendPlainText(
                f"PyTorch {event.get('torch_version')} · CUDA {event.get('torch_cuda_version')} · "
                f"GPU: {', '.join(str(value) for value in devices)}"
            )
        elif kind == "evaluation_progress":
            self.status.setText(
                f"Leerbilder auswerten: {event.get('done')}/{event.get('total')}"
            )
        elif kind == "cache_progress":
            done = int(event.get("done", 0))
            total = int(event.get("total", 1))
            self.progress.setRange(0, total)
            self.progress.setValue(done)
            self.status.setText(f"Trainingscache: {done}/{total}")
        elif kind == "cache_ready":
            reused = "wiederverwendet" if event.get("reused") else "erzeugt"
            self.status.setText(f"Trainingscache {reused}; starte GPU-Training …")
        elif kind == "completed":
            self._result_directory = Path(str(event["run_directory"]))
            self.open_result_button.setEnabled(True)
            rate = float(event.get("empty_false_positive_rate", 0.0)) * 100
            test_metrics = event.get("test_metrics", {})
            map50 = None
            map50_95 = None
            if isinstance(test_metrics, dict):
                map50 = test_metrics.get("metrics/mAP50(B)")
                map50_95 = test_metrics.get("metrics/mAP50-95(B)")
            metrics_text = ""
            if map50 is not None and map50_95 is not None:
                metrics_text = f" · Test mAP50 {float(map50):.3f}, mAP50–95 {float(map50_95):.3f}"
            self.status.setText(
                f"Training und Test abgeschlossen{metrics_text} · "
                f"Leerbild-Fehlalarme {rate:.1f} %"
            )
        elif kind == "error":
            self.status.setText(f"Trainingsfehler: {event.get('message')}")
        self.log.appendPlainText(json.dumps(event, ensure_ascii=False))

    @pyqtSlot(int, QProcess.ExitStatus)
    def _training_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_training_output()
        if exit_code != 0 and not self.status.text().startswith("Trainingsfehler"):
            self.status.setText(
                f"Training beendet mit Fehlercode {exit_code}. Details im Protokoll."
            )
        elif exit_code == 0 and self._result_directory is None:
            self.status.setText("Trainingsprozess erfolgreich beendet.")
        if self._process is not None:
            self._process.deleteLater()
        self._process = None
        self._set_busy(False)

    def stop(self) -> None:
        if self._build_worker is not None:
            self._build_worker.cancel()
            self.stop_button.setEnabled(False)
            self.status.setText("Abbruch der Datensatzaufbereitung angefordert …")
        elif self._process is not None:
            process = self._process
            self.status.setText("Beende Training geordnet …")
            self._terminate_training_process(force=False)
            QTimer.singleShot(
                5000,
                lambda: self._terminate_training_process(force=True)
                if process.state() != QProcess.ProcessState.NotRunning
                else None,
            )

    def _terminate_training_process(self, force: bool) -> None:
        process = self._process
        if process is None or process.state() == QProcess.ProcessState.NotRunning:
            return
        if sys.platform != "win32":
            process.kill() if force else process.terminate()
            return
        arguments = ["taskkill", "/PID", str(process.processId()), "/T"]
        if force:
            arguments.append("/F")
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.run(
            arguments,
            capture_output=True,
            check=False,
            timeout=3,
            creationflags=creation_flags,
        )

    def _set_busy(self, busy: bool) -> None:
        self.load_button.setEnabled(not busy)
        self.build_button.setEnabled(not busy)
        self.train_button.setEnabled(not busy and self._dataset_directory is not None)
        self.stop_button.setEnabled(busy)
        self.pose1_path.setEnabled(not busy)
        self.pose2_path.setEnabled(not busy)
        self.output_path.setEnabled(not busy)

    def _open_dataset(self) -> None:
        if self._dataset_directory is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._dataset_directory)))

    def _open_results(self) -> None:
        if self._result_directory is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._result_directory)))

    def shutdown(self) -> None:
        if self._build_worker is not None:
            self._build_worker.cancel()
        if self._process is not None:
            self._terminate_training_process(force=True)
            self._process.waitForFinished(3000)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._build_thread is not None or self._process is not None:
            QMessageBox.information(
                self,
                "Vorgang läuft",
                "Bitte den Vorgang zuerst mit „Stoppen“ beenden.",
            )
            event.ignore()
            return
        super().closeEvent(event)
