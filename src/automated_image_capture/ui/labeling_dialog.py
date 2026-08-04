from __future__ import annotations

import threading
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, QThread, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCloseEvent, QDesktopServices, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from automated_image_capture.labeling import (
    LabelingCancelled,
    LabelingConfig,
    LabelingResult,
    generate_obb_dataset,
)
from automated_image_capture.settings import SettingsStore


class LabelingWorker(QObject):
    progress = pyqtSignal(int, int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, config: LabelingConfig) -> None:
        super().__init__()
        self.config = config
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = generate_obb_dataset(
                self.config,
                lambda done, total, message: self.progress.emit(done, total, message),
                self._cancelled.is_set,
            )
        except LabelingCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc) or type(exc).__name__)
        else:
            self.completed.emit(result)


class LabelingDialog(QDialog):
    def __init__(
        self,
        settings_store: SettingsStore,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings_store = settings_store
        defaults = settings_store.load_labeling()
        output_directory = defaults.output_directory
        suffix = 2
        while output_directory.exists() and any(output_directory.iterdir()):
            output_directory = defaults.output_directory.with_name(
                f"{defaults.output_directory.name}_{suffix}"
            )
            suffix += 1
        self._thread: QThread | None = None
        self._worker: LabelingWorker | None = None
        self._result: LabelingResult | None = None
        self.setWindowTitle("Automatische YOLO-OBB-Labels")
        self.setMinimumSize(780, 720)

        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Das Tool paart jede Bauteilaufnahme mit dem Leerbild derselben Pose und "
            "Beleuchtung. Aus allen Beleuchtungen einer Pose entsteht eine gemeinsame OBB; "
            "abweichende Segmentierungen werden im Prüfbericht markiert."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        form = QFormLayout()
        self.foreground = QLineEdit(str(defaults.foreground_directory))
        self.background = QLineEdit(str(defaults.background_directory))
        self.output = QLineEdit(str(output_directory))
        form.addRow("Bauteil-Aufnahme", self._path_row(self.foreground, self._browse_foreground))
        form.addRow("Leere Rutsche", self._path_row(self.background, self._browse_background))
        form.addRow("YOLO-Ausgabe", self._path_row(self.output, self._browse_output))

        self.class_name = QLineEdit(defaults.class_name)
        self.class_id = QSpinBox()
        self.class_id.setRange(0, 9999)
        self.class_id.setValue(defaults.class_id)
        class_row = QHBoxLayout()
        class_row.addWidget(self.class_name, 1)
        class_row.addWidget(QLabel("ID"))
        class_row.addWidget(self.class_id)
        class_widget = QWidget()
        class_widget.setLayout(class_row)
        form.addRow("Klasse", class_widget)

        self.validation = QSpinBox()
        self.validation.setRange(0, 50)
        self.validation.setSuffix(" %")
        self.validation.setValue(round(defaults.validation_fraction * 100))
        form.addRow("Validierung nach Posen", self.validation)

        self.minimum_difference = QSpinBox()
        self.minimum_difference.setRange(1, 255)
        self.minimum_difference.setValue(defaults.minimum_difference)
        self.minimum_difference.setToolTip(
            "Für Kk1 real geprüft: 80. Kleinere Werte reagieren stärker auf Rasterverschiebungen."
        )
        self.consensus = QDoubleSpinBox()
        self.consensus.setRange(0.05, 0.95)
        self.consensus.setSingleStep(0.05)
        self.consensus.setDecimals(2)
        self.consensus.setValue(defaults.consensus_fraction)
        advanced_row = QHBoxLayout()
        advanced_row.addWidget(QLabel("Differenz"))
        advanced_row.addWidget(self.minimum_difference)
        advanced_row.addWidget(QLabel("Konsens"))
        advanced_row.addWidget(self.consensus)
        advanced_widget = QWidget()
        advanced_widget.setLayout(advanced_row)
        form.addRow("Segmentierung", advanced_widget)
        layout.addLayout(form)

        self.background_negatives = QCheckBox(
            "Leerbilder als negative Trainingsbeispiele übernehmen"
        )
        self.background_negatives.setChecked(defaults.include_background_negatives)
        self.hardlinks = QCheckBox("Speicherplatzsparende Hardlinks bevorzugen")
        self.hardlinks.setChecked(defaults.prefer_hardlinks)
        layout.addWidget(self.background_negatives)
        layout.addWidget(self.hardlinks)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.status = QLabel("Bereit. Der Ausgabeordner muss leer oder noch nicht vorhanden sein.")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)

        self.preview = QLabel("Nach Abschluss erscheint hier eine Prüfübersicht.")
        self.preview.setMinimumHeight(230)
        self.preview.setScaledContents(False)
        self.preview.setStyleSheet("background:#111827; color:#d1d5db; padding:8px;")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.preview, 1)

        action_row = QHBoxLayout()
        self.start_button = QPushButton("OBB-Labels erzeugen")
        self.cancel_button = QPushButton("Abbrechen")
        self.cancel_button.setEnabled(False)
        self.open_button = QPushButton("Ausgabeordner öffnen")
        self.open_button.setEnabled(False)
        self.start_button.clicked.connect(self._start)
        self.cancel_button.clicked.connect(self._cancel)
        self.open_button.clicked.connect(self._open_output)
        action_row.addWidget(self.start_button)
        action_row.addWidget(self.cancel_button)
        action_row.addStretch(1)
        action_row.addWidget(self.open_button)
        layout.addLayout(action_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _path_row(field: QLineEdit, callback: object) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("…")
        browse.setFixedWidth(40)
        browse.clicked.connect(callback)  # type: ignore[arg-type]
        row.addWidget(field, 1)
        row.addWidget(browse)
        return widget

    def _browse_foreground(self) -> None:
        self._choose_directory(self.foreground, "Bauteil-Aufnahme auswählen")

    def _browse_background(self) -> None:
        self._choose_directory(self.background, "Leeraufnahme auswählen")

    def _browse_output(self) -> None:
        self._choose_directory(self.output, "Leeren YOLO-Ausgabeordner auswählen")

    def _choose_directory(self, field: QLineEdit, title: str) -> None:
        selected = QFileDialog.getExistingDirectory(self, title, field.text())
        if selected:
            field.setText(selected)

    def _config(self) -> LabelingConfig:
        return LabelingConfig(
            foreground_directory=Path(self.foreground.text().strip()),
            background_directory=Path(self.background.text().strip()),
            output_directory=Path(self.output.text().strip()),
            class_name=self.class_name.text().strip(),
            class_id=self.class_id.value(),
            validation_fraction=self.validation.value() / 100.0,
            minimum_difference=self.minimum_difference.value(),
            consensus_fraction=self.consensus.value(),
            include_background_negatives=self.background_negatives.isChecked(),
            prefer_hardlinks=self.hardlinks.isChecked(),
        ).validated()

    def _start(self) -> None:
        if self._thread is not None:
            return
        if not self.foreground.text().strip() or not self.background.text().strip():
            QMessageBox.warning(self, "Pfad fehlt", "Bitte beide Aufnahmeordner auswählen.")
            return
        if not self.output.text().strip():
            QMessageBox.warning(self, "Pfad fehlt", "Bitte einen Ausgabeordner auswählen.")
            return
        try:
            config = self._config()
        except Exception as exc:
            QMessageBox.warning(self, "Ungültige Label-Einstellung", str(exc))
            return
        self.settings_store.save_labeling(config)
        self._result = None
        self.progress.setRange(0, 0)
        self.status.setText("Prüfe und registriere die Bildpaare …")
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.open_button.setEnabled(False)

        thread = QThread(self)
        worker = LabelingWorker(config)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._progress)
        worker.completed.connect(self._completed)
        worker.failed.connect(self._failed)
        worker.cancelled.connect(self._cancelled)
        for signal in (worker.completed, worker.failed, worker.cancelled):
            signal.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    @pyqtSlot(int, int, str)
    def _progress(self, done: int, total: int, message: str) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.status.setText(f"{message} · {done}/{total}")

    @pyqtSlot(object)
    def _completed(self, result: object) -> None:
        if not isinstance(result, LabelingResult):
            self._failed("Unbekanntes Ergebnis der Label-Erzeugung.")
            return
        self._result = result
        self.status.setText(
            f"Fertig: {result.positive_images} positive und {result.negative_images} negative "
            f"Bilder, {result.poses} Posen, {result.flagged_images} Bilder zur Nachprüfung.\n"
            f"Bericht: {result.report_path}"
        )
        self.open_button.setEnabled(True)
        previews = sorted(result.review_directory.glob("pose_*_obb.jpg"))
        if previews:
            pixmap = QPixmap(str(previews[0])).scaled(
                740,
                260,
                aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio,
                transformMode=Qt.TransformationMode.SmoothTransformation,
            )
            self.preview.setPixmap(pixmap)

    @pyqtSlot(str)
    def _failed(self, message: str) -> None:
        self.status.setText(f"Fehler: {message}")
        QMessageBox.critical(self, "OBB-Labeling fehlgeschlagen", message)

    @pyqtSlot()
    def _cancelled(self) -> None:
        self.status.setText("Label-Erzeugung abgebrochen.")

    @pyqtSlot()
    def _thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def _cancel(self) -> None:
        if self._worker is not None:
            self.status.setText("Abbruch angefordert …")
            self._worker.cancel()
            self.cancel_button.setEnabled(False)

    def _open_output(self) -> None:
        if self._result is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._result.output_directory)))

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._thread is not None:
            self._cancel()
            event.ignore()
            return
        super().closeEvent(event)
