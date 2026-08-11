from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from automated_image_capture.cleanup import (
    CleanupPlan,
    CleanupResult,
    CleanupSettings,
    analyze_cleanup,
    execute_cleanup,
)
from automated_image_capture.settings import SettingsStore


def _size_text(size: int) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GiB"
    if size >= 1024**2:
        return f"{size / 1024**2:.1f} MiB"
    if size >= 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size} B"


class CleanupWorker(QObject):
    progress = pyqtSignal(int, int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, settings: CleanupSettings, plan: CleanupPlan | None = None) -> None:
        super().__init__()
        self.settings = settings
        self.plan = plan
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            if self.plan is None:
                result = analyze_cleanup(
                    self.settings,
                    lambda done, total, message: self.progress.emit(done, total, message),
                    self._cancelled.is_set,
                )
            else:
                result = execute_cleanup(
                    self.plan,
                    lambda done, total, message: self.progress.emit(done, total, message),
                    self._cancelled.is_set,
                )
        except Exception as exc:
            self.failed.emit(str(exc) or type(exc).__name__)
        else:
            self.completed.emit(result)


class CleanupDialog(QDialog):
    def __init__(
        self,
        settings_store: SettingsStore,
        acquisition_running: Callable[[], bool] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Speicher bereinigen")
        self.resize(850, 690)
        self.settings_store = settings_store
        self.acquisition_running = acquisition_running or (lambda: False)
        self._plan: CleanupPlan | None = None
        self._thread: QThread | None = None
        self._worker: CleanupWorker | None = None
        self._build_ui(settings_store.load_cleanup())
        self._set_busy(False)

    def _build_ui(self, defaults: CleanupSettings) -> None:
        root = QVBoxLayout(self)
        explanation = QLabel(
            "Die Analyse verändert nichts. Sie berücksichtigt NTFS-Hardlinks und verarbeitet "
            "nur abgeschlossene Aufnahme-, OBB- und YOLO-Datensätze. Erst nach einer zweiten "
            "Bestätigung werden Bilder ersetzt und rekonstruierbare Caches endgültig gelöscht."
        )
        explanation.setWordWrap(True)
        root.addWidget(explanation)

        form = QFormLayout()
        self.path = QLineEdit(str(defaults.root_directory))
        browse = QPushButton("Ordner …")
        browse.clicked.connect(self._browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self.path, 1)
        path_row.addWidget(browse)
        path_widget = QWidget()
        path_widget.setLayout(path_row)
        form.addRow("Zu bereinigender Ordner", path_widget)

        self.output_format = QComboBox()
        self.output_format.addItem("PNG · verlustfrei", "png")
        self.output_format.addItem("WebP · verlustfrei", "webp_lossless")
        self.output_format.addItem("WebP · Qualitätsstufe", "webp")
        self.output_format.addItem("JPEG · Qualitätsstufe", "jpeg")
        self.output_format.setCurrentIndex(
            max(0, self.output_format.findData(defaults.output_format))
        )
        form.addRow("Dateiformat", self.output_format)

        self.max_edge = QSpinBox()
        self.max_edge.setRange(0, 8192)
        self.max_edge.setSpecialValueText("Originalgröße")
        self.max_edge.setSuffix(" px")
        self.max_edge.setValue(defaults.max_edge)
        self.max_edge.setToolTip("0 behält die Auflösung; andere Werte müssen mindestens 128 sein.")
        self.max_edge.editingFinished.connect(self._normalize_edge)
        form.addRow("Maximale Kantenlänge", self.max_edge)

        self.png_compression = QSpinBox()
        self.png_compression.setRange(0, 9)
        self.png_compression.setValue(defaults.png_compression)
        form.addRow("PNG-Kompression", self.png_compression)
        self.quality = QSpinBox()
        self.quality.setRange(1, 100)
        self.quality.setValue(defaults.quality)
        form.addRow("JPEG/WebP-Qualität", self.quality)
        root.addLayout(form)

        self.remove_caches = QCheckBox(
            "Validierte _imgsz-Trainingscaches und Ultralytics-*.cache löschen"
        )
        self.remove_caches.setChecked(defaults.remove_caches)
        self.deduplicate = QCheckBox("Exakte Duplikate pfaderhaltend als Hardlinks zusammenführen")
        self.deduplicate.setChecked(defaults.deduplicate)
        root.addWidget(self.remove_caches)
        root.addWidget(self.deduplicate)

        warning = QLabel(
            "Verkleinerung, JPEG und WebP mit Qualitätsstufe verändern Bildinformationen "
            "irreversibel. Die sichere Voreinstellung ist Originalgröße + PNG."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color:#b45309; font-weight:600;")
        root.addWidget(warning)

        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlaceholderText("Noch keine Analyse durchgeführt.")
        root.addWidget(self.summary, 1)
        self.status = QLabel("Bereit zur Nur-Analyse.")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        buttons = QHBoxLayout()
        self.analyze_button = QPushButton("Ordner analysieren")
        self.execute_button = QPushButton("Bereinigung ausführen")
        self.cancel_button = QPushButton("Abbrechen")
        close_button = QPushButton("Schließen")
        self.analyze_button.clicked.connect(self._analyze)
        self.execute_button.clicked.connect(self._execute)
        self.cancel_button.clicked.connect(self._cancel)
        close_button.clicked.connect(self.close)
        buttons.addWidget(self.analyze_button)
        buttons.addWidget(self.execute_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        buttons.addWidget(close_button)
        root.addLayout(buttons)

        for widget in (
            self.path,
            self.output_format,
            self.max_edge,
            self.png_compression,
            self.quality,
            self.remove_caches,
            self.deduplicate,
        ):
            if isinstance(widget, QLineEdit):
                widget.textChanged.connect(self._invalidate)
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._invalidate)
            elif isinstance(widget, QSpinBox):
                widget.valueChanged.connect(self._invalidate)
            else:
                widget.toggled.connect(self._invalidate)
        self.output_format.currentIndexChanged.connect(self._update_option_state)
        self._update_option_state()

    def _normalize_edge(self) -> None:
        if 0 < self.max_edge.value() < 128:
            self.max_edge.setValue(128)

    def _update_option_state(self) -> None:
        selected = self.output_format.currentData()
        self.png_compression.setEnabled(selected == "png")
        self.quality.setEnabled(selected in {"webp", "jpeg"})

    def _browse(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Ordner auswählen", self.path.text())
        if selected:
            self.path.setText(selected)

    def _settings(self) -> CleanupSettings:
        self._normalize_edge()
        return CleanupSettings(
            root_directory=Path(self.path.text().strip()),
            output_format=self.output_format.currentData(),
            max_edge=self.max_edge.value(),
            png_compression=self.png_compression.value(),
            quality=self.quality.value(),
            remove_caches=self.remove_caches.isChecked(),
            deduplicate=self.deduplicate.isChecked(),
        ).validated()

    def _invalidate(self, *_args: object) -> None:
        self._plan = None
        if hasattr(self, "execute_button"):
            self.execute_button.setEnabled(False)

    def _analyze(self) -> None:
        try:
            settings = self._settings()
        except Exception as exc:
            QMessageBox.warning(self, "Ungültige Einstellung", str(exc))
            return
        self.settings_store.save_cleanup(settings)
        self._start_worker(settings)

    def _execute(self) -> None:
        if self._plan is None:
            return
        if self.acquisition_running():
            QMessageBox.warning(
                self,
                "Aufnahme läuft",
                "Während einer automatischen Aufnahme darf kein Aufnahmeordner bereinigt werden.",
            )
            return
        answer = QMessageBox.warning(
            self,
            "Endgültige Bereinigung bestätigen",
            f"Voraussichtlich werden {_size_text(self._plan.estimated_savings)} freigegeben.\n\n"
            f"{len(self._plan.cache_directories)} Cache-Ordner und "
            f"{len(self._plan.cache_files)} Cache-Dateien werden endgültig gelöscht. "
            "Die Bildersetzung kann nicht rückgängig gemacht werden. Fortfahren?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._start_worker(self._plan.settings, self._plan)

    def _start_worker(self, settings: CleanupSettings, plan: CleanupPlan | None = None) -> None:
        self._set_busy(True)
        self.progress.setRange(0, 0)
        self.status.setText("Bereinigung läuft …" if plan else "Analyse läuft …")
        thread = QThread(self)
        worker = CleanupWorker(settings, plan)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._progress)
        worker.completed.connect(self._completed)
        worker.failed.connect(self._failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._thread_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _progress(self, done: int, total: int, message: str) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.status.setText(message)

    def _completed(self, result: object) -> None:
        if isinstance(result, CleanupPlan):
            self._plan = result
            lines = [
                f"Logische Größe: { _size_text(result.logical_bytes_before) }",
                f"Physische Größe: { _size_text(result.physical_bytes_before) }",
                f"Geschätzte Zielgröße: { _size_text(result.estimated_physical_bytes_after) }",
                f"Geschätzte Einsparung: { _size_text(result.estimated_savings) }",
                "",
                f"Erkannte Projektbilder: {result.managed_image_count}",
                f"Physische Bildgruppen: {len(result.image_actions)}",
                f"Übersprungene sonstige Bilder: {result.skipped_image_count}",
                f"Rekonstruierbare Cache-Ordner: {len(result.cache_directories)}",
                f"Cache-Dateien: {len(result.cache_files)}",
            ]
            if result.warnings:
                lines.extend(["", "Hinweise:", *(f"- {item}" for item in result.warnings[:50])])
            self.summary.setPlainText("\n".join(lines))
            self.status.setText("Analyse abgeschlossen. Bitte Ergebnis vor Ausführung prüfen.")
            self.execute_button.setEnabled(bool(result.image_actions or result.cache_directories))
        elif isinstance(result, CleanupResult):
            self.status.setText(
                (
                    "Bereinigung abgebrochen; abgeschlossene Gruppen bleiben konsistent. "
                    if result.cancelled
                    else "Bereinigung abgeschlossen. "
                )
                + f"Freigegeben: {_size_text(result.freed_bytes)}. Bericht: {result.report_path}"
            )
            self.summary.appendPlainText(
                f"\nTatsächlich freigegeben: {_size_text(result.freed_bytes)}\n"
                f"Verarbeitete Bildpfade: {result.processed_images}\n"
                f"Bericht: {result.report_path}"
            )
            self._plan = None
        self.progress.setRange(0, 1)
        self.progress.setValue(1)

    def _failed(self, message: str) -> None:
        self.status.setText(f"Fehler: {message}")
        QMessageBox.critical(self, "Bereinigung fehlgeschlagen", message)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)

    def _thread_finished(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self._set_busy(False)

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.status.setText("Abbruch wird nach der aktuellen Bildgruppe ausgeführt …")

    def _set_busy(self, busy: bool) -> None:
        self.analyze_button.setEnabled(not busy)
        self.execute_button.setEnabled(not busy and self._plan is not None)
        self.cancel_button.setEnabled(busy)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.status.setText("Bitte warten, bis der Worker konsistent beendet wurde …")
            event.ignore()
            return
        super().closeEvent(event)
