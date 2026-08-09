from __future__ import annotations

import threading
from pathlib import Path

from PyQt6.QtCore import QObject, QSize, Qt, QThread, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCloseEvent, QDesktopServices, QIcon, QPixmap
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
    QListView,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from automated_image_capture.dataset import default_build_config
from automated_image_capture.labeling import (
    AnchorReviewItem,
    LabelingCancelled,
    LabelingConfig,
    LabelingResult,
    LabelSource,
    VisibilityReviewItem,
    generate_obb_dataset,
)
from automated_image_capture.settings import SettingsStore


class LabelingWorker(QObject):
    progress = pyqtSignal(int, int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    anchors_ready = pyqtSignal(object)
    visibility_ready = pyqtSignal(object)

    def __init__(self, config: LabelingConfig) -> None:
        super().__init__()
        self.config = config
        self._cancelled = threading.Event()
        self._anchor_decision = threading.Event()
        self._anchors_accepted = False
        self._visibility_decision = threading.Event()
        self._visibility_selection: frozenset[Path] | None = None

    def cancel(self) -> None:
        self._cancelled.set()
        self._anchor_decision.set()
        self._visibility_decision.set()

    def resolve_anchor_review(self, accepted: bool) -> None:
        self._anchors_accepted = accepted
        self._anchor_decision.set()

    def _review_anchors(self, items: tuple[AnchorReviewItem, ...]) -> bool:
        self._anchor_decision.clear()
        self.anchors_ready.emit(items)
        while not self._anchor_decision.wait(0.1):
            if self._cancelled.is_set():
                return False
        return self._anchors_accepted and not self._cancelled.is_set()

    def resolve_visibility_review(self, selected: frozenset[Path] | None) -> None:
        self._visibility_selection = selected
        self._visibility_decision.set()

    def _review_visibility(
        self, items: tuple[VisibilityReviewItem, ...]
    ) -> frozenset[Path] | None:
        self._visibility_decision.clear()
        self._visibility_selection = None
        self.visibility_ready.emit(items)
        while not self._visibility_decision.wait(0.1):
            if self._cancelled.is_set():
                return None
        if self._cancelled.is_set():
            return None
        return self._visibility_selection

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = generate_obb_dataset(
                self.config,
                lambda done, total, message: self.progress.emit(done, total, message),
                self._cancelled.is_set,
                self._review_anchors,
                self._review_visibility,
            )
        except LabelingCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc) or type(exc).__name__)
        else:
            self.completed.emit(result)


class LabelSourceRow(QWidget):
    remove_requested = pyqtSignal(object)

    def __init__(self, source: LabelSource, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.is_empty = source.is_empty
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        self.name = QLineEdit(source.name)
        self.name.setMinimumWidth(120)
        self.name.setReadOnly(source.is_empty)
        self.kind = QLabel("Negativ" if source.is_empty else "Klasse")
        self.kind.setMinimumWidth(65)
        self.directory = QLineEdit("" if source.directory == Path() else str(source.directory))
        browse = QPushButton("Ordner …")
        browse.clicked.connect(self._browse)
        self.remove_button = QPushButton("Entfernen")
        self.remove_button.clicked.connect(lambda: self.remove_requested.emit(self))
        self.remove_button.setVisible(not source.is_empty)
        row.addWidget(self.name)
        row.addWidget(self.kind)
        row.addWidget(self.directory, 1)
        row.addWidget(browse)
        row.addWidget(self.remove_button)

    def set_class_id(self, class_id: int) -> None:
        if not self.is_empty:
            self.kind.setText(f"Klasse {class_id}")

    def source(self) -> LabelSource:
        return LabelSource(
            self.name.text().strip(),
            Path(self.directory.text().strip()),
            self.is_empty,
        )

    def _browse(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            f"Ordner für {self.name.text().strip() or 'Quelle'} auswählen",
            self.directory.text(),
        )
        if selected:
            self.directory.setText(selected)


class AnchorReviewDialog(QDialog):
    def __init__(
        self,
        items: tuple[AnchorReviewItem, ...],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("OBB-Anker bestätigen")
        self.resize(1050, 760)
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Grün markiert eine zuverlässige Einzelsegmentierung (ANKER), Orange eine daraus "
            "berechnete Box (BAHNMODELL). Pro Roboterwinkel wird der Mittelpunkt auf einer "
            "geraden Förderbandbahn geführt. Bitte nur übernehmen, wenn alle Boxen das "
            "Bauteil eng und vollständig umschließen."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        content = QVBoxLayout(container)
        for item in items:
            title = QLabel(
                f"Klasse {item.class_id}: {item.class_name} · "
                f"UR {item.pose_id / 10.0:.1f}° · "
                f"{item.anchor_count}/{item.image_count} sichere Anker"
            )
            title.setStyleSheet("font-weight:600;")
            content.addWidget(title)
            image = QLabel()
            pixmap = QPixmap(str(item.preview_path))
            image.setPixmap(
                pixmap.scaled(
                    980,
                    430,
                    aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio,
                    transformMode=Qt.TransformationMode.SmoothTransformation,
                )
            )
            image.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image.setStyleSheet("background:#111827; padding:6px;")
            content.addWidget(image)
        content.addStretch(1)
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)
        buttons = QDialogButtonBox()
        accept = buttons.addButton("Anker übernehmen", QDialogButtonBox.ButtonRole.AcceptRole)
        reject = buttons.addButton(
            "Ablehnen / nichts exportieren", QDialogButtonBox.ButtonRole.RejectRole
        )
        accept.clicked.connect(self.accept)
        reject.clicked.connect(self.reject)
        layout.addWidget(buttons)


class VisibilityReviewDialog(QDialog):
    def __init__(
        self,
        items: tuple[VisibilityReviewItem, ...],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.items = items
        self.setWindowTitle("Schlecht sichtbare Bilder aussortieren")
        self.resize(1120, 800)
        layout = QVBoxLayout(self)
        recommended = sum(item.recommended_exclude for item in items)
        explanation = QLabel(
            f"{len(items)} Bilder wirken sehr dunkel, überbelichtet oder lokal kontrastarm. "
            f"{recommended} eindeutig unbrauchbare Bilder sind bereits zum Ausschluss "
            "markiert. Häkchen bedeutet: nicht in den YOLO-Datensatz übernehmen."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        self.gallery = QListWidget()
        self.gallery.setViewMode(QListView.ViewMode.IconMode)
        self.gallery.setResizeMode(QListView.ResizeMode.Adjust)
        self.gallery.setMovement(QListView.Movement.Static)
        self.gallery.setIconSize(QSize(310, 190))
        self.gallery.setGridSize(QSize(350, 285))
        self.gallery.setWordWrap(True)
        for review in items:
            entry = QListWidgetItem(QIcon(str(review.preview_path)), "")
            entry.setText(
                f"Klasse {review.class_id}: {review.class_name} · "
                f"UR {review.pose_id / 10.0:.1f}°\n"
                f"Score {review.score:.2f} · {review.reason}\n{review.source_path.name}"
            )
            entry.setData(Qt.ItemDataRole.UserRole, str(review.source_path))
            entry.setFlags(
                entry.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
            )
            entry.setCheckState(
                Qt.CheckState.Checked
                if review.recommended_exclude
                else Qt.CheckState.Unchecked
            )
            self.gallery.addItem(entry)
        layout.addWidget(self.gallery, 1)

        selection_row = QHBoxLayout()
        recommended_button = QPushButton("Nur Empfehlungen markieren")
        all_button = QPushButton("Alle markieren")
        none_button = QPushButton("Keine markieren")
        recommended_button.clicked.connect(self._mark_recommended)
        all_button.clicked.connect(lambda: self._set_all(Qt.CheckState.Checked))
        none_button.clicked.connect(lambda: self._set_all(Qt.CheckState.Unchecked))
        selection_row.addWidget(recommended_button)
        selection_row.addWidget(all_button)
        selection_row.addWidget(none_button)
        selection_row.addStretch(1)
        layout.addLayout(selection_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Auswahl übernehmen")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_all(self, state: Qt.CheckState) -> None:
        for index in range(self.gallery.count()):
            self.gallery.item(index).setCheckState(state)

    def _mark_recommended(self) -> None:
        for index, review in enumerate(self.items):
            self.gallery.item(index).setCheckState(
                Qt.CheckState.Checked
                if review.recommended_exclude
                else Qt.CheckState.Unchecked
            )

    def excluded_paths(self) -> frozenset[Path]:
        return frozenset(
            Path(str(self.gallery.item(index).data(Qt.ItemDataRole.UserRole)))
            for index in range(self.gallery.count())
            if self.gallery.item(index).checkState() == Qt.CheckState.Checked
        )


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
            "Jede Pose in der Liste wird eine eigene YOLO-Klasse. Das Tool paart ihre "
            "Aufnahmen mit dem Leerbild derselben UR-Ansicht und Beleuchtung. Bei klassischen "
            "Rasterserien entsteht aus wiederholten Bildern derselben Ansicht eine gemeinsame "
            "OBB. Synchronisierte Förderbandserien werden dagegen winkelweise als vollständige "
            "Bahn ausgewertet: Die gemessene ADS-Position stabilisiert die OBB und ergänzt "
            "visuell zu dunkle Samples. Solche Ergänzungen erscheinen ausdrücklich als REVIEW."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        source_header = QHBoxLayout()
        source_header.addWidget(QLabel("Bildquellen"))
        source_header.addStretch(1)
        self.add_pose_button = QPushButton("Pose hinzufügen")
        self.add_pose_button.clicked.connect(self._add_pose)
        source_header.addWidget(self.add_pose_button)
        layout.addLayout(source_header)

        source_scroll = QScrollArea()
        source_scroll.setWidgetResizable(True)
        source_scroll.setMinimumHeight(150)
        source_scroll.setMaximumHeight(260)
        self.source_container = QWidget()
        self.source_layout = QVBoxLayout(self.source_container)
        self.source_layout.setContentsMargins(4, 4, 4, 4)
        self.source_rows: list[LabelSourceRow] = []
        for source in defaults.sources:
            self._add_source_row(source)
        self.source_layout.addStretch(1)
        source_scroll.setWidget(self.source_container)
        layout.addWidget(source_scroll)

        form = QFormLayout()
        self.output = QLineEdit(str(output_directory))
        form.addRow("YOLO-Ausgabe", self._path_row(self.output, self._browse_output))

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

    def _browse_output(self) -> None:
        self._choose_directory(self.output, "Leeren YOLO-Ausgabeordner auswählen")

    def _choose_directory(self, field: QLineEdit, title: str) -> None:
        selected = QFileDialog.getExistingDirectory(self, title, field.text())
        if selected:
            field.setText(selected)

    def _add_source_row(self, source: LabelSource) -> None:
        row = LabelSourceRow(source, self.source_container)
        row.remove_requested.connect(self._remove_source_row)
        if source.is_empty or not self.source_rows:
            self.source_rows.append(row)
            self.source_layout.addWidget(row)
        else:
            empty_index = next(
                (index for index, item in enumerate(self.source_rows) if item.is_empty),
                len(self.source_rows),
            )
            self.source_rows.insert(empty_index, row)
            self.source_layout.insertWidget(empty_index, row)
        self._refresh_source_rows()

    def _add_pose(self) -> None:
        used_names = {row.name.text().strip().casefold() for row in self.source_rows}
        number = 1
        while f"pose {number}" in used_names:
            number += 1
        self._add_source_row(LabelSource(f"Pose {number}", Path()))

    @pyqtSlot(object)
    def _remove_source_row(self, row: object) -> None:
        if not isinstance(row, LabelSourceRow) or row.is_empty:
            return
        pose_rows = [item for item in self.source_rows if not item.is_empty]
        if len(pose_rows) <= 1:
            QMessageBox.warning(self, "Pose erforderlich", "Mindestens eine Pose muss bleiben.")
            return
        self.source_rows.remove(row)
        self.source_layout.removeWidget(row)
        row.deleteLater()
        self._refresh_source_rows()

    def _refresh_source_rows(self) -> None:
        pose_rows = [row for row in self.source_rows if not row.is_empty]
        for class_id, row in enumerate(pose_rows):
            row.set_class_id(class_id)
            row.remove_button.setEnabled(len(pose_rows) > 1)

    def _config(self) -> LabelingConfig:
        return LabelingConfig(
            sources=tuple(row.source() for row in self.source_rows),
            output_directory=Path(self.output.text().strip()),
            validation_fraction=self.validation.value() / 100.0,
            minimum_difference=self.minimum_difference.value(),
            consensus_fraction=self.consensus.value(),
            include_background_negatives=self.background_negatives.isChecked(),
            prefer_hardlinks=self.hardlinks.isChecked(),
        ).validated()

    def _start(self) -> None:
        if self._thread is not None:
            return
        if any(not row.directory.text().strip() for row in self.source_rows):
            QMessageBox.warning(self, "Pfad fehlt", "Bitte für jede Quelle einen Ordner auswählen.")
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
        self.source_container.setEnabled(False)
        self.add_pose_button.setEnabled(False)

        thread = QThread(self)
        worker = LabelingWorker(config)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._progress)
        worker.anchors_ready.connect(self._review_anchors)
        worker.visibility_ready.connect(self._review_visibility)
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

    @pyqtSlot(object)
    def _review_anchors(self, value: object) -> None:
        items = tuple(item for item in value if isinstance(item, AnchorReviewItem))
        worker = self._worker
        if worker is None:
            return
        self.status.setText(
            "Segmentierung fertig. Bitte die vorgeschlagenen OBB-Anker kontrollieren."
        )
        accepted = (
            bool(items)
            and AnchorReviewDialog(items, self).exec() == QDialog.DialogCode.Accepted
        )
        worker.resolve_anchor_review(accepted)

    @pyqtSlot(object)
    def _review_visibility(self, value: object) -> None:
        items = tuple(item for item in value if isinstance(item, VisibilityReviewItem))
        worker = self._worker
        if worker is None:
            return
        self.status.setText(
            "Bitte sehr dunkle, überbelichtete oder kontrastarme Bilder aussortieren."
        )
        dialog = VisibilityReviewDialog(items, self)
        if items and dialog.exec() == QDialog.DialogCode.Accepted:
            worker.resolve_visibility_review(dialog.excluded_paths())
        else:
            worker.resolve_visibility_review(None)

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
        training_defaults = default_build_config()
        training_paths = self.settings_store.load_training_paths(training_defaults)
        output_root = training_paths["output_root"] or training_defaults.output_root
        self.settings_store.save_training_paths(
            result.output_directory,
            Path(output_root),
            None,
        )
        self.status.setText(
            f"Fertig: {result.positive_images} positive und {result.negative_images} negative "
            f"Bilder, {result.classes} Klassen und {result.poses} UR-Ansichten, "
            f"{result.flagged_images} Bilder zur Nachprüfung. "
            f"{result.excluded_images} schlecht sichtbare Bilder ausgeschlossen. "
            f"Positionsbahn: {result.position_tracked_images} Bilder, "
            f"davon {result.position_corrected_images} stabilisiert und "
            f"{result.position_interpolated_images} ohne sichere Einzelsegmentierung "
            "interpoliert. Der Datensatz ist als Quelle im YOLO-Training vorgemerkt.\n"
            f"Bericht: {result.report_path}"
        )
        self.open_button.setEnabled(True)
        previews = sorted(result.review_directory.glob("class_*_ur_*_obb.jpg"))
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
        self.source_container.setEnabled(True)
        self.add_pose_button.setEnabled(True)

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
