from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

import qasync
from PyQt6.QtWidgets import QApplication, QMessageBox

from automated_image_capture.hardware_lease import HardwareLease
from automated_image_capture.logging_setup import setup_logging
from automated_image_capture.ui import MainWindow


def _exception_hook(exc_type: type[BaseException], exc: BaseException, traceback: Any) -> None:
    logging.getLogger("uncaught").critical(
        "Unbehandelte Ausnahme", exc_info=(exc_type, exc, traceback)
    )
    sys.__excepthook__(exc_type, exc, traceback)


def main() -> int:
    log_path = setup_logging()
    sys.excepthook = _exception_hook
    app = QApplication(sys.argv)
    app.setApplicationName("Automated Image Capture")
    app.setOrganizationName("Leibniz Universität Hannover")
    lease = HardwareLease.acquire()
    if lease is None:
        QMessageBox.critical(
            None,
            "Hardware bereits belegt",
            "BiBaZu Reorientation Control oder eine weitere Automated-Image-Capture-"
            "Instanz verwendet bereits die Baumer-Kamera und Neewer-Panels. Bitte "
            "zuerst die andere Anwendung schließen.",
        )
        return 2
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    try:
        window = MainWindow()
        window.show()
        window.statusBar().showMessage(f"Logdatei: {log_path}", 5000)
        with loop:
            loop.run_forever()
            loop.run_until_complete(window.shutdown_async())
        return 0
    finally:
        lease.close()


if __name__ == "__main__":
    raise SystemExit(main())
