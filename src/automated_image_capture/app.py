from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

import qasync
from PyQt6.QtWidgets import QApplication

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
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    window = MainWindow()
    window.show()
    window.statusBar().showMessage(f"Logdatei: {log_path}", 5000)

    with loop:
        loop.run_forever()
        loop.run_until_complete(window.shutdown_async())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

