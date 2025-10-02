from __future__ import annotations
# Allow running as a script: `python audiobook\__main__.py`
import sys as _sys
from pathlib import Path as _Path
if __package__ is None or __package__ == "":
    _sys.path.append(str(_Path(__file__).resolve().parents[1]))

import sys
import logging
import faulthandler
import atexit


_LOG_DIR = _Path(__file__).resolve().parent
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_PATH = _LOG_DIR / "audiobook.log"
_FAULT_LOG_PATH = _LOG_DIR / "audiobook_fault.log"
_FAULT_HANDLE = _FAULT_LOG_PATH.open("a", buffering=1)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(_LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)

faulthandler.enable(_FAULT_HANDLE, all_threads=True)


@atexit.register
def _close_fault_handle() -> None:
    try:
        _FAULT_HANDLE.close()
    except Exception:
        pass

logger = logging.getLogger(__name__)
from PySide6.QtWidgets import QApplication
from audiobook.app_gui.main_window import MainWindow


def main() -> None:
    logger.info("Starting Audiobook GUI")
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    exit_code = app.exec()
    logger.info("Application exited with code %s", exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
