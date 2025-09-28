from __future__ import annotations
# Allow running as a script: `python audiobook\__main__.py`
import sys as _sys
from pathlib import Path as _Path
if __package__ is None or __package__ == "":
    _sys.path.append(str(_Path(__file__).resolve().parents[1]))

import sys
from PySide6.QtWidgets import QApplication
from audiobook.app_gui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

