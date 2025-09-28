from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QProgressBar, QLabel


class ProgressPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.total = QProgressBar(self)
        self.chapter = QProgressBar(self)
        self.label = QLabel("Idle", self)
        lay = QVBoxLayout(self)
        lay.addWidget(self.label)
        lay.addWidget(self.total)
        lay.addWidget(self.chapter)

    def update_progress(self, ci: int, ct: int, li: int, lt: int) -> None:
        self.total.setMaximum(ct)
        self.total.setValue(ci)
        self.chapter.setMaximum(max(lt, 1))
        self.chapter.setValue(li)
        self.label.setText(f"Chapter {ci}/{ct}  Line {li}/{lt}")
