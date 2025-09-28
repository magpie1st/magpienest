from __future__ import annotations

from typing import List
from PySide6.QtWidgets import QWidget, QVBoxLayout, QListWidget, QListWidgetItem

from audiobook.core.epub_reader import Chapter


class TocPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.list = QListWidget(self)
        self.list.setSelectionMode(self.list.MultiSelection)
        lay = QVBoxLayout(self)
        lay.addWidget(self.list)

    def load(self, chapters: List[Chapter]) -> None:
        self.list.clear()
        for ch in chapters:
            item = QListWidgetItem(f"{ch.index:03d}  {ch.title}")
            item.setCheckState(0)  # unchecked
            self.list.addItem(item)

    def selected(self) -> List[int]:
        idx: List[int] = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.checkState():
                idx.append(i)
        return idx
