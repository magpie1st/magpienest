from __future__ import annotations

from typing import List
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QAbstractItemView

from audiobook.core.epub_reader import Chapter


class TocPanel(QWidget):
    current_changed = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.list = QListWidget(self)
        self.list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.list.itemChanged.connect(self._on_item_changed)
        self.list.currentRowChanged.connect(self.current_changed.emit)

        layout = QVBoxLayout(self)
        layout.addWidget(self.list)

        self._block_item_signal = False

    def load(self, chapters: List[Chapter]) -> None:
        self._block_item_signal = True
        self.list.clear()
        for ch in chapters:
            item = QListWidgetItem(f"{ch.index:03d}  {ch.title}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.list.addItem(item)
        self._block_item_signal = False
        if chapters:
            self.list.setCurrentRow(0)

    def selected(self) -> List[int]:
        selected_indices: List[int] = []
        for i in range(self.list.count()):
            if self.list.item(i).checkState() == Qt.Checked:
                selected_indices.append(i)
        return selected_indices

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._block_item_signal:
            return
        row = self.list.row(item)
        self.current_changed.emit(row)
