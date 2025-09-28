from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QMainWindow, QFileDialog, QWidget, QHBoxLayout, QPushButton, QLabel

from audiobook.config import DEFAULT_LANGUAGE
from audiobook.core.epub_reader import load_epub, BookMeta, Chapter
from audiobook.core.pipeline import SynthesisPlan  # type: ignore
from audiobook.app_gui.widgets.toc_panel import TocPanel
from audiobook.app_gui.widgets.progress_panel import ProgressPanel
from audiobook.app_gui.workers import BookWorker


@dataclass
class Plan:
    book: BookMeta
    chapters: list[Chapter]
    out_dir: Path
    format: str = "mp3"
    pause_ms: int = 350
    speaker_wav: Optional[str] = None


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Audiobook TTS")
        self.resize(1200, 800)

        self._speaker: Optional[str] = None
        self._out_dir: Optional[Path] = None
        self._plan: Optional[Plan] = None

        central = QWidget(self)
        self.setCentralWidget(central)
        lay = QHBoxLayout(central)

        self.toc = TocPanel(self)
        lay.addWidget(self.toc)

        self.progress = ProgressPanel(self)
        lay.addWidget(self.progress)

        self.status = QLabel("Ready", self)
        lay.addWidget(self.status)

        self._thread: Optional[QThread] = None

        self._setup_menu()

    def _setup_menu(self) -> None:
        bar = self.menuBar()
        filem = bar.addMenu("File")
        open_epub = filem.addAction("Open EPUB")
        open_epub.triggered.connect(self._open_epub)
        out_dir = filem.addAction("Output Folder")
        out_dir.triggered.connect(self._pick_out_dir)
        speaker = filem.addAction("Select Speaker")
        speaker.triggered.connect(self._pick_speaker)
        run = filem.addAction("Synthesize Selected")
        run.triggered.connect(self._start_book)

    def _open_epub(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open EPUB", str(Path.home()), "EPUB (*.epub)")
        if not path:
            return
        meta, chapters = load_epub(path)
        self._plan = Plan(book=meta, chapters=chapters, out_dir=self._out_dir or Path.cwd())
        self.toc.load(chapters)
        self.status.setText(f"Loaded: {meta.title}")

    def _pick_out_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder", str(Path.home()))
        if path:
            self._out_dir = Path(path)
            self.status.setText(f"Output: {path}")

    def _pick_speaker(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Speaker", str(Path.home()), "Audio (*.wav *.mp3 *.m4a)")
        if path:
            self._speaker = path
            self.status.setText(f"Speaker: {Path(path).name}")

    def _start_book(self) -> None:
        if not self._plan:
            self.status.setText("Open an EPUB first")
            return
        if not self._out_dir:
            self._out_dir = Path.cwd()
        # narrow to selected chapters if any checked
        sel = self.toc.selected()
        if sel:
            chapters = [c for i, c in enumerate(self._plan.chapters) if i in sel]
        else:
            chapters = self._plan.chapters
        plan = Plan(book=self._plan.book, chapters=chapters, out_dir=self._out_dir, speaker_wav=self._speaker)
        cancel = threading.Event()
        worker = BookWorker(plan, cancel)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.progress.update_progress)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        thread.start()
        self.status.setText("Synthesis started")

    def _on_finished(self, path: str) -> None:
        self.status.setText(f"Done: {Path(path).name}")

    def _on_failed(self, msg: str) -> None:
        self.status.setText(f"Failed: {msg}")
