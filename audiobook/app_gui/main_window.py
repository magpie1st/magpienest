from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QThread, Qt, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QSlider,
)

from audiobook.config import DEFAULT_EPUB, DEFAULT_SPEAKER, FFMPEG_PATH
from audiobook.core.audio import configure_ffmpeg
from audiobook.core.epub_reader import load_epub, BookMeta, Chapter
from audiobook.app_gui.widgets.progress_panel import ProgressPanel
from audiobook.app_gui.widgets.toc_panel import TocPanel
from audiobook.app_gui.workers import BookWorker


configure_ffmpeg(FFMPEG_PATH)


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
        self.resize(1400, 860)

        self._speaker: Optional[str] = DEFAULT_SPEAKER
        self._out_dir: Optional[Path] = None
        self._plan: Optional[Plan] = None
        self._epub_path: Optional[Path] = None
        self._ffmpeg_path: Optional[str] = FFMPEG_PATH

        self._player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_output)
        self._slider_is_pressed = False
        self._current_audio_path: Optional[Path] = None
        self._current_duration_ms = 0

        central = QWidget(self)
        self.setCentralWidget(central)
        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(8, 8, 8, 8)
        outer_layout.setSpacing(8)

        self._setup_menu()
        toolbar = self._create_toolbar()
        outer_layout.addWidget(toolbar)

        main_split = QHBoxLayout()
        outer_layout.addLayout(main_split, stretch=1)

        self.toc = TocPanel(self)
        self.toc.setMinimumWidth(280)
        self.toc.current_changed.connect(self._show_chapter_text)
        main_split.addWidget(self.toc, stretch=0)

        right_panel = QVBoxLayout()
        right_panel.setSpacing(8)
        main_split.addLayout(right_panel, stretch=1)

        right_panel.addWidget(self._create_actions_group())
        right_panel.addWidget(self._create_summary_group())

        self.reader_view = QTextEdit(self)
        self.reader_view.setReadOnly(True)
        self.reader_view.setPlaceholderText("EPUB not loaded yet.")
        right_panel.addWidget(self.reader_view, stretch=1)

        self.status = QLabel("Ready", self)
        right_panel.addWidget(self.status)

        self.progress = ProgressPanel(self)
        right_panel.addWidget(self.progress)

        self.audio_group = self._create_audio_preview_group()
        outer_layout.addWidget(self.audio_group)
        self._apply_audio_group_size()

        self.log_view = QTextEdit(self)
        self.log_view.setReadOnly(True)
        self.log_view.setFixedHeight(160)
        self.log_view.setPlaceholderText("Log output...")
        outer_layout.addWidget(self.log_view)

        self._thread: Optional[QThread] = None

        self._load_default_epub()
        self._update_summary_labels()

    def _setup_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("File")
        self.action_open = file_menu.addAction("Open EPUB")
        self.action_open.triggered.connect(self._open_epub)
        self.action_out_dir = file_menu.addAction("Output Folder")
        self.action_out_dir.triggered.connect(self._pick_out_dir)
        self.action_speaker = file_menu.addAction("Select Speaker")
        self.action_speaker.triggered.connect(self._pick_speaker)
        self.action_ffmpeg = file_menu.addAction("FFmpeg Path")
        self.action_ffmpeg.triggered.connect(self._pick_ffmpeg)
        file_menu.addSeparator()
        self.action_synth = file_menu.addAction("Synthesize Selected")
        self.action_synth.triggered.connect(self._start_book)

    def _create_toolbar(self) -> QToolBar:
        toolbar = QToolBar("Actions", self)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toolbar.addAction(self.action_open)
        toolbar.addAction(self.action_out_dir)
        toolbar.addAction(self.action_speaker)
        toolbar.addAction(self.action_ffmpeg)
        toolbar.addSeparator()
        toolbar.addAction(self.action_synth)
        return toolbar

    def _create_actions_group(self) -> QWidget:
        group = QGroupBox("Quick Actions", self)
        layout = QHBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        def make_btn(text: str, slot) -> QToolButton:
            btn = QToolButton(group)
            btn.setText(text)
            btn.clicked.connect(slot)
            btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            return btn

        layout.addWidget(make_btn("Open EPUB", self._open_epub))
        layout.addWidget(make_btn("Output Folder", self._pick_out_dir))
        layout.addWidget(make_btn("Select Speaker", self._pick_speaker))
        layout.addWidget(make_btn("FFmpeg Path", self._pick_ffmpeg))
        layout.addWidget(make_btn("Synthesize", self._start_book))
        layout.addStretch(1)
        return group

    def _create_summary_group(self) -> QWidget:
        group = QGroupBox("Current Selection", self)
        form = QFormLayout(group)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(6)

        self.label_book = QLabel("—")
        self.label_output = QLabel("(not set)")
        self.label_speaker = QLabel("(not set)")
        self.label_ffmpeg = QLabel(self._ffmpeg_path or "(PATH)")

        form.addRow("Book:", self.label_book)
        form.addRow("Output:", self.label_output)
        form.addRow("Speaker:", self.label_speaker)
        form.addRow("FFmpeg:", self.label_ffmpeg)
        return group

    def _create_audio_preview_group(self) -> QWidget:
        group = QGroupBox("Audio Preview", self)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.waveform = pg.PlotWidget()
        self.waveform.setBackground("w")
        self.waveform.showGrid(x=True, y=True, alpha=0.15)
        self.waveform.setLabel("bottom", "Time", units="s")
        self.waveform.setLabel("left", "Amplitude")
        layout.addWidget(self.waveform)

        self.position_slider = QSlider(Qt.Horizontal, self)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderPressed.connect(self._on_slider_pressed)
        self.position_slider.sliderReleased.connect(self._on_slider_released)
        self.position_slider.sliderMoved.connect(self._on_slider_moved)
        layout.addWidget(self.position_slider)

        btn_row = QHBoxLayout()
        self.btn_play = QPushButton("Play", self)
        self.btn_play.clicked.connect(self._play_audio)
        btn_row.addWidget(self.btn_play)

        self.btn_stop = QPushButton("Stop", self)
        self.btn_stop.clicked.connect(self._stop_audio)
        btn_row.addWidget(self.btn_stop)

        self.btn_save = QPushButton("Save As…", self)
        self.btn_save.clicked.connect(self._save_audio)
        btn_row.addWidget(self.btn_save)

        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self._player.positionChanged.connect(self._on_player_position)
        self._player.durationChanged.connect(self._on_player_duration)

        self._update_audio_controls(False)
        return group

    def _load_default_epub(self) -> None:
        pass

    def _append_log(self, message: str) -> None:
        self.log_view.append(str(message))

    def _load_default_epub(self) -> None:
        if DEFAULT_EPUB and DEFAULT_EPUB.exists():
            try:
                self._load_book(DEFAULT_EPUB)
                self._append_log(f"Loaded default EPUB: {DEFAULT_EPUB}")
            except Exception as exc:  # pragma: no cover
                QMessageBox.warning(self, "Failed to load default EPUB", str(exc))
                self._append_log(f"Failed to load default EPUB: {exc}")

    def _load_book(self, path: Path) -> None:
        meta, chapters = load_epub(path)
        self._epub_path = path
        out_dir = self._out_dir or Path.cwd()
        self._plan = Plan(book=meta, chapters=chapters, out_dir=out_dir, speaker_wav=self._speaker)
        self.toc.load(chapters)
        self.status.setText(f"Loaded: {meta.title}")
        self._append_log(f"Loaded book '{meta.title}' ({len(chapters)} chapters)")
        if chapters:
            self._show_chapter_text(0)
        self._update_summary_labels()

    def _open_epub(self) -> None:
        start_dir = str(self._epub_path.parent) if self._epub_path else str(Path.home())
        path_str, _ = QFileDialog.getOpenFileName(self, "Open EPUB", start_dir, "EPUB (*.epub)")
        if not path_str:
            return
        self._load_book(Path(path_str))

    def _pick_out_dir(self) -> None:
        start_dir = str(self._out_dir or Path.home())
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder", start_dir)
        if path:
            self._out_dir = Path(path)
            self.status.setText(f"Output: {path}")
            self._append_log(f"Output directory set to {path}")
            self._update_summary_labels()

    def _pick_speaker(self) -> None:
        start_dir = str(Path(self._speaker).parent) if self._speaker else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Select Speaker", start_dir, "Audio (*.wav *.mp3 *.m4a)")
        if path:
            self._speaker = path
            self.status.setText(f"Speaker: {Path(path).name}")
            self._append_log(f"Speaker set to {path}")
            self._update_summary_labels()

    def _pick_ffmpeg(self) -> None:
        start_dir = str(Path(self._ffmpeg_path).parent) if self._ffmpeg_path else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Select ffmpeg executable", start_dir, "Executable (*.exe);;All Files (*)")
        if not path:
            return
        self._ffmpeg_path = path
        configure_ffmpeg(path)
        self._append_log(f"FFmpeg set to {path}")
        self._update_summary_labels()

    def _start_book(self) -> None:
        if not self._plan:
            self.status.setText("Open an EPUB first")
            return
        if not self._out_dir:
            self._out_dir = Path.cwd()
        selected = self.toc.selected()
        chapters = [c for idx, c in enumerate(self._plan.chapters) if idx in selected] if selected else self._plan.chapters
        if not chapters:
            self.status.setText("No chapters selected")
            return
        plan = Plan(book=self._plan.book, chapters=chapters, out_dir=self._out_dir, speaker_wav=self._speaker)
        cancel_event = threading.Event()
        worker = BookWorker(plan, cancel_event)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.progress.update_progress)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        thread.start()
        self.status.setText("Synthesis started")
        self._append_log("Synthesis started")

    def _on_progress(self, ci: int, ct: int, li: int, lt: int) -> None:
        self._append_log(f"Progress — Chapter {ci}/{ct}, chunk {li}/{lt}")

    def _show_chapter_text(self, index: int) -> None:
        if not self._plan or index < 0 or index >= len(self._plan.chapters):
            self.reader_view.clear()
            return
        chapter = self._plan.chapters[index]
        header = f"Chapter {chapter.index}: {chapter.title}\n\n"
        self.reader_view.setPlainText(header + chapter.text)

    def _on_finished(self, path: str) -> None:
        self.status.setText(f"Done: {Path(path).name}")
        self._append_log(f"Finished synthesis → {path}")
        self._set_audio_preview(Path(path))

    def _on_failed(self, msg: str) -> None:
        self.status.setText(f"Failed: {msg}")
        self._append_log(f"Synthesis failed: {msg}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_audio_group_size()


    def _update_summary_labels(self) -> None:
        book_str = self._plan.book.title if self._plan else "—"
        output_str = str(self._out_dir) if self._out_dir else "(not set)"
        speaker_str = str(self._speaker) if self._speaker else "(not set)"
        ffmpeg_str = self._ffmpeg_path or "(PATH)"
        self.label_book.setText(book_str)
        self.label_output.setText(output_str)
        self.label_speaker.setText(speaker_str)
        self.label_ffmpeg.setText(ffmpeg_str)

    def _apply_audio_group_size(self) -> None:
        if not hasattr(self, 'audio_group'):
            return
        target = max(140, self.height() // 5)
        self.audio_group.setMinimumHeight(target)
        self.audio_group.setMaximumHeight(target)
        if hasattr(self, 'waveform'):
            self.waveform.setMinimumHeight(max(80, int(target * 0.6)))

    # --- Audio preview helpers -------------------------------------------------
    def _update_audio_controls(self, enabled: bool) -> None:
        self.btn_play.setEnabled(enabled)
        self.btn_stop.setEnabled(enabled)
        self.btn_save.setEnabled(enabled)
        self.position_slider.setEnabled(enabled)

    def _set_audio_preview(self, path: Path) -> None:
        if not path.exists():
            return
        self._current_audio_path = path
        try:
            from pydub import AudioSegment

            seg = AudioSegment.from_file(path)
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            if seg.channels > 1:
                samples = samples.reshape((-1, seg.channels)).mean(axis=1)
            max_val = np.max(np.abs(samples)) or 1.0
            samples /= max_val
            times = np.linspace(0, len(samples) / seg.frame_rate, num=len(samples), endpoint=False)
            self.waveform.clear()
            self.waveform.plot(times, samples, pen=pg.mkPen(color="#2E86AB", width=1))
            self.waveform.setXRange(0, times[-1] if times.size else 1, padding=0.01)
            self.waveform.setYRange(-1.05, 1.05, padding=0.02)
            duration_ms = int(seg.duration_seconds * 1000)
            self._current_duration_ms = duration_ms
            self.position_slider.blockSignals(True)
            self.position_slider.setRange(0, duration_ms)
            self.position_slider.setValue(0)
            self.position_slider.blockSignals(False)
            self._player.setSource(QUrl.fromLocalFile(str(path.resolve())))
            self._update_audio_controls(True)
            self._append_log(f"Loaded audio preview: {path}")
        except Exception as exc:  # pragma: no cover
            self._append_log(f"Failed to load audio preview: {exc}")

    def _play_audio(self) -> None:
        if not self._current_audio_path:
            return
        self._player.play()
        self._append_log("Playback started")

    def _stop_audio(self) -> None:
        self._player.stop()
        self._append_log("Playback stopped")

    def _save_audio(self) -> None:
        if not self._current_audio_path:
            return
        dest, _ = QFileDialog.getSaveFileName(self, "Save audio", str(self._current_audio_path.parent), "Audio (*.mp3 *.wav)")
        if not dest:
            return
        out_path = Path(dest)
        out_path.write_bytes(self._current_audio_path.read_bytes())
        self._append_log(f"Saved copy to {out_path}")
    def _on_slider_pressed(self) -> None:
        self._slider_is_pressed = True

    def _on_slider_released(self) -> None:
        self._slider_is_pressed = False
        self._player.setPosition(self.position_slider.value())

    def _on_slider_moved(self, position: int) -> None:
        if self._slider_is_pressed:
            self.status.setText(f"Seeking to {position / 1000:.1f}s")

    def _on_player_position(self, position: int) -> None:
        if not self._slider_is_pressed:
            self.position_slider.blockSignals(True)
            self.position_slider.setValue(position)
            self.position_slider.blockSignals(False)

    def _on_player_duration(self, duration: int) -> None:
        self._current_duration_ms = duration
        self.position_slider.blockSignals(True)
        self.position_slider.setRange(0, max(duration, 0))
        self.position_slider.blockSignals(False)






