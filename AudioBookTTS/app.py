from __future__ import annotations

import os
import sys
import threading
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import numpy as np
import soundfile as sf
from PySide6.QtCore import Qt, Signal, QObject, QThread, QUrl
from PySide6.QtGui import QAction
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout, QWidget, QSlider)
import pyqtgraph as pg
from TTS.api import TTS
os.environ.setdefault("COQUI_TTS_LOG_LEVEL", "INFO")
DEFAULT_SPEAKER_PATH = Path(r"D:/Downloads/coqui_voice_pack_v2/voice_pack_v2/voice/my_reader.wav")
DEFAULT_LANGUAGE = "en"
class TTSCache:
    """Thread-safe lazy loader for the XTTS model."""
    _model: Optional[TTS] = None
    _lock = threading.Lock()
    @classmethod
    def get_model(cls) -> TTS:
        with cls._lock:
            if cls._model is None:
                cls._model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
            return cls._model
class GenerationWorker(QObject):
    finished = Signal(str, np.ndarray, int)
    failed = Signal(str)
    started = Signal()
    status = Signal(str)
    def __init__(self, text: str, language: str, speaker_path: Optional[str] = None) -> None:
        super().__init__()
        self._text = text
        self._language = language
        self._speaker_path = speaker_path
    def _emit_status(self, message: str) -> None:
        self.status.emit(message)
        print(message, flush=True)
    def run(self) -> None:
        self.started.emit()
        self._emit_status("Preparing XTTS model... (first run may download files)")
        try:
            model = TTSCache.get_model()
            self._emit_status("Synthesizing audio...")
            kwargs = {"text": self._text, "language": self._language}
            if self._speaker_path:
                kwargs["speaker_wav"] = self._speaker_path
            self._emit_status("TTS parameters: chars=%d, language=%s, speaker=%s" % (len(self._text), self._language, "yes" if self._speaker_path else "no"))
            wav = model.tts(**kwargs)
            sample_rate = getattr(model, "output_sample_rate", None)
            if sample_rate is None and hasattr(model, "synthesizer"):
                sample_rate = getattr(model.synthesizer, "output_sample_rate", None)
            if sample_rate is None:
                sample_rate = 24000
            wav = np.asarray(wav, dtype=np.float32)
            if wav.ndim > 1:
                wav = np.mean(wav, axis=1)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                sf.write(tmp_file.name, wav, sample_rate)
                temp_path = tmp_file.name
            self._emit_status("Synthesis completed. Starting playback.")
            self.finished.emit(temp_path, wav, sample_rate)
        except Exception as exc:  # pylint: disable=broad-except
            self._emit_status("An error occurred during synthesis.")
            self.failed.emit(str(exc))

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AudioBook TTS")
        self.resize(1100, 700)
        self._current_audio_path: Optional[str] = None
        self._current_sample_rate: Optional[int] = None
        self._current_audio_data: Optional[np.ndarray] = None
        self._speaker_path: Optional[str] = None
        self._worker_thread: Optional[QThread] = None
        self._current_worker: Optional[GenerationWorker] = None
        self._language: str = DEFAULT_LANGUAGE
        self._last_text: str = ""
        self._pending_text: Optional[str] = None
        self._slider_is_pressed: bool = False
        self._player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._audio_output.setVolume(1.0)
        self._player.setAudioOutput(self._audio_output)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        self._text_input = QPlainTextEdit()
        self._text_input.setPlaceholderText("Paste English text to narrate...")
        self._text_input.setMinimumHeight(180)
        root_layout.addWidget(self._text_input, stretch=1)
        waveform_container = QWidget()
        waveform_layout = QVBoxLayout(waveform_container)
        waveform_layout.setContentsMargins(0, 0, 0, 0)
        waveform_layout.setSpacing(6)
        self._waveform = pg.PlotWidget()
        self._waveform.setBackground("w")
        self._waveform.showGrid(x=True, y=True, alpha=0.15)
        self._waveform.setLabel("bottom", "Time", units="s")
        self._waveform.setLabel("left", "Amplitude")
        self._waveform.setFixedHeight(160)
        waveform_layout.addWidget(self._waveform)
        self._position_slider = QSlider(Qt.Horizontal)
        self._position_slider.setRange(0, 0)
        self._position_slider.setSingleStep(100)
        self._position_slider.setPageStep(1000)
        self._position_slider.setEnabled(False)
        self._position_slider.sliderPressed.connect(self._on_slider_pressed)
        self._position_slider.sliderReleased.connect(self._on_slider_released)
        self._position_slider.sliderMoved.connect(self._on_slider_moved)
        waveform_layout.addWidget(self._position_slider)
        root_layout.addWidget(waveform_container)
        button_row = QHBoxLayout()
        root_layout.addLayout(button_row)
        self._speaker_button = QPushButton("Select Speaker")
        self._speaker_button.clicked.connect(self._pick_speaker)  # type: ignore[arg-type]
        button_row.addWidget(self._speaker_button)
        self._read_button = QPushButton("Synthesize")
        self._read_button.clicked.connect(self._on_read_clicked)  # type: ignore[arg-type]
        button_row.addWidget(self._read_button)
        self._stop_button = QPushButton("Stop")
        self._stop_button.clicked.connect(self._on_stop_clicked)  # type: ignore[arg-type]
        button_row.addWidget(self._stop_button)
        self._save_button = QPushButton("Save Audio")
        self._save_button.clicked.connect(self._on_save_clicked)  # type: ignore[arg-type]
        button_row.addWidget(self._save_button)
        button_row.addStretch(1)
        self._status_label = QLabel("Ready")
        root_layout.addWidget(self._status_label)
        self._log_output = QPlainTextEdit()
        self._log_output.setReadOnly(True)
        self._log_output.setPlaceholderText("Logs will appear here...")
        self._log_output.document().setMaximumBlockCount(500)
        self._log_output.setFixedHeight(160)
        root_layout.addWidget(self._log_output)
        self._cleanup_paths: List[Path] = []
        self._setup_menu()
        self._apply_default_speaker()
        self._append_log("Application ready.")
    def _setup_menu(self) -> None:
        select_speaker_action = QAction("Select Speaker", self)
        select_speaker_action.triggered.connect(self._pick_speaker)  # type: ignore[arg-type]
        self._select_speaker_action = select_speaker_action
        clear_text_action = QAction("Clear Text", self)
        clear_text_action.triggered.connect(self._text_input.clear)  # type: ignore[arg-type]
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(select_speaker_action)
        file_menu.addAction(clear_text_action)
    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self._log_output.appendPlainText(entry)
        print(entry, flush=True)
    def _on_thread_started(self) -> None:
        self._append_log("Worker thread started.")
    def _on_worker_started(self) -> None:
        self._append_log("Worker signaled start.")
    def _apply_default_speaker(self) -> None:
        if DEFAULT_SPEAKER_PATH.exists():
            self._speaker_path = str(DEFAULT_SPEAKER_PATH)
            self._speaker_button.setEnabled(False)
            self._speaker_button.setText("Speaker Locked")
            self._speaker_button.setToolTip(f"Default speaker: {DEFAULT_SPEAKER_PATH.name}")
            if hasattr(self, "_select_speaker_action"):
                self._select_speaker_action.setEnabled(False)
            self._status_label.setText(f"Speaker: {DEFAULT_SPEAKER_PATH.name}")
            self._append_log(f"Using default speaker file: {DEFAULT_SPEAKER_PATH}")
        else:
            message = f"Default speaker file not found: {DEFAULT_SPEAKER_PATH}"
            self._append_log(message)
            self._status_label.setText("Speaker file missing. Use the button to select one.")
            self._speaker_button.setEnabled(True)
            self._speaker_button.setText("Select Speaker")
            self._speaker_button.setToolTip("Default speaker not found. Select a file manually.")
            self._select_speaker_action.setEnabled(True)
    def _on_worker_status(self, message: str) -> None:
        self._status_label.setText(message)
        self._append_log(message)
    def _on_position_changed(self, position: int) -> None:
        if not self._slider_is_pressed:
            self._position_slider.setValue(position)
    def _on_duration_changed(self, duration: int) -> None:
        max_duration = max(duration, 0)
        self._position_slider.blockSignals(True)
        self._position_slider.setRange(0, max_duration)
        self._position_slider.setValue(0)
        self._position_slider.blockSignals(False)
        self._position_slider.setEnabled(max_duration > 0)
    def _on_slider_pressed(self) -> None:
        self._slider_is_pressed = True
    def _on_slider_released(self) -> None:
        self._slider_is_pressed = False
        self._player.setPosition(self._position_slider.value())
        self._player.play()
    def _on_slider_moved(self, position: int) -> None:
        if self._slider_is_pressed:
            self._status_label.setText(f"Seeking to {position / 1000:.1f}s")
    def _pick_speaker(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select speaker audio", str(Path.home()), "Audio Files (*.wav *.mp3 *.m4a)")
        if not path:
            return
        self._speaker_path = path
        self._status_label.setText(f"Speaker: {Path(path).name}")
        self._append_log(f"Speaker selected: {path}")
    def _on_read_clicked(self) -> None:
        if self._worker_thread is not None:
            QMessageBox.warning(self, "In progress", "Please wait for the previous synthesis to finish.")
        text = self._text_input.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "Text required", "Enter text to synthesize before starting.")
        if not self._speaker_path or not Path(self._speaker_path).exists():
            QMessageBox.information(self, "Speaker required", "XTTS v2 needs a valid speaker reference before synthesis.")
        if text == self._last_text and self._current_audio_path and Path(self._current_audio_path).exists():
            self._append_log("Text unchanged; replaying cached audio.")
            self._status_label.setText("Replaying cached audio")
            self._player.stop()
            self._slider_is_pressed = False
            self._position_slider.blockSignals(True)
            self._position_slider.blockSignals(False)
            self._player.setPosition(0)
            self._save_button.setEnabled(True)
            self._play_audio()
        self._pending_text = text
        self._status_label.setText("Preparing XTTS model...")
        self._read_button.setEnabled(False)
        self._save_button.setEnabled(False)
        self._append_log(f"Starting synthesis: {len(text)} chars, speaker={self._speaker_path}, language={self._language}")
        worker = GenerationWorker(text, language=self._language, speaker_path=self._speaker_path)
        self._current_worker = worker
        thread = QThread(self)
        worker.moveToThread(thread)
        worker.status.connect(self._on_worker_status)
        worker.started.connect(self._on_worker_started)
        thread.started.connect(self._on_thread_started)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_generation_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(self._on_generation_failed)
        worker.failed.connect(thread.quit)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._worker_thread = thread
        thread.start()
    def _on_thread_finished(self) -> None:
        if self._worker_thread is None:
            self._worker_thread.wait()
        self._worker_thread.deleteLater()
        self._worker_thread = None
        self._current_worker = None
        self._read_button.setEnabled(True)
        self._save_button.setEnabled(self._current_audio_path is not None)
        self._append_log("Synthesis worker finished.")
    def _on_generation_finished(self, path: str, audio: np.ndarray, sample_rate: int) -> None:
        self._status_label.setText("Synthesis finished. Playing...")
        self._append_log("Synthesis completed; starting playback.")
        self._set_audio_data(path, audio, sample_rate)
        if self._pending_text is not None:
            self._last_text = self._pending_text
        self._pending_text = None
        self._play_audio()

    def _on_generation_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Synthesis failed", message)
        self._status_label.setText("An error occurred.")
        self._append_log(f"Synthesis error: {message}")
        self._pending_text = None
        self._read_button.setEnabled(True)

    def _on_stop_clicked(self) -> None:
        self._player.stop()
        self._status_label.setText("Stopped")
        self._append_log("Playback stopped.")

    def _play_audio(self) -> None:
        if not self._current_audio_path:
            return
        url = QUrl.fromLocalFile(str(Path(self._current_audio_path).resolve()))
        self._slider_is_pressed = False
        self._player.setSource(url)
        self._player.setPosition(0)
        self._position_slider.blockSignals(True)
        self._position_slider.setValue(0)
        self._position_slider.blockSignals(False)
        self._player.play()
        self._append_log("Playback started.")

    def _set_audio_data(self, path: str, audio: np.ndarray, sample_rate: int) -> None:
        if self._current_audio_path:
            self._cleanup_paths.append(Path(self._current_audio_path))
        self._current_audio_path = path
        self._current_audio_data = audio
        self._current_sample_rate = sample_rate

        self._waveform.clear()
        if audio.size == 0:
            self._position_slider.blockSignals(True)
            self._position_slider.setRange(0, 0)
            self._position_slider.setValue(0)
            self._position_slider.setEnabled(False)
            self._position_slider.blockSignals(False)
            return

        duration = audio.size / sample_rate
        times = np.linspace(0.0, duration, num=audio.size, endpoint=False)
        self._waveform.plot(times, audio, pen=pg.mkPen(color="#2E86AB", width=1))
        self._waveform.setXRange(0.0, max(duration, 1.0), padding=0.01)
        min_amp = float(np.min(audio))
        max_amp = float(np.max(audio))
        if min_amp == max_amp:
            max_amp = max_amp + 1.0 if max_amp >= 0 else max_amp - 1.0
        self._waveform.setYRange(min_amp, max_amp, padding=0.1)

        total_ms = int(duration * 1000)
        self._position_slider.blockSignals(True)
        self._position_slider.setRange(0, max(total_ms, 0))
        self._position_slider.setValue(0)
        self._position_slider.setEnabled(True)
        self._position_slider.blockSignals(False)

    def _on_save_clicked(self) -> None:
        if not self._current_audio_path:
            QMessageBox.information(self, "Nothing to save", "Generate speech before saving.")
            return

        default_dir = str(Path.home())
        filter_spec = "WAV (*.wav)"
        target_path, _ = QFileDialog.getSaveFileName(self, "Save audio", default_dir, filter_spec)
        if not target_path:
            return

        self._save_audio_file(Path(target_path))

    def _save_audio_file(self, target_path: Path) -> None:
        target_path = target_path.with_suffix(".wav")
        try:
            shutil.copyfile(self._current_audio_path, target_path)
            self._status_label.setText(f"Saved: {target_path.name}")
            self._append_log(f"WAV file saved to {target_path}")
        except Exception as exc:  # pylint: disable=broad-except
            QMessageBox.critical(self, "Save failed", str(exc))
            self._append_log(f"Save failed: {exc}")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._player.stop()
        for path in self._cleanup_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        if self._current_audio_path:
            try:
                Path(self._current_audio_path).unlink(missing_ok=True)
            except OSError:
                pass
        super().closeEvent(event)
def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
if __name__ == "__main__":
    main()

