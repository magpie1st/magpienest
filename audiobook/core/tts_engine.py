from __future__ import annotations

import threading
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np  # type: ignore
import soundfile as sf  # type: ignore
from TTS.api import TTS  # type: ignore

from audiobook.config import DEFAULT_MODEL, DEFAULT_LANGUAGE


class XTTSEngine:
    _lock = threading.Lock()
    _model: Optional[TTS] = None

    def __init__(self, model: str = DEFAULT_MODEL, gpu: bool = True, speaker_wav: Optional[str] = None) -> None:
        self.model_name = model
        self.use_cuda = gpu
        self.speaker_wav = speaker_wav
        self.language = DEFAULT_LANGUAGE

    @classmethod
    def _ensure_model(cls, model_name: str) -> TTS:
        with cls._lock:
            if cls._model is None:
                cls._model = TTS(model_name=model_name)
            return cls._model

    def synthesize_line(self, text: str, language: str | None = None) -> Tuple[Path, int]:
        lang = language or self.language
        model = self._ensure_model(self.model_name)
        kwargs = {"text": text, "language": lang}
        if self.speaker_wav:
            kwargs["speaker_wav"] = self.speaker_wav
        wav = model.tts(**kwargs)
        # try to get SR from model
        sr = getattr(model, "output_sample_rate", None)
        if sr is None and hasattr(model, "synthesizer"):
            sr = getattr(model.synthesizer, "output_sample_rate", None)
        if sr is None:
            sr = 24000
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
        sf.write(str(tmp), wav, sr)
        return tmp, int(sr)
