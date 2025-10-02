from __future__ import annotations

import threading
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np  # type: ignore
import soundfile as sf  # type: ignore
from TTS.api import TTS  # type: ignore
import logging

# Compatibility shims for newer dependency stacks.
try:  # pragma: no cover - dependent on transformers version
    from transformers.generation import GenerationMixin  # type: ignore
    from TTS.tts.layers.xtts import gpt as xtts_gpt  # type: ignore

    if GenerationMixin not in xtts_gpt.GPT2InferenceModel.__mro__:
        xtts_gpt.GPT2InferenceModel.__bases__ = (
            xtts_gpt.GPT2InferenceModel.__bases__ + (GenerationMixin,)
        )
except Exception:  # pragma: no cover - safe fallback if transformers layout changes
    pass

# Newer torch builds require explicitly allow-listing config classes for safe loading.
try:  # pragma: no cover - dependent on torch version
    from torch.serialization import add_safe_globals  # type: ignore
    from TTS.tts.configs.xtts_config import XttsConfig  # type: ignore
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs  # type: ignore
    from TTS.config.shared_configs import BaseDatasetConfig  # type: ignore

    add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])
except Exception:  # pragma: no cover - safe fallback if torch lacks APIs
    pass

from audiobook.config import DEFAULT_MODEL, DEFAULT_LANGUAGE, USE_GPU


logger = logging.getLogger(__name__)


class XTTSEngine:
    _lock = threading.Lock()
    _model: Optional[TTS] = None
    _model_name: Optional[str] = None
    _model_gpu: Optional[bool] = None

    def __init__(self, model: str = DEFAULT_MODEL, gpu: bool = USE_GPU, speaker_wav: Optional[str] = None) -> None:
        self.model_name = model
        self.use_cuda = gpu
        self.speaker_wav = speaker_wav
        self.language = DEFAULT_LANGUAGE

    @classmethod
    def _ensure_model(cls, model_name: str, use_gpu: bool) -> TTS:
        with cls._lock:
            if (
                cls._model is None
                or cls._model_name != model_name
                or cls._model_gpu != use_gpu
            ):
                logger.info(
                    "Loading TTS model: %s (gpu=%s)",
                    model_name,
                    use_gpu,
                )
                cls._model = TTS(model_name=model_name, gpu=use_gpu)
                cls._model_name = model_name
                cls._model_gpu = use_gpu
            return cls._model

    def synthesize_line(self, text: str, language: str | None = None) -> Tuple[Path, int]:
        lang = language or self.language
        model = self._ensure_model(self.model_name, self.use_cuda)
        kwargs = {"text": text, "language": lang}
        if self.speaker_wav:
            kwargs["speaker_wav"] = self.speaker_wav
        logger.info(
            "Synthesizing line len=%d language=%s speaker=%s",
            len(text),
            lang,
            bool(self.speaker_wav),
        )
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
