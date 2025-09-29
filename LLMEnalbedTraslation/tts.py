"""Coqui XTTS v2 text to speech helper module."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Optional

from pydub import AudioSegment

try:
    from TTS.api import TTS  # type: ignore
    from TTS.tts.configs.xtts_config import XttsConfig  # type: ignore
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs  # type: ignore
    from TTS.config.shared_configs import BaseDatasetConfig  # type: ignore

    try:  # Register XTTS-related config classes for torch.load when weights_only=True (PyTorch >= 2.6)
        from torch.serialization import add_safe_globals

        add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])
    except Exception:
        pass  # Older torch versions or failures fall back to default behaviour
except Exception as exc:  # pragma: no cover - optional dependency load
    raise ImportError(
        "The 'TTS' package is required for Coqui XTTS support. Install with 'pip install git+https://github.com/coqui-ai/TTS'"
        " and 'pip install pydub'."
    ) from exc


DEFAULT_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_LANGUAGE = "en"


class CoquiTTSService:
    """Thin wrapper around Coqui XTTS for audio synthesis."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        use_cuda: bool = False,
    ) -> None:
        self._model_name = model_name
        self._use_cuda = use_cuda
        self._tts: Optional[TTS] = None

    def _ensure_model(self) -> None:
        if self._tts is None:
            self._tts = TTS(model_name=self._model_name, gpu=self._use_cuda)

    def _available_speakers(self) -> list[str]:
        self._ensure_model()
        speakers = []
        if not self._tts:
            return speakers

        attr = getattr(self._tts, "speakers", None)
        if isinstance(attr, dict):
            speakers.extend(attr.keys())
        elif isinstance(attr, (list, tuple)):
            speakers.extend(attr)

        manager = getattr(self._tts, "speaker_manager", None)
        if manager is not None:
            names = getattr(manager, "speaker_names", None)
            if isinstance(names, (list, tuple)):
                speakers.extend(names)

        seen = set()
        unique: list[str] = []
        for name in speakers:
            if not isinstance(name, str):
                continue
            if name not in seen:
                seen.add(name)
                unique.append(name)
        return unique

    def _resolve_speaker(self, requested: Optional[str], speaker_wav: Optional[str]) -> Optional[str]:
        if speaker_wav:
            return None  # XTTS will derive speaker embedding from wav
        if requested:
            return requested
        speakers = self._available_speakers()
        return speakers[0] if speakers else None

    def synthesize_to_mp3(
        self,
        text: str,
        *,
        language: str = DEFAULT_LANGUAGE,
        speaker_wav: Optional[str] = None,
        speaker: Optional[str] = None,
        progress_callbacks: Optional[Iterable[Callable[[], None]]] = None,
    ) -> bytes:
        """Generate MP3 audio for the supplied text.

        Args:
            text: English text to read out.
            language: Language code accepted by XTTS (default: 'en').
            speaker_wav: Optional reference WAV/MP3 file for voice cloning.
            progress_callbacks: Optional iterables of callables invoked at key steps.

        Returns:
            Raw MP3 bytes.
        """

        self._ensure_model()
        assert self._tts is not None  # for mypy

        callbacks = list(progress_callbacks or [])

        def _notify(index: int) -> None:
            if 0 <= index < len(callbacks):
                try:
                    callbacks[index]()
                except Exception:  # pragma: no cover - defensive best effort
                    pass

        _notify(0)  # loading start
        resolved_speaker = self._resolve_speaker(speaker, speaker_wav)

        with tempfile.TemporaryDirectory(prefix="xtts_") as tmpdir:
            tmp_wav = Path(tmpdir) / "output.wav"
            tts_kwargs = {
                "text": text,
                "file_path": str(tmp_wav),
                "language": language,
            }
            if speaker_wav:
                tts_kwargs["speaker_wav"] = speaker_wav
            if resolved_speaker:
                tts_kwargs["speaker"] = resolved_speaker

            self._tts.tts_to_file(**tts_kwargs)
            _notify(1)  # synthesis complete
            audio = AudioSegment.from_file(tmp_wav, format="wav")
            mp3_path = Path(tmpdir) / "output.mp3"
            audio.export(mp3_path, format="mp3")
            _notify(2)  # export complete
            data = mp3_path.read_bytes()
        return data

    @staticmethod
    def encode_base64(audio_bytes: bytes) -> str:
        """Return base64 string for MP3 bytes."""

        return base64.b64encode(audio_bytes).decode("ascii")
