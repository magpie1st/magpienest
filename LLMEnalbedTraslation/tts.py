"""Coqui XTTS v2 text to speech helper module."""

from __future__ import annotations

import base64
import datetime as _dt
import logging
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Callable, Iterable, Optional

from pydub import AudioSegment

try:
    from TTS.api import TTS  # type: ignore
    from TTS.tts.configs.xtts_config import XttsConfig  # type: ignore
    from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig  # type: ignore
    from TTS.config.shared_configs import BaseDatasetConfig  # type: ignore

    try:  # Register XTTS-related config classes for torch.load when weights_only=True (PyTorch >= 2.6)
        from torch.serialization import add_safe_globals

        add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])
    except Exception:  # pragma: no cover - torch < 2.6 or unavailable helper
        pass
except Exception as exc:  # pragma: no cover - optional dependency load
    raise ImportError(
        "The 'TTS' package is required for Coqui XTTS support. Install with 'pip install git+https://github.com/coqui-ai/TTS'"
        " and 'pip install pydub'."
    ) from exc


DEFAULT_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_LANGUAGE = "en"
DEFAULT_SPEAKER = "english_female_0"
DEFAULT_FFMPEG_PATH = "/usr/bin/ffmpeg"
DEFAULT_OUTPUT_DIR = Path.home() / "LLMTTS" / "audio"


logger = logging.getLogger(__name__)


def _configure_ffmpeg(path: Optional[str]) -> None:
    if not path:
        logger.debug("[XTTS] no explicit FFmpeg path provided; using defaults")
        return
    ffmpeg_path = Path(path).expanduser()
    if ffmpeg_path.is_file():
        converter = ffmpeg_path
        directory = ffmpeg_path.parent
    else:
        directory = ffmpeg_path
        converter = directory / "ffmpeg"

    if converter.exists():
        AudioSegment.converter = str(converter)
        AudioSegment.ffmpeg = str(converter)
        logger.debug("[XTTS] ffmpeg converter set to %s", converter)
    probe = (directory / "ffprobe") if directory else None
    if probe and probe.exists():
        AudioSegment.ffprobe = str(probe)
        logger.debug("[XTTS] ffprobe set to %s", probe)


@dataclass
class TTSSettings:
    ffmpeg_path: Optional[str] = DEFAULT_FFMPEG_PATH
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    default_language: str = DEFAULT_LANGUAGE
    default_speaker: Optional[str] = DEFAULT_SPEAKER
    use_cuda: bool = False

    def to_dict(self, *, include_available: bool = False, available_speakers: Optional[list[str]] = None) -> dict:
        data = asdict(self)
        if include_available:
            data["available_speakers"] = available_speakers or []
        return data


@dataclass
class TTSResult:
    audio_bytes: bytes
    language: str
    speaker: Optional[str]
    saved_path: Optional[Path]
    filename: Optional[str]

    @property
    def mime(self) -> str:  # pragma: no cover - trivial
        return "audio/mpeg"


class CoquiTTSService:
    """Thin wrapper around Coqui XTTS for audio synthesis."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        settings: Optional[TTSSettings] = None,
    ) -> None:
        self._model_name = model_name
        self._settings = settings or TTSSettings()
        _configure_ffmpeg(self._settings.ffmpeg_path)
        self._tts: Optional[TTS] = None
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Settings management
    # ------------------------------------------------------------------
    def get_settings(self) -> TTSSettings:
        return TTSSettings(**asdict(self._settings))

    def update_settings(
        self,
        *,
        ffmpeg_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        default_language: Optional[str] = None,
        default_speaker: Optional[str] = None,
        use_cuda: Optional[bool] = None,
    ) -> TTSSettings:
        changed_cuda = False
        if ffmpeg_path is not None:
            self._settings.ffmpeg_path = ffmpeg_path or None
            _configure_ffmpeg(self._settings.ffmpeg_path)
        if output_dir is not None:
            self._settings.output_dir = output_dir
        if default_language:
            self._settings.default_language = default_language
        if default_speaker is not None:
            self._settings.default_speaker = default_speaker or None
        if use_cuda is not None and use_cuda != self._settings.use_cuda:
            self._settings.use_cuda = use_cuda
            changed_cuda = True
        logger.debug(
            "[XTTS] settings updated | ffmpeg=%s output=%s lang=%s speaker=%s cuda=%s",
            self._settings.ffmpeg_path,
            self._settings.output_dir,
            self._settings.default_language,
            self._settings.default_speaker,
            self._settings.use_cuda,
        )
        if changed_cuda:
            self._tts = None  # force reload with new device preference
            logger.debug("[XTTS] reset cached model because CUDA preference changed")
        return self.get_settings()

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------
    def _ensure_model(self) -> None:
        with self._lock:
            if self._tts is None:
                logger.debug(
                    "[XTTS] loading model %s (use_cuda=%s)",
                    self._model_name,
                    self._settings.use_cuda,
                )
                self._tts = TTS(model_name=self._model_name, gpu=self._settings.use_cuda)

    def available_speakers(self) -> list[str]:
        self._ensure_model()
        tts = self._tts
        if not tts:
            return []

        speakers: list[str] = []
        attr = getattr(tts, "speakers", None)
        if isinstance(attr, dict):
            speakers.extend(attr.keys())
        elif isinstance(attr, (list, tuple)):
            speakers.extend(attr)

        manager = getattr(tts, "speaker_manager", None)
        if manager is not None:
            names = getattr(manager, "speaker_names", None)
            if isinstance(names, (list, tuple)):
                speakers.extend(names)

        seen: set[str] = set()
        ordered: list[str] = []
        for name in speakers:
            if isinstance(name, str) and name not in seen:
                seen.add(name)
                ordered.append(name)
        logger.debug("[XTTS] available_speakers -> %s", ordered)
        return ordered

    def _resolve_speaker(self, requested: Optional[str], speaker_wav: Optional[str]) -> Optional[str]:
        if speaker_wav:
            return None  # XTTS will derive speaker embedding from wav file
        if requested:
            return requested
        default = self._settings.default_speaker
        if default:
            return default
        speakers = self.available_speakers()
        resolved = speakers[0] if speakers else None
        logger.debug("[XTTS] resolved speaker=%s", resolved)
        return resolved

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------
    def synthesize_to_mp3(
        self,
        text: str,
        *,
        language: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        speaker: Optional[str] = None,
        save_to_disk: bool = False,
        filename: Optional[str] = None,
        progress_callbacks: Optional[Iterable[Callable[[], None]]] = None,
    ) -> TTSResult:
        """Generate MP3 audio for the supplied text."""

        self._ensure_model()
        if not self._tts:  # pragma: no cover - defensive
            raise RuntimeError("XTTS model failed to initialise")

        lang = (language or self._settings.default_language or DEFAULT_LANGUAGE).strip() or DEFAULT_LANGUAGE
        callbacks = list(progress_callbacks or [])

        def _notify(index: int) -> None:
            if 0 <= index < len(callbacks):
                try:
                    callbacks[index]()
                except Exception:  # pragma: no cover - best effort logging only, ignore
                    pass

        _notify(0)  # loading start
        resolved_speaker = self._resolve_speaker(speaker, speaker_wav)
        logger.debug(
            "[XTTS] synthesize | len=%d lang=%s resolved_speaker=%s speaker_wav=%s save=%s filename=%s",
            len(text),
            lang,
            resolved_speaker,
            bool(speaker_wav),
            save_to_disk,
            filename,
        )

        with tempfile.TemporaryDirectory(prefix="xtts_") as tmpdir:
            tmp_wav = Path(tmpdir) / "output.wav"
            tts_kwargs = {
                "text": text,
                "file_path": str(tmp_wav),
                "language": lang,
            }
            if speaker_wav:
                tts_kwargs["speaker_wav"] = speaker_wav
            if resolved_speaker:
                tts_kwargs["speaker"] = resolved_speaker

            start_time = _dt.datetime.now()
            with self._lock:
                self._tts.tts_to_file(**tts_kwargs)
            elapsed = (_dt.datetime.now() - start_time).total_seconds()
            logger.debug("[XTTS] synthesis complete (%.2fs)", elapsed)
            _notify(1)  # synthesis complete
            audio = AudioSegment.from_file(tmp_wav, format="wav")
            mp3_path = Path(tmpdir) / "output.mp3"
            audio.export(mp3_path, format="mp3")
            _notify(2)  # export complete
            data = mp3_path.read_bytes()

        saved_path: Optional[Path] = None
        safe_filename: Optional[str] = None
        if save_to_disk:
            out_dir = Path(self._settings.output_dir).expanduser()
            out_dir.mkdir(parents=True, exist_ok=True)
            if filename:
                candidate = Path(filename).name
            else:
                timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                candidate = f"tts-{timestamp}.mp3"
            safe_filename = candidate
            saved_path = out_dir / candidate
            saved_path.write_bytes(data)
            logger.debug("[XTTS] saved MP3 to %s", saved_path)

        return TTSResult(
            audio_bytes=data,
            language=lang,
            speaker=resolved_speaker,
            saved_path=saved_path,
            filename=safe_filename,
        )

    @staticmethod
    def encode_base64(audio_bytes: bytes) -> str:
        return base64.b64encode(audio_bytes).decode("ascii")
