from __future__ import annotations

from pathlib import Path

DEFAULT_LANGUAGE = "en"
DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
CHUNK_MAX_CHARS = 250
PAUSE_MS_DEFAULT = 350
OUTPUT_FORMAT_DEFAULT = "mp3"  # or "wav"
CACHE_DIRNAME = ".cache"

# Optional default speaker file; set to None to force selection
DEFAULT_SPEAKER: str | None = None
