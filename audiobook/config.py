from __future__ import annotations

from pathlib import Path

DEFAULT_LANGUAGE = "en"
DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
CHUNK_MAX_CHARS = 250
PAUSE_MS_DEFAULT = 350
OUTPUT_FORMAT_DEFAULT = "mp3"
CACHE_DIRNAME = ".cache"

PACKAGE_DIR = Path(__file__).resolve().parent

DEFAULT_EPUB: Path | None = PACKAGE_DIR / "test.epub"
if DEFAULT_EPUB and not DEFAULT_EPUB.exists():
    DEFAULT_EPUB = None

DEFAULT_SPEAKER: str | None = str(PACKAGE_DIR / "my_reader.wav")
if DEFAULT_SPEAKER and not Path(DEFAULT_SPEAKER).exists():
    DEFAULT_SPEAKER = None

FFMPEG_PATH: str | None = None
