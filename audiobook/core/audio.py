from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import logging

from pydub import AudioSegment  # type: ignore

from audiobook.config import FFMPEG_PATH


logger = logging.getLogger(__name__)


def configure_ffmpeg(path: Optional[str]) -> None:
    if path:
        exe = Path(path)
        if not exe.exists():
            logger.warning("Configured ffmpeg path does not exist: %s", exe)
            return
        AudioSegment.converter = str(exe)
        AudioSegment.ffmpeg = str(exe)
        ffprobe = exe.with_name("ffprobe.exe")
        if ffprobe.exists():
            AudioSegment.ffprobe = str(ffprobe)
        logger.info("Configured ffmpeg converter: %s", exe)
    else:
        logger.info("Using system ffmpeg resolution")
        return


configure_ffmpeg(FFMPEG_PATH)


def merge_wavs(paths: List[Path], out: Path, pause_ms: int = 350, out_format: str = "mp3") -> Path:
    logger.info(
        "Merging %d audio segments -> %s (format=%s, pause_ms=%d)",
        len(paths),
        out,
        out_format,
        pause_ms,
    )
    silence = AudioSegment.silent(duration=max(0, int(pause_ms)))
    combined = AudioSegment.silent(duration=0)
    for p in paths:
        seg = AudioSegment.from_file(p)
        combined += seg + silence
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = out_format.lower()
    if fmt == "wav":
        dst = out.with_suffix(".wav")
        combined.export(str(dst), format="wav")
        return dst
    dst = out.with_suffix(".mp3")
    combined.export(str(dst), format="mp3", bitrate="192k")
    return dst
