from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydub import AudioSegment  # type: ignore

from audiobook.config import FFMPEG_PATH


def configure_ffmpeg(path: Optional[str]) -> None:
    if path:
        exe = Path(path)
        AudioSegment.converter = str(exe)
        AudioSegment.ffmpeg = str(exe)
        ffprobe = exe.with_name("ffprobe.exe")
        if ffprobe.exists():
            AudioSegment.ffprobe = str(ffprobe)
    else:
        # fallback to default resolution
        for attr in ("converter", "ffmpeg", "ffprobe"):
            if hasattr(AudioSegment, attr):
                try:
                    delattr(AudioSegment, attr)
                except AttributeError:
                    pass


configure_ffmpeg(FFMPEG_PATH)


def merge_wavs(paths: List[Path], out: Path, pause_ms: int = 350, out_format: str = "mp3") -> Path:
    silence = AudioSegment.silent(duration=max(0, int(pause_ms)))
    combined = AudioSegment.silent(duration=0)
    for p in paths:
        seg = AudioSegment.from_wav(p)
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
