from __future__ import annotations

from pathlib import Path
from typing import List

from pydub import AudioSegment  # type: ignore


def merge_wavs(paths: List[Path], out: Path, pause_ms: int = 350, out_format: str = "mp3") -> Path:
    silence = AudioSegment.silent(duration=max(0, int(pause_ms)))
    output = AudioSegment.silent(duration=0)
    for p in paths:
        seg = AudioSegment.from_wav(p)
        output += seg + silence
    out.parent.mkdir(parents=True, exist_ok=True)
    if out_format.lower() == "wav":
        output.export(str(out.with_suffix(".wav")), format="wav")
        return out.with_suffix(".wav")
    # default mp3 192k
    out_mp3 = out.with_suffix(".mp3")
    output.export(str(out_mp3), format="mp3", bitrate="192k")
    return out_mp3
