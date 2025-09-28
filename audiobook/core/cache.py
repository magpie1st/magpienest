from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from audiobook.config import CACHE_DIRNAME


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def cache_dir(out_dir: Path) -> Path:
    d = out_dir / CACHE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_key(value: str) -> str:
    return _sha256(value)


def get(out_dir: Path, key: str) -> Optional[Path]:
    d = cache_dir(out_dir)
    for ext in (".mp3", ".wav"):
        p = d / f"{key}{ext}"
        if p.exists():
            return p
    return None


def put(out_dir: Path, key: str, src_path: Path, ext: str = "mp3") -> Path:
    d = cache_dir(out_dir)
    dst = d / f"{key}.{ext.strip('.')}"
    dst.write_bytes(Path(src_path).read_bytes())
    return dst
