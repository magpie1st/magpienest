from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

try:
    from unidecode import unidecode  # type: ignore
except Exception:  # pragma: no cover
    def unidecode(s: str) -> str:  # type: ignore
        return s

_slug_re = re.compile(r"[^a-z0-9]+")


def slugify(s: str) -> str:
    s = unidecode(s or "").lower().strip()
    s = _slug_re.sub("-", s)
    return s.strip("-") or "untitled"


def book_dir(root: Path, author: Optional[str], title: str) -> Path:
    name = f"{author or 'anon'}_{title}"
    return root / slugify(name)


def chapter_filename(index: int, title: str, ext: str = "mp3") -> str:
    return f"{index:03d}_" + slugify(title) + f".{ext}"
