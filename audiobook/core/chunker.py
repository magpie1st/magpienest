from __future__ import annotations

import re
from typing import List

SENT_SPLIT = re.compile(r"(?<=[\.\?!])\s+")


def chunk_paragraphs(text: str, max_chars: int = 250) -> List[str]:
    blocks: List[str] = []
    for para in text.splitlines():
        para = para.strip()
        if not para:
            continue
        sentences = SENT_SPLIT.split(para)
        buf = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if not buf:
                buf = s
            elif len(buf) + 1 + len(s) <= max_chars:
                buf = f"{buf} {s}"
            else:
                blocks.append(buf)
                buf = s
        if buf:
            blocks.append(buf)
            buf = ""
    return [b.strip() for b in blocks if b.strip()]
