"""Helper utilities for text chunking and token estimation."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

PARAGRAPH_BREAK = "\u0000PARA_BREAK\u0000"
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])[\s\u200b]+(?=[A-Z0-9\"'\(])")
_REPEAT_PATTERN = re.compile(r"([A-Za-z가-힣])\1{4,}")


@dataclass
class TokenEstimator:
    """Estimate token counts using an optional Hugging Face tokenizer."""

    tokenizer_path: str | None = None
    _tokenizer: Optional["Tokenizer"] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._tokenizer = None
        path = self.tokenizer_path or os.getenv("QWEN_TOKENIZER_PATH")
        if not path:
            return
        try:
            from tokenizers import Tokenizer  # type: ignore[import-untyped]

            self._tokenizer = Tokenizer.from_file(path)
        except Exception:
            self._tokenizer = None

    def estimate(self, text: str) -> int:
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text).ids)
            except Exception:
                pass
        word_like = re.findall(r"\w+|[^\w\s]", text)
        # Rough fallback: assume average 4 characters per token.
        fallback = max(len(text) // 4, len(word_like), 1)
        return fallback


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_into_sentences(paragraph: str) -> List[str]:
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    parts = _SENTENCE_SPLIT_RE.split(paragraph)
    sentences: List[str] = []
    for part in parts:
        cleaned = normalize_text(part)
        if cleaned:
            sentences.append(cleaned)
    if not sentences:
        sentences.append(paragraph.strip())
    return sentences


def iter_sentences_with_breaks(text: str) -> Iterator[str]:
    raw_parts = re.split(r"\n{2,}", text)
    last_index = len(raw_parts) - 1
    for idx, part in enumerate(raw_parts):
        sentences = split_into_sentences(part)
        for sentence in sentences:
            yield sentence
        if idx != last_index and sentences:
            yield PARAGRAPH_BREAK


def chunk_text(
    text: str,
    *,
    estimator: TokenEstimator,
    max_tokens: int,
    max_sentences: int,
) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    sentence_count = 0
    token_count = 0

    for segment in iter_sentences_with_breaks(text):
        if segment == PARAGRAPH_BREAK:
            if current and current[-1] != "":
                current.append("")
            continue

        segment_tokens = estimator.estimate(segment)

        if current and (
            sentence_count >= max_sentences or token_count + segment_tokens > max_tokens
        ):
            chunks.append("\n".join(s for s in current if s is not None).strip())
            current = []
            sentence_count = 0
            token_count = 0

        current.append(segment)
        sentence_count += 1
        token_count += segment_tokens

    if current:
        chunks.append("\n".join(s for s in current if s is not None).strip())

    return [chunk for chunk in chunks if chunk]


def is_garbage_translation(text: str) -> bool:
    cleaned = normalize_text(text)
    if not cleaned:
        return True
    if len(set(cleaned)) <= 3 and len(cleaned) >= 10:
        return True
    if _REPEAT_PATTERN.search(cleaned):
        return True
    if cleaned.upper() in {"N/A", "UNKNOWN"}:
        return True
    return False
