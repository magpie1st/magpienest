from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, List

from audiobook.config import CHUNK_MAX_CHARS, PAUSE_MS_DEFAULT
from audiobook.core.chunker import chunk_paragraphs
from audiobook.core.epub_reader import Chapter, BookMeta
from audiobook.core.tts_engine import XTTSEngine
from audiobook.core.audio import merge_wavs
from audiobook.core import cache
from audiobook.core import naming


def synthesize_chapter(plan, chapter: Chapter, engine: XTTSEngine) -> Path:
    # simple cache key: book+chapter+hash(text)
    key_str = f"{plan.book.uid}|{chapter.index}|{engine.model_name}|{engine.language}|{bool(engine.speaker_wav)}|{len(chapter.text)}"
    key = cache.make_key(key_str)
    hit = cache.get(plan.out_dir, key)
    if hit:
        return hit
    parts: List[Path] = []
    try:
        chunks = chunk_paragraphs(chapter.text, max_chars=CHUNK_MAX_CHARS)
        for line in chunks:
            wav_path, _ = engine.synthesize_line(line, plan.book.language or engine.language)
            parts.append(wav_path)
        book_path = naming.book_dir(plan.out_dir, plan.book.author, plan.book.title)
        book_path.mkdir(parents=True, exist_ok=True)
        out = book_path / naming.chapter_filename(chapter.index, chapter.title, plan.format)
        merged = merge_wavs(parts, out, pause_ms=plan.pause_ms, out_format=plan.format)
        cache.put(plan.out_dir, key, merged, ext=plan.format)
        return merged
    finally:
        for p in parts:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


def synthesize_book(plan, on_progress: Callable[[int,int,int,int], None], cancel_flag: threading.Event) -> Path:
    engine = XTTSEngine(speaker_wav=str(getattr(plan, 'speaker_wav', None) or ''))
    outputs: List[Path] = []
    total_ch = len(plan.chapters)
    for ci, ch in enumerate(plan.chapters, start=1):
        if cancel_flag.is_set():
            break
        on_progress(ci, total_ch, 0, 0)
        out = synthesize_chapter(plan, ch, engine)
        outputs.append(out)
        on_progress(ci, total_ch, 1, 1)
    # merge into full
    full_dir = naming.book_dir(plan.out_dir, plan.book.author, plan.book.title)
    full_path = full_dir / (naming.slugify(plan.book.title) + "__full." + plan.format)
    return merge_wavs([Path(p) for p in outputs], full_path, pause_ms=plan.pause_ms, out_format=plan.format)
