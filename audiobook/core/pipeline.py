from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, List, Optional

from audiobook.config import CHUNK_MAX_CHARS
from audiobook.core.chunker import chunk_paragraphs
from audiobook.core.epub_reader import Chapter
from audiobook.core.tts_engine import XTTSEngine
from audiobook.core.audio import merge_wavs
from audiobook.core import cache
from audiobook.core import naming


LogFn = Optional[Callable[[str], None]]


def _log(log: LogFn, message: str) -> None:
    if log:
        log(message)


def synthesize_chapter(plan, chapter: Chapter, engine: XTTSEngine, *, log: LogFn = None,
                       on_chunk_progress: Optional[Callable[[int, int], None]] = None) -> Path:
    key_str = f"{plan.book.uid}|{chapter.index}|{engine.model_name}|{engine.language}|{bool(engine.speaker_wav)}|{len(chapter.text)}"
    key = cache.make_key(key_str)
    hit = cache.get(plan.out_dir, key)
    if hit:
        _log(log, f"Cache hit for chapter {chapter.index}: {chapter.title}")
        if on_chunk_progress:
            on_chunk_progress(1, 1)
        return hit

    parts: List[Path] = []
    chunks = chunk_paragraphs(chapter.text, max_chars=CHUNK_MAX_CHARS)
    total_chunks = max(len(chunks), 1)
    _log(log, f"Synthesizing chapter {chapter.index}: {chapter.title} ({total_chunks} chunks)")
    try:
        for idx, line in enumerate(chunks, start=1):
            _log(log, f"  Chunk {idx}/{total_chunks}…")
            wav_path, _ = engine.synthesize_line(line, plan.book.language or engine.language)
            parts.append(wav_path)
            if on_chunk_progress:
                on_chunk_progress(idx, total_chunks)
        book_path = naming.book_dir(plan.out_dir, plan.book.author, plan.book.title)
        book_path.mkdir(parents=True, exist_ok=True)
        out = book_path / naming.chapter_filename(chapter.index, chapter.title, plan.format)
        merged = merge_wavs(parts, out, pause_ms=plan.pause_ms, out_format=plan.format)
        cache.put(plan.out_dir, key, merged, ext=plan.format)
        _log(log, f"Chapter {chapter.index} complete → {merged}")
        return merged
    finally:
        for p in parts:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


def synthesize_book(plan, on_progress: Callable[[int, int, int, int], None], cancel_flag: threading.Event,
                    log: LogFn = None) -> Path:
    engine = XTTSEngine(speaker_wav=str(getattr(plan, 'speaker_wav', None) or ''))
    outputs: List[Path] = []
    total_ch = len(plan.chapters)
    for ci, ch in enumerate(plan.chapters, start=1):
        if cancel_flag.is_set():
            _log(log, "Cancellation requested; stopping synthesis")
            break
        _log(log, f"Starting chapter {ci}/{total_ch}: {ch.title}")
        on_progress(ci, total_ch, 0, 1)

        def chunk_progress(li: int, lt: int) -> None:
            on_progress(ci, total_ch, li, lt)

        out = synthesize_chapter(plan, ch, engine, log=log, on_chunk_progress=chunk_progress)
        outputs.append(out)
        on_progress(ci, total_ch, 1, 1)
    full_dir = naming.book_dir(plan.out_dir, plan.book.author, plan.book.title)
    full_path = full_dir / (naming.slugify(plan.book.title) + "__full." + plan.format)
    _log(log, "Merging chapters into full audiobook")
    merged = merge_wavs([Path(p) for p in outputs], full_path, pause_ms=plan.pause_ms, out_format=plan.format)
    _log(log, f"Full audiobook ready → {merged}")
    return merged
