from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

from bs4 import BeautifulSoup  # type: ignore
from ebooklib import epub  # type: ignore


@dataclass
class Chapter:
    index: int
    title: str
    text: str
    relpath: Optional[str] = None


@dataclass
class BookMeta:
    title: str
    author: Optional[str]
    language: Optional[str]
    uid: Optional[str]


def _html_to_text(html: bytes | str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # remove scripts/styles
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n")
    # normalize whitespace
    lines = [" ".join(line.split()) for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return text


def load_epub(path: str | Path) -> Tuple[BookMeta, List[Chapter]]:
    book = epub.read_epub(str(path))
    title = (book.get_metadata("DC", "title") or [("Unknown", {})])[0][0]
    authors = book.get_metadata("DC", "creator")
    author = authors[0][0] if authors else None
    language = (book.get_metadata("DC", "language") or [(None, {})])[0][0]
    uid = book.uid

    chapters: List[Chapter] = []
    i = 0
    for item in book.get_items_of_type(epub.ITEM_DOCUMENT):
        # Use the file name as a fallback title
        doc_title = Path(item.get_name()).stem
        text = _html_to_text(item.get_content())
        if not text.strip():
            continue
        chapters.append(Chapter(index=i, title=doc_title, text=text, relpath=item.get_name()))
        i += 1

    meta = BookMeta(title=title, author=author, language=language, uid=uid)
    return meta, chapters
