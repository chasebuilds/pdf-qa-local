"""Tokenizing and chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass

STOPWORDS = frozenset(
    "a an the of to in on at for and or is are was were be been by with as it its this that "
    "these those what which who whom how when where why does do did from has have had not can "
    "will would should may about into than then there their they them you your we our i".split()
)
_WORD = re.compile(r"[a-z0-9][a-z0-9\-\.]*[a-z0-9]|[a-z0-9]")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stopwords dropped, trailing plural 's' stripped."""
    out: list[str] = []
    for raw in _WORD.findall(text.lower()):
        tok = raw.rstrip(".-")
        if not tok or tok in STOPWORDS:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


@dataclass(frozen=True)
class Chunk:
    file: str
    page: int
    index: int
    text: str


def split_page(text: str, max_chars: int = 700, overlap: int = 100) -> list[str]:
    """Split one page into pieces of about max_chars, breaking on sentence ends."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    cur = ""
    for sent in sentences:
        while len(sent) > max_chars:  # one huge sentence: hard split
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(sent[:max_chars])
            sent = sent[max_chars - overlap :]
        if cur and len(cur) + 1 + len(sent) > max_chars:
            pieces.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = (tail.split(" ", 1)[-1] + " " + sent).strip() if tail else sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def chunk_pages(
    file: str, pages: list[str], max_chars: int = 700, overlap: int = 100
) -> list[Chunk]:
    """Chunk a document. Chunks never cross a page, so page references stay exact."""
    chunks: list[Chunk] = []
    for page_no, page_text in enumerate(pages, start=1):
        for piece in split_page(page_text, max_chars, overlap):
            chunks.append(Chunk(file, page_no, len(chunks), piece))
    return chunks
