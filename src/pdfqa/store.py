"""PDF reading and the SQLite index (chunks plus float32 vectors as blobs)."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pypdf import PdfReader

from .embed import Embedder
from .text import Chunk, chunk_pages

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS docs (file TEXT PRIMARY KEY, sha256 TEXT, pages INTEGER);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY, file TEXT, page INTEGER, idx INTEGER, text TEXT, vec BLOB
);
CREATE INDEX IF NOT EXISTS chunks_file ON chunks(file);
"""


def read_pdf_pages(path: Path) -> list[str]:
    """Text of each page, in order. Pages with no text layer come back empty."""
    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]


@dataclass
class IndexStats:
    files_indexed: int = 0
    files_skipped: int = 0
    files_removed: int = 0
    chunks_added: int = 0
    empty_pages: int = 0


class Index:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = db_path
        self.db = sqlite3.connect(str(db_path))
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def _meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return str(row[0]) if row else None

    def _check_embedder(self, embedder: Embedder) -> None:
        have = self._meta("embedder")
        if have is None:
            self.db.execute("INSERT INTO meta VALUES ('embedder', ?)", (embedder.name,))
        elif have != embedder.name:
            raise ValueError(
                f"Index was built with {have}, not {embedder.name}. "
                "Delete the index file or pass the same embedder."
            )

    def index_folder(self, folder: Path, embedder: Embedder) -> IndexStats:
        self._check_embedder(embedder)
        stats = IndexStats()
        pdfs = sorted(folder.rglob("*.pdf"))
        seen: set[str] = set()
        for pdf in pdfs:
            name = str(pdf.relative_to(folder)).replace("\\", "/")
            seen.add(name)
            sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
            row = self.db.execute("SELECT sha256 FROM docs WHERE file=?", (name,)).fetchone()
            if row and row[0] == sha:
                stats.files_skipped += 1
                continue
            pages = read_pdf_pages(pdf)
            stats.empty_pages += sum(1 for p in pages if not p.strip())
            chunks = chunk_pages(name, pages)
            vecs = embedder.embed_documents([c.text for c in chunks])
            self.db.execute("DELETE FROM chunks WHERE file=?", (name,))
            self.db.executemany(
                "INSERT INTO chunks (file, page, idx, text, vec) VALUES (?,?,?,?,?)",
                [
                    (c.file, c.page, c.index, c.text, vecs[i].astype(np.float32).tobytes())
                    for i, c in enumerate(chunks)
                ],
            )
            self.db.execute("INSERT OR REPLACE INTO docs VALUES (?,?,?)", (name, sha, len(pages)))
            stats.files_indexed += 1
            stats.chunks_added += len(chunks)
        for (gone,) in self.db.execute("SELECT file FROM docs").fetchall():
            if gone not in seen:
                self.db.execute("DELETE FROM chunks WHERE file=?", (gone,))
                self.db.execute("DELETE FROM docs WHERE file=?", (gone,))
                stats.files_removed += 1
        self.db.commit()
        return stats

    def load(self) -> tuple[list[Chunk], npt.NDArray[np.float32]]:
        rows = self.db.execute(
            "SELECT file, page, idx, text, vec FROM chunks ORDER BY file, page, idx"
        ).fetchall()
        chunks = [Chunk(r[0], r[1], r[2], r[3]) for r in rows]
        if not rows:
            return chunks, np.zeros((0, 1), np.float32)
        mat = np.stack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
        return chunks, mat

    def embedder_name(self) -> str | None:
        return self._meta("embedder")
