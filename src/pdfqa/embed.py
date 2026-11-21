"""Embedders: a deterministic stub for tests and a local Ollama client."""

from __future__ import annotations

import hashlib
import os
from typing import Protocol
from urllib.parse import urlparse

import httpx
import numpy as np
import numpy.typing as npt

from .text import tokenize

Vectors = npt.NDArray[np.float32]
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


def ensure_local(url: str) -> str:
    """Refuse non-loopback URLs, so documents cannot be sent off the machine by mistake."""
    host = urlparse(url).hostname or ""
    if host not in LOCAL_HOSTS and os.environ.get("PDFQA_ALLOW_REMOTE") != "1":
        raise ValueError(f"{url} is not a local address. Set PDFQA_ALLOW_REMOTE=1 to override.")
    return url.rstrip("/")


class Embedder(Protocol):
    name: str
    min_cosine: float

    def embed_documents(self, texts: list[str]) -> Vectors: ...

    def embed_query(self, text: str) -> Vectors: ...


def _normalize(m: Vectors) -> Vectors:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out: Vectors = (m / norms).astype(np.float32)
    return out


class HashEmbedder:
    """Hashed bag of words. No model, no network. Used by tests and the stub benchmark."""

    min_cosine = 0.45

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def _one(self, text: str) -> Vectors:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0
        return v

    def embed_documents(self, texts: list[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return _normalize(np.stack([self._one(t) for t in texts]))

    def embed_query(self, text: str) -> Vectors:
        return _normalize(self._one(text)[None, :])


class OllamaEmbedder:
    """Embeddings from a local Ollama server (POST /api/embed)."""

    min_cosine = 0.62

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or os.environ.get("PDFQA_EMBED_MODEL", "nomic-embed-text")
        url = base_url or os.environ.get("PDFQA_OLLAMA_URL", "http://localhost:11434")
        self.base = ensure_local(url)
        self.name = f"ollama:{self.model}"
        self._nomic = "nomic" in self.model

    def _embed(self, texts: list[str]) -> Vectors:
        out: list[list[float]] = []
        for i in range(0, len(texts), 16):
            r = httpx.post(
                f"{self.base}/api/embed",
                json={"model": self.model, "input": texts[i : i + 16]},
                timeout=120,
            )
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
        return _normalize(np.asarray(out, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> Vectors:
        if not texts:
            return np.zeros((0, 1), np.float32)
        pre = "search_document: " if self._nomic else ""
        return self._embed([pre + t for t in texts])

    def embed_query(self, text: str) -> Vectors:
        pre = "search_query: " if self._nomic else ""
        return self._embed([pre + text])
