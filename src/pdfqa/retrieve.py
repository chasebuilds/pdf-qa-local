"""Hybrid retrieval: BM25 keyword scores plus vector cosine, fused by reciprocal rank."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .embed import Embedder
from .text import Chunk, tokenize

RRF_K = 60


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # fused rank score
    cosine: float
    bm25: float
    coverage: float  # share of query terms that appear in the chunk
    phrase_coverage: float = 1.0  # share of adjacent query term pairs that appear in the chunk


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        vectors: npt.NDArray[np.float32],
        embedder: Embedder,
        min_coverage: float = 0.6,
        min_phrase: float = 0.5,
        min_cosine: float | None = None,
    ) -> None:
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder
        self.min_coverage = min_coverage
        self.min_phrase = min_phrase
        self.min_cosine = embedder.min_cosine if min_cosine is None else min_cosine
        self._tokens = [tokenize(c.text) for c in chunks]
        self._sets = [set(t) for t in self._tokens]
        self._pairs = [set(zip(t, t[1:], strict=False)) for t in self._tokens]
        self._tf = [Counter(t) for t in self._tokens]
        n = max(len(chunks), 1)
        df: Counter[str] = Counter()
        for s in self._sets:
            df.update(s)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self._avg_len = sum(len(t) for t in self._tokens) / n or 1.0

    def _bm25(self, q_terms: list[str]) -> npt.NDArray[np.float64]:
        k1, b = 1.5, 0.75
        scores = np.zeros(len(self.chunks))
        for i, tf in enumerate(self._tf):
            length = len(self._tokens[i])
            for t in q_terms:
                f = tf.get(t, 0)
                if f:
                    denom = f + k1 * (1 - b + b * length / self._avg_len)
                    scores[i] += self._idf[t] * f * (k1 + 1) / denom
        return scores

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if not self.chunks:
            return []
        q_ordered = tokenize(query)
        q_pairs = set(zip(q_ordered, q_ordered[1:], strict=False))
        q_terms = sorted(set(q_ordered))
        cos = (self.vectors @ self.embedder.embed_query(query)[0]).astype(np.float64)
        bm = self._bm25(q_terms)
        pool = min(max(k * 4, 20), len(self.chunks))
        vec_rank = np.argsort(-cos)[:pool]
        kw_rank = [i for i in np.argsort(-bm)[:pool] if bm[i] > 0]
        fused: dict[int, float] = {}
        for rank, i in enumerate(vec_rank):
            fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, i in enumerate(kw_rank):
            fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + rank + 1)
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        hits = []
        for i in top:
            cov = len(set(q_terms) & self._sets[i]) / len(q_terms) if q_terms else 0.0
            phrase = len(q_pairs & self._pairs[i]) / len(q_pairs) if q_pairs else cov
            hits.append(Hit(self.chunks[i], fused[i], float(cos[i]), float(bm[i]), cov, phrase))
        return hits

    def is_answerable(self, hits: list[Hit]) -> bool:
        """Weak unless a top hit matches the query words and word pairs, or is close by vector.

        Word pairs catch a name that is nearly, but not exactly, in the text.
        """
        return any(
            (h.coverage >= self.min_coverage and h.phrase_coverage >= self.min_phrase)
            or h.cosine >= self.min_cosine
            for h in hits[:3]
        )
