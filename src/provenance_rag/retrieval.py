"""Retrieval engine with BM25 keyword search and TF-IDF vector similarity."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from provenance_rag.models import Chunk, RetrievalResult, RetrievalTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray


class RetrievalEngine:
    """Hybrid retrieval engine combining BM25 and TF-IDF vector similarity."""

    def __init__(self, bm25_weight: float = 0.5, tfidf_weight: float = 0.5) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._tfidf_vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix: NDArray[np.float64] | None = None
        self._bm25_weight = bm25_weight
        self._tfidf_weight = tfidf_weight

    def index(self, chunks: list[Chunk]) -> None:
        """Build search indices from a list of chunks."""
        if not chunks:
            return

        self._chunks = list(chunks)
        corpus = [c.content for c in self._chunks]
        tokenized = [doc.lower().split() for doc in corpus]

        self._bm25 = BM25Okapi(tokenized)

        self._tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus)

    def search(
        self,
        query: str,
        top_k: int = 5,
        methods: list[str] | None = None,
    ) -> RetrievalTrace:
        """Search for chunks matching the query using configured methods."""
        start_time = time.monotonic()

        if methods is None:
            methods = ["bm25", "tfidf"]

        if not self._chunks:
            return RetrievalTrace(
                query=query,
                methods_used=methods,
                duration_ms=0.0,
            )

        scores: dict[int, dict[str, float]] = {}

        if "bm25" in methods and self._bm25 is not None:
            bm25_scores = self._bm25.get_scores(query.lower().split())
            max_bm25 = float(np.max(bm25_scores)) if np.max(bm25_scores) > 0 else 1.0
            for idx, score in enumerate(bm25_scores):
                scores.setdefault(idx, {})
                scores[idx]["bm25"] = float(score) / max_bm25

        tfidf_ready = self._tfidf_vectorizer is not None and self._tfidf_matrix is not None
        if "tfidf" in methods and tfidf_ready:
            query_vec = self._tfidf_vectorizer.transform([query])
            sim_scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
            for idx, score in enumerate(sim_scores):
                scores.setdefault(idx, {})
                scores[idx]["tfidf"] = float(score)

        combined: list[tuple[int, float]] = []
        active_weight_sum = (
            (self._bm25_weight if "bm25" in methods else 0.0)
            + (self._tfidf_weight if "tfidf" in methods else 0.0)
        ) or 1.0
        for idx, method_scores in scores.items():
            bm25_s = method_scores.get("bm25", 0.0)
            tfidf_s = method_scores.get("tfidf", 0.0)
            raw = self._bm25_weight * bm25_s + self._tfidf_weight * tfidf_s
            final = raw / active_weight_sum
            combined.append((idx, final))

        combined.sort(key=lambda x: x[1], reverse=True)
        top_results = combined[:top_k]

        results = [
            RetrievalResult(
                chunk=self._chunks[idx],
                score=score,
                retrieval_method="+".join(methods),
            )
            for idx, score in top_results
            if score > 0
        ]

        elapsed = (time.monotonic() - start_time) * 1000

        return RetrievalTrace(
            query=query,
            results=results,
            methods_used=methods,
            total_candidates=len(self._chunks),
            duration_ms=round(elapsed, 2),
        )
