"""Confidence scoring, uncertainty detection, and conflict detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from provenance_rag.models import ConfidenceResult, SourceReliability

if TYPE_CHECKING:
    from provenance_rag.ingestion import DocumentStore
    from provenance_rag.models import RetrievalTrace
    from provenance_rag.temporal import TemporalScorer


_RELIABILITY_SCORES: dict[SourceReliability, float] = {
    SourceReliability.HIGH: 1.0,
    SourceReliability.MEDIUM: 0.7,
    SourceReliability.LOW: 0.3,
    SourceReliability.UNKNOWN: 0.5,
}


class ConfidenceScorer:
    """Score confidence of retrieval results."""

    def __init__(
        self,
        document_store: DocumentStore,
        temporal_scorer: TemporalScorer | None = None,
    ) -> None:
        self._store = document_store
        self._temporal = temporal_scorer

    def score(self, trace: RetrievalTrace) -> ConfidenceResult:
        """Compute confidence for a set of retrieval results."""
        if not trace.results:
            return ConfidenceResult(
                overall_confidence=0.0,
                source_reliability_score=0.0,
                freshness_score=0.0,
                agreement_score=0.0,
            )

        reliability_score = self._compute_reliability(trace)
        freshness_score = self._compute_freshness(trace)
        agreement_score, has_conflicts, conflict_details = self._compute_agreement(trace)

        overall = 0.4 * reliability_score + 0.3 * freshness_score + 0.3 * agreement_score

        return ConfidenceResult(
            overall_confidence=round(overall, 4),
            source_reliability_score=round(reliability_score, 4),
            freshness_score=round(freshness_score, 4),
            agreement_score=round(agreement_score, 4),
            has_conflicts=has_conflicts,
            conflict_details=conflict_details,
        )

    def _compute_reliability(self, trace: RetrievalTrace) -> float:
        scores: list[float] = []
        for result in trace.results:
            doc = self._store.get_document(result.chunk.document_id)
            if doc:
                scores.append(_RELIABILITY_SCORES.get(doc.source.reliability, 0.5))
            else:
                scores.append(0.5)
        return sum(scores) / len(scores) if scores else 0.0

    def _compute_freshness(self, trace: RetrievalTrace) -> float:
        if self._temporal is None:
            return 0.5

        scores: list[float] = []
        for result in trace.results:
            doc = self._store.get_document(result.chunk.document_id)
            if doc:
                scores.append(self._temporal.freshness_score(doc))
        return sum(scores) / len(scores) if scores else 0.5

    def _compute_agreement(
        self,
        trace: RetrievalTrace,
    ) -> tuple[float, bool, list[str]]:
        """Detect conflicts by comparing source documents.

        Simple heuristic: if retrieval results come from multiple sources and
        scores vary widely, flag potential conflicts.
        """
        if len(trace.results) < 2:
            return 1.0, False, []

        source_scores: dict[str, list[float]] = {}
        for result in trace.results:
            doc = self._store.get_document(result.chunk.document_id)
            source_name = doc.source.name if doc else "unknown"
            source_scores.setdefault(source_name, []).append(result.score)

        if len(source_scores) < 2:
            return 1.0, False, []

        avg_scores = {src: sum(s) / len(s) for src, s in source_scores.items()}
        values = list(avg_scores.values())
        score_range = max(values) - min(values)

        has_conflicts = score_range > 0.5
        conflicts: list[str] = []
        if has_conflicts:
            sorted_sources = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            top = sorted_sources[0]
            bottom = sorted_sources[-1]
            conflicts.append(
                f"Source '{top[0]}' (avg score {top[1]:.2f}) conflicts with "
                f"'{bottom[0]}' (avg score {bottom[1]:.2f})"
            )

        agreement = 1.0 - min(score_range, 1.0)
        return round(agreement, 4), has_conflicts, conflicts
