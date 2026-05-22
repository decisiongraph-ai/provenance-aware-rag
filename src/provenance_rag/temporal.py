"""Temporal awareness: freshness scoring and validity tracking."""

from __future__ import annotations

import math
from datetime import UTC, datetime

from provenance_rag.models import Document


class TemporalScorer:
    """Score documents based on temporal relevance."""

    def __init__(self, half_life_days: float = 180.0) -> None:
        self._half_life_days = half_life_days

    def freshness_score(self, document: Document, reference_time: datetime | None = None) -> float:
        """Compute a freshness score based on document age.

        Uses exponential decay: score = 2^(-age_days / half_life_days).
        """
        now = reference_time or datetime.now(UTC)
        age = now - document.updated_at
        age_days = max(age.total_seconds() / 86400, 0.0)
        return math.pow(2, -age_days / self._half_life_days)

    def is_valid(self, document: Document, reference_time: datetime | None = None) -> bool:
        """Check if a document is within its temporal validity window."""
        now = reference_time or datetime.now(UTC)
        if document.valid_from and now < document.valid_from:
            return False
        if document.valid_until and now > document.valid_until:
            return False
        return True

    def is_stale(
        self,
        document: Document,
        stale_threshold: float = 0.25,
        reference_time: datetime | None = None,
    ) -> bool:
        """Check whether a document's freshness is below the staleness threshold."""
        return self.freshness_score(document, reference_time) < stale_threshold

    def filter_valid(
        self,
        documents: list[Document],
        reference_time: datetime | None = None,
    ) -> list[Document]:
        """Return only documents within their validity window."""
        return [d for d in documents if self.is_valid(d, reference_time)]
