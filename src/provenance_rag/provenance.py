"""Provenance tracking: source attribution and lineage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from provenance_rag.models import ProvenanceRecord

if TYPE_CHECKING:
    from provenance_rag.ingestion import DocumentStore
    from provenance_rag.models import ConfidenceResult, RetrievalTrace


class ProvenanceTracker:
    """Tracks provenance for retrieval operations."""

    def __init__(self, document_store: DocumentStore) -> None:
        self._store = document_store
        self._records: list[ProvenanceRecord] = []

    def track(
        self,
        query: str,
        trace: RetrievalTrace,
        confidence: ConfidenceResult,
    ) -> ProvenanceRecord:
        """Create a provenance record from a retrieval trace."""
        doc_ids: list[str] = []
        attributions: dict[str, float] = {}

        for result in trace.results:
            doc_id = result.chunk.document_id
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)

            doc = self._store.get_document(doc_id)
            source_name = doc.source.name if doc else "unknown"
            attributions[source_name] = max(
                attributions.get(source_name, 0.0),
                result.score,
            )

        record = ProvenanceRecord(
            query=query,
            retrieval_trace=trace,
            source_documents=doc_ids,
            source_attributions=attributions,
            confidence_score=confidence.overall_confidence,
        )
        self._records.append(record)
        return record

    def get_records(self) -> list[ProvenanceRecord]:
        return list(self._records)

    def get_record(self, record_id: str) -> ProvenanceRecord | None:
        for r in self._records:
            if r.record_id == record_id:
                return r
        return None
