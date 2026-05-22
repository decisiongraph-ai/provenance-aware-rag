"""Response generation with citations and confidence indicators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from provenance_rag.models import Citation, GeneratedResponse

if TYPE_CHECKING:
    from provenance_rag.ingestion import DocumentStore
    from provenance_rag.models import ConfidenceResult, ProvenanceRecord, RetrievalTrace


class ResponseGenerator:
    """Generate structured responses with inline citations."""

    def __init__(self, document_store: DocumentStore) -> None:
        self._store = document_store

    def generate(
        self,
        query: str,
        trace: RetrievalTrace,
        confidence: ConfidenceResult,
        provenance: ProvenanceRecord,
    ) -> GeneratedResponse:
        """Build a response from retrieval results with citations.

        For the MVP this constructs an extractive answer from top chunks
        rather than calling an LLM.
        """
        citations: list[Citation] = []
        answer_parts: list[str] = []

        for i, result in enumerate(trace.results):
            doc = self._store.get_document(result.chunk.document_id)
            title = doc.title if doc else "Unknown"

            citation = Citation(
                source_document_id=result.chunk.document_id,
                source_title=title,
                chunk_content=result.chunk.content,
                relevance_score=result.score,
            )
            citations.append(citation)

            answer_parts.append(f"[{i + 1}] {result.chunk.content}")

        if not answer_parts:
            answer = "No relevant information found for your query."
        else:
            answer = self._format_answer(query, answer_parts, confidence)

        return GeneratedResponse(
            query=query,
            answer=answer,
            citations=citations,
            confidence=confidence,
            provenance=provenance,
        )

    def _format_answer(
        self,
        query: str,
        parts: list[str],
        confidence: ConfidenceResult,
    ) -> str:
        lines: list[str] = []

        conf_label = _confidence_label(confidence.overall_confidence)
        lines.append(f"**Confidence: {conf_label} ({confidence.overall_confidence:.0%})**\n")

        if confidence.has_conflicts:
            lines.append("**Warning: Conflicting sources detected.**\n")
            for detail in confidence.conflict_details:
                lines.append(f"  - {detail}")
            lines.append("")

        lines.append("### Relevant excerpts\n")
        lines.extend(parts)

        return "\n".join(lines)


def _confidence_label(score: float) -> str:
    if score >= 0.8:
        return "High"
    if score >= 0.5:
        return "Medium"
    return "Low"
