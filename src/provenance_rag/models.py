"""Pydantic models for the provenance-aware RAG system."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    POLICY = "policy"
    TECHNICAL = "technical"
    LEGAL = "legal"
    GENERAL = "general"
    INTERNAL = "internal"


class SourceReliability(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class Source(BaseModel):
    """Metadata about a document source."""

    source_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    author: str = ""
    url: str = ""
    reliability: SourceReliability = SourceReliability.UNKNOWN
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Document(BaseModel):
    """A document with full metadata for provenance tracking."""

    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    source: Source
    document_type: DocumentType = DocumentType.GENERAL
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A chunk of text derived from a document."""

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """A single retrieval result with scoring."""

    chunk: Chunk
    score: float
    retrieval_method: str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RetrievalTrace(BaseModel):
    """Full trace of a retrieval operation for audit purposes."""

    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    results: list[RetrievalResult] = Field(default_factory=list)
    methods_used: list[str] = Field(default_factory=list)
    total_candidates: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_ms: float = 0.0


class ProvenanceRecord(BaseModel):
    """Provenance information linking a response to its sources."""

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    retrieval_trace: RetrievalTrace
    source_documents: list[str] = Field(default_factory=list)
    source_attributions: dict[str, float] = Field(default_factory=dict)
    confidence_score: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConfidenceResult(BaseModel):
    """Confidence assessment for a set of retrieval results."""

    overall_confidence: float = Field(ge=0.0, le=1.0)
    source_reliability_score: float = Field(ge=0.0, le=1.0)
    freshness_score: float = Field(ge=0.0, le=1.0)
    agreement_score: float = Field(ge=0.0, le=1.0)
    has_conflicts: bool = False
    conflict_details: list[str] = Field(default_factory=list)


class GeneratedResponse(BaseModel):
    """A generated response with citations and provenance."""

    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: ConfidenceResult
    provenance: ProvenanceRecord
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Citation(BaseModel):
    """A citation linking part of a response to a source."""

    citation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_document_id: str
    source_title: str
    chunk_content: str
    relevance_score: float = 0.0


class AuditEntry(BaseModel):
    """An entry in the audit log."""

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    query: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


# Rebuild models that have forward references
GeneratedResponse.model_rebuild()
