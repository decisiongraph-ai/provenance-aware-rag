"""FastAPI REST API for provenance-aware RAG."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from provenance_rag.audit import AuditLogger
from provenance_rag.confidence import ConfidenceScorer
from provenance_rag.generation import ResponseGenerator
from provenance_rag.ingestion import DocumentStore
from provenance_rag.models import DocumentType, SourceReliability
from provenance_rag.provenance import ProvenanceTracker
from provenance_rag.retrieval import RetrievalEngine
from provenance_rag.temporal import TemporalScorer

app = FastAPI(
    title="Provenance-Aware RAG",
    description=(
        "Trustworthy enterprise RAG with source lineage, auditability, and confidence scoring"
    ),
    version="0.1.0",
)

# Application state — initialised at module level for the MVP.
store = DocumentStore()
engine = RetrievalEngine()
temporal = TemporalScorer()
confidence_scorer = ConfidenceScorer(store, temporal)
provenance_tracker = ProvenanceTracker(store)
generator = ResponseGenerator(store)
audit = AuditLogger()


# ---------- Request / Response schemas ----------


class IngestRequest(BaseModel):
    title: str
    content: str
    source_name: str
    author: str = ""
    url: str = ""
    document_type: str = "general"
    reliability: str = "unknown"
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)


class IngestResponse(BaseModel):
    document_id: str
    chunk_count: int


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, gt=0, le=50)
    methods: list[str] = Field(default_factory=lambda: ["bm25", "tfidf"])


class QueryResponse(BaseModel):
    response_id: str
    query: str
    answer: str
    confidence: dict[str, Any]
    citations: list[dict[str, Any]]
    provenance: dict[str, Any]


# ---------- Endpoints ----------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def ingest_document(req: IngestRequest) -> IngestResponse:
    doc_type = DocumentType(req.document_type)
    reliability = SourceReliability(req.reliability)

    doc = store.ingest(
        title=req.title,
        content=req.content,
        source_name=req.source_name,
        author=req.author,
        url=req.url,
        document_type=doc_type,
        reliability=reliability,
        valid_from=req.valid_from,
        valid_until=req.valid_until,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
    )

    # Re-index all chunks
    engine.index(store.get_all_chunks())

    audit.log("ingest", details={"document_id": doc.document_id, "title": req.title})

    chunks = store.get_chunks(doc.document_id)
    return IngestResponse(document_id=doc.document_id, chunk_count=len(chunks))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if store.document_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents ingested yet",
        )

    trace = engine.search(req.query, top_k=req.top_k, methods=req.methods)
    conf = confidence_scorer.score(trace)
    prov = provenance_tracker.track(req.query, trace, conf)
    resp = generator.generate(req.query, trace, conf, prov)

    audit.log(
        "query",
        query=req.query,
        details={
            "response_id": resp.response_id,
            "confidence": conf.overall_confidence,
            "result_count": len(trace.results),
        },
    )

    return QueryResponse(
        response_id=resp.response_id,
        query=resp.query,
        answer=resp.answer,
        confidence=conf.model_dump(),
        citations=[c.model_dump() for c in resp.citations],
        provenance=prov.model_dump(),
    )


@app.get("/documents")
def list_documents() -> list[dict[str, Any]]:
    docs = store.get_all_documents()
    return [
        {
            "document_id": d.document_id,
            "title": d.title,
            "source": d.source.name,
            "type": d.document_type.value,
            "created_at": d.created_at.isoformat(),
        }
        for d in docs
    ]


@app.get("/documents/{document_id}")
def get_document(document_id: str) -> dict[str, Any]:
    doc = store.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc.model_dump()


@app.get("/audit")
def get_audit_log(event_type: str | None = None) -> list[dict[str, Any]]:
    entries = audit.get_entries(event_type=event_type)
    return [e.model_dump() for e in entries]


@app.get("/provenance")
def get_provenance_records() -> list[dict[str, Any]]:
    records = provenance_tracker.get_records()
    return [r.model_dump() for r in records]


@app.get("/provenance/{record_id}")
def get_provenance_record(record_id: str) -> dict[str, Any]:
    record = provenance_tracker.get_record(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Provenance record not found")
    return record.model_dump()
