"""Tests for the FastAPI REST API."""

import pytest
from fastapi.testclient import TestClient

from provenance_rag.api import app, audit, engine, provenance_tracker, store


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset application state between tests."""
    store._documents.clear()
    store._chunks.clear()
    engine._chunks.clear()
    engine._bm25 = None
    engine._tfidf_vectorizer = None
    engine._tfidf_matrix = None
    audit._entries.clear()
    provenance_tracker._records.clear()
    yield


@pytest.fixture()
def client():
    return TestClient(app)


SAMPLE_DOC = {
    "title": "Test Document",
    "content": (
        "Machine learning is a subset of artificial intelligence. "
        "It enables systems to learn from data and improve over time. "
        "Common algorithms include decision trees, random forests, and neural networks."
    ),
    "source_name": "AI Team",
    "author": "Test Author",
    "document_type": "technical",
    "reliability": "high",
}


class TestHealthEndpoint:
    def test_health(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestIngestEndpoint:
    def test_ingest_success(self, client: TestClient) -> None:
        resp = client.post("/ingest", json=SAMPLE_DOC)
        assert resp.status_code == 201
        data = resp.json()
        assert "document_id" in data
        assert data["chunk_count"] > 0

    def test_ingest_minimal(self, client: TestClient) -> None:
        resp = client.post(
            "/ingest",
            json={"title": "Min", "content": "Hello world", "source_name": "Test"},
        )
        assert resp.status_code == 201


class TestQueryEndpoint:
    def test_query_no_documents(self, client: TestClient) -> None:
        resp = client.post("/query", json={"query": "test"})
        assert resp.status_code == 400

    def test_query_success(self, client: TestClient) -> None:
        client.post("/ingest", json=SAMPLE_DOC)
        resp = client.post("/query", json={"query": "machine learning algorithms"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "confidence" in data
        assert "citations" in data
        assert "provenance" in data
        assert data["confidence"]["overall_confidence"] >= 0

    def test_query_with_top_k(self, client: TestClient) -> None:
        client.post("/ingest", json=SAMPLE_DOC)
        resp = client.post(
            "/query",
            json={"query": "neural networks", "top_k": 1},
        )
        assert resp.status_code == 200
        assert len(resp.json()["citations"]) <= 1


class TestDocumentsEndpoint:
    def test_list_documents_empty(self, client: TestClient) -> None:
        resp = client.get("/documents")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_documents(self, client: TestClient) -> None:
        client.post("/ingest", json=SAMPLE_DOC)
        resp = client.get("/documents")
        assert resp.status_code == 200
        docs = resp.json()
        assert len(docs) == 1
        assert docs[0]["title"] == "Test Document"

    def test_get_document_not_found(self, client: TestClient) -> None:
        resp = client.get("/documents/nonexistent")
        assert resp.status_code == 404


class TestAuditEndpoint:
    def test_audit_log_empty(self, client: TestClient) -> None:
        resp = client.get("/audit")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_audit_log_after_ingest(self, client: TestClient) -> None:
        client.post("/ingest", json=SAMPLE_DOC)
        resp = client.get("/audit")
        entries = resp.json()
        assert len(entries) == 1
        assert entries[0]["event_type"] == "ingest"

    def test_audit_log_filter(self, client: TestClient) -> None:
        client.post("/ingest", json=SAMPLE_DOC)
        client.post("/query", json={"query": "test"})
        resp = client.get("/audit?event_type=query")
        entries = resp.json()
        assert all(e["event_type"] == "query" for e in entries)


class TestProvenanceEndpoints:
    def test_provenance_list_empty(self, client: TestClient) -> None:
        resp = client.get("/provenance")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_provenance_after_query(self, client: TestClient) -> None:
        client.post("/ingest", json=SAMPLE_DOC)
        client.post("/query", json={"query": "machine learning"})
        resp = client.get("/provenance")
        records = resp.json()
        assert len(records) == 1
        assert records[0]["query"] == "machine learning"

    def test_provenance_record_not_found(self, client: TestClient) -> None:
        resp = client.get("/provenance/nonexistent")
        assert resp.status_code == 404
