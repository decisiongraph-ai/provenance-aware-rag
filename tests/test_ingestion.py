"""Tests for document ingestion and chunking."""

from provenance_rag.ingestion import DocumentStore
from provenance_rag.models import DocumentType, SourceReliability


def _make_store_with_doc() -> tuple:
    store = DocumentStore()
    doc = store.ingest(
        title="Security Policy",
        content="All systems must use encryption. " * 50,
        source_name="InfoSec Team",
        author="Alice",
        document_type=DocumentType.POLICY,
        reliability=SourceReliability.HIGH,
    )
    return store, doc


class TestDocumentStore:
    def test_ingest_creates_document(self) -> None:
        store, doc = _make_store_with_doc()
        assert store.document_count == 1
        assert doc.title == "Security Policy"
        assert doc.source.name == "InfoSec Team"
        assert doc.source.reliability == SourceReliability.HIGH

    def test_ingest_creates_chunks(self) -> None:
        store, doc = _make_store_with_doc()
        chunks = store.get_chunks(doc.document_id)
        assert len(chunks) > 0
        assert all(c.document_id == doc.document_id for c in chunks)

    def test_get_document_returns_none_for_missing(self) -> None:
        store = DocumentStore()
        assert store.get_document("nonexistent") is None

    def test_get_all_chunks(self) -> None:
        store, _ = _make_store_with_doc()
        store.ingest(
            title="Second Doc",
            content="Another document with some content. " * 30,
            source_name="Team B",
        )
        assert store.document_count == 2
        assert store.chunk_count > 0
        all_chunks = store.get_all_chunks()
        assert len(all_chunks) == store.chunk_count

    def test_ingest_with_custom_chunk_size(self) -> None:
        store = DocumentStore()
        doc = store.ingest(
            title="Small Chunks",
            content="Word " * 200,
            source_name="Test",
            chunk_size=100,
            chunk_overlap=10,
        )
        chunks = store.get_chunks(doc.document_id)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.content) <= 110  # small margin for boundary logic
