"""Tests for the retrieval engine."""

from provenance_rag.ingestion import DocumentStore
from provenance_rag.models import DocumentType, SourceReliability
from provenance_rag.retrieval import RetrievalEngine


def _build_engine() -> tuple[DocumentStore, RetrievalEngine]:
    store = DocumentStore()
    store.ingest(
        title="Python Guide",
        content=(
            "Python is a versatile programming language used for web development, "
            "data science, machine learning, and automation. It has a rich ecosystem "
            "of libraries including NumPy, pandas, and scikit-learn."
        ),
        source_name="DevDocs",
        reliability=SourceReliability.HIGH,
        document_type=DocumentType.TECHNICAL,
    )
    store.ingest(
        title="Security Policy",
        content=(
            "All systems must use TLS 1.2 or higher. Passwords must be at least "
            "12 characters. Multi-factor authentication is mandatory for all admin "
            "accounts. Regular security audits must be performed quarterly."
        ),
        source_name="InfoSec",
        reliability=SourceReliability.HIGH,
        document_type=DocumentType.POLICY,
    )
    store.ingest(
        title="Cooking Guide",
        content=(
            "To make pasta, boil water and add salt. Cook the pasta for 8-10 minutes. "
            "Drain and serve with your favorite sauce. Italian cuisine is known for "
            "its simplicity and fresh ingredients."
        ),
        source_name="FoodBlog",
        reliability=SourceReliability.LOW,
        document_type=DocumentType.GENERAL,
    )

    eng = RetrievalEngine()
    eng.index(store.get_all_chunks())
    return store, eng


class TestRetrievalEngine:
    def test_search_returns_results(self) -> None:
        _, eng = _build_engine()
        trace = eng.search("Python programming")
        assert len(trace.results) > 0
        assert trace.query == "Python programming"

    def test_search_relevance(self) -> None:
        _, eng = _build_engine()
        trace = eng.search("security password TLS")
        top_chunk = trace.results[0].chunk
        assert "TLS" in top_chunk.content or "password" in top_chunk.content.lower()

    def test_search_top_k(self) -> None:
        _, eng = _build_engine()
        trace = eng.search("python", top_k=1)
        assert len(trace.results) <= 1

    def test_search_bm25_only(self) -> None:
        _, eng = _build_engine()
        trace = eng.search("pasta cooking", methods=["bm25"])
        assert trace.methods_used == ["bm25"]
        assert len(trace.results) > 0

    def test_search_tfidf_only(self) -> None:
        _, eng = _build_engine()
        trace = eng.search("machine learning", methods=["tfidf"])
        assert trace.methods_used == ["tfidf"]
        assert len(trace.results) > 0

    def test_search_empty_index(self) -> None:
        eng = RetrievalEngine()
        trace = eng.search("anything")
        assert len(trace.results) == 0

    def test_trace_has_duration(self) -> None:
        _, eng = _build_engine()
        trace = eng.search("security")
        assert trace.duration_ms >= 0
