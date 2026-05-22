"""Tests for confidence scoring and temporal awareness."""

from datetime import UTC, datetime, timedelta

from provenance_rag.confidence import ConfidenceScorer
from provenance_rag.ingestion import DocumentStore
from provenance_rag.models import DocumentType, SourceReliability
from provenance_rag.retrieval import RetrievalEngine
from provenance_rag.temporal import TemporalScorer


def _setup() -> tuple[DocumentStore, RetrievalEngine, ConfidenceScorer, TemporalScorer]:
    store = DocumentStore()
    store.ingest(
        title="High Reliability Doc",
        content="Enterprise security requires encryption at rest and in transit. " * 10,
        source_name="Security Team",
        reliability=SourceReliability.HIGH,
        document_type=DocumentType.POLICY,
    )
    store.ingest(
        title="Low Reliability Doc",
        content="Some random tips about security and passwords. " * 10,
        source_name="Blog Post",
        reliability=SourceReliability.LOW,
        document_type=DocumentType.GENERAL,
    )

    engine = RetrievalEngine()
    engine.index(store.get_all_chunks())
    temporal = TemporalScorer()
    scorer = ConfidenceScorer(store, temporal)
    return store, engine, scorer, temporal


class TestConfidenceScorer:
    def test_score_returns_valid_range(self) -> None:
        _, engine, scorer, _ = _setup()
        trace = engine.search("security encryption")
        conf = scorer.score(trace)
        assert 0.0 <= conf.overall_confidence <= 1.0
        assert 0.0 <= conf.source_reliability_score <= 1.0
        assert 0.0 <= conf.freshness_score <= 1.0
        assert 0.0 <= conf.agreement_score <= 1.0

    def test_empty_results_zero_confidence(self) -> None:
        store = DocumentStore()
        scorer = ConfidenceScorer(store)
        engine = RetrievalEngine()
        trace = engine.search("anything")
        conf = scorer.score(trace)
        assert conf.overall_confidence == 0.0

    def test_high_reliability_boosts_score(self) -> None:
        store, engine, _, temporal = _setup()
        trace = engine.search("encryption at rest")
        scorer_with_temporal = ConfidenceScorer(store, temporal)
        conf = scorer_with_temporal.score(trace)
        assert conf.source_reliability_score > 0.5


class TestTemporalScorer:
    def test_fresh_document_high_score(self) -> None:
        store = DocumentStore()
        doc = store.ingest(title="Fresh", content="New content", source_name="Test")
        temporal = TemporalScorer()
        score = temporal.freshness_score(doc)
        assert score > 0.9

    def test_old_document_low_score(self) -> None:
        store = DocumentStore()
        doc = store.ingest(title="Old", content="Old content", source_name="Test")
        temporal = TemporalScorer(half_life_days=30)
        future = datetime.now(UTC) + timedelta(days=365)
        score = temporal.freshness_score(doc, reference_time=future)
        assert score < 0.1

    def test_validity_window(self) -> None:
        store = DocumentStore()
        now = datetime.now(UTC)
        doc = store.ingest(
            title="Temporal",
            content="Content",
            source_name="Test",
            valid_from=now - timedelta(days=10),
            valid_until=now + timedelta(days=10),
        )
        temporal = TemporalScorer()
        assert temporal.is_valid(doc) is True

    def test_expired_document(self) -> None:
        store = DocumentStore()
        now = datetime.now(UTC)
        doc = store.ingest(
            title="Expired",
            content="Content",
            source_name="Test",
            valid_from=now - timedelta(days=100),
            valid_until=now - timedelta(days=1),
        )
        temporal = TemporalScorer()
        assert temporal.is_valid(doc) is False

    def test_stale_detection(self) -> None:
        store = DocumentStore()
        doc = store.ingest(title="Doc", content="Content", source_name="Test")
        temporal = TemporalScorer(half_life_days=30)
        future = datetime.now(UTC) + timedelta(days=365)
        assert temporal.is_stale(doc, reference_time=future) is True

    def test_filter_valid(self) -> None:
        store = DocumentStore()
        now = datetime.now(UTC)
        doc1 = store.ingest(
            title="Valid",
            content="A",
            source_name="T",
            valid_from=now - timedelta(days=5),
            valid_until=now + timedelta(days=5),
        )
        doc2 = store.ingest(
            title="Expired",
            content="B",
            source_name="T",
            valid_from=now - timedelta(days=20),
            valid_until=now - timedelta(days=1),
        )
        temporal = TemporalScorer()
        valid = temporal.filter_valid([doc1, doc2])
        assert len(valid) == 1
        assert valid[0].document_id == doc1.document_id
