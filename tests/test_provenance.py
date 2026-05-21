"""Tests for provenance tracking."""

from provenance_rag.confidence import ConfidenceScorer
from provenance_rag.ingestion import DocumentStore
from provenance_rag.models import SourceReliability
from provenance_rag.provenance import ProvenanceTracker
from provenance_rag.retrieval import RetrievalEngine
from provenance_rag.temporal import TemporalScorer


def _setup() -> tuple[DocumentStore, RetrievalEngine, ProvenanceTracker, ConfidenceScorer]:
    store = DocumentStore()
    store.ingest(
        title="Architecture Guide",
        content=(
            "Microservices should communicate via well-defined APIs. "
            "Each service owns its data and exposes it through contracts. "
            "Service discovery and load balancing are essential."
        ),
        source_name="Platform Team",
        reliability=SourceReliability.HIGH,
    )
    store.ingest(
        title="Deployment Runbook",
        content=(
            "Deploy using blue-green strategy. Rollback within 5 minutes if errors exceed 1%. "
            "Monitor dashboards during deployment windows."
        ),
        source_name="SRE Team",
        reliability=SourceReliability.MEDIUM,
    )

    engine = RetrievalEngine()
    engine.index(store.get_all_chunks())
    temporal = TemporalScorer()
    scorer = ConfidenceScorer(store, temporal)
    tracker = ProvenanceTracker(store)
    return store, engine, tracker, scorer


class TestProvenanceTracker:
    def test_track_creates_record(self) -> None:
        _, engine, tracker, scorer = _setup()
        trace = engine.search("microservices API")
        conf = scorer.score(trace)
        record = tracker.track("microservices API", trace, conf)

        assert record.query == "microservices API"
        assert len(record.source_documents) > 0
        assert record.confidence_score == conf.overall_confidence

    def test_track_source_attributions(self) -> None:
        _, engine, tracker, scorer = _setup()
        trace = engine.search("deployment rollback")
        conf = scorer.score(trace)
        record = tracker.track("deployment rollback", trace, conf)

        assert len(record.source_attributions) > 0

    def test_get_records(self) -> None:
        _, engine, tracker, scorer = _setup()
        trace = engine.search("API")
        conf = scorer.score(trace)
        tracker.track("API", trace, conf)
        tracker.track("deployment", trace, conf)

        records = tracker.get_records()
        assert len(records) == 2

    def test_get_record_by_id(self) -> None:
        _, engine, tracker, scorer = _setup()
        trace = engine.search("deploy")
        conf = scorer.score(trace)
        record = tracker.track("deploy", trace, conf)

        found = tracker.get_record(record.record_id)
        assert found is not None
        assert found.record_id == record.record_id

    def test_get_record_not_found(self) -> None:
        _, _, tracker, _ = _setup()
        assert tracker.get_record("nonexistent") is None
