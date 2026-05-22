# Provenance-Aware RAG

Building trustworthy enterprise retrieval systems with source lineage, auditability, temporal awareness, and confidence-aware AI generation.

## Vision

Enterprise AI systems must explain:
- where information came from
- why an answer was generated
- whether information is still valid
- what sources were used
- how trustworthy the response is

This project explores provenance-aware retrieval systems for enterprise AI.

## MVP Goals

### Source-Aware Retrieval
Every response includes:
- source references
- retrieval lineage
- supporting context
- confidence indicators

### Temporal Awareness
The system tracks:
- source freshness
- historical validity
- evolving organizational knowledge

### Confidence-Aware Responses
Responses include:
- confidence scoring
- uncertainty indicators
- conflicting source detection

### Enterprise Auditability
The system maintains:
- retrieval traces
- prompt lineage
- source attribution
- audit logs

## Proposed Architecture

Knowledge Sources -> Processing Layer -> Retrieval Engine -> Provenance Layer -> Generation Layer -> Response + Citations

## Quick Start

### Prerequisites

- Python 3.11+
- pip

### Install

```bash
# Clone the repository
git clone https://github.com/decisiongraph-ai/provenance-aware-rag.git
cd provenance-aware-rag

# Install with dev dependencies
pip install -e ".[dev]"
```

### Run the API

```bash
uvicorn provenance_rag.api:app --reload --port 8000
```

The API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Run with Docker

```bash
docker compose up --build
```

### Run Tests

```bash
pytest
```

### Lint

```bash
ruff check src/ tests/
ruff format --check src/ tests/
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/ingest` | Ingest a document with metadata |
| `POST` | `/query` | Query with provenance-aware retrieval |
| `GET` | `/documents` | List all documents |
| `GET` | `/documents/{id}` | Get a specific document |
| `GET` | `/audit` | Retrieve audit log entries |
| `GET` | `/provenance` | List provenance records |
| `GET` | `/provenance/{id}` | Get a specific provenance record |

### Example: Ingest a Document

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Security Policy",
    "content": "All systems must use TLS 1.2 or higher...",
    "source_name": "InfoSec Team",
    "author": "Security Lead",
    "document_type": "policy",
    "reliability": "high"
  }'
```

### Example: Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the TLS requirements?", "top_k": 5}'
```

## Repository Structure

```
provenance-aware-rag/
├── src/
│   └── provenance_rag/
│       ├── __init__.py
│       ├── models.py           # Pydantic models
│       ├── ingestion.py        # Document ingestion with metadata
│       ├── chunking.py         # Smart text chunking with overlap
│       ├── retrieval.py        # BM25 + TF-IDF hybrid retrieval
│       ├── provenance.py       # Source attribution and lineage
│       ├── confidence.py       # Confidence scoring and conflict detection
│       ├── temporal.py         # Freshness scoring and validity tracking
│       ├── audit.py            # Audit logging
│       ├── generation.py       # Response generation with citations
│       └── api.py              # FastAPI REST API
├── tests/
├── docs/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Tech Stack

- **FastAPI** + uvicorn — REST API
- **Pydantic v2** — data validation and models
- **scikit-learn** — TF-IDF vectorization
- **rank-bm25** — BM25 keyword retrieval
- **pytest** — testing

## Contributors

- Akash Raj
- Prem Kumar
