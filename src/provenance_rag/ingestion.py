"""Document ingestion with metadata extraction."""

from __future__ import annotations

from datetime import datetime

from provenance_rag.chunking import chunk_document
from provenance_rag.models import (
    Chunk,
    Document,
    DocumentType,
    Source,
    SourceReliability,
)


class DocumentStore:
    """In-memory document store for the MVP."""

    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}
        self._chunks: dict[str, list[Chunk]] = {}

    def ingest(
        self,
        title: str,
        content: str,
        source_name: str,
        author: str = "",
        url: str = "",
        document_type: DocumentType = DocumentType.GENERAL,
        reliability: SourceReliability = SourceReliability.UNKNOWN,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> Document:
        """Ingest a document: store it and produce chunks."""
        source = Source(
            name=source_name,
            author=author,
            url=url,
            reliability=reliability,
        )
        doc = Document(
            title=title,
            content=content,
            source=source,
            document_type=document_type,
            valid_from=valid_from,
            valid_until=valid_until,
        )
        self._documents[doc.document_id] = doc

        chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._chunks[doc.document_id] = chunks

        return doc

    def get_document(self, document_id: str) -> Document | None:
        return self._documents.get(document_id)

    def get_chunks(self, document_id: str) -> list[Chunk]:
        return self._chunks.get(document_id, [])

    def get_all_chunks(self) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for chunks in self._chunks.values():
            all_chunks.extend(chunks)
        return all_chunks

    def get_all_documents(self) -> list[Document]:
        return list(self._documents.values())

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def chunk_count(self) -> int:
        return sum(len(c) for c in self._chunks.values())
