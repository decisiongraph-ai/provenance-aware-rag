"""Smart text chunking with configurable overlap."""

from __future__ import annotations

from provenance_rag.models import Chunk, Document


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[tuple[str, int, int]]:
    """Split text into overlapping chunks.

    Returns list of (chunk_text, start_char, end_char) tuples.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    if not text.strip():
        return []

    chunks: list[tuple[str, int, int]] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to break at sentence or word boundary
        if end < text_len:
            boundary = _find_boundary(text, start, end)
            if boundary > start:
                end = boundary

        chunk_content = text[start:end].strip()
        if chunk_content:
            chunks.append((chunk_content, start, end))

        if end >= text_len:
            break

        start = end - chunk_overlap

    return chunks


def _find_boundary(text: str, start: int, end: int) -> int:
    """Find the best boundary (sentence > newline > word) near end."""
    search_region = text[max(start, end - 100) : end]

    for sep in (". ", ".\n", "\n\n", "\n", " "):
        idx = search_region.rfind(sep)
        if idx != -1:
            return max(start, end - 100) + idx + len(sep)

    return end


def chunk_document(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Chunk a document into Chunk models."""
    raw_chunks = chunk_text(document.content, chunk_size, chunk_overlap)
    return [
        Chunk(
            document_id=document.document_id,
            content=content,
            chunk_index=i,
            start_char=start,
            end_char=end,
            metadata={"title": document.title, "source": document.source.name},
        )
        for i, (content, start, end) in enumerate(raw_chunks)
    ]
