"""Audit logging for all retrieval operations."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from provenance_rag.models import AuditEntry

logger = logging.getLogger(__name__)


class AuditLogger:
    """Log every query, retrieval trace, and generation event."""

    def __init__(self, log_dir: str | Path | None = None) -> None:
        self._entries: list[AuditEntry] = []
        self._log_dir: Path | None = Path(log_dir) if log_dir else None
        if self._log_dir:
            self._log_dir.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        event_type: str,
        query: str = "",
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Record an audit event."""
        entry = AuditEntry(
            event_type=event_type,
            query=query,
            details=details or {},
        )
        self._entries.append(entry)
        logger.info("audit event=%s query=%r", event_type, query)

        if self._log_dir:
            self._write_to_file(entry)

        return entry

    def get_entries(
        self,
        event_type: str | None = None,
        since: datetime | None = None,
    ) -> list[AuditEntry]:
        """Retrieve audit entries, optionally filtered."""
        entries = self._entries
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]
        if since:
            entries = [e for e in entries if e.timestamp >= since]
        return entries

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def _write_to_file(self, entry: AuditEntry) -> None:
        if not self._log_dir:
            return
        filename = f"audit_{entry.timestamp.strftime('%Y%m%d')}.jsonl"
        filepath = self._log_dir / filename
        with filepath.open("a") as f:
            f.write(json.dumps(entry.model_dump(), default=str) + "\n")
