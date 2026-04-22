from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .metadata_schema import LedgerRecord, LedgerSchemaError, validate_ledger_record


MAIN_CLUSTER_HEAD_ROLE = "main_cluster_head"
SUBCLUSTER_HEAD_ROLE = "subcluster_head"
LEAF_CLIENT_ROLE = "leaf_client"


class LedgerAccessError(PermissionError):
    """Raised when a non-main-cluster actor attempts to write to the ledger."""


@dataclass(frozen=True)
class LedgerAppendResult:
    ledger_path: Path
    record: LedgerRecord
    line_count: int
    size_bytes: int


class JSONLMockLedger:
    def __init__(self, ledger_path: str | Path) -> None:
        self.ledger_path = Path(ledger_path)

    def append_record(
        self,
        record: Mapping[str, Any] | LedgerRecord,
        *,
        actor_role: str,
        actor_identity: str,
    ) -> LedgerAppendResult:
        self._require_maincluster_writer(actor_role=actor_role, actor_identity=actor_identity)
        validated_record = validate_ledger_record(record)
        self._require_identity_match(validated_record, actor_identity=actor_identity)

        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(validated_record.to_json_line() + "\n")

        return LedgerAppendResult(
            ledger_path=self.ledger_path,
            record=validated_record,
            line_count=self.line_count(),
            size_bytes=self.size_bytes(),
        )

    def read_records(self) -> list[LedgerRecord]:
        if not self.ledger_path.exists():
            return []

        records: list[LedgerRecord] = []
        with self.ledger_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise LedgerSchemaError(
                        f"{self.ledger_path}: invalid JSONL record at line {line_number}."
                    ) from exc
                records.append(validate_ledger_record(payload))
        return records

    def line_count(self) -> int:
        return len(self.read_records())

    def size_bytes(self) -> int:
        if not self.ledger_path.exists():
            return 0
        return int(self.ledger_path.stat().st_size)

    def _require_maincluster_writer(self, *, actor_role: str, actor_identity: str) -> None:
        normalized_role = str(actor_role).strip().lower()
        if normalized_role != MAIN_CLUSTER_HEAD_ROLE:
            raise LedgerAccessError(
                "Only main-cluster heads may write ledger records. "
                f"Observed actor_role={actor_role!r} actor_identity={actor_identity!r}."
            )

    def _require_identity_match(self, record: LedgerRecord, *, actor_identity: str) -> None:
        if actor_identity != record.cluster_head_id or actor_identity != record.submitter_identity:
            raise LedgerAccessError(
                "Ledger writer identity must match both cluster_head_id and submitter_identity. "
                f"Observed actor_identity={actor_identity!r}, "
                f"cluster_head_id={record.cluster_head_id!r}, "
                f"submitter_identity={record.submitter_identity!r}."
            )
