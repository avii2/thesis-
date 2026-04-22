from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REQUIRED_LEDGER_FIELDS = (
    "round_id",
    "cluster_id",
    "cluster_head_id",
    "model_version",
    "previous_main_model_hash",
    "new_main_model_hash",
    "clustering_method",
    "clustering_configuration_hash",
    "subcluster_count",
    "subcluster_membership_hash",
    "fl_method",
    "aggregation_rule",
    "effective_sample_count",
    "participant_count",
    "timestamp_start",
    "timestamp_end",
    "submitter_identity",
)

OPTIONAL_LEDGER_FIELDS = ("subcluster_digest",)
ALLOWED_LEDGER_FIELDS = REQUIRED_LEDGER_FIELDS + OPTIONAL_LEDGER_FIELDS

FORBIDDEN_LEDGER_FIELD_PATTERNS = (
    "raw",
    "weight",
    "weights",
    "gradient",
    "tensor",
    "state_dict",
    "optimizer",
    "descriptor",
    "sample",
    "dataset",
)

HASH_FIELDS = (
    "previous_main_model_hash",
    "new_main_model_hash",
    "clustering_configuration_hash",
    "subcluster_membership_hash",
    "subcluster_digest",
)


class LedgerSchemaError(ValueError):
    """Raised when a ledger record violates the approved metadata schema."""


@dataclass(frozen=True)
class LedgerRecord:
    round_id: int
    cluster_id: int
    cluster_head_id: str
    model_version: str
    previous_main_model_hash: str | None
    new_main_model_hash: str
    clustering_method: str
    clustering_configuration_hash: str
    subcluster_count: int
    subcluster_membership_hash: str
    fl_method: str
    aggregation_rule: str
    effective_sample_count: int
    participant_count: int
    timestamp_start: str
    timestamp_end: str
    submitter_identity: str
    subcluster_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["subcluster_digest"] is None:
            payload.pop("subcluster_digest")
        return payload

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def validate_ledger_record(record: Mapping[str, Any] | LedgerRecord) -> LedgerRecord:
    payload = record.to_dict() if isinstance(record, LedgerRecord) else dict(record)
    _require_mapping_fields(payload)
    _reject_forbidden_payload_keys(payload)
    _reject_unknown_fields(payload)

    round_id = _require_int(payload, "round_id", minimum=0)
    cluster_id = _require_int(payload, "cluster_id", minimum=1, maximum=3)
    cluster_head_id = _require_nonempty_string(payload, "cluster_head_id")
    model_version = _require_nonempty_string(payload, "model_version")
    previous_main_model_hash = _require_optional_hash(payload, "previous_main_model_hash")
    new_main_model_hash = _require_hash(payload, "new_main_model_hash")
    clustering_method = _require_nonempty_string(payload, "clustering_method")
    clustering_configuration_hash = _require_hash(payload, "clustering_configuration_hash")
    subcluster_count = _require_int(payload, "subcluster_count", minimum=1)
    subcluster_membership_hash = _require_hash(payload, "subcluster_membership_hash")
    fl_method = _require_nonempty_string(payload, "fl_method")
    aggregation_rule = _require_nonempty_string(payload, "aggregation_rule")
    effective_sample_count = _require_int(payload, "effective_sample_count", minimum=0)
    participant_count = _require_int(payload, "participant_count", minimum=0)
    timestamp_start = _require_iso8601_string(payload, "timestamp_start")
    timestamp_end = _require_iso8601_string(payload, "timestamp_end")
    submitter_identity = _require_nonempty_string(payload, "submitter_identity")
    subcluster_digest = _require_optional_hash(payload, "subcluster_digest")

    if submitter_identity != cluster_head_id:
        raise LedgerSchemaError(
            "Ledger submitter_identity must match cluster_head_id for metadata-only main-cluster submissions."
        )
    if participant_count > effective_sample_count and effective_sample_count >= 0:
        raise LedgerSchemaError(
            "Ledger participant_count cannot exceed effective_sample_count."
        )

    return LedgerRecord(
        round_id=round_id,
        cluster_id=cluster_id,
        cluster_head_id=cluster_head_id,
        model_version=model_version,
        previous_main_model_hash=previous_main_model_hash,
        new_main_model_hash=new_main_model_hash,
        clustering_method=clustering_method,
        clustering_configuration_hash=clustering_configuration_hash,
        subcluster_count=subcluster_count,
        subcluster_membership_hash=subcluster_membership_hash,
        fl_method=fl_method,
        aggregation_rule=aggregation_rule,
        effective_sample_count=effective_sample_count,
        participant_count=participant_count,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        submitter_identity=submitter_identity,
        subcluster_digest=subcluster_digest,
    )


def canonical_sha256(payload: Any) -> str:
    canonical_payload = _canonicalize_for_hash(payload)
    encoded = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def model_version_for_round(cluster_id: int, round_id: int) -> str:
    if cluster_id <= 0:
        raise LedgerSchemaError(f"cluster_id must be positive for model version construction. Observed {cluster_id}.")
    if round_id < 0:
        raise LedgerSchemaError(f"round_id must be non-negative for model version construction. Observed {round_id}.")
    return f"C{cluster_id}_R{round_id:03d}"


def default_ledger_path(experiment_id: str, *, repo_root: str | Path = ".") -> Path:
    experiment_id = str(experiment_id).strip()
    if not experiment_id:
        raise LedgerSchemaError("experiment_id must be non-empty for ledger path construction.")
    return Path(repo_root) / "outputs" / "ledgers" / f"{experiment_id}_ledger.jsonl"


def _require_mapping_fields(payload: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_LEDGER_FIELDS if field not in payload]
    if missing:
        raise LedgerSchemaError(f"Ledger record is missing required fields: {missing}.")


def _reject_unknown_fields(payload: Mapping[str, Any]) -> None:
    unknown = [field for field in payload.keys() if field not in ALLOWED_LEDGER_FIELDS]
    if unknown:
        raise LedgerSchemaError(
            f"Ledger payload schema matches approved metadata schema only. Unknown fields: {sorted(unknown)}."
        )


def _reject_forbidden_payload_keys(payload: Mapping[str, Any]) -> None:
    forbidden = [
        field for field in payload.keys()
        if field not in ALLOWED_LEDGER_FIELDS
        and any(pattern in field.lower() for pattern in FORBIDDEN_LEDGER_FIELD_PATTERNS)
    ]
    if forbidden:
        raise LedgerSchemaError(
            f"Ledger records must remain metadata-only. Forbidden payload fields present: {sorted(forbidden)}."
        )


def _require_int(
    payload: Mapping[str, Any],
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise LedgerSchemaError(f"Ledger field {field!r} must be an integer. Observed {value!r}.")
    if minimum is not None and value < minimum:
        raise LedgerSchemaError(f"Ledger field {field!r} must be >= {minimum}. Observed {value}.")
    if maximum is not None and value > maximum:
        raise LedgerSchemaError(f"Ledger field {field!r} must be <= {maximum}. Observed {value}.")
    return int(value)


def _require_nonempty_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise LedgerSchemaError(f"Ledger field {field!r} must be a non-empty string. Observed {value!r}.")
    return value


def _require_hash(payload: Mapping[str, Any], field: str) -> str:
    value = _require_nonempty_string(payload, field)
    if not value.startswith("sha256:") or len(value) != len("sha256:") + 64:
        raise LedgerSchemaError(
            f"Ledger field {field!r} must be a sha256-prefixed hash string. Observed {value!r}."
        )
    return value


def _require_optional_hash(payload: Mapping[str, Any], field: str) -> str | None:
    if field not in payload or payload[field] is None:
        return None
    return _require_hash(payload, field)


def _require_iso8601_string(payload: Mapping[str, Any], field: str) -> str:
    value = _require_nonempty_string(payload, field)
    normalized = value.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise LedgerSchemaError(
            f"Ledger field {field!r} must be ISO-8601 compatible. Observed {value!r}."
        ) from exc
    return value


def _canonicalize_for_hash(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): _canonicalize_for_hash(value) for key, value in sorted(payload.items(), key=lambda item: str(item[0]))}
    if isinstance(payload, (list, tuple)):
        return [_canonicalize_for_hash(value) for value in payload]
    if isinstance(payload, np.ndarray):
        return {
            "dtype": str(payload.dtype),
            "shape": list(payload.shape),
            "values": payload.tolist(),
        }
    if isinstance(payload, np.generic):
        return payload.item()
    if isinstance(payload, Path):
        return str(payload)
    if payload is None or isinstance(payload, (str, int, float, bool)):
        return payload
    raise LedgerSchemaError(
        f"Unsupported payload type for canonical SHA256 hashing: {type(payload).__name__}."
    )
