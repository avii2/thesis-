from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ledger.metadata_schema import (  # noqa: E402
    REQUIRED_LEDGER_FIELDS,
    canonical_sha256,
)
from src.ledger.mock_ledger import JSONLMockLedger, MAIN_CLUSTER_HEAD_ROLE  # noqa: E402


def valid_record_payload() -> dict[str, object]:
    return {
        "round_id": 0,
        "cluster_id": 1,
        "cluster_head_id": "MC1_HEAD",
        "model_version": "C1_R000",
        "previous_main_model_hash": None,
        "new_main_model_hash": canonical_sha256({"weights_digest": "new"}),
        "clustering_method": "AgglomerativeClustering",
        "clustering_configuration_hash": canonical_sha256({"method": "ward", "metric": "euclidean"}),
        "subcluster_count": 2,
        "subcluster_membership_hash": canonical_sha256({"membership_file": "cluster1_memberships.json"}),
        "fl_method": "FedBN",
        "aggregation_rule": "weighted_non_bn_mean",
        "effective_sample_count": 128,
        "participant_count": 2,
        "timestamp_start": "2026-04-22T10:00:00+05:30",
        "timestamp_end": "2026-04-22T10:00:05+05:30",
        "submitter_identity": "MC1_HEAD",
    }


class LedgerMetadataOnlyTests(unittest.TestCase):
    def test_jsonl_mock_ledger_appends_one_valid_metadata_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = JSONLMockLedger(Path(tmpdir) / "outputs" / "ledgers" / "A_C1_ledger.jsonl")
            record = valid_record_payload()

            result = ledger.append_record(
                record,
                actor_role=MAIN_CLUSTER_HEAD_ROLE,
                actor_identity="MC1_HEAD",
            )

            self.assertTrue(result.ledger_path.exists())
            self.assertEqual(result.line_count, 1)
            self.assertGreater(result.size_bytes, 0)

            raw_lines = result.ledger_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(raw_lines), 1)
            payload = json.loads(raw_lines[0])
            self.assertEqual(set(payload.keys()), set(REQUIRED_LEDGER_FIELDS))
            self.assertEqual(payload["cluster_id"], 1)
            self.assertEqual(payload["new_main_model_hash"][:7], "sha256:")
            self.assertNotIn("raw_data", payload)
            self.assertNotIn("full_model_weights", payload)

            records = ledger.read_records()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].cluster_head_id, "MC1_HEAD")

    def test_ledger_rejects_forbidden_non_metadata_payload_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = JSONLMockLedger(Path(tmpdir) / "outputs" / "ledgers" / "P_C1_ledger.jsonl")
            record = valid_record_payload()
            record["full_model_weights"] = {"conv_weight": [1.0, 2.0]}

            with self.assertRaisesRegex(ValueError, "metadata-only|Unknown fields"):
                ledger.append_record(
                    record,
                    actor_role=MAIN_CLUSTER_HEAD_ROLE,
                    actor_identity="MC1_HEAD",
                )


if __name__ == "__main__":
    unittest.main()
