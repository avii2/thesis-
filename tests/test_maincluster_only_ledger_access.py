from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ledger.metadata_schema import canonical_sha256  # noqa: E402
from src.ledger.mock_ledger import (  # noqa: E402
    JSONLMockLedger,
    LEAF_CLIENT_ROLE,
    MAIN_CLUSTER_HEAD_ROLE,
    SUBCLUSTER_HEAD_ROLE,
)


def record_payload() -> dict[str, object]:
    return {
        "round_id": 1,
        "cluster_id": 3,
        "cluster_head_id": "MC3_HEAD",
        "model_version": "C3_R001",
        "previous_main_model_hash": canonical_sha256({"weights_digest": "prev"}),
        "new_main_model_hash": canonical_sha256({"weights_digest": "new"}),
        "clustering_method": "AgglomerativeClustering",
        "clustering_configuration_hash": canonical_sha256({"method": "ward", "metric": "euclidean"}),
        "subcluster_count": 3,
        "subcluster_membership_hash": canonical_sha256({"membership_file": "cluster3_memberships.json"}),
        "fl_method": "SCAFFOLD",
        "aggregation_rule": "weighted_arithmetic_mean",
        "effective_sample_count": 960,
        "participant_count": 15,
        "timestamp_start": "2026-04-22T11:00:00+05:30",
        "timestamp_end": "2026-04-22T11:00:12+05:30",
        "submitter_identity": "MC3_HEAD",
    }


class MainClusterOnlyLedgerAccessTests(unittest.TestCase):
    def test_only_maincluster_heads_may_write_ledger_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = JSONLMockLedger(Path(tmpdir) / "outputs" / "ledgers" / "P_C3_ledger.jsonl")

            ok = ledger.append_record(
                record_payload(),
                actor_role=MAIN_CLUSTER_HEAD_ROLE,
                actor_identity="MC3_HEAD",
            )
            self.assertEqual(ok.line_count, 1)

            with self.assertRaisesRegex(PermissionError, "Only main-cluster heads may write ledger records"):
                ledger.append_record(
                    record_payload(),
                    actor_role=SUBCLUSTER_HEAD_ROLE,
                    actor_identity="W1_HEAD",
                )

            with self.assertRaisesRegex(PermissionError, "Only main-cluster heads may write ledger records"):
                ledger.append_record(
                    record_payload(),
                    actor_role=LEAF_CLIENT_ROLE,
                    actor_identity="C3_L001",
                )

    def test_maincluster_writer_identity_must_match_record_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = JSONLMockLedger(Path(tmpdir) / "outputs" / "ledgers" / "B_C3_ledger.jsonl")

            with self.assertRaisesRegex(PermissionError, "must match both cluster_head_id and submitter_identity"):
                ledger.append_record(
                    record_payload(),
                    actor_role=MAIN_CLUSTER_HEAD_ROLE,
                    actor_identity="MC3_SHADOW",
                )


if __name__ == "__main__":
    unittest.main()
