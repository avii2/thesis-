from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.clustering.agglomerative import run_offline_agglomerative_clustering  # noqa: E402
from src.fl.subcluster import load_frozen_membership  # noqa: E402


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.write_text(
        "\n".join(",".join(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_cluster_config(path: Path, raw_dir: Path) -> None:
    config = {
        "config_version": 1,
        "cluster": {
            "id": 1,
            "key": "C1",
            "dataset_key": "TEST_CLUSTERING",
            "dataset_name": "Frozen Membership Test",
            "audit_report": "outputs/reports/test_clustering.json",
        },
        "data": {
            "data_root_env_var": "FCFL_DATA_ROOT",
            "default_data_root": str(raw_dir.parent),
            "current_raw_input_dir": str(raw_dir),
            "current_raw_files": ["sample.csv"],
            "training_input_mode": "raw_csv_glob",
            "training_input_glob": None,
            "training_input_path": None,
            "expected_processed_input_path": None,
            "label_column": "attack",
            "label_column_confirmed_from_audit": True,
            "candidate_label_columns_present": ["attack"],
            "timestamp_or_order_columns": ["time"],
            "excluded_columns": ["attack", "time"],
            "exclude_if_present": [],
        },
        "partitioning": {
            "candidate_leaf_clients": 4,
            "split_ratios": {"train": 0.5, "validation": 0.25, "test": 0.25},
        },
        "clustering": {
            "fixed_subclusters": 2,
            "fixed_subcluster_ids": ["H1", "H2"],
        },
        "runtime_validation": {
            "require_training_input_to_exist": True,
            "require_label_column_to_exist": True,
            "error_on_missing_label_column": "CONFIGURED_LABEL_COLUMN_MISSING",
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


class FixedMembershipTests(unittest.TestCase):
    def test_existing_membership_file_is_reused_without_reclustering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()
            write_csv(
                raw_dir / "sample.csv",
                [
                    ["time", "sensor", "attack"],
                    ["2026-01-01 00:00:01", "1", "0"],
                    ["2026-01-01 00:00:02", "2", "1"],
                    ["2026-01-01 00:00:03", "3", "0"],
                    ["2026-01-01 00:00:04", "4", "1"],
                    ["2026-01-01 00:00:05", "100", "0"],
                    ["2026-01-01 00:00:06", "101", "1"],
                    ["2026-01-01 00:00:07", "102", "1"],
                    ["2026-01-01 00:00:08", "103", "0"],
                    ["2026-01-01 00:00:09", "1000", "0"],
                    ["2026-01-01 00:00:10", "1001", "1"],
                    ["2026-01-01 00:00:11", "1002", "0"],
                    ["2026-01-01 00:00:12", "1003", "1"],
                    ["2026-01-01 00:00:13", "2000", "0"],
                    ["2026-01-01 00:00:14", "2001", "1"],
                    ["2026-01-01 00:00:15", "2002", "1"],
                    ["2026-01-01 00:00:16", "2003", "0"],
                ],
            )
            config_path = tmp_path / "cluster.yaml"
            write_cluster_config(config_path, raw_dir)
            membership_path = tmp_path / "memberships.json"
            scaler_path = tmp_path / "descriptor_scaler.pkl"
            client_metadata_path = tmp_path / "clients.json"

            first = run_offline_agglomerative_clustering(
                config_path,
                membership_path=membership_path,
                client_metadata_path=client_metadata_path,
                descriptor_scaler_path=scaler_path,
            )
            first_contents = membership_path.read_text(encoding="utf-8")

            write_csv(
                raw_dir / "sample.csv",
                [
                    ["time", "sensor", "attack"],
                    ["2026-01-01 00:00:01", "9", "0"],
                    ["2026-01-01 00:00:02", "9", "1"],
                    ["2026-01-01 00:00:03", "9", "0"],
                    ["2026-01-01 00:00:04", "9", "1"],
                ],
            )

            second = run_offline_agglomerative_clustering(
                config_path,
                membership_path=membership_path,
                client_metadata_path=client_metadata_path,
                descriptor_scaler_path=scaler_path,
            )

            self.assertFalse(first.reused_existing_membership)
            self.assertTrue(second.reused_existing_membership)
            self.assertEqual(first_contents, membership_path.read_text(encoding="utf-8"))

    def test_hierarchical_baseline_and_proposed_configs_reuse_same_membership_files(self) -> None:
        baseline = yaml.safe_load((REPO_ROOT / "configs/baseline_hierarchical.yaml").read_text(encoding="utf-8"))
        proposed = yaml.safe_load((REPO_ROOT / "configs/proposed.yaml").read_text(encoding="utf-8"))

        baseline_by_cluster = {
            Path(entry["cluster_config"]).name: entry["membership_file"]
            for entry in baseline["clusters"]
        }
        proposed_by_cluster = {
            Path(entry["cluster_config"]).name: entry["membership_file"]
            for entry in proposed["clusters"]
        }

        self.assertEqual(baseline_by_cluster, proposed_by_cluster)
        self.assertEqual(
            proposed_by_cluster,
            {
                "cluster1_hai.yaml": "outputs/clustering/cluster1_memberships.json",
                "cluster2_ton_iot.yaml": "outputs/clustering/cluster2_memberships.json",
                "cluster3_wustl.yaml": "outputs/clustering/cluster3_memberships.json",
            },
        )

    def test_frozen_membership_requires_reuse_for_baseline_and_proposed_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            membership_path = Path(tmpdir) / "cluster1_memberships.json"
            membership_path.write_text(
                json.dumps(
                    {
                        "cluster_id": 1,
                        "dataset": "HAI 21.03",
                        "frozen": True,
                        "n_subclusters": 2,
                        "fixed_subcluster_ids": ["H1", "H2"],
                        "membership_hash": "deadbeef",
                        "reuse_for_experiment_groups": ["baseline_uniform_hierarchical"],
                        "subclusters": [
                            {"subcluster_id": "H1", "client_ids": ["c1"]},
                            {"subcluster_id": "H2", "client_ids": ["c2"]},
                        ],
                        "clients": [
                            {"client_id": "c1", "subcluster_id": "H1"},
                            {"client_id": "c2", "subcluster_id": "H2"},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as context:
                load_frozen_membership(
                    membership_path,
                    expected_cluster_id=1,
                    expected_n_subclusters=2,
                    expected_client_ids=["c1", "c2"],
                )

            self.assertIn("proposed_specialized_hierarchical", str(context.exception))


if __name__ == "__main__":
    unittest.main()
