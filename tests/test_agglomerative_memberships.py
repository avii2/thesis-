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
from src.data.descriptors import build_cluster_descriptors  # noqa: E402


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
            "dataset_name": "Agglomerative Membership Test",
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


class AgglomerativeMembershipTests(unittest.TestCase):
    def test_descriptors_use_train_split_only_and_memberships_use_fixed_subcluster_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()
            write_csv(
                raw_dir / "sample.csv",
                [
                    ["time", "sensor", "attack"],
                    ["2026-01-01 00:00:16", "2003", "0"],
                    ["2026-01-01 00:00:01", "1", "0"],
                    ["2026-01-01 00:00:15", "2002", "1"],
                    ["2026-01-01 00:00:02", "2", "1"],
                    ["2026-01-01 00:00:14", "2001", "1"],
                    ["2026-01-01 00:00:03", "3", "0"],
                    ["2026-01-01 00:00:13", "2000", "0"],
                    ["2026-01-01 00:00:04", "4", "1"],
                    ["2026-01-01 00:00:12", "1003", "1"],
                    ["2026-01-01 00:00:05", "100", "0"],
                    ["2026-01-01 00:00:11", "1002", "0"],
                    ["2026-01-01 00:00:06", "101", "1"],
                    ["2026-01-01 00:00:10", "1001", "1"],
                    ["2026-01-01 00:00:07", "102", "1"],
                    ["2026-01-01 00:00:09", "1000", "0"],
                    ["2026-01-01 00:00:08", "103", "0"],
                ],
            )
            config_path = tmp_path / "cluster.yaml"
            write_cluster_config(config_path, raw_dir)
            membership_path = tmp_path / "memberships.json"
            scaler_path = tmp_path / "descriptor_scaler.pkl"
            client_metadata_path = tmp_path / "clients.json"

            descriptor_result = build_cluster_descriptors(
                config_path,
                client_metadata_path=client_metadata_path,
                descriptor_scaler_path=scaler_path,
            )
            self.assertEqual(descriptor_result.descriptor_dim, 2)
            self.assertEqual(descriptor_result.output_feature_names, ("sensor",))
            self.assertEqual(descriptor_result.raw_descriptor_matrix.shape, (4, 2))

            result = run_offline_agglomerative_clustering(
                config_path,
                membership_path=membership_path,
                client_metadata_path=client_metadata_path,
                descriptor_scaler_path=scaler_path,
                force_recompute=True,
            )

            payload = json.loads(membership_path.read_text(encoding="utf-8"))
            memberships = {item["client_id"]: item["subcluster_id"] for item in payload["clients"]}

            self.assertFalse(result.reused_existing_membership)
            self.assertEqual(payload["n_subclusters"], 2)
            self.assertEqual(set(payload["fixed_subcluster_ids"]), {"H1", "H2"})
            self.assertEqual(set(memberships), {"C1_L001", "C1_L002", "C1_L003", "C1_L004"})
            self.assertEqual(memberships["C1_L001"], memberships["C1_L002"])
            self.assertEqual(memberships["C1_L003"], memberships["C1_L004"])
            self.assertNotEqual(memberships["C1_L001"], memberships["C1_L003"])
            self.assertEqual({item["subcluster_id"] for item in payload["subclusters"]}, {"H1", "H2"})


if __name__ == "__main__":
    unittest.main()
