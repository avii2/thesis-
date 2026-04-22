import csv
from pathlib import Path
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(relative_path: str) -> dict:
    return yaml.safe_load((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


class ConfigFileTests(unittest.TestCase):
    def test_cluster_configs_match_audited_dataset_facts(self) -> None:
        cluster1 = load_yaml("configs/cluster1_hai.yaml")
        self.assertEqual(cluster1["data"]["label_column"], "attack")
        self.assertTrue(cluster1["data"]["label_column_confirmed_from_audit"])
        self.assertEqual(cluster1["data"]["timestamp_or_order_columns"], ["time"])
        self.assertEqual(cluster1["partitioning"]["candidate_leaf_clients"], 12)
        self.assertEqual(cluster1["clustering"]["fixed_subclusters"], 2)
        self.assertIn("attack_P3", cluster1["data"]["excluded_columns"])
        self.assertIn("attack_P4", cluster1["data"]["exclude_if_present"])

        cluster2 = load_yaml("configs/cluster2_ton_iot.yaml")
        self.assertEqual(cluster2["data"]["label_column"], "label")
        self.assertTrue(cluster2["data"]["label_column_confirmed_from_audit"])
        self.assertEqual(cluster2["data"]["training_input_mode"], "combined_processed_csv_required")
        self.assertEqual(
            cluster2["data"]["expected_processed_input_path"],
            "outputs/processed/cluster2_ton_iot_combined.csv",
        )
        self.assertFalse(cluster2["data"]["schema_consistent_across_files"])
        self.assertEqual(cluster2["partitioning"]["candidate_leaf_clients"], 15)
        self.assertEqual(cluster2["clustering"]["fixed_subclusters"], 3)
        self.assertIn("type", cluster2["data"]["excluded_columns"])
        self.assertEqual(
            cluster2["runtime_validation"]["error_on_missing_expected_processed_input"],
            "TON_IOT_COMBINED_TELEMETRY_REQUIRED",
        )

        cluster3 = load_yaml("configs/cluster3_wustl.yaml")
        self.assertEqual(cluster3["data"]["label_column"], "Target")
        self.assertTrue(cluster3["data"]["label_column_confirmed_from_audit"])
        self.assertEqual(cluster3["data"]["timestamp_or_order_columns"], ["StartTime", "LastTime"])
        self.assertEqual(cluster3["partitioning"]["candidate_leaf_clients"], 15)
        self.assertEqual(cluster3["clustering"]["fixed_subclusters"], 3)
        self.assertIn("Traffic", cluster3["data"]["excluded_columns"])
        self.assertIn("Traffic", cluster3["data"]["auxiliary_label_like_columns"])

    def test_experiment_configs_match_experiment_matrix(self) -> None:
        expected_rows = {}
        with (REPO_ROOT / "docs/EXPERIMENT_MATRIX.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["experiment_id"].startswith(("A_", "B_", "P_")):
                    expected_rows[row["experiment_id"]] = row

        experiment_files = [
            "configs/baseline_flat.yaml",
            "configs/baseline_hierarchical.yaml",
            "configs/proposed.yaml",
        ]

        for relative_path in experiment_files:
            config = load_yaml(relative_path)
            self.assertTrue(config["ledger"]["metadata_only"])
            for cluster_entry in config["clusters"]:
                expected = expected_rows[cluster_entry["experiment_id"]]
                self.assertEqual(cluster_entry["model_family"], expected["model"])
                self.assertEqual(cluster_entry["fl_method"], expected["fl_method"])
                self.assertEqual(cluster_entry["aggregation"], expected["aggregation"])
                self.assertEqual(cluster_entry["hierarchy"], expected["hierarchy"])
                self.assertEqual(cluster_entry["clustering_method"], expected["clustering_method"])
                self.assertEqual(str(cluster_entry["n_subclusters"]), expected["n_subclusters"])


if __name__ == "__main__":
    unittest.main()
