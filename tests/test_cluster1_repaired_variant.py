from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import yaml

from src.data.cluster1_repaired import prepare_cluster1_repaired
from src.fl.maincluster import build_flat_federated_clients


def _write_hai_csv(path: Path, *, rows: int, attack_rows: set[int]) -> None:
    lines = ["time,sensor_a,sensor_b,attack,attack_P1,attack_P2,attack_P3"]
    for index in range(rows):
        attack = 1 if index in attack_rows else 0
        lines.append(f"{index},{index % 7},{(index * 3) % 11},{attack},0,0,0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class Cluster1RepairedVariantTests(unittest.TestCase):
    def test_repaired_pretraining_keeps_heldout_files_out_of_training_and_balances_attacks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "hai-21.03"
            output_root = root / "outputs_c1_repaired"
            raw_dir.mkdir()

            normal_only = set()
            attack_some = {5, 6, 15, 16, 25, 26}
            heldout_attack = {4, 5, 20, 21}
            for name in ("train1.csv", "train2.csv", "train3.csv"):
                _write_hai_csv(raw_dir / name, rows=32, attack_rows=normal_only)
            for name in ("test1.csv", "test2.csv", "test3.csv"):
                _write_hai_csv(raw_dir / name, rows=32, attack_rows=attack_some)
            for name in ("test4.csv", "test5.csv"):
                _write_hai_csv(raw_dir / name, rows=32, attack_rows=heldout_attack)

            config_path = root / "cluster1_hai_repaired.yaml"
            config = {
                "config_version": 1,
                "cluster": {
                    "id": 1,
                    "key": "C1",
                    "dataset_key": "HAI_2103_REPAIRED",
                    "dataset_name": "HAI 21.03 repaired supervised variant",
                    "audit_report": str(output_root / "reports/data_profile_cluster1_repaired.json"),
                },
                "data": {
                    "data_root_env_var": "FCFL_DATA_ROOT",
                    "default_data_root": "data",
                    "current_raw_input_dir": str(raw_dir),
                    "current_raw_files": [
                        "train1.csv",
                        "train2.csv",
                        "train3.csv",
                        "test1.csv",
                        "test2.csv",
                        "test3.csv",
                        "test4.csv",
                        "test5.csv",
                    ],
                    "train_validation_files": [
                        "train1.csv",
                        "train2.csv",
                        "train3.csv",
                        "test1.csv",
                        "test2.csv",
                        "test3.csv",
                    ],
                    "heldout_test_files": ["test4.csv", "test5.csv"],
                    "repaired_output_root": str(output_root),
                    "training_input_mode": "raw_csv_glob",
                    "training_input_glob": str(raw_dir / "*.csv"),
                    "schema_consistent_across_files": True,
                    "label_column": "attack",
                    "label_column_confirmed_from_audit": True,
                    "candidate_label_columns_present": ["attack"],
                    "timestamp_or_order_columns": ["time"],
                    "excluded_columns": ["attack", "time", "attack_P1", "attack_P2", "attack_P3"],
                    "exclude_if_present": ["attack_P4"],
                },
                "partitioning": {
                    "candidate_leaf_clients": 4,
                    "strategy": "window_first_attack_aware",
                    "validation_ratio": 0.20,
                },
                "clustering": {
                    "fixed_subclusters": 2,
                    "fixed_subcluster_ids": ["H1", "H2"],
                },
                "preprocessing": {
                    "input_type": "multivariate_time_series",
                    "window_length": 4,
                    "stride": 2,
                    "window_label_rule": "any_positive_row",
                },
                "runtime_validation": {
                    "require_training_input_to_exist": True,
                    "require_schema_consistency_across_files": True,
                    "require_label_column_to_exist": True,
                },
            }
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            result = prepare_cluster1_repaired(config_path)
            profile = result["profile"]
            self.assertFalse(profile["test_leakage_prevention"]["heldout_test_files_used_for_training"])
            self.assertEqual(profile["heldout_test_source_files"], ["test4.csv", "test5.csv"])
            self.assertEqual(profile["clients_with_positive_train_or_validation"], 4)

            clients = json.loads((output_root / "clients/cluster1_leaf_clients.json").read_text(encoding="utf-8"))
            for client in clients["clients"]:
                positive_count = client["train_label_counts"].get("1", 0) + client["val_label_counts"].get("1", 0)
                self.assertGreater(positive_count, 0)

            validation_summary = output_root / "reports/cluster1_repaired_validation_summary.md"
            self.assertTrue(validation_summary.exists())
            validation_summary_text = validation_summary.read_text(encoding="utf-8")
            self.assertIn("Every client received at least one positive training window: `YES`", validation_summary_text)

            membership = json.loads((output_root / "clustering/cluster1_memberships.json").read_text(encoding="utf-8"))
            self.assertEqual(membership["n_subclusters"], 2)
            self.assertTrue(membership["frozen"])

            clients, model_config, data_summary = build_flat_federated_clients(config_path)
            self.assertEqual(len(clients), 4)
            self.assertEqual(model_config.input_length, 4)
            self.assertEqual(data_summary["variant"], "cluster1_repaired")
            self.assertFalse(data_summary["heldout_test_files_used_for_training"])
            self.assertEqual(data_summary["heldout_test_source_files"], ["test4.csv", "test5.csv"])
            for client in clients:
                positive_count = int((client.train.labels == 1).sum() + (client.validation.labels == 1).sum())
                self.assertGreater(positive_count, 0)


if __name__ == "__main__":
    unittest.main()
