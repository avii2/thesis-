from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from src.data.cluster1_balanced import (
    EXPECTED_FILE_ORDER,
    prepare_cluster1_balanced_train,
)
from src.fl.maincluster import build_flat_federated_clients


def _write_hai_csv(path: Path, *, rows: int, attack_rows: set[int]) -> None:
    lines = ["time,sensor_a,sensor_b,attack,attack_P1,attack_P2,attack_P3"]
    for index in range(rows):
        attack = 1 if index in attack_rows else 0
        lines.append(f"{index},{index % 7},{(index * 3) % 11},{attack},0,0,0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class Cluster1BalancedVariantTests(unittest.TestCase):
    def test_balanced_variant_keeps_heldout_files_out_and_loads_preprocessed_clients(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "hai-21.03"
            output_root = root / "outputs_c1_balanced_train"
            variant_root = root / "variants"
            raw_dir.mkdir()

            expected_counts: dict[str, tuple[int, int, int]] = {}
            for name in ("train1.csv", "train2.csv", "train3.csv"):
                _write_hai_csv(raw_dir / name, rows=64, attack_rows=set())
                expected_counts[name] = (64, 64, 0)
            for name in ("test1.csv", "test2.csv", "test3.csv"):
                attacks = {20, 21}
                _write_hai_csv(raw_dir / name, rows=64, attack_rows=attacks)
                expected_counts[name] = (64, 62, 2)
            for name in ("test4.csv", "test5.csv"):
                attacks = {18, 19}
                _write_hai_csv(raw_dir / name, rows=64, attack_rows=attacks)
                expected_counts[name] = (64, 62, 2)

            result = prepare_cluster1_balanced_train(
                raw_dir=raw_dir,
                output_root=output_root,
                variant_root=variant_root,
                ratios=(("ratio_1to1", 1),),
                expected_counts=expected_counts,
                num_clients=4,
                write_configs=False,
            )
            self.assertEqual(result["status"], "ok")

            metadata = json.loads((variant_root / "ratio_1to1" / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["class_counts"]["train"]["0"], metadata["class_counts"]["train"]["1"])
            self.assertFalse(metadata["leakage_prevention"]["test4_test5_used_for_training"])
            self.assertFalse(metadata["leakage_prevention"]["validation_balanced"])
            self.assertFalse(metadata["leakage_prevention"]["heldout_test_balanced"])
            self.assertAlmostEqual(metadata["positive_class_weight"], 1.0)

            with np.load(variant_root / "ratio_1to1" / "train_windows.npz") as train_npz:
                train_sources = set(train_npz["source_file"].astype(str).tolist())
                self.assertTrue(train_sources <= {"train1.csv", "train2.csv", "train3.csv", "test1.csv", "test2.csv"})
                self.assertEqual(train_npz["inputs"].shape[1:], (2, 32))
                self.assertTrue(np.isfinite(train_npz["inputs"]).all())
            with np.load(variant_root / "ratio_1to1" / "val_windows.npz") as val_npz:
                self.assertEqual(set(val_npz["source_file"].astype(str).tolist()), {"test3.csv"})
            with np.load(variant_root / "ratio_1to1" / "test_windows.npz") as test_npz:
                self.assertEqual(set(test_npz["source_file"].astype(str).tolist()), {"test4.csv", "test5.csv"})

            client_metadata = json.loads(
                (output_root / "ratio_1to1" / "clients" / "cluster1_leaf_clients.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(client_metadata["clients"]), 4)
            for client in client_metadata["clients"]:
                self.assertGreater(client["train_positive_count"], 0)
                self.assertNotIn("test4.csv", client["train_source_files"])
                self.assertNotIn("test5.csv", client["train_source_files"])

            config_path = root / "cluster1_hai_balanced_1to1.yaml"
            config = {
                "config_version": 1,
                "cluster": {
                    "id": 1,
                    "key": "C1",
                    "dataset_key": "HAI_2103_BALANCED_1TO1",
                    "dataset_name": "HAI 21.03 balanced-training variant 1to1",
                    "audit_report": str(output_root / "reports/hai_file_audit.json"),
                },
                "data": {
                    "data_root_env_var": "FCFL_DATA_ROOT",
                    "default_data_root": "data",
                    "current_raw_input_dir": str(raw_dir),
                    "current_raw_files": list(EXPECTED_FILE_ORDER),
                    "balanced_variant_dir": str(variant_root / "ratio_1to1"),
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
                    "strategy": "balanced_window_npz",
                    "negative_to_positive_ratio": 1,
                },
                "clustering": {
                    "fixed_subclusters": 2,
                    "fixed_subcluster_ids": ["H1", "H2"],
                    "membership_file": str(output_root / "ratio_1to1/clustering/cluster1_memberships.json"),
                },
                "preprocessing": {
                    "input_type": "multivariate_time_series",
                    "window_length": 32,
                    "stride": 8,
                    "window_label_rule": "any_positive_row",
                    "preprocessed_npz": True,
                },
                "runtime_validation": {
                    "require_training_input_to_exist": True,
                    "require_schema_consistency_across_files": True,
                    "require_label_column_to_exist": True,
                },
            }
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            clients, model_config, data_summary = build_flat_federated_clients(config_path)
            self.assertEqual(len(clients), 4)
            self.assertEqual(model_config.input_channels, 2)
            self.assertEqual(model_config.input_length, 32)
            self.assertEqual(data_summary["variant"], "cluster1_balanced_training")
            self.assertFalse(data_summary["heldout_test_files_used_for_training"])
            for client in clients:
                self.assertGreater(int((client.train.labels == 1).sum()), 0)


if __name__ == "__main__":
    unittest.main()
