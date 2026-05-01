from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

import yaml

from src.data.cluster1_batadal import (
    DEFAULT_OUTPUT_ROOT,
    LABEL_COLUMN,
    TIMESTAMP_COLUMN,
    _build_windows,
    _parse_datetime,
    prepare_cluster1_batadal,
)
from src.data.schema_validation import DatasetSchemaError
from src.fl.maincluster import build_flat_federated_clients


FEATURE_COLUMNS = (
    tuple(f"L_T{index}" for index in range(1, 8))
    + tuple(f"F_PU{index}" for index in range(1, 12))
    + ("F_V2",)
    + tuple(f"S_PU{index}" for index in range(1, 12))
    + ("S_V2",)
    + tuple(f"P_J{index:03d}" for index in range(1, 13))
)


def _write_batadal_csv(path: Path, start: datetime, rows: int, positive_rows: set[int]) -> None:
    header = [TIMESTAMP_COLUMN, *FEATURE_COLUMNS, LABEL_COLUMN]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row_index in range(rows):
            timestamp = start + timedelta(hours=row_index)
            values: list[object] = [timestamp.strftime("%d/%m/%y %H")]
            for feature_index, feature_name in enumerate(FEATURE_COLUMNS):
                if feature_name.startswith("S_"):
                    values.append((row_index + feature_index) % 2)
                else:
                    values.append(float(row_index + 1) + feature_index / 100.0)
            values.append(1 if row_index in positive_rows else 0)
            writer.writerow(values)


def _write_config(path: Path, raw_dir: Path, output_root: Path, *, window_length: int = 4) -> None:
    config = {
        "config_version": 1,
        "cluster": {
            "id": 1,
            "key": "C1",
            "dataset_key": "BATADAL",
            "dataset_name": "BATADAL",
            "audit_report": str(output_root / "reports/data_profile_cluster1_batadal.json"),
        },
        "data": {
            "data_root_env_var": "FCFL_DATA_ROOT",
            "default_data_root": "data",
            "current_raw_input_dir": str(raw_dir),
            "current_raw_files": [
                "training_dataset_1.csv",
                "training_dataset_2.csv",
                "test_dataset.csv",
            ],
            "batadal_output_root": str(output_root),
            "training_input_mode": "raw_csv_glob",
            "training_input_glob": str(raw_dir / "*.csv"),
            "schema_consistent_across_files": True,
            "label_column": "ATT_FLAG",
            "label_column_confirmed_from_audit": True,
            "candidate_label_columns_present": ["ATT_FLAG"],
            "timestamp_or_order_columns": ["DATETIME"],
            "excluded_columns": ["ATT_FLAG", "DATETIME"],
            "exclude_if_present": [],
        },
        "partitioning": {
            "candidate_leaf_clients": 12,
            "strategy": "batadal_controlled_emulation",
        },
        "clustering": {
            "fixed_subclusters": 2,
            "fixed_subcluster_ids": ["H1", "H2"],
            "include_attack_ratio_in_descriptor": True,
        },
        "preprocessing": {
            "input_type": "multivariate_time_series",
            "window_length": window_length,
            "stride": 1,
            "window_label_rule": "last_row",
            "validation_cutoff": "2016-11-29 05:00",
        },
        "experiment_defaults": {"proposed_model_family": "cnn1d_bn"},
        "runtime_validation": {
            "require_training_input_to_exist": True,
            "require_schema_consistency_across_files": True,
            "require_label_column_to_exist": True,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_fixture(root: Path) -> tuple[Path, Path]:
    raw_dir = root / "raw" / "batadal"
    output_root = root / "outputs_c1_batadal"
    raw_dir.mkdir(parents=True)
    _write_batadal_csv(raw_dir / "training_dataset_1.csv", datetime(2016, 1, 1), 30, set())
    _write_batadal_csv(
        raw_dir / "training_dataset_2.csv",
        datetime(2016, 11, 28),
        80,
        {index for index in range(4, 76, 2)},
    )
    _write_batadal_csv(
        raw_dir / "test_dataset.csv",
        datetime(2017, 1, 4),
        60,
        {index for index in range(5, 60, 6)},
    )
    config_path = root / "cluster1_batadal.yaml"
    _write_config(config_path, raw_dir, output_root)
    return config_path, output_root


class Cluster1BatadalTests(unittest.TestCase):
    def test_datetime_parsing_uses_day_first_format(self) -> None:
        parsed = _parse_datetime("04/07/16 00", path=Path("x.csv"), row_number=2)
        self.assertEqual(parsed, datetime(2016, 7, 4, 0))

    def test_batadal_schema_validation_requires_att_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, _ = _write_fixture(root)
            raw_file = root / "raw" / "batadal" / "training_dataset_1.csv"
            text = raw_file.read_text(encoding="utf-8").replace(",ATT_FLAG\n", "\n")
            raw_file.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(DatasetSchemaError, "ATT_FLAG"):
                prepare_cluster1_batadal(config_path)

    def test_prepare_batadal_outputs_splits_clients_membership_and_no_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path, output_root = _write_fixture(Path(tmp))
            result = prepare_cluster1_batadal(config_path)
            profile = result["profile"]

            self.assertEqual(profile["dataset"], "BATADAL")
            self.assertEqual(profile["output_root"], str(output_root))
            self.assertEqual(profile["row_counts"]["train"], 59)
            self.assertEqual(profile["row_counts"]["validation"], 51)
            self.assertEqual(profile["row_counts"]["test"], 60)
            self.assertEqual(profile["window_label_rule"], "last_row")
            self.assertFalse(profile["test_leakage_prevention"]["test_dataset_used_for_training"])
            self.assertFalse(profile["test_leakage_prevention"]["test_dataset_used_for_clustering"])
            self.assertFalse(profile["test_leakage_prevention"]["test_dataset_used_for_threshold_tuning"])

            clients = json.loads((output_root / "clients/cluster1_leaf_clients.json").read_text(encoding="utf-8"))
            self.assertEqual(clients["num_leaf_clients"], 12)
            self.assertEqual([client["client_id"] for client in clients["clients"]], [f"C1_L{i:03d}" for i in range(1, 13)])
            self.assertTrue(all(client["train_label_counts"].get("1", 0) > 0 for client in clients["clients"]))

            membership = json.loads((output_root / "clustering/cluster1_memberships.json").read_text(encoding="utf-8"))
            self.assertEqual(membership["n_subclusters"], 2)
            self.assertEqual(membership["fixed_subcluster_ids"], ["H1", "H2"])
            self.assertTrue(membership["frozen"])

            for relative_path in (
                "reports/data_profile_cluster1_batadal.json",
                "reports/file_split_cluster1_batadal.md",
                "reports/client_balance_cluster1_batadal.md",
                "reports/subcluster_profile_cluster1_batadal.md",
                "reports/cluster1_batadal_validation_summary.md",
                "preprocessing/cluster1_batadal_preprocessor.pkl",
            ):
                self.assertTrue((output_root / relative_path).exists(), relative_path)

            built_clients, model_config, data_summary = build_flat_federated_clients(config_path)
            self.assertEqual(len(built_clients), 12)
            self.assertEqual(model_config.input_length, 4)
            self.assertEqual(data_summary["variant"], "cluster1_batadal")
            self.assertFalse(data_summary["test_dataset_used_for_preprocessing_fit"])

    def test_last_row_window_labeling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path, _ = _write_fixture(Path(tmp))
            result = prepare_cluster1_batadal(config_path)
            profile = result["profile"]
            self.assertEqual(profile["window_label_rule"], "last_row")
            self.assertEqual(profile["window_label_counts"]["train"].get("1", 0), 13)

    def test_active_configs_and_matrix_use_batadal_paths_and_ids(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        active_files = [
            repo_root / "configs/cluster1_batadal.yaml",
            repo_root / "configs/baseline_flat.yaml",
            repo_root / "configs/baseline_hierarchical.yaml",
            repo_root / "configs/proposed.yaml",
            repo_root / "docs/EXPERIMENT_MATRIX.csv",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in active_files)
        self.assertIn("BATADAL", combined)
        self.assertIn("A_C1_BATADAL", combined)
        self.assertIn("B_C1_BATADAL", combined)
        self.assertIn("P_C1_BATADAL", combined)
        self.assertIn(str(DEFAULT_OUTPUT_ROOT), combined)
        self.assertNotIn("data/raw/" + "hai_2103", combined)
        self.assertNotIn("HAI" + "_2103", combined)


if __name__ == "__main__":
    unittest.main()
