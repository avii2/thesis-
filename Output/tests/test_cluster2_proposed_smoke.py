from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train_cluster2_proposed import run_cluster2_proposed  # noqa: E402


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(",".join(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_membership(path: Path, *, subclusters: list[tuple[str, list[str]]]) -> None:
    payload = {
        "cluster_id": 2,
        "dataset": "TON IoT Smoke",
        "status": "ok",
        "clustering_method": "AgglomerativeClustering",
        "linkage": "ward",
        "metric": "euclidean",
        "descriptor": "feature_mean_std",
        "descriptor_dim": 12,
        "descriptor_source_split": "train",
        "n_subclusters": len(subclusters),
        "fixed_subcluster_ids": [subcluster_id for subcluster_id, _ in subclusters],
        "frozen": True,
        "membership_hash": "synthetic-cluster2-fedprox",
        "client_metadata_path": "outputs/clients/cluster2_leaf_clients.json",
        "descriptor_scaler_path": "outputs/clustering/cluster2_descriptor_scaler.pkl",
        "membership_file": str(path),
        "reuse_for_experiment_groups": [
            "baseline_uniform_hierarchical",
            "proposed_specialized_hierarchical",
        ],
        "subclusters": [
            {"subcluster_id": subcluster_id, "client_ids": client_ids}
            for subcluster_id, client_ids in subclusters
        ],
        "clients": [
            {"client_id": client_id, "subcluster_id": subcluster_id}
            for subcluster_id, client_ids in subclusters
            for client_id in client_ids
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class Cluster2ProposedSmokeTests(unittest.TestCase):
    def test_cluster2_proposed_runs_and_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            outputs_dir = tmp_path / "outputs"
            configs_dir = tmp_path / "configs"
            raw_root = tmp_path / "raw"
            processed_dir = outputs_dir / "processed"
            clustering_dir = outputs_dir / "clustering"
            configs_dir.mkdir()
            raw_root.mkdir()
            processed_dir.mkdir(parents=True)
            clustering_dir.mkdir(parents=True)

            combined_path = processed_dir / "cluster2_ton_iot_combined.csv"
            for raw_name in (
                "Train_Test_IoT_Fridge.csv",
                "Train_Test_IoT_GPS_Tracker.csv",
                "Train_Test_IoT_Garage_Door.csv",
            ):
                write_csv(raw_root / raw_name, [["date", "time", "label", "type"], ["2026-01-01", "00:00:00", "0", "normal"]])
            rows = [[
                "date",
                "time",
                "door_state",
                "fridge_temperature",
                "label",
                "latitude",
                "longitude",
                "sphone_signal",
                "temp_condition",
                "type",
                "source",
            ]]
            for index in range(150):
                client_block = index // 10
                local_index = index % 10
                if local_index < 7:
                    label = str((client_block + local_index) % 2)
                elif local_index == 7:
                    label = str(client_block % 2)
                elif local_index == 8:
                    label = str((client_block + 1) % 2)
                else:
                    label = str(client_block % 2)
                rows.append(
                    [
                        "2026-01-01",
                        f"00:{index // 60:02d}:{index % 60:02d}",
                        "open" if index % 3 == 0 else "closed",
                        f"{4.0 + (index % 7) * 0.5:.1f}",
                        label,
                        f"{10.0 + index * 0.01:.2f}",
                        f"{20.0 + index * 0.02:.2f}",
                        str(55 + (index % 5)),
                        "cold" if index % 4 < 2 else "warm",
                        "normal" if label == "0" else "scan",
                        "synthetic_device",
                    ]
                )
            write_csv(combined_path, rows)

            cluster2_config = configs_dir / "cluster2_ton_iot.yaml"
            cluster2_config.write_text(
                yaml.safe_dump(
                    {
                        "config_version": 1,
                        "cluster": {
                            "id": 2,
                            "key": "C2",
                            "dataset_key": "TON_IoT_combined_telemetry",
                            "dataset_name": "TON IoT Smoke",
                            "audit_report": "outputs/reports/test_cluster2.json",
                        },
                        "data": {
                            "data_root_env_var": "FCFL_DATA_ROOT",
                            "default_data_root": str(tmp_path),
                            "current_raw_input_dir": str(raw_root),
                            "current_raw_files": [
                                "Train_Test_IoT_Fridge.csv",
                                "Train_Test_IoT_GPS_Tracker.csv",
                                "Train_Test_IoT_Garage_Door.csv",
                            ],
                            "training_input_mode": "combined_processed_csv_required",
                            "training_input_glob": None,
                            "training_input_path": None,
                            "expected_processed_input_path": str(combined_path),
                            "label_column": "label",
                            "label_column_confirmed_from_audit": True,
                            "candidate_label_columns_present": ["label", "type"],
                            "timestamp_or_order_columns": ["date", "time"],
                            "excluded_columns": ["label", "type", "date", "time"],
                            "exclude_if_present": ["source", "device", "id"],
                        },
                        "partitioning": {
                            "candidate_leaf_clients": 15,
                        },
                        "clustering": {
                            "fixed_subclusters": 3,
                            "fixed_subcluster_ids": ["T1", "T2", "T3"],
                        },
                        "preprocessing": {
                            "input_type": "fixed_length_tabular",
                            "windowing": "none",
                        },
                        "runtime_validation": {
                            "require_expected_processed_input_to_exist": True,
                            "require_label_column_to_exist": True,
                            "error_on_missing_expected_processed_input": "TON_IOT_COMBINED_TELEMETRY_REQUIRED",
                            "error_on_training_from_current_raw_per_device_files": "TON_IOT_RAW_PER_DEVICE_FILES_ARE_PROFILE_ONLY",
                            "error_on_missing_label_column": "CONFIGURED_LABEL_COLUMN_MISSING",
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            membership_path = clustering_dir / "cluster2_memberships.json"
            write_membership(
                membership_path,
                subclusters=[
                    ("T1", ["C2_L001", "C2_L002", "C2_L003", "C2_L004", "C2_L005"]),
                    ("T2", ["C2_L006", "C2_L007", "C2_L008", "C2_L009", "C2_L010"]),
                    ("T3", ["C2_L011", "C2_L012", "C2_L013", "C2_L014", "C2_L015"]),
                ],
            )
            membership_before = membership_path.read_text(encoding="utf-8")

            proposed_config = configs_dir / "proposed.yaml"
            proposed_config.write_text(
                yaml.safe_dump(
                    {
                        "config_version": 1,
                        "experiment_group": "proposed_specialized_hierarchical",
                        "training_defaults": {
                            "rounds": 2,
                            "local_epochs": 1,
                            "batch_size": 8,
                            "seeds": [42],
                        },
                        "smoke_test_defaults": {
                            "rounds": 2,
                            "local_epochs": 1,
                            "batch_size": 8,
                            "seed": 42,
                        },
                        "clusters": [
                            {
                                "experiment_id": "P_C2",
                                "cluster_config": str(cluster2_config),
                                "hierarchy": "hierarchical_fixed",
                                "clustering_method": "agglomerative",
                                "n_subclusters": 3,
                                "descriptor": "feature_mean_std",
                                "membership_file": str(membership_path),
                                "model_family": "compact_mlp",
                                "fl_method": "FedProx",
                                "aggregation": "weighted_arithmetic_mean",
                            }
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            old_cwd = Path.cwd()
            os.chdir(tmp_path)
            try:
                result = run_cluster2_proposed(
                    proposed_config_path=proposed_config,
                    smoke_test=True,
                    max_train_examples_per_client=8,
                    max_eval_examples_per_client=8,
                    output_root=outputs_dir,
                )
            finally:
                os.chdir(old_cwd)

            self.assertEqual(result.experiment_id, "P_C2")
            run_dir = outputs_dir / "runs" / "P_C2"
            metrics_csv = outputs_dir / "metrics" / "P_C2_metrics.csv"
            self.assertTrue(run_dir.exists())
            self.assertTrue((run_dir / "round_metrics.csv").exists())
            self.assertTrue((run_dir / "run_summary.json").exists())
            self.assertTrue(metrics_csv.exists())

            with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["experiment_id"], "P_C2")

            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["model_family"], "compact_mlp")
            self.assertEqual(summary["fl_method"], "FedProx")
            self.assertEqual(summary["aggregation"], "weighted_arithmetic_mean")
            self.assertEqual(summary["input_adapter"], "feature_vector_as_sequence")
            self.assertEqual(summary["model_input_layout"], "batch_x_features")
            self.assertTrue(summary["subcluster_layer_used"])
            self.assertFalse(summary["membership_file_changed"])
            self.assertEqual(summary["n_subclusters"], 3)
            self.assertAlmostEqual(float(summary["mu"]), 0.01)
            self.assertEqual(membership_before, membership_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
