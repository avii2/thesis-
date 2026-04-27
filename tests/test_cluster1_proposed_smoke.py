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

from src.train_cluster1_proposed import run_cluster1_proposed  # noqa: E402


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(",".join(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_membership(path: Path, *, subclusters: list[tuple[str, list[str]]]) -> None:
    payload = {
        "cluster_id": 1,
        "dataset": "HAI Smoke",
        "status": "ok",
        "clustering_method": "AgglomerativeClustering",
        "linkage": "ward",
        "metric": "euclidean",
        "descriptor": "feature_mean_std",
        "descriptor_dim": 8,
        "descriptor_source_split": "train",
        "n_subclusters": len(subclusters),
        "fixed_subcluster_ids": [subcluster_id for subcluster_id, _ in subclusters],
        "frozen": True,
        "membership_hash": "synthetic-cluster1-fedbn",
        "client_metadata_path": "outputs/clients/cluster1_leaf_clients.json",
        "descriptor_scaler_path": "outputs/clustering/cluster1_descriptor_scaler.pkl",
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


class Cluster1ProposedSmokeTests(unittest.TestCase):
    def test_cluster1_proposed_runs_and_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            outputs_dir = tmp_path / "outputs"
            configs_dir = tmp_path / "configs"
            raw_root = tmp_path / "raw"
            clustering_dir = outputs_dir / "clustering"
            configs_dir.mkdir()
            raw_root.mkdir()
            clustering_dir.mkdir(parents=True)

            cluster1_dir = raw_root / "cluster1"
            cluster1_dir.mkdir()

            rows = [["time", "sensor_a", "sensor_b", "attack"]]
            attack_indices = {
                10, 11, 18, 19, 26, 27,
                44, 45,
                52, 53,
                70, 71, 78, 79, 86, 87,
                104, 105,
                112, 113,
            }
            for index in range(120):
                label = "1" if index in attack_indices else "0"
                rows.append(
                    [
                        f"2026-01-01 00:{index // 60:02d}:{index % 60:02d}",
                        str(10 + index),
                        str(20 + (index % 9)),
                        label,
                    ]
                )
            write_csv(cluster1_dir / "hai.csv", rows)

            cluster1_config = configs_dir / "cluster1_hai.yaml"
            cluster1_config.write_text(
                yaml.safe_dump(
                    {
                        "config_version": 1,
                        "cluster": {
                            "id": 1,
                            "key": "C1",
                            "dataset_key": "HAI_2103",
                            "dataset_name": "HAI Smoke",
                            "audit_report": "outputs/reports/test_cluster1.json",
                        },
                        "data": {
                            "data_root_env_var": "FCFL_DATA_ROOT",
                            "default_data_root": str(raw_root),
                            "current_raw_input_dir": str(cluster1_dir),
                            "current_raw_files": ["hai.csv"],
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
                            "candidate_leaf_clients": 2,
                        },
                        "preprocessing": {
                            "input_type": "multivariate_time_series",
                            "window_length": 8,
                            "stride": 2,
                            "window_label_rule": "any_positive_row",
                        },
                        "runtime_validation": {
                            "require_training_input_to_exist": True,
                            "require_label_column_to_exist": True,
                            "error_on_missing_label_column": "CONFIGURED_LABEL_COLUMN_MISSING",
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            membership_path = clustering_dir / "cluster1_memberships.json"
            write_membership(
                membership_path,
                subclusters=[("H1", ["C1_L001"]), ("H2", ["C1_L002"])],
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
                                "experiment_id": "P_C1",
                                "cluster_config": str(cluster1_config),
                                "hierarchy": "hierarchical_fixed",
                                "clustering_method": "agglomerative",
                                "n_subclusters": 2,
                                "descriptor": "feature_mean_std",
                                "membership_file": str(membership_path),
                                "model_family": "tcn",
                                "fl_method": "FedBN",
                                "aggregation": "weighted_non_bn_mean",
                                "model_hyperparameters": {
                                    "block_channels": [64, 64, 64],
                                    "hidden_dim": 64,
                                    "dropout": 0.2,
                                },
                                "training_hyperparameters": {
                                    "positive_class_weight_scale": 0.5,
                                },
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
                result = run_cluster1_proposed(
                    proposed_config_path=proposed_config,
                    smoke_test=True,
                    max_train_examples_per_client=8,
                    max_eval_examples_per_client=8,
                    output_root=outputs_dir,
                )
            finally:
                os.chdir(old_cwd)

            self.assertEqual(result.experiment_id, "P_C1")
            run_dir = outputs_dir / "runs" / "P_C1"
            metrics_csv = outputs_dir / "metrics" / "P_C1_metrics.csv"
            self.assertTrue(run_dir.exists())
            self.assertTrue((run_dir / "round_metrics.csv").exists())
            self.assertTrue((run_dir / "run_summary.json").exists())
            self.assertTrue(metrics_csv.exists())

            with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["experiment_id"], "P_C1")

            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["model_family"], "tcn")
            self.assertEqual(summary["fl_method"], "FedBN")
            self.assertEqual(summary["aggregation"], "weighted_non_bn_mean")
            self.assertEqual(summary["tcn_block_channels"], [64, 64, 64])
            self.assertEqual(summary["tcn_hidden_dim"], 64)
            self.assertEqual(summary["tcn_dropout"], 0.2)
            self.assertEqual(summary["positive_class_weight_scale"], 0.5)
            self.assertAlmostEqual(
                summary["positive_class_weight"],
                summary["computed_positive_class_weight"] * 0.5,
            )
            self.assertEqual(summary["input_adapter"], "sliding_window_feature_channels")
            self.assertTrue(summary["subcluster_layer_used"])
            self.assertFalse(summary["membership_file_changed"])
            self.assertEqual(summary["n_subclusters"], 2)
            self.assertEqual(membership_before, membership_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
