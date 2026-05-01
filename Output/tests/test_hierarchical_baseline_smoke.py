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

from src.train_hierarchical_baseline import run_hierarchical_baseline_experiments  # noqa: E402


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(",".join(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_membership(
    path: Path,
    *,
    cluster_id: int,
    dataset: str,
    subclusters: list[tuple[str, list[str]]],
) -> None:
    payload = {
        "cluster_id": cluster_id,
        "dataset": dataset,
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
        "membership_hash": f"synthetic-cluster-{cluster_id}",
        "client_metadata_path": f"outputs/clients/cluster{cluster_id}_leaf_clients.json",
        "descriptor_scaler_path": f"outputs/clustering/cluster{cluster_id}_descriptor_scaler.pkl",
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


def cluster_config(
    *,
    cluster_id: int,
    dataset_name: str,
    raw_dir: Path,
    raw_files: list[str],
    label_column: str,
    excluded_columns: list[str],
    timestamp_or_order_columns: list[str],
    training_input_mode: str,
    training_input_path: str | None,
    expected_processed_input_path: str | None,
    candidate_leaf_clients: int,
    preprocessing: dict[str, object],
) -> dict[str, object]:
    return {
        "config_version": 1,
        "cluster": {
            "id": cluster_id,
            "key": f"C{cluster_id}",
            "dataset_key": f"TEST_CLUSTER_{cluster_id}",
            "dataset_name": dataset_name,
            "audit_report": f"outputs/reports/test_cluster_{cluster_id}.json",
        },
        "data": {
            "data_root_env_var": "FCFL_DATA_ROOT",
            "default_data_root": str(raw_dir.parent),
            "current_raw_input_dir": str(raw_dir),
            "current_raw_files": raw_files,
            "training_input_mode": training_input_mode,
            "training_input_glob": None,
            "training_input_path": training_input_path,
            "expected_processed_input_path": expected_processed_input_path,
            "label_column": label_column,
            "label_column_confirmed_from_audit": True,
            "candidate_label_columns_present": [label_column],
            "timestamp_or_order_columns": timestamp_or_order_columns,
            "excluded_columns": excluded_columns,
            "exclude_if_present": [],
        },
        "partitioning": {
            "candidate_leaf_clients": candidate_leaf_clients,
        },
        "preprocessing": preprocessing,
        "runtime_validation": {
            "require_training_input_to_exist": True,
            "require_label_column_to_exist": True,
            "error_on_missing_label_column": "CONFIGURED_LABEL_COLUMN_MISSING",
            "error_on_missing_expected_processed_input": "EXPECTED_PROCESSED_INPUT_REQUIRED",
        },
    }


class HierarchicalBaselineSmokeTests(unittest.TestCase):
    def test_hierarchical_baseline_runs_all_three_clusters_and_preserves_frozen_memberships(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            outputs_dir = tmp_path / "outputs"
            configs_dir = tmp_path / "configs"
            raw_root = tmp_path / "raw"
            processed_root = tmp_path / "processed"
            clustering_dir = outputs_dir / "clustering"
            configs_dir.mkdir()
            raw_root.mkdir()
            processed_root.mkdir()
            clustering_dir.mkdir(parents=True)

            cluster1_dir = raw_root / "cluster1"
            cluster2_dir = raw_root / "cluster2"
            cluster3_dir = raw_root / "cluster3"
            cluster1_dir.mkdir()
            cluster2_dir.mkdir()
            cluster3_dir.mkdir()

            cluster1_rows = [["time", "sensor_a", "sensor_b", "attack"]]
            for index in range(48):
                label = "1" if index in {6, 10, 18, 22, 30, 34, 42, 46} else "0"
                cluster1_rows.append(
                    [
                        f"2026-01-01 00:00:{index:02d}",
                        str(10 + index),
                        str(20 + (index % 7)),
                        label,
                    ]
                )
            write_csv(cluster1_dir / "hai.csv", cluster1_rows)

            cluster2_rows = [["date", "time", "feat_a", "feat_b", "label", "type"]]
            for index in range(36):
                label = "1" if index % 4 in {1, 2} else "0"
                cluster2_rows.append(
                    [
                        "01-Jan-26",
                        f"00:00:{index:02d}",
                        str(index),
                        str(index % 5),
                        label,
                        "scan" if label == "1" else "normal",
                    ]
                )
            combined_path = processed_root / "cluster2_combined.csv"
            write_csv(combined_path, cluster2_rows)

            cluster3_rows = [[
                "StartTime",
                "LastTime",
                "SrcAddr",
                "DstAddr",
                "sIpId",
                "dIpId",
                "feat_a",
                "feat_b",
                "Target",
                "Traffic",
            ]]
            for index in range(36):
                label = "1" if index % 4 in {1, 2} else "0"
                cluster3_rows.append(
                    [
                        str(index),
                        str(index + 1),
                        f"10.0.0.{index % 4}",
                        f"10.0.1.{index % 4}",
                        str(index),
                        str(index + 100),
                        str(index % 11),
                        str(index % 3),
                        label,
                        "dos" if label == "1" else "normal",
                    ]
                )
            write_csv(cluster3_dir / "wustl.csv", cluster3_rows)

            cluster1_config = configs_dir / "cluster1.yaml"
            cluster2_config = configs_dir / "cluster2.yaml"
            cluster3_config = configs_dir / "cluster3.yaml"
            cluster1_config.write_text(
                yaml.safe_dump(
                    cluster_config(
                        cluster_id=1,
                        dataset_name="HAI Smoke",
                        raw_dir=cluster1_dir,
                        raw_files=["hai.csv"],
                        label_column="attack",
                        excluded_columns=["attack", "time"],
                        timestamp_or_order_columns=["time"],
                        training_input_mode="raw_csv_glob",
                        training_input_path=None,
                        expected_processed_input_path=None,
                        candidate_leaf_clients=2,
                        preprocessing={
                            "input_type": "multivariate_time_series",
                            "window_length": 4,
                            "stride": 2,
                            "window_label_rule": "any_positive_row",
                        },
                    ),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            cluster2_config.write_text(
                yaml.safe_dump(
                    cluster_config(
                        cluster_id=2,
                        dataset_name="TON Smoke",
                        raw_dir=cluster2_dir,
                        raw_files=["profile_only.csv"],
                        label_column="label",
                        excluded_columns=["label", "type", "date", "time"],
                        timestamp_or_order_columns=["date", "time"],
                        training_input_mode="combined_processed_csv_required",
                        training_input_path=None,
                        expected_processed_input_path=str(combined_path),
                        candidate_leaf_clients=3,
                        preprocessing={
                            "input_type": "fixed_length_tabular",
                            "windowing": "none",
                        },
                    ),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            cluster3_config.write_text(
                yaml.safe_dump(
                    cluster_config(
                        cluster_id=3,
                        dataset_name="WUSTL Smoke",
                        raw_dir=cluster3_dir,
                        raw_files=["wustl.csv"],
                        label_column="Target",
                        excluded_columns=[
                            "Target",
                            "Traffic",
                            "StartTime",
                            "LastTime",
                            "SrcAddr",
                            "DstAddr",
                            "sIpId",
                            "dIpId",
                        ],
                        timestamp_or_order_columns=["StartTime", "LastTime"],
                        training_input_mode="single_csv",
                        training_input_path=str(cluster3_dir / "wustl.csv"),
                        expected_processed_input_path=None,
                        candidate_leaf_clients=3,
                        preprocessing={
                            "input_type": "fixed_length_network_flow_vector",
                            "windowing": "none",
                        },
                    ),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            membership_c1 = clustering_dir / "cluster1_memberships.json"
            membership_c2 = clustering_dir / "cluster2_memberships.json"
            membership_c3 = clustering_dir / "cluster3_memberships.json"
            write_membership(
                membership_c1,
                cluster_id=1,
                dataset="HAI Smoke",
                subclusters=[("H1", ["C1_L001"]), ("H2", ["C1_L002"])],
            )
            write_membership(
                membership_c2,
                cluster_id=2,
                dataset="TON Smoke",
                subclusters=[
                    ("T1", ["C2_L001"]),
                    ("T2", ["C2_L002"]),
                    ("T3", ["C2_L003"]),
                ],
            )
            write_membership(
                membership_c3,
                cluster_id=3,
                dataset="WUSTL Smoke",
                subclusters=[
                    ("W1", ["C3_L001"]),
                    ("W2", ["C3_L002"]),
                    ("W3", ["C3_L003"]),
                ],
            )
            before_memberships = {
                "B_C1": membership_c1.read_text(encoding="utf-8"),
                "B_C2": membership_c2.read_text(encoding="utf-8"),
                "B_C3": membership_c3.read_text(encoding="utf-8"),
            }

            baseline_config = configs_dir / "baseline_hierarchical.yaml"
            baseline_config.write_text(
                yaml.safe_dump(
                    {
                        "config_version": 1,
                        "experiment_group": "baseline_uniform_hierarchical",
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
                                "experiment_id": "B_C1",
                                "cluster_config": str(cluster1_config),
                                "hierarchy": "hierarchical_fixed",
                                "clustering_method": "agglomerative",
                                "n_subclusters": 2,
                                "descriptor": "feature_mean_std",
                                "membership_file": str(membership_c1),
                                "model_family": "cnn1d",
                                "fl_method": "FedAvg",
                                "aggregation": "weighted_arithmetic_mean",
                            },
                            {
                                "experiment_id": "B_C2",
                                "cluster_config": str(cluster2_config),
                                "hierarchy": "hierarchical_fixed",
                                "clustering_method": "agglomerative",
                                "n_subclusters": 3,
                                "descriptor": "feature_mean_std",
                                "membership_file": str(membership_c2),
                                "model_family": "cnn1d",
                                "fl_method": "FedAvg",
                                "aggregation": "weighted_arithmetic_mean",
                            },
                            {
                                "experiment_id": "B_C3",
                                "cluster_config": str(cluster3_config),
                                "hierarchy": "hierarchical_fixed",
                                "clustering_method": "agglomerative",
                                "n_subclusters": 3,
                                "descriptor": "feature_mean_std",
                                "membership_file": str(membership_c3),
                                "model_family": "cnn1d",
                                "fl_method": "FedAvg",
                                "aggregation": "weighted_arithmetic_mean",
                            },
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            old_cwd = Path.cwd()
            os.chdir(tmp_path)
            try:
                results = run_hierarchical_baseline_experiments(
                    baseline_config,
                    smoke_test=True,
                    max_train_examples_per_client=8,
                    max_eval_examples_per_client=8,
                    output_root=outputs_dir,
                )
            finally:
                os.chdir(old_cwd)

            self.assertEqual([result.experiment_id for result in results], ["B_C1", "B_C2", "B_C3"])
            for experiment_id in ("B_C1", "B_C2", "B_C3"):
                run_dir = outputs_dir / "runs" / experiment_id
                metrics_csv = outputs_dir / "metrics" / f"{experiment_id}_metrics.csv"
                self.assertTrue(run_dir.exists(), experiment_id)
                self.assertTrue((run_dir / "round_metrics.csv").exists(), experiment_id)
                self.assertTrue((run_dir / "run_summary.json").exists(), experiment_id)
                self.assertTrue(metrics_csv.exists(), experiment_id)

                with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["experiment_id"], experiment_id)
                self.assertEqual(rows[0]["hierarchy"], "hierarchical_fixed")

                summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["hierarchy"], "hierarchical_fixed")
                self.assertTrue(summary["subcluster_layer_used"])
                self.assertFalse(summary["membership_file_changed"])
                self.assertIsNotNone(summary["membership_file_used"])

            summary_c1 = json.loads((outputs_dir / "runs" / "B_C1" / "run_summary.json").read_text(encoding="utf-8"))
            summary_c2 = json.loads((outputs_dir / "runs" / "B_C2" / "run_summary.json").read_text(encoding="utf-8"))
            summary_c3 = json.loads((outputs_dir / "runs" / "B_C3" / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary_c1["input_adapter"], "sliding_window_feature_channels")
            self.assertEqual(summary_c2["input_adapter"], "feature_vector_as_sequence")
            self.assertEqual(summary_c3["input_adapter"], "feature_vector_as_sequence")
            self.assertEqual(summary_c1["n_subclusters"], 2)
            self.assertEqual(summary_c2["n_subclusters"], 3)
            self.assertEqual(summary_c3["n_subclusters"], 3)
            self.assertEqual(before_memberships["B_C1"], membership_c1.read_text(encoding="utf-8"))
            self.assertEqual(before_memberships["B_C2"], membership_c2.read_text(encoding="utf-8"))
            self.assertEqual(before_memberships["B_C3"], membership_c3.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
