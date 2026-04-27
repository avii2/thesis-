from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train_ablation import (  # noqa: E402
    run_cluster1_fedavg_tcn_ablation,
    run_cluster2_fedavg_mlp_ablation,
    run_cluster3_fedavg_cnn1d_ablation,
)


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _write_membership(
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
        "membership_hash": f"synthetic-ablation-cluster-{cluster_id}",
        "client_metadata_path": f"outputs/clients/cluster{cluster_id}_leaf_clients.json",
        "descriptor_scaler_path": f"outputs/clustering/cluster{cluster_id}_descriptor_scaler.pkl",
        "membership_file": str(path),
        "reuse_for_experiment_groups": [
            "baseline_uniform_hierarchical",
            "proposed_specialized_hierarchical",
            "ablation_fl_method",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _cluster_config(
    *,
    cluster_id: int,
    dataset_name: str,
    raw_dir: Path,
    raw_files: list[str],
    label_column: str,
    excluded_columns: list[str],
    timestamp_or_order_columns: list[str],
    training_input_mode: str,
    training_input_path: Path | None,
    expected_processed_input_path: Path | None,
    candidate_leaf_clients: int,
    preprocessing: dict[str, object],
    audit_report: Path,
) -> dict[str, object]:
    return {
        "config_version": 1,
        "cluster": {
            "id": cluster_id,
            "key": f"C{cluster_id}",
            "dataset_key": f"TEST_CLUSTER_{cluster_id}",
            "dataset_name": dataset_name,
            "audit_report": str(audit_report),
        },
        "data": {
            "data_root_env_var": "FCFL_DATA_ROOT",
            "default_data_root": str(raw_dir.parent),
            "current_raw_input_dir": str(raw_dir),
            "current_raw_files": raw_files,
            "training_input_mode": training_input_mode,
            "training_input_glob": str(raw_dir / "*.csv") if training_input_mode == "raw_csv_glob" else None,
            "training_input_path": str(training_input_path) if training_input_path is not None else None,
            "expected_processed_input_path": str(expected_processed_input_path)
            if expected_processed_input_path is not None
            else None,
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
        "clustering": {
            "fixed_subclusters": 2 if cluster_id == 1 else 3,
        },
        "preprocessing": preprocessing,
        "runtime_validation": {
            "require_training_input_to_exist": training_input_path is not None or training_input_mode == "raw_csv_glob",
            "require_expected_processed_input_to_exist": expected_processed_input_path is not None,
            "require_label_column_to_exist": True,
            "error_on_missing_label_column": "CONFIGURED_LABEL_COLUMN_MISSING",
            "error_on_missing_expected_processed_input": "EXPECTED_PROCESSED_INPUT_REQUIRED",
        },
    }


def _ablation_config(
    *,
    path: Path,
    ablation_id: str,
    comparison_id: str,
    experiment_id: str,
    run_source: str,
    cluster_id: int,
    cluster_config: Path,
    membership_file: Path,
    n_subclusters: int,
    model_family: str,
    learning_rate: float,
    treatment_id: str,
) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "config_version": 1,
                "ablation_id": ablation_id,
                "training_defaults": {
                    "rounds": 1,
                    "local_epochs": 1,
                    "batch_size": 4,
                    "seeds": [42],
                },
                "smoke_test_defaults": {
                    "rounds": 1,
                    "local_epochs": 1,
                    "batch_size": 4,
                    "seed": 42,
                },
                "comparisons": [
                    {
                        "comparison_id": comparison_id,
                        "control": {
                            "experiment_id": experiment_id,
                            "run_source": run_source,
                            "cluster_id": cluster_id,
                            "cluster_config": str(cluster_config),
                            "membership_file": str(membership_file),
                            "hierarchy": "hierarchical_fixed",
                            "clustering_method": "agglomerative",
                            "n_subclusters": n_subclusters,
                            "descriptor": "feature_mean_std",
                            "model_family": model_family,
                            "fl_method": "FedAvg",
                            "aggregation": "weighted_arithmetic_mean",
                            "learning_rate": learning_rate,
                        },
                        "treatment": {
                            "experiment_id": treatment_id,
                            "run_source": "standard",
                        },
                    }
                ],
                "runtime_validation": {
                    "reuse_frozen_memberships": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


class AblationSmokeExecutionTests(unittest.TestCase):
    def test_custom_ablation_controls_run_smoke_and_preserve_memberships(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            outputs = tmp_path / "outputs"
            configs = tmp_path / "configs"
            raw_root = tmp_path / "raw"
            processed = outputs / "processed"
            clustering = outputs / "clustering"
            reports = outputs / "reports"
            configs.mkdir()
            raw_root.mkdir()
            processed.mkdir(parents=True)
            clustering.mkdir(parents=True)
            reports.mkdir(parents=True)

            c1_raw = raw_root / "c1"
            c1_raw.mkdir()
            c1_rows: list[list[object]] = [["time", "sensor_a", "sensor_b", "attack"]]
            for index in range(96):
                label = 1 if index % 48 in {5, 6, 36, 42} else 0
                c1_rows.append([f"2026-01-01 00:{index // 60:02d}:{index % 60:02d}", index, index % 9, label])
            _write_csv(c1_raw / "hai.csv", c1_rows)

            c2_raw = raw_root / "c2"
            c2_raw.mkdir()
            for raw_name in (
                "Train_Test_IoT_Fridge.csv",
                "Train_Test_IoT_GPS_Tracker.csv",
                "Train_Test_IoT_Garage_Door.csv",
            ):
                _write_csv(c2_raw / raw_name, [["date", "time", "label", "type"], ["2026-01-01", "00:00:00", 0, "normal"]])
            c2_processed = processed / "cluster2_combined.csv"
            c2_rows: list[list[object]] = [["date", "time", "feat_a", "feat_b", "label", "type", "source"]]
            for index in range(90):
                label = 1 if index % 4 in {1, 2} else 0
                c2_rows.append(["2026-01-01", f"00:00:{index:02d}", index % 11, index % 5, label, "scan" if label else "normal", "synthetic"])
            _write_csv(c2_processed, c2_rows)

            c3_raw = raw_root / "c3"
            c3_raw.mkdir()
            c3_rows: list[list[object]] = [[
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
            for index in range(90):
                label = 1 if index % 4 in {1, 2} else 0
                c3_rows.append([index, index + 1, "10.0.0.1", "10.0.0.2", index, index + 100, index % 13, index % 7, label, "dos" if label else "normal"])
            _write_csv(c3_raw / "wustl.csv", c3_rows)

            c1_config = configs / "cluster1.yaml"
            c2_config = configs / "cluster2.yaml"
            c3_config = configs / "cluster3.yaml"
            c1_config.write_text(
                yaml.safe_dump(
                    _cluster_config(
                        cluster_id=1,
                        dataset_name="HAI Smoke",
                        raw_dir=c1_raw,
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
                        audit_report=reports / "c1.json",
                    ),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            c2_config.write_text(
                yaml.safe_dump(
                    _cluster_config(
                        cluster_id=2,
                        dataset_name="TON Smoke",
                        raw_dir=c2_raw,
                        raw_files=[
                            "Train_Test_IoT_Fridge.csv",
                            "Train_Test_IoT_GPS_Tracker.csv",
                            "Train_Test_IoT_Garage_Door.csv",
                        ],
                        label_column="label",
                        excluded_columns=["label", "type", "date", "time"],
                        timestamp_or_order_columns=["date", "time"],
                        training_input_mode="combined_processed_csv_required",
                        training_input_path=None,
                        expected_processed_input_path=c2_processed,
                        candidate_leaf_clients=3,
                        preprocessing={"input_type": "fixed_length_tabular", "windowing": "none"},
                        audit_report=reports / "c2.json",
                    ),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            c3_config.write_text(
                yaml.safe_dump(
                    _cluster_config(
                        cluster_id=3,
                        dataset_name="WUSTL Smoke",
                        raw_dir=c3_raw,
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
                        training_input_path=c3_raw / "wustl.csv",
                        expected_processed_input_path=None,
                        candidate_leaf_clients=3,
                        preprocessing={"input_type": "fixed_length_network_flow_vector", "windowing": "none"},
                        audit_report=reports / "c3.json",
                    ),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            c1_membership = clustering / "cluster1_memberships.json"
            c2_membership = clustering / "cluster2_memberships.json"
            c3_membership = clustering / "cluster3_memberships.json"
            _write_membership(c1_membership, cluster_id=1, dataset="HAI Smoke", subclusters=[("H1", ["C1_L001"]), ("H2", ["C1_L002"])])
            _write_membership(c2_membership, cluster_id=2, dataset="TON Smoke", subclusters=[("T1", ["C2_L001"]), ("T2", ["C2_L002"]), ("T3", ["C2_L003"])])
            _write_membership(c3_membership, cluster_id=3, dataset="WUSTL Smoke", subclusters=[("W1", ["C3_L001"]), ("W2", ["C3_L002"]), ("W3", ["C3_L003"])])
            membership_before = {
                c1_membership: c1_membership.read_text(encoding="utf-8"),
                c2_membership: c2_membership.read_text(encoding="utf-8"),
                c3_membership: c3_membership.read_text(encoding="utf-8"),
            }

            c1_ablation = _ablation_config(
                path=configs / "ab_c1.yaml",
                ablation_id="ablation_cluster1_fedavg_vs_fedbn",
                comparison_id="ablation_cluster1_fedavg_vs_fedbn",
                experiment_id="AB_C1_FEDAVG_TCN",
                run_source="custom_cluster1_fedavg_tcn",
                cluster_id=1,
                cluster_config=c1_config,
                membership_file=c1_membership,
                n_subclusters=2,
                model_family="tcn",
                learning_rate=0.01,
                treatment_id="P_C1",
            )
            c2_ablation = _ablation_config(
                path=configs / "ab_c2.yaml",
                ablation_id="ablation_cluster2_fedavg_vs_fedprox",
                comparison_id="ablation_cluster2_fedavg_vs_fedprox",
                experiment_id="AB_C2_FEDAVG_MLP",
                run_source="custom_cluster2_fedavg_mlp",
                cluster_id=2,
                cluster_config=c2_config,
                membership_file=c2_membership,
                n_subclusters=3,
                model_family="compact_mlp",
                learning_rate=0.001,
                treatment_id="P_C2",
            )
            c3_ablation = _ablation_config(
                path=configs / "ab_c3.yaml",
                ablation_id="ablation_cluster3_fedavg_vs_scaffold",
                comparison_id="ablation_cluster3_fedavg_vs_scaffold",
                experiment_id="AB_C3_FEDAVG_CNN1D",
                run_source="custom_cluster3_fedavg_cnn1d",
                cluster_id=3,
                cluster_config=c3_config,
                membership_file=c3_membership,
                n_subclusters=3,
                model_family="cnn1d",
                learning_rate=0.05,
                treatment_id="P_C3",
            )

            results = [
                run_cluster1_fedavg_tcn_ablation(
                    c1_ablation,
                    smoke_test=True,
                    output_root=outputs,
                    max_train_examples_per_client=12,
                    max_eval_examples_per_client=8,
                ),
                run_cluster2_fedavg_mlp_ablation(
                    c2_ablation,
                    smoke_test=True,
                    output_root=outputs,
                    max_train_examples_per_client=12,
                    max_eval_examples_per_client=8,
                ),
                run_cluster3_fedavg_cnn1d_ablation(
                    c3_ablation,
                    smoke_test=True,
                    output_root=outputs,
                    max_train_examples_per_client=12,
                    max_eval_examples_per_client=8,
                ),
            ]

            for result in results:
                self.assertTrue(result.summary_path.exists())
                self.assertTrue(result.round_metrics_path.exists())
                self.assertTrue(result.metrics_csv_path.exists())
                self.assertEqual(result.summary["hierarchy"], "hierarchical_fixed")
                self.assertEqual(result.summary["fl_method"], "FedAvg")
                self.assertEqual(result.summary["aggregation"], "weighted_arithmetic_mean")
                self.assertIn("prediction_outputs", result.summary)

            self.assertEqual(results[0].summary["model_family"], "tcn")
            self.assertEqual(results[1].summary["model_family"], "compact_mlp")
            self.assertEqual(results[2].summary["model_family"], "cnn1d")
            self.assertIn("positive_class_weight", results[1].summary)

            for membership_path, before in membership_before.items():
                self.assertEqual(membership_path.read_text(encoding="utf-8"), before)


if __name__ == "__main__":
    unittest.main()
