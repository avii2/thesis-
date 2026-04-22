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

from src.train_cluster3_proposed import run_cluster3_proposed  # noqa: E402


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(",".join(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_membership(path: Path, *, subclusters: list[tuple[str, list[str]]]) -> None:
    payload = {
        "cluster_id": 3,
        "dataset": "WUSTL Smoke",
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
        "membership_hash": "synthetic-cluster3-scaffold",
        "client_metadata_path": "outputs/clients/cluster3_leaf_clients.json",
        "descriptor_scaler_path": "outputs/clustering/cluster3_descriptor_scaler.pkl",
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


class Cluster3ProposedSmokeTests(unittest.TestCase):
    def test_cluster3_proposed_runs_and_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            outputs_dir = tmp_path / "outputs"
            configs_dir = tmp_path / "configs"
            raw_root = tmp_path / "raw"
            clustering_dir = outputs_dir / "clustering"
            configs_dir.mkdir()
            raw_root.mkdir()
            clustering_dir.mkdir(parents=True)

            raw_path = raw_root / "wustl_iiot_2021.csv"
            rows = [[
                "StartTime",
                "LastTime",
                "SrcAddr",
                "DstAddr",
                "sIpId",
                "dIpId",
                "feat_a",
                "feat_b",
                "feat_c",
                "Target",
                "Traffic",
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
                        str(index),
                        str(index + 1),
                        f"10.0.0.{index % 7}",
                        f"10.0.1.{index % 5}",
                        str(100 + index),
                        str(200 + index),
                        str(index % 11),
                        str((index * 2) % 13),
                        str((index * 3) % 17),
                        label,
                        "dos" if label == "1" else "normal",
                    ]
                )
            write_csv(raw_path, rows)

            cluster3_config = configs_dir / "cluster3_wustl.yaml"
            cluster3_config.write_text(
                yaml.safe_dump(
                    {
                        "config_version": 1,
                        "cluster": {
                            "id": 3,
                            "key": "C3",
                            "dataset_key": "WUSTL_IIOT_2021",
                            "dataset_name": "WUSTL Smoke",
                            "audit_report": "outputs/reports/test_cluster3.json",
                        },
                        "data": {
                            "data_root_env_var": "FCFL_DATA_ROOT",
                            "default_data_root": str(tmp_path),
                            "current_raw_input_dir": str(raw_root),
                            "current_raw_files": ["wustl_iiot_2021.csv"],
                            "training_input_mode": "single_csv",
                            "training_input_glob": None,
                            "training_input_path": str(raw_path),
                            "expected_processed_input_path": None,
                            "label_column": "Target",
                            "label_column_confirmed_from_audit": True,
                            "candidate_label_columns_present": ["Target", "Traffic"],
                            "auxiliary_label_like_columns": ["Traffic"],
                            "timestamp_or_order_columns": ["StartTime", "LastTime"],
                            "excluded_columns": [
                                "Target",
                                "Traffic",
                                "StartTime",
                                "LastTime",
                                "SrcAddr",
                                "DstAddr",
                                "sIpId",
                                "dIpId",
                            ],
                            "exclude_if_present": ["attack_type", "traffic_class", "time", "date"],
                        },
                        "partitioning": {
                            "candidate_leaf_clients": 15,
                        },
                        "clustering": {
                            "fixed_subclusters": 3,
                            "fixed_subcluster_ids": ["W1", "W2", "W3"],
                        },
                        "preprocessing": {
                            "input_type": "fixed_length_network_flow_vector",
                            "windowing": "none",
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

            membership_path = clustering_dir / "cluster3_memberships.json"
            write_membership(
                membership_path,
                subclusters=[
                    ("W1", ["C3_L001", "C3_L002", "C3_L003", "C3_L004", "C3_L005"]),
                    ("W2", ["C3_L006", "C3_L007", "C3_L008", "C3_L009", "C3_L010"]),
                    ("W3", ["C3_L011", "C3_L012", "C3_L013", "C3_L014", "C3_L015"]),
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
                                "experiment_id": "P_C3",
                                "cluster_config": str(cluster3_config),
                                "hierarchy": "hierarchical_fixed",
                                "clustering_method": "agglomerative",
                                "n_subclusters": 3,
                                "descriptor": "feature_mean_std",
                                "membership_file": str(membership_path),
                                "model_family": "cnn1d",
                                "fl_method": "SCAFFOLD",
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
                result = run_cluster3_proposed(
                    proposed_config_path=proposed_config,
                    smoke_test=True,
                    max_train_examples_per_client=8,
                    max_eval_examples_per_client=8,
                    output_root=outputs_dir,
                )
            finally:
                os.chdir(old_cwd)

            self.assertEqual(result.experiment_id, "P_C3")
            run_dir = outputs_dir / "runs" / "P_C3"
            metrics_csv = outputs_dir / "metrics" / "P_C3_metrics.csv"
            self.assertTrue(run_dir.exists())
            self.assertTrue((run_dir / "round_metrics.csv").exists())
            self.assertTrue((run_dir / "run_summary.json").exists())
            self.assertTrue(metrics_csv.exists())

            with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["experiment_id"], "P_C3")

            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["model_family"], "cnn1d")
            self.assertEqual(summary["fl_method"], "SCAFFOLD")
            self.assertEqual(summary["aggregation"], "weighted_arithmetic_mean")
            self.assertEqual(summary["input_adapter"], "feature_vector_as_sequence")
            self.assertTrue(summary["subcluster_layer_used"])
            self.assertFalse(summary["membership_file_changed"])
            self.assertEqual(summary["n_subclusters"], 3)
            self.assertGreater(int(summary["control_variate_bytes"]), 0)
            self.assertGreater(float(summary["final_server_control_variate_l2_norm"]), 0.0)
            self.assertGreater(int(summary["num_nonzero_client_control_variates"]), 0)
            self.assertEqual(membership_before, membership_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
