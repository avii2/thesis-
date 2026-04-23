from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ledger.metadata_schema import canonical_sha256  # noqa: E402
from src.train import SUPPORTED_EXPERIMENT_IDS, load_config_registry, load_experiment_matrix, run_experiments  # noqa: E402


@dataclass(frozen=True)
class FakeRunResult:
    experiment_id: str
    cluster_id: int
    dataset: str
    output_dir: Path
    summary_path: Path
    round_metrics_path: Path
    metrics_csv_path: Path
    summary: dict[str, object]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fake_ablation_result(
    output_root: Path,
    *,
    experiment_id: str,
    cluster_id: int,
    dataset: str,
    model_family: str,
    fl_method: str,
    aggregation: str,
) -> FakeRunResult:
    run_dir = output_root / "runs" / experiment_id
    round_metrics_path = run_dir / "round_metrics.csv"
    summary_path = run_dir / "run_summary.json"
    metrics_csv_path = output_root / "metrics" / f"{experiment_id}_metrics.csv"
    run_dir.mkdir(parents=True, exist_ok=True)

    round_rows = [
        {
            "round": 1,
            "train_loss_local_mean": 0.53,
            "train_accuracy": 0.70,
            "train_f1": 0.61,
            "validation_loss": 0.48,
            "validation_accuracy": 0.72,
            "validation_f1": 0.58,
            "test_loss": 0.46,
            "test_accuracy": 0.71,
            "test_f1": 0.57,
            "communication_cost_bytes": 2048,
        },
        {
            "round": 2,
            "train_loss_local_mean": 0.41,
            "train_accuracy": 0.77,
            "train_f1": 0.69,
            "validation_loss": 0.37,
            "validation_accuracy": 0.78,
            "validation_f1": 0.66,
            "test_loss": 0.35,
            "test_accuracy": 0.76,
            "test_f1": 0.64,
            "communication_cost_bytes": 2048,
        },
    ]
    _write_csv(round_metrics_path, round_rows)

    _write_csv(
        metrics_csv_path,
        [
            {
                "experiment_id": experiment_id,
                "cluster_id": cluster_id,
                "dataset": dataset,
                "hierarchy": "hierarchical_fixed",
                "model_family": model_family,
                "fl_method": fl_method,
                "aggregation": aggregation,
                "clustering_method": "agglomerative",
                "membership_hash": canonical_sha256({"experiment_id": experiment_id, "cluster_id": cluster_id}),
                "input_adapter": "sliding_window_feature_channels" if cluster_id == 1 else "feature_vector_as_sequence",
                "num_leaf_clients": 4,
                "n_subclusters": 2 if cluster_id == 1 else 3,
                "rounds": 2,
                "best_validation_round": 2,
                "best_validation_f1": 0.66,
                "test_accuracy": 0.76,
                "test_precision": 0.71,
                "test_recall": 0.62,
                "test_f1": 0.64,
                "test_auroc": 0.75,
                "test_pr_auc": 0.70,
                "test_fpr": 0.17,
                "test_confusion_matrix": json.dumps([[17, 3], [5, 9]]),
                "communication_cost_per_round_bytes": 2048,
                "total_communication_cost_bytes": 4096,
                "wall_clock_training_seconds": 1.3,
            }
        ],
    )

    summary: dict[str, object] = {
        "experiment_id": experiment_id,
        "cluster_id": cluster_id,
        "dataset": dataset,
        "hierarchy": "hierarchical_fixed",
        "subcluster_layer_used": True,
        "membership_hash": canonical_sha256({"experiment_id": experiment_id, "cluster_id": cluster_id}),
        "membership_file_used": f"outputs/clustering/cluster{cluster_id}_memberships.json",
        "model_family": model_family,
        "fl_method": fl_method,
        "aggregation": aggregation,
        "clustering_method": "agglomerative",
        "input_adapter": "sliding_window_feature_channels" if cluster_id == 1 else "feature_vector_as_sequence",
        "num_leaf_clients": 4,
        "n_subclusters": 2 if cluster_id == 1 else 3,
        "best_validation_round": 2,
        "best_validation_f1": 0.66,
        "best_round_train_metrics": {"f1": 0.69},
        "best_round_validation_metrics": {"f1": 0.66},
        "best_round_test_metrics": {
            "accuracy": 0.76,
            "precision": 0.71,
            "recall": 0.62,
            "f1": 0.64,
            "auroc": 0.75,
            "pr_auc": 0.70,
            "fpr": 0.17,
            "false_positive_rate": 0.17,
            "confusion_matrix": [[17, 3], [5, 9]],
        },
        "best_round_subcluster_train_sample_counts": {"S1": 8, "S2": 8} if cluster_id == 1 else {"S1": 8, "S2": 8, "S3": 8},
        "best_round_subcluster_client_counts": {"S1": 1, "S2": 1} if cluster_id == 1 else {"S1": 1, "S2": 1, "S3": 1},
        "data_summary": {
            "client_train_sample_counts": {
                f"C{cluster_id}_L001": 8,
                f"C{cluster_id}_L002": 8,
                f"C{cluster_id}_L003": 8,
                f"C{cluster_id}_L004": 8,
            }
        },
        "round_metrics_path": str(round_metrics_path),
        "metrics_csv_path": str(metrics_csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return FakeRunResult(
        experiment_id=experiment_id,
        cluster_id=cluster_id,
        dataset=dataset,
        output_dir=run_dir,
        summary_path=summary_path,
        round_metrics_path=round_metrics_path,
        metrics_csv_path=metrics_csv_path,
        summary=summary,
    )


class AblationExperimentRegistryTests(unittest.TestCase):
    def test_all_ablation_experiment_ids_are_supported_and_configured(self) -> None:
        matrix = load_experiment_matrix(REPO_ROOT / "docs" / "EXPERIMENT_MATRIX.csv")
        registry = load_config_registry()
        for experiment_id in (
            "AB_C1_FEDAVG_TCN",
            "AB_C2_FEDAVG_MLP",
            "AB_C3_FEDAVG_CNN1D",
        ):
            self.assertIn(experiment_id, SUPPORTED_EXPERIMENT_IDS)
            self.assertIn(experiment_id, matrix)
            self.assertIn(experiment_id, registry)
            self.assertEqual(matrix[experiment_id].run_category, "ablation_fl_method")
            self.assertEqual(registry[experiment_id].experiment_group, "ablation_fl_method")
            self.assertTrue(registry[experiment_id].cluster_config_path.exists())

    def test_smoke_mode_creates_metrics_ledger_and_plot_outputs_for_all_ablation_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "outputs"

            def fake_ab_c1(**kwargs: object) -> FakeRunResult:
                del kwargs
                return _fake_ablation_result(
                    output_root,
                    experiment_id="AB_C1_FEDAVG_TCN",
                    cluster_id=1,
                    dataset="HAI 21.03",
                    model_family="tcn",
                    fl_method="FedAvg",
                    aggregation="weighted_arithmetic_mean",
                )

            def fake_ab_c2(**kwargs: object) -> FakeRunResult:
                del kwargs
                return _fake_ablation_result(
                    output_root,
                    experiment_id="AB_C2_FEDAVG_MLP",
                    cluster_id=2,
                    dataset="TON IoT combined telemetry",
                    model_family="compact_mlp",
                    fl_method="FedAvg",
                    aggregation="weighted_arithmetic_mean",
                )

            def fake_ab_c3(**kwargs: object) -> FakeRunResult:
                del kwargs
                return _fake_ablation_result(
                    output_root,
                    experiment_id="AB_C3_FEDAVG_CNN1D",
                    cluster_id=3,
                    dataset="WUSTL-IIOT-2021",
                    model_family="cnn1d",
                    fl_method="FedAvg",
                    aggregation="weighted_arithmetic_mean",
                )

            with patch("src.train.run_cluster1_fedavg_tcn_ablation", side_effect=fake_ab_c1), patch(
                "src.train.run_cluster2_fedavg_mlp_ablation",
                side_effect=fake_ab_c2,
            ), patch(
                "src.train.run_cluster3_fedavg_cnn1d_ablation",
                side_effect=fake_ab_c3,
            ):
                batch = run_experiments(
                    experiment_ids=[
                        "AB_C1_FEDAVG_TCN",
                        "AB_C2_FEDAVG_MLP",
                        "AB_C3_FEDAVG_CNN1D",
                    ],
                    smoke_test=True,
                    output_root=output_root,
                )

            self.assertEqual(len(batch.experiments), 3)
            with batch.summary_csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["experiment_id"] for row in rows}, {
                "AB_C1_FEDAVG_TCN",
                "AB_C2_FEDAVG_MLP",
                "AB_C3_FEDAVG_CNN1D",
            })

            for experiment_id in ("AB_C1_FEDAVG_TCN", "AB_C2_FEDAVG_MLP", "AB_C3_FEDAVG_CNN1D"):
                self.assertTrue((output_root / "runs" / experiment_id / "run_summary.json").exists())
                self.assertTrue((output_root / "runs" / experiment_id / "round_metrics.csv").exists())
                self.assertTrue((output_root / "metrics" / f"{experiment_id}_metrics.csv").exists())
                self.assertTrue((output_root / "ledgers" / f"{experiment_id}_ledger.jsonl").exists())
                self.assertTrue((output_root / "plots" / f"convergence_{experiment_id}.svg").exists())


if __name__ == "__main__":
    unittest.main()
