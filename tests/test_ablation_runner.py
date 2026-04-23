from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ablation import run_ablation_configs  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fake_raw_result(
    output_root: Path,
    *,
    experiment_id: str,
    cluster_id: int,
    dataset: str,
    hierarchy: str,
    model_family: str,
    fl_method: str,
    aggregation: str,
    clustering_method: str,
    best_validation_f1: float,
    test_f1: float,
) -> SimpleNamespace:
    run_dir = output_root / "runs" / experiment_id
    round_metrics_path = run_dir / "round_metrics.csv"
    metrics_csv_path = output_root / "metrics" / f"{experiment_id}_metrics.csv"
    summary_path = run_dir / "run_summary.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        round_metrics_path,
        [
            {
                "round": 1,
                "train_loss_local_mean": 0.4,
                "train_accuracy": 0.7,
                "train_f1": 0.68,
                "validation_loss": 0.35,
                "validation_accuracy": 0.71,
                "validation_f1": best_validation_f1,
                "test_loss": 0.33,
                "test_accuracy": 0.72,
                "test_f1": test_f1,
                "communication_cost_bytes": 1024,
            }
        ],
    )
    _write_csv(
        metrics_csv_path,
        [
            {
                "experiment_id": experiment_id,
                "cluster_id": cluster_id,
                "dataset": dataset,
                "hierarchy": hierarchy,
                "model_family": model_family,
                "fl_method": fl_method,
                "aggregation": aggregation,
                "clustering_method": clustering_method,
                "rounds": 1,
                "best_validation_round": 1,
                "best_validation_f1": best_validation_f1,
                "test_accuracy": 0.72,
                "test_precision": 0.70,
                "test_recall": 0.69,
                "test_f1": test_f1,
                "test_auroc": 0.73,
                "test_pr_auc": 0.71,
                "test_fpr": 0.18,
                "communication_cost_per_round_bytes": 1024,
                "total_communication_cost_bytes": 1024,
                "wall_clock_training_seconds": 1.1,
            }
        ],
    )
    summary = {
        "experiment_id": experiment_id,
        "cluster_id": cluster_id,
        "dataset": dataset,
        "hierarchy": hierarchy,
        "subcluster_layer_used": hierarchy != "flat",
        "membership_hash": "0" * 64 if hierarchy != "flat" else None,
        "membership_file_used": f"outputs/clustering/cluster{cluster_id}_memberships.json" if hierarchy != "flat" else None,
        "model_family": model_family,
        "fl_method": fl_method,
        "aggregation": aggregation,
        "clustering_method": clustering_method,
        "n_subclusters": 2 if cluster_id == 1 and hierarchy != "flat" else (3 if hierarchy != "flat" else 0),
        "best_validation_round": 1,
        "best_round_validation_metrics": {"f1": best_validation_f1},
        "best_round_test_metrics": {"f1": test_f1},
        "best_round_subcluster_train_sample_counts": {"S1": 8, "S2": 8} if hierarchy != "flat" else None,
        "best_round_subcluster_client_counts": {"S1": 1, "S2": 1} if hierarchy != "flat" else None,
        "data_summary": {
            "client_train_sample_counts": {
                f"C{cluster_id}_L001": 8,
                f"C{cluster_id}_L002": 8,
            }
        },
        "round_metrics_path": str(round_metrics_path),
        "metrics_csv_path": str(metrics_csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return SimpleNamespace(
        experiment_id=experiment_id,
        cluster_id=cluster_id,
        dataset=dataset,
        output_dir=run_dir,
        summary_path=summary_path,
        round_metrics_path=round_metrics_path,
        metrics_csv_path=metrics_csv_path,
        summary=summary,
    )


class AblationRunnerTests(unittest.TestCase):
    def test_runner_creates_required_ablation_csvs_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "outputs"

            def fake_dispatch(
                experiment_id: str,
                *,
                registry: object,
                smoke_test: bool,
                rounds: int | None,
                local_epochs: int | None,
                batch_size: int | None,
                seed: int | None,
                max_train_examples_per_client: int | None,
                max_eval_examples_per_client: int | None,
                output_root: Path,
            ) -> SimpleNamespace:
                del registry, smoke_test, rounds, local_epochs, batch_size, seed
                del max_train_examples_per_client, max_eval_examples_per_client
                if experiment_id == "A_C1":
                    return _fake_raw_result(output_root, experiment_id=experiment_id, cluster_id=1, dataset="HAI", hierarchy="flat", model_family="cnn1d", fl_method="FedAvg", aggregation="weighted_arithmetic_mean", clustering_method="none", best_validation_f1=0.50, test_f1=0.49)
                if experiment_id == "B_C1":
                    return _fake_raw_result(output_root, experiment_id=experiment_id, cluster_id=1, dataset="HAI", hierarchy="hierarchical_fixed", model_family="cnn1d", fl_method="FedAvg", aggregation="weighted_arithmetic_mean", clustering_method="agglomerative", best_validation_f1=0.55, test_f1=0.54)
                if experiment_id == "A_C2":
                    return _fake_raw_result(output_root, experiment_id=experiment_id, cluster_id=2, dataset="TON", hierarchy="flat", model_family="cnn1d", fl_method="FedAvg", aggregation="weighted_arithmetic_mean", clustering_method="none", best_validation_f1=0.60, test_f1=0.58)
                if experiment_id == "B_C2":
                    return _fake_raw_result(output_root, experiment_id=experiment_id, cluster_id=2, dataset="TON", hierarchy="hierarchical_fixed", model_family="cnn1d", fl_method="FedAvg", aggregation="weighted_arithmetic_mean", clustering_method="agglomerative", best_validation_f1=0.63, test_f1=0.61)
                if experiment_id == "A_C3":
                    return _fake_raw_result(output_root, experiment_id=experiment_id, cluster_id=3, dataset="WUSTL", hierarchy="flat", model_family="cnn1d", fl_method="FedAvg", aggregation="weighted_arithmetic_mean", clustering_method="none", best_validation_f1=0.57, test_f1=0.56)
                if experiment_id == "B_C3":
                    return _fake_raw_result(output_root, experiment_id=experiment_id, cluster_id=3, dataset="WUSTL", hierarchy="hierarchical_fixed", model_family="cnn1d", fl_method="FedAvg", aggregation="weighted_arithmetic_mean", clustering_method="agglomerative", best_validation_f1=0.65, test_f1=0.64)
                if experiment_id == "P_C1":
                    return _fake_raw_result(output_root, experiment_id=experiment_id, cluster_id=1, dataset="HAI", hierarchy="hierarchical_fixed", model_family="tcn", fl_method="FedBN", aggregation="weighted_non_bn_mean", clustering_method="agglomerative", best_validation_f1=0.62, test_f1=0.60)
                if experiment_id == "P_C2":
                    return _fake_raw_result(output_root, experiment_id=experiment_id, cluster_id=2, dataset="TON", hierarchy="hierarchical_fixed", model_family="compact_mlp", fl_method="FedProx", aggregation="weighted_arithmetic_mean", clustering_method="agglomerative", best_validation_f1=0.67, test_f1=0.66)
                if experiment_id == "P_C3":
                    return _fake_raw_result(output_root, experiment_id=experiment_id, cluster_id=3, dataset="WUSTL", hierarchy="hierarchical_fixed", model_family="cnn1d", fl_method="SCAFFOLD", aggregation="weighted_arithmetic_mean", clustering_method="agglomerative", best_validation_f1=0.72, test_f1=0.70)
                raise AssertionError(f"Unexpected standard experiment_id {experiment_id}")

            def fake_cluster1(*args: object, **kwargs: object) -> SimpleNamespace:
                del args, kwargs
                return _fake_raw_result(output_root, experiment_id="AB_C1_FEDAVG_TCN", cluster_id=1, dataset="HAI", hierarchy="hierarchical_fixed", model_family="tcn", fl_method="FedAvg", aggregation="weighted_arithmetic_mean", clustering_method="agglomerative", best_validation_f1=0.57, test_f1=0.55)

            def fake_cluster2(*args: object, **kwargs: object) -> SimpleNamespace:
                del args, kwargs
                return _fake_raw_result(output_root, experiment_id="AB_C2_FEDAVG_MLP", cluster_id=2, dataset="TON", hierarchy="hierarchical_fixed", model_family="compact_mlp", fl_method="FedAvg", aggregation="weighted_arithmetic_mean", clustering_method="agglomerative", best_validation_f1=0.61, test_f1=0.60)

            def fake_cluster3(*args: object, **kwargs: object) -> SimpleNamespace:
                del args, kwargs
                return _fake_raw_result(output_root, experiment_id="AB_C3_FEDAVG_CNN1D", cluster_id=3, dataset="WUSTL", hierarchy="hierarchical_fixed", model_family="cnn1d", fl_method="FedAvg", aggregation="weighted_arithmetic_mean", clustering_method="agglomerative", best_validation_f1=0.65, test_f1=0.64)

            with patch("scripts.run_ablation._dispatch_experiment", side_effect=fake_dispatch), patch(
                "scripts.run_ablation.run_cluster1_fedavg_tcn_ablation",
                side_effect=fake_cluster1,
            ), patch(
                "scripts.run_ablation.run_cluster2_fedavg_mlp_ablation",
                side_effect=fake_cluster2,
            ), patch(
                "scripts.run_ablation.run_cluster3_fedavg_cnn1d_ablation",
                side_effect=fake_cluster3,
            ):
                artifacts = run_ablation_configs(
                    run_all=True,
                    smoke_test=True,
                    output_root=output_root,
                )

            self.assertEqual(len(artifacts), 6)
            expected_csvs = {
                output_root / "metrics" / "ablation_hierarchy_effect.csv",
                output_root / "metrics" / "ablation_cluster1_fedavg_vs_fedbn.csv",
                output_root / "metrics" / "ablation_cluster2_fedavg_vs_fedprox.csv",
                output_root / "metrics" / "ablation_cluster3_fedavg_vs_scaffold.csv",
            }
            for path in expected_csvs:
                self.assertTrue(path.exists(), path)

            cluster3_csv = output_root / "metrics" / "ablation_cluster3_fedavg_vs_scaffold.csv"
            with cluster3_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["control_experiment_id"], "AB_C3_FEDAVG_CNN1D")
            self.assertEqual(rows[0]["control_source_experiment_id"], "AB_C3_FEDAVG_CNN1D")
            self.assertEqual(rows[0]["treatment_experiment_id"], "P_C3")

            plot_names = {artifact.plot_path.name for artifact in artifacts}
            self.assertIn("ablation_hierarchy_effect_c1.svg", plot_names)
            self.assertIn("ablation_cluster1_fedavg_vs_fedbn.svg", plot_names)
            self.assertIn("ablation_cluster2_fedavg_vs_fedprox.svg", plot_names)
            self.assertIn("ablation_cluster3_fedavg_vs_scaffold.svg", plot_names)


if __name__ == "__main__":
    unittest.main()
