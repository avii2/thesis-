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
from src.train import DEFAULT_RUN_ALL_EXPERIMENT_IDS, SUPPORTED_EXPERIMENT_IDS, load_config_registry, load_experiment_matrix, run_experiments  # noqa: E402


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
    if not rows:
        raise ValueError(f"{path}: rows must be non-empty.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fake_result(
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
    n_subclusters: int,
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
            "train_f1": 0.61,
            "validation_f1": 0.58,
            "test_f1": 0.57,
            "communication_cost_bytes": 1024,
        },
        {
            "round": 2,
            "train_loss_local_mean": 0.44,
            "train_f1": 0.69,
            "validation_f1": 0.66,
            "test_f1": 0.64,
            "communication_cost_bytes": 1024,
        },
    ]
    _write_csv(round_metrics_path, round_rows)

    summary_row = {
        "experiment_id": experiment_id,
        "cluster_id": cluster_id,
        "dataset": dataset,
        "hierarchy": hierarchy,
        "model_family": model_family,
        "fl_method": fl_method,
        "aggregation": aggregation,
        "clustering_method": clustering_method,
        "rounds": 2,
        "best_validation_round": 2,
        "best_validation_f1": 0.66,
        "test_accuracy": 0.70,
        "test_precision": 0.68,
        "test_recall": 0.65,
        "test_f1": 0.64,
        "test_auroc": 0.72,
        "test_pr_auc": 0.69,
        "test_fpr": 0.18,
        "communication_cost_per_round_bytes": 1024,
        "total_communication_cost_bytes": 2048,
        "wall_clock_training_seconds": 1.2,
    }
    _write_csv(metrics_csv_path, [summary_row])

    summary: dict[str, object] = {
        "experiment_id": experiment_id,
        "cluster_id": cluster_id,
        "dataset": dataset,
        "hierarchy": hierarchy,
        "subcluster_layer_used": hierarchy != "flat",
        "model_family": model_family,
        "fl_method": fl_method,
        "aggregation": aggregation,
        "clustering_method": clustering_method,
        "input_adapter": "feature_vector_as_sequence" if cluster_id != 1 else "sliding_window_feature_channels",
        "num_leaf_clients": 4,
        "n_subclusters": n_subclusters,
        "best_validation_round": 2,
        "best_validation_f1": 0.66,
        "best_round_train_metrics": {"f1": 0.69},
        "best_round_validation_metrics": {"f1": 0.66},
        "best_round_test_metrics": {"f1": 0.64},
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
    if hierarchy != "flat":
        summary["membership_hash"] = canonical_sha256(
            {
                "experiment_id": experiment_id,
                "subclusters": n_subclusters,
            }
        ).split(":", 1)[1]
        summary["membership_file_used"] = f"outputs/clustering/cluster{cluster_id}_memberships.json"
        summary["best_round_subcluster_train_sample_counts"] = {
            f"S{subcluster_index + 1}": 8
            for subcluster_index in range(n_subclusters)
        }
        summary["best_round_subcluster_client_counts"] = {
            f"S{subcluster_index + 1}": 1
            for subcluster_index in range(n_subclusters)
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


class ExperimentMatrixTests(unittest.TestCase):
    def test_repo_matrix_and_configs_cover_required_supported_experiments(self) -> None:
        matrix = load_experiment_matrix(REPO_ROOT / "docs" / "EXPERIMENT_MATRIX.csv")
        registry = load_config_registry(
            baseline_flat_config_path=REPO_ROOT / "configs" / "baseline_flat.yaml",
            baseline_hierarchical_config_path=REPO_ROOT / "configs" / "baseline_hierarchical.yaml",
            proposed_config_path=REPO_ROOT / "configs" / "proposed.yaml",
        )
        self.assertEqual(list(SUPPORTED_EXPERIMENT_IDS), list(SUPPORTED_EXPERIMENT_IDS))
        for experiment_id in SUPPORTED_EXPERIMENT_IDS:
            self.assertIn(experiment_id, matrix)
            self.assertIn(experiment_id, registry)

    def test_run_all_dispatches_supported_matrix_and_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "outputs"

            def fake_flat_runner(**kwargs: object) -> list[FakeRunResult]:
                experiment_ids = list(kwargs["experiment_ids"])
                output_root_arg = Path(kwargs["output_root"])
                return [
                    _fake_result(
                        output_root_arg,
                        experiment_id=experiment_id,
                        cluster_id=int(experiment_id[-1]),
                        dataset=f"Dataset {experiment_id}",
                        hierarchy="flat",
                        model_family="cnn1d",
                        fl_method="FedAvg",
                        aggregation="weighted_arithmetic_mean",
                        clustering_method="none",
                        n_subclusters=0,
                    )
                    for experiment_id in experiment_ids
                ]

            def fake_hierarchical_runner(**kwargs: object) -> list[FakeRunResult]:
                experiment_ids = list(kwargs["experiment_ids"])
                output_root_arg = Path(kwargs["output_root"])
                subclusters = {"B_C1": 2, "B_C2": 3, "B_C3": 3}
                return [
                    _fake_result(
                        output_root_arg,
                        experiment_id=experiment_id,
                        cluster_id=int(experiment_id[-1]),
                        dataset=f"Dataset {experiment_id}",
                        hierarchy="hierarchical_fixed",
                        model_family="cnn1d",
                        fl_method="FedAvg",
                        aggregation="weighted_arithmetic_mean",
                        clustering_method="agglomerative",
                        n_subclusters=subclusters[experiment_id],
                    )
                    for experiment_id in experiment_ids
                ]

            def fake_cluster1_runner(**kwargs: object) -> FakeRunResult:
                return _fake_result(
                    Path(kwargs["output_root"]),
                    experiment_id="P_C1",
                    cluster_id=1,
                    dataset="Dataset P_C1",
                    hierarchy="hierarchical_fixed",
                    model_family="tcn",
                    fl_method="FedBN",
                    aggregation="weighted_non_bn_mean",
                    clustering_method="agglomerative",
                    n_subclusters=2,
                )

            def fake_cluster2_runner(**kwargs: object) -> FakeRunResult:
                return _fake_result(
                    Path(kwargs["output_root"]),
                    experiment_id="P_C2",
                    cluster_id=2,
                    dataset="Dataset P_C2",
                    hierarchy="hierarchical_fixed",
                    model_family="compact_mlp",
                    fl_method="FedProx",
                    aggregation="weighted_arithmetic_mean",
                    clustering_method="agglomerative",
                    n_subclusters=3,
                )

            def fake_cluster3_runner(**kwargs: object) -> FakeRunResult:
                return _fake_result(
                    Path(kwargs["output_root"]),
                    experiment_id="P_C3",
                    cluster_id=3,
                    dataset="Dataset P_C3",
                    hierarchy="hierarchical_fixed",
                    model_family="cnn1d",
                    fl_method="SCAFFOLD",
                    aggregation="weighted_arithmetic_mean",
                    clustering_method="agglomerative",
                    n_subclusters=3,
                )

            with patch("src.train.run_flat_baseline_experiments", side_effect=fake_flat_runner), patch(
                "src.train.run_hierarchical_baseline_experiments",
                side_effect=fake_hierarchical_runner,
            ), patch("src.train.run_cluster1_proposed", side_effect=fake_cluster1_runner), patch(
                "src.train.run_cluster2_proposed",
                side_effect=fake_cluster2_runner,
            ), patch("src.train.run_cluster3_proposed", side_effect=fake_cluster3_runner):
                batch = run_experiments(
                    run_all=True,
                    smoke_test=True,
                    matrix_path=REPO_ROOT / "docs" / "EXPERIMENT_MATRIX.csv",
                    baseline_flat_config_path=REPO_ROOT / "configs" / "baseline_flat.yaml",
                    baseline_hierarchical_config_path=REPO_ROOT / "configs" / "baseline_hierarchical.yaml",
                    proposed_config_path=REPO_ROOT / "configs" / "proposed.yaml",
                    output_root=output_root,
                )

            self.assertEqual(len(batch.experiments), len(DEFAULT_RUN_ALL_EXPERIMENT_IDS))
            self.assertTrue(batch.summary_csv_path.exists())
            self.assertEqual(len(batch.comparison_plot_paths), 2)
            for path in batch.comparison_plot_paths:
                self.assertTrue(path.exists())

            with batch.summary_csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(DEFAULT_RUN_ALL_EXPERIMENT_IDS))
            self.assertEqual({row["experiment_id"] for row in rows}, set(DEFAULT_RUN_ALL_EXPERIMENT_IDS))

            for artifact in batch.experiments:
                self.assertTrue(artifact.model_manifest_path.exists())
                self.assertTrue(artifact.ledger_path.exists())
                self.assertTrue(artifact.convergence_plot_path.exists())
                with artifact.ledger_path.open("r", encoding="utf-8") as handle:
                    ledger_lines = [line for line in handle.read().splitlines() if line.strip()]
                self.assertEqual(len(ledger_lines), 2)
                payload = json.loads(ledger_lines[0])
                self.assertEqual(payload["cluster_id"], artifact.cluster_id)
                self.assertNotIn("full_model_weights", payload)

    def test_requesting_unknown_experiment_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "supports the required A/B/P experiments plus the dedicated AB_\\* ablations"):
            run_experiments(
                experiment_ids=["UNKNOWN_EXPERIMENT"],
                matrix_path=REPO_ROOT / "docs" / "EXPERIMENT_MATRIX.csv",
                baseline_flat_config_path=REPO_ROOT / "configs" / "baseline_flat.yaml",
                baseline_hierarchical_config_path=REPO_ROOT / "configs" / "baseline_hierarchical.yaml",
                proposed_config_path=REPO_ROOT / "configs" / "proposed.yaml",
                output_root=REPO_ROOT / "outputs",
            )


if __name__ == "__main__":
    unittest.main()
