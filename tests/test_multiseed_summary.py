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

from src.train import (  # noqa: E402
    DEFAULT_MULTI_SEED_VALUES,
    DEFAULT_PAPER_SUITE_EXPERIMENT_IDS,
    parse_args,
    run_experiments,
    write_multiseed_reports,
)


def _write_seed_metrics(
    output_root: Path,
    *,
    experiment_id: str,
    seed: int,
    test_accuracy: float,
    test_precision: float,
    test_recall: float,
    test_f1: float,
    test_auroc: float,
    test_pr_auc: float,
    test_fpr: float,
    wall_clock_training_seconds: float,
    total_communication_cost_bytes: int,
) -> Path:
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{experiment_id}_seed_{seed}_metrics.csv"
    row = {
        "experiment_id": experiment_id,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_auroc": test_auroc,
        "test_pr_auc": test_pr_auc,
        "test_fpr": test_fpr,
        "wall_clock_training_seconds": wall_clock_training_seconds,
        "total_communication_cost_bytes": total_communication_cost_bytes,
    }
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return metrics_path


def _write_raw_result(
    output_root: Path,
    *,
    experiment_id: str,
    cluster_id: int,
    seed: int,
) -> SimpleNamespace:
    run_dir = output_root / "runs" / experiment_id
    round_metrics_path = run_dir / "round_metrics.csv"
    summary_path = run_dir / "run_summary.json"
    metrics_csv_path = output_root / "metrics" / f"{experiment_id}_metrics.csv"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)

    seed_offset = {42: 0.00, 123: 0.03, 2025: 0.06}[int(seed)]
    round_rows = [
        {
            "round": 1,
            "train_loss_local_mean": 0.5 - seed_offset,
            "train_f1": 0.60 + seed_offset,
            "validation_f1": 0.58 + seed_offset,
            "test_f1": 0.57 + seed_offset,
            "communication_cost_bytes": 1000 + int(seed_offset * 1000),
        },
        {
            "round": 2,
            "train_loss_local_mean": 0.4 - seed_offset,
            "train_f1": 0.66 + seed_offset,
            "validation_f1": 0.64 + seed_offset,
            "test_f1": 0.63 + seed_offset,
            "communication_cost_bytes": 1000 + int(seed_offset * 1000),
        },
    ]
    with round_metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(round_rows[0].keys()))
        writer.writeheader()
        writer.writerows(round_rows)

    hierarchy = "flat" if experiment_id.startswith("A_") else "hierarchical_fixed"
    n_subclusters = 0 if hierarchy == "flat" else (2 if cluster_id == 1 else 3)
    model_family = (
        "tcn"
        if experiment_id in {"P_C1", "AB_C1_FEDAVG_TCN"}
        else "compact_mlp"
        if experiment_id in {"P_C2", "AB_C2_FEDAVG_MLP"}
        else "cnn1d"
    )
    fl_method = (
        "FedBN"
        if experiment_id == "P_C1"
        else "FedProx"
        if experiment_id == "P_C2"
        else "SCAFFOLD"
        if experiment_id == "P_C3"
        else "FedAvg"
    )
    aggregation = "weighted_non_bn_mean" if experiment_id == "P_C1" else "weighted_arithmetic_mean"
    clustering_method = "none" if hierarchy == "flat" else "agglomerative"
    metrics_row = {
        "experiment_id": experiment_id,
        "cluster_id": cluster_id,
        "dataset": f"Dataset {cluster_id}",
        "hierarchy": hierarchy,
        "model_family": model_family,
        "fl_method": fl_method,
        "aggregation": aggregation,
        "clustering_method": clustering_method,
        "rounds": 2,
        "best_validation_round": 2,
        "best_validation_f1": 0.64 + seed_offset,
        "test_accuracy": 0.70 + seed_offset,
        "test_precision": 0.68 + seed_offset,
        "test_recall": 0.65 + seed_offset,
        "test_f1": 0.63 + seed_offset,
        "test_auroc": 0.72 + seed_offset,
        "test_pr_auc": 0.69 + seed_offset,
        "test_fpr": 0.18 - seed_offset / 10,
        "test_confusion_matrix": json.dumps([[17, 3], [5, 9]]),
        "communication_cost_per_round_bytes": 1000 + int(seed_offset * 1000),
        "total_communication_cost_bytes": 2000 + int(seed_offset * 2000),
        "wall_clock_training_seconds": 1.0 + seed_offset,
    }
    with metrics_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_row.keys()))
        writer.writeheader()
        writer.writerow(metrics_row)

    summary = {
        "experiment_id": experiment_id,
        "cluster_id": cluster_id,
        "dataset": f"Dataset {cluster_id}",
        "hierarchy": hierarchy,
        "subcluster_layer_used": hierarchy != "flat",
        "model_family": model_family,
        "fl_method": fl_method,
        "aggregation": aggregation,
        "clustering_method": clustering_method,
        "n_subclusters": n_subclusters,
        "best_validation_round": 2,
        "best_round_train_metrics": {"f1": 0.66 + seed_offset},
        "best_round_validation_metrics": {"f1": 0.64 + seed_offset},
        "best_round_test_metrics": {"f1": 0.63 + seed_offset},
        "data_summary": {
            "client_train_sample_counts": {
                f"C{cluster_id}_L001": 8,
                f"C{cluster_id}_L002": 8,
            }
        },
        "round_metrics_path": str(round_metrics_path),
        "metrics_csv_path": str(metrics_csv_path),
    }
    if hierarchy != "flat":
        summary["membership_hash"] = f"{cluster_id}" * 64
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

    return SimpleNamespace(
        experiment_id=experiment_id,
        cluster_id=cluster_id,
        dataset=f"Dataset {cluster_id}",
        output_dir=run_dir,
        summary_path=summary_path,
        round_metrics_path=round_metrics_path,
        metrics_csv_path=metrics_csv_path,
        summary=summary,
    )


class MultiSeedSummaryTests(unittest.TestCase):
    def test_paper_multiseed_cli_defaults_to_required_seeds(self) -> None:
        args = parse_args(["--paper-suite", "--paper-multiseed"])

        self.assertTrue(args.paper_suite)
        self.assertTrue(args.paper_multiseed)
        self.assertEqual(DEFAULT_MULTI_SEED_VALUES, (42, 123, 2025))

    def test_mean_std_calculations_are_correct_on_synthetic_metric_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "outputs"
            markdown_path = Path(tmpdir) / "RESULTS_SUMMARY_MEAN_STD.md"

            _write_seed_metrics(
                output_root,
                experiment_id="A_C1",
                seed=42,
                test_accuracy=0.80,
                test_precision=0.70,
                test_recall=0.60,
                test_f1=0.65,
                test_auroc=0.75,
                test_pr_auc=0.72,
                test_fpr=0.10,
                wall_clock_training_seconds=10.0,
                total_communication_cost_bytes=1000,
            )
            _write_seed_metrics(
                output_root,
                experiment_id="A_C1",
                seed=123,
                test_accuracy=0.90,
                test_precision=0.80,
                test_recall=0.70,
                test_f1=0.75,
                test_auroc=0.85,
                test_pr_auc=0.82,
                test_fpr=0.20,
                wall_clock_training_seconds=20.0,
                total_communication_cost_bytes=2000,
            )
            _write_seed_metrics(
                output_root,
                experiment_id="A_C1",
                seed=2025,
                test_accuracy=1.00,
                test_precision=0.90,
                test_recall=0.80,
                test_f1=0.85,
                test_auroc=0.95,
                test_pr_auc=0.92,
                test_fpr=0.30,
                wall_clock_training_seconds=30.0,
                total_communication_cost_bytes=3000,
            )

            summary_path, table_path, written_markdown_path = write_multiseed_reports(
                experiment_ids=["A_C1"],
                seeds=[42, 123, 2025],
                output_root=output_root,
                markdown_output_path=markdown_path,
            )

            self.assertEqual(summary_path.resolve(), (output_root / "metrics" / "summary_all_experiments_mean_std.csv").resolve())
            self.assertEqual(table_path.resolve(), (output_root / "reports" / "results_mean_std_table.csv").resolve())
            self.assertEqual(written_markdown_path.resolve(), markdown_path.resolve())
            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))

            self.assertEqual(row["experiment_id"], "A_C1")
            self.assertEqual(row["status"], "COMPLETE")
            self.assertEqual(row["successful_seeds"], "42,123,2025")
            self.assertEqual(row["missing_seeds"], "")
            self.assertAlmostEqual(float(row["test_accuracy_mean"]), 0.9)
            self.assertAlmostEqual(float(row["test_accuracy_std"]), 0.08164965809277258)
            self.assertAlmostEqual(float(row["test_f1_mean"]), 0.75)
            self.assertAlmostEqual(float(row["test_f1_std"]), 0.08164965809277258)
            self.assertAlmostEqual(float(row["wall_clock_training_seconds_mean"]), 20.0)
            self.assertAlmostEqual(float(row["wall_clock_training_seconds_std"]), 8.16496580927726)
            self.assertAlmostEqual(float(row["total_communication_cost_bytes_mean"]), 2000.0)
            self.assertAlmostEqual(float(row["total_communication_cost_bytes_std"]), 816.496580927726)

    def test_missing_and_failed_seeds_are_reported_not_silently_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "outputs"
            markdown_path = Path(tmpdir) / "RESULTS_SUMMARY_MEAN_STD.md"
            _write_seed_metrics(
                output_root,
                experiment_id="A_C1",
                seed=42,
                test_accuracy=0.75,
                test_precision=0.65,
                test_recall=0.55,
                test_f1=0.60,
                test_auroc=0.70,
                test_pr_auc=0.68,
                test_fpr=0.12,
                wall_clock_training_seconds=12.0,
                total_communication_cost_bytes=1200,
            )

            failed_dir = output_root / "runs" / "A_C1" / "seed_123"
            failed_dir.mkdir(parents=True, exist_ok=True)
            (failed_dir / "FAILED.json").write_text(
                '{"experiment_id":"A_C1","seed":123,"status":"FAILED","error":"synthetic failure"}\n',
                encoding="utf-8",
            )

            summary_path, _, written_markdown_path = write_multiseed_reports(
                experiment_ids=["A_C1"],
                seeds=[42, 123, 2025],
                output_root=output_root,
                markdown_output_path=markdown_path,
            )

            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))

            self.assertEqual(row["status"], "PARTIAL")
            self.assertEqual(row["successful_seeds"], "42")
            self.assertEqual(row["failed_seeds"], "123")
            self.assertEqual(row["missing_seeds"], "2025")
            self.assertIn("seed 123 FAILED: synthetic failure", row["notes"])
            self.assertIn("seed 2025 MISSING", row["notes"])
            markdown_text = written_markdown_path.read_text(encoding="utf-8")
            self.assertIn("A_C1", markdown_text)
            self.assertIn("FAILED", markdown_text)
            self.assertIn("MISSING", markdown_text)

    def test_run_experiments_multiseed_writes_seed_specific_artifacts_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "outputs"
            markdown_path = Path(tmpdir) / "RESULTS_SUMMARY_MEAN_STD.md"
            real_write_multiseed_reports = write_multiseed_reports

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
                del registry, smoke_test, rounds, local_epochs, batch_size
                del max_train_examples_per_client, max_eval_examples_per_client
                self.assertIsNotNone(seed)
                cluster_id = next(
                    int(token[1:])
                    for token in str(experiment_id).split("_")
                    if token.startswith("C") and token[1:].isdigit()
                )
                return _write_raw_result(
                    Path(output_root),
                    experiment_id=experiment_id,
                    cluster_id=cluster_id,
                    seed=int(seed),
                )

            def write_reports_with_temp_markdown(**kwargs: object) -> tuple[Path, Path, Path]:
                return real_write_multiseed_reports(
                    **kwargs,
                    markdown_output_path=markdown_path,
                )

            with patch("src.train._dispatch_experiment", side_effect=fake_dispatch), patch(
                "src.train.write_multiseed_reports",
                side_effect=write_reports_with_temp_markdown,
            ):
                batch = run_experiments(
                    paper_suite=True,
                    seeds=DEFAULT_MULTI_SEED_VALUES,
                    smoke_test=True,
                    output_root=output_root,
                )

            self.assertEqual(
                len(batch.experiments),
                len(DEFAULT_PAPER_SUITE_EXPERIMENT_IDS) * len(DEFAULT_MULTI_SEED_VALUES),
            )
            for experiment_id in DEFAULT_PAPER_SUITE_EXPERIMENT_IDS:
                for seed in DEFAULT_MULTI_SEED_VALUES:
                    seed_label = f"seed_{seed}"
                    self.assertTrue((output_root / "runs" / experiment_id / seed_label / "run_summary.json").exists())
                    self.assertTrue((output_root / "runs" / experiment_id / seed_label / "round_metrics.csv").exists())
                    self.assertTrue((output_root / "metrics" / f"{experiment_id}_{seed_label}_metrics.csv").exists())
                    self.assertTrue((output_root / "ledgers" / f"{experiment_id}_{seed_label}_ledger.jsonl").exists())
                    self.assertTrue((output_root / "plots" / f"convergence_{experiment_id}_{seed_label}.svg").exists())

            self.assertTrue(batch.mean_std_summary_csv_path.exists())
            self.assertTrue(batch.mean_std_results_csv_path.exists())
            self.assertEqual(batch.mean_std_markdown_path.resolve(), markdown_path.resolve())
            self.assertTrue(markdown_path.exists())
            self.assertTrue(batch.confusion_matrix_report_path.exists())

            with batch.mean_std_summary_csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(DEFAULT_PAPER_SUITE_EXPERIMENT_IDS))
            self.assertTrue(all(row["status"] == "COMPLETE" for row in rows))
            self.assertTrue(all(row["successful_seeds"] == "42,123,2025" for row in rows))


if __name__ == "__main__":
    unittest.main()
