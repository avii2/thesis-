from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train import write_multiseed_reports  # noqa: E402


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


class MultiSeedSummaryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
