from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import generate_ieee_plots as plotgen  # noqa: E402


EXPERIMENT_IDS = (
    "A_C1",
    "A_C2",
    "A_C3",
    "B_C1",
    "B_C2",
    "B_C3",
    "P_C1",
    "P_C2",
    "P_C3",
    "AB_C1_FEDAVG_TCN",
    "AB_C2_FEDAVG_MLP",
    "AB_C3_FEDAVG_CNN1D",
)


def _cluster_id(experiment_id: str) -> int:
    return next(int(token[1:]) for token in experiment_id.split("_") if token.startswith("C") and token[1:].isdigit())


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class IEEEPlotsTests(unittest.TestCase):
    def test_generate_plots_writes_required_ieee_artifacts_from_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            output_root = repo_root / "outputs"
            docs_dir = repo_root / "docs"
            matrix_path = docs_dir / "EXPERIMENT_MATRIX.csv"
            docs_dir.mkdir(parents=True, exist_ok=True)

            _write_csv(
                matrix_path,
                [
                    {"experiment_id": experiment_id, "cluster_id": _cluster_id(experiment_id)}
                    for experiment_id in EXPERIMENT_IDS
                ],
            )
            _write_csv(
                output_root / "metrics" / "summary_all_experiments.csv",
                [
                    {
                        "experiment_id": experiment_id,
                        "test_accuracy": 0.80 + _cluster_id(experiment_id) * 0.01,
                        "test_precision": 0.70 + _cluster_id(experiment_id) * 0.01,
                        "test_recall": 0.60 + _cluster_id(experiment_id) * 0.01,
                        "test_f1": 0.65 + _cluster_id(experiment_id) * 0.01,
                        "test_auroc": 0.75 + _cluster_id(experiment_id) * 0.01,
                        "test_pr_auc": 0.55 + _cluster_id(experiment_id) * 0.01,
                        "test_fpr": 0.10 + _cluster_id(experiment_id) * 0.01,
                        "wall_clock_training_seconds": 10.0 + _cluster_id(experiment_id),
                        "total_communication_cost_bytes": 1000 * _cluster_id(experiment_id),
                    }
                    for experiment_id in EXPERIMENT_IDS
                ],
            )
            for experiment_id in EXPERIMENT_IDS:
                ledger_path = output_root / "ledgers" / f"{experiment_id}_ledger.jsonl"
                ledger_path.parent.mkdir(parents=True, exist_ok=True)
                ledger_path.write_text(
                    "\n".join(
                        [
                            '{"timestamp_start":"2026-01-01T00:00:00+00:00","timestamp_end":"2026-01-01T00:00:00.001000+00:00"}',
                            '{"timestamp_start":"2026-01-01T00:00:01+00:00","timestamp_end":"2026-01-01T00:00:01.002000+00:00"}',
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
            for experiment_id in ("A_C1", "A_C2", "A_C3", "B_C1", "B_C2", "B_C3", "P_C1", "P_C2", "P_C3"):
                _write_csv(
                    output_root / "runs" / experiment_id / "round_metrics.csv",
                    [
                        {"round": 1, "validation_f1": 0.20},
                        {"round": 2, "validation_f1": 0.30},
                    ],
                )

            original_paths = (
                plotgen.REPO_ROOT,
                plotgen.OUTPUT_ROOT,
                plotgen.PLOTS_DIR,
                plotgen.README_PATH,
                plotgen.MATRIX_PATH,
                plotgen.SUMMARY_MEAN_STD_PATH,
                plotgen.SUMMARY_SINGLE_PATH,
            )
            try:
                plotgen.REPO_ROOT = repo_root
                plotgen.OUTPUT_ROOT = output_root
                plotgen.PLOTS_DIR = output_root / "plots_ieee"
                plotgen.README_PATH = docs_dir / "PLOTS_README.md"
                plotgen.MATRIX_PATH = matrix_path
                plotgen.SUMMARY_MEAN_STD_PATH = output_root / "metrics" / "summary_all_experiments_mean_std.csv"
                plotgen.SUMMARY_SINGLE_PATH = output_root / "metrics" / "summary_all_experiments.csv"

                report = plotgen.generate_plots()
            finally:
                (
                    plotgen.REPO_ROOT,
                    plotgen.OUTPUT_ROOT,
                    plotgen.PLOTS_DIR,
                    plotgen.README_PATH,
                    plotgen.MATRIX_PATH,
                    plotgen.SUMMARY_MEAN_STD_PATH,
                    plotgen.SUMMARY_SINGLE_PATH,
                ) = original_paths

            required_stems = {
                "fig_f1_comparison",
                "fig_auroc_comparison",
                "fig_pr_auc_comparison",
                "fig_fpr_comparison",
                "fig_ablation_delta_f1",
                "fig_convergence_cluster1",
                "fig_convergence_cluster2",
                "fig_convergence_cluster3",
                "fig_communication_cost",
                "fig_ledger_overhead",
            }
            generated = {Path(path).stem for path in report["generated_files"]}
            for stem in required_stems:
                self.assertIn(stem, generated)
                self.assertTrue((output_root / "plots_ieee" / f"{stem}.pdf").exists())
                self.assertTrue((output_root / "plots_ieee" / f"{stem}.png").exists())
            self.assertTrue((docs_dir / "PLOTS_README.md").exists())


if __name__ == "__main__":
    unittest.main()
