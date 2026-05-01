from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.cluster1_batadal_calibration import (  # noqa: E402
    EXPERIMENT_ID,
    POSITIVE_CLASS_WEIGHT_SCALES,
    THRESHOLD_SELECTION_MODES,
    PredictionSet,
    _select_validation_threshold_row,
    evaluate_threshold_modes,
    write_calibration_report,
    write_csv_rows,
    write_threshold_calibration_report,
)


class Cluster1BatadalCalibrationTests(unittest.TestCase):
    def test_threshold_modes_select_from_validation_with_fpr_constraints(self) -> None:
        validation_labels = np.asarray([1, 1, 0, 0], dtype=np.int8)
        validation_probabilities = np.asarray([0.9, 0.4, 0.6, 0.1], dtype=np.float32)

        unconstrained = _select_validation_threshold_row(
            validation_labels,
            validation_probabilities,
            mode="max_validation_f1",
        )
        constrained = _select_validation_threshold_row(
            validation_labels,
            validation_probabilities,
            mode="max_validation_f1_with_validation_fpr_le_0.05",
        )

        self.assertAlmostEqual(float(unconstrained["threshold"]), 0.4)
        self.assertLessEqual(float(constrained["validation_fpr"]), 0.05)
        self.assertNotAlmostEqual(float(constrained["threshold"]), float(unconstrained["threshold"]))

    def test_evaluate_threshold_modes_reports_test_metrics_without_test_tuning(self) -> None:
        prediction_set = PredictionSet(
            validation_labels=np.asarray([1, 1, 0, 0], dtype=np.int8),
            validation_probabilities=np.asarray([0.9, 0.4, 0.6, 0.1], dtype=np.float32),
            test_labels=np.asarray([1, 1, 0, 0], dtype=np.int8),
            test_probabilities=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            validation_predictions_path=Path("validation_predictions_seed_42.npz"),
            test_predictions_path=Path("test_predictions_seed_42.npz"),
        )

        rows = evaluate_threshold_modes(
            prediction_set,
            seed=42,
            positive_class_weight_scale=0.25,
            threshold_modes=THRESHOLD_SELECTION_MODES,
        )

        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(row["experiment_id"], EXPERIMENT_ID)
            self.assertEqual(row["threshold_selected_on"], "validation")
            self.assertFalse(row["test_dataset_used_for_threshold_tuning"])
            self.assertIn("test_accuracy", row)
            self.assertIn("test_precision", row)
            self.assertIn("test_recall", row)
            self.assertIn("test_f1", row)
            self.assertIn("test_auroc", row)
            self.assertIn("test_pr_auc", row)
            self.assertIn("test_fpr", row)
            self.assertIn("test_confusion_matrix", row)

    def test_requested_class_weight_scales_are_declared(self) -> None:
        self.assertEqual(POSITIVE_CLASS_WEIGHT_SCALES, (0.25, 0.50, 1.00))

    def test_csv_and_report_outputs_are_written(self) -> None:
        rows = [
            {
                "experiment_id": EXPERIMENT_ID,
                "threshold_selection_mode": "max_validation_f1",
                "positive_class_weight_scale": 1.0,
                "selected_threshold": 0.5,
                "validation_f1": 0.7,
                "validation_fpr": 0.1,
                "test_accuracy": 0.8,
                "test_precision": 0.6,
                "test_recall": 0.5,
                "test_f1": 0.55,
                "test_auroc": 0.9,
                "test_pr_auc": 0.4,
                "test_fpr": 0.2,
                "test_confusion_matrix": "[[8, 2], [1, 9]]",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            threshold_csv = write_csv_rows(tmp_path / "p_c1_batadal_threshold_sweep.csv", rows)
            ablation_csv = write_csv_rows(tmp_path / "p_c1_batadal_class_weight_ablation.csv", rows)
            report_path = write_calibration_report(
                tmp_path / "p_c1_batadal_calibration_report.md",
                threshold_rows=rows,
                class_weight_rows=rows,
                threshold_csv_path=threshold_csv,
                class_weight_csv_path=ablation_csv,
            )
            threshold_report_path = write_threshold_calibration_report(
                tmp_path / "p_c1_batadal_threshold_calibration.md",
                threshold_rows=rows,
                threshold_csv_path=threshold_csv,
            )

            with threshold_csv.open("r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            report = report_path.read_text(encoding="utf-8")
            threshold_report = threshold_report_path.read_text(encoding="utf-8")

        self.assertEqual(csv_rows[0]["experiment_id"], EXPERIMENT_ID)
        self.assertIn("P_C1_BATADAL Calibration Report", report)
        self.assertIn("Held-out BATADAL test predictions are used only after", report)
        self.assertIn("P_C1_BATADAL Threshold Calibration", threshold_report)
        self.assertIn("Test data is not used for threshold tuning.", threshold_report)


if __name__ == "__main__":
    unittest.main()
