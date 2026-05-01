from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.cluster1_batadal_tuning import (  # noqa: E402
    FIRST_TUNING_SPEC,
    FPR_REDUCTION_SPEC,
    RECALL_RECOVERY_SPEC,
    select_final_candidate,
)


class Cluster1BatadalTuningTests(unittest.TestCase):
    def test_first_requested_tuning_spec_is_exact(self) -> None:
        self.assertEqual(FIRST_TUNING_SPEC.rounds, 50)
        self.assertEqual(FIRST_TUNING_SPEC.local_epochs, 1)
        self.assertEqual(FIRST_TUNING_SPEC.batch_size, 64)
        self.assertAlmostEqual(FIRST_TUNING_SPEC.learning_rate, 0.001)
        self.assertAlmostEqual(FIRST_TUNING_SPEC.dropout, 0.20)
        self.assertAlmostEqual(FIRST_TUNING_SPEC.positive_class_weight_scale, 0.10)
        self.assertEqual(FIRST_TUNING_SPEC.threshold_mode, "max_validation_f1_with_validation_fpr_le_0.10")
        self.assertEqual(FIRST_TUNING_SPEC.seed, 42)
        self.assertEqual(
            FIRST_TUNING_SPEC.output_root.as_posix(),
            "outputs_c1_batadal_tuned/scale_010_lr_0001_dropout_020",
        )

    def test_conditional_tuning_specs_are_exact(self) -> None:
        self.assertAlmostEqual(FPR_REDUCTION_SPEC.positive_class_weight_scale, 0.05)
        self.assertEqual(FPR_REDUCTION_SPEC.threshold_mode, "max_validation_f1_with_validation_fpr_le_0.05")
        self.assertEqual(
            FPR_REDUCTION_SPEC.output_root.as_posix(),
            "outputs_c1_batadal_tuned/scale_005_lr_0001_dropout_020",
        )
        self.assertAlmostEqual(RECALL_RECOVERY_SPEC.positive_class_weight_scale, 0.25)
        self.assertAlmostEqual(RECALL_RECOVERY_SPEC.dropout, 0.10)
        self.assertEqual(RECALL_RECOVERY_SPEC.threshold_mode, "max_validation_f1_with_validation_fpr_le_0.10")
        self.assertEqual(
            RECALL_RECOVERY_SPEC.output_root.as_posix(),
            "outputs_c1_batadal_tuned/scale_025_lr_0001_dropout_010",
        )

    def test_final_selection_uses_validation_constraint_before_test_metrics(self) -> None:
        rows = [
            {
                "experiment_id": "P_C1_BATADAL",
                "source_type": "tuned_proposed",
                "label": "high-test-low-validation",
                "validation_f1": 0.80,
                "validation_fpr": 0.50,
                "test_f1": 0.99,
            },
            {
                "experiment_id": "P_C1_BATADAL",
                "source_type": "tuned_proposed",
                "label": "valid-selection",
                "validation_f1": 0.60,
                "validation_fpr": 0.08,
                "test_f1": 0.10,
            },
        ]

        selected, rule = select_final_candidate(rows)

        self.assertEqual(selected["label"], "valid-selection")
        self.assertIn("validation FPR <= 0.10", rule)


if __name__ == "__main__":
    unittest.main()

