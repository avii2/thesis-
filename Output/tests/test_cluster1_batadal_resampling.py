from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.cluster1_batadal_resampling import (  # noqa: E402
    POSITIVE_CLASS_WEIGHT_SCALES,
    build_resampling_specs,
    select_by_validation,
)
from src.fl.sampling import positive_window_oversample_indices  # noqa: E402


class Cluster1BatadalResamplingTests(unittest.TestCase):
    def test_positive_window_oversampling_reaches_balanced_fraction(self) -> None:
        labels = np.asarray([0] * 10 + [1] * 2, dtype=np.int8)
        indices = positive_window_oversample_indices(
            labels,
            rng=np.random.default_rng(42),
            target_positive_fraction=0.5,
        )
        sampled_labels = labels[indices]

        self.assertEqual(int(np.sum(sampled_labels == 0)), 10)
        self.assertEqual(int(np.sum(sampled_labels == 1)), 10)

    def test_resampling_specs_use_requested_scales_and_threshold_mode(self) -> None:
        specs = build_resampling_specs(output_root="outputs_c1_batadal_tuned/resampling")

        self.assertEqual(tuple(spec.positive_class_weight_scale for spec in specs), POSITIVE_CLASS_WEIGHT_SCALES)
        self.assertTrue(all(spec.training_resampling == "positive_window_oversampling" for spec in specs))
        self.assertTrue(
            all(spec.threshold_mode == "max_validation_f1_with_validation_fpr_le_0.10" for spec in specs)
        )

    def test_selection_uses_validation_only(self) -> None:
        rows = [
            {
                "run_id": "bad_fpr_high_test",
                "validation_f1": 0.90,
                "validation_fpr": 0.20,
                "test_f1": 0.99,
            },
            {
                "run_id": "selected",
                "validation_f1": 0.50,
                "validation_fpr": 0.05,
                "test_f1": 0.10,
            },
        ]

        selected = select_by_validation(rows)

        self.assertEqual(selected["run_id"], "selected")


if __name__ == "__main__":
    unittest.main()

