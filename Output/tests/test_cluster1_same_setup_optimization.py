from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_cluster1_same_setup_optimization import (  # noqa: E402
    CONSTANT_TRAINING_FEATURES,
    _all_candidate_space,
    _feature_variant_columns,
)


class Cluster1SameSetupOptimizationTests(unittest.TestCase):
    def test_candidate_space_is_staged_from_requested_values(self) -> None:
        self.assertEqual(len(_all_candidate_space()), 3240)

    def test_feature_variant_selection_drops_constants_and_lower_scored_correlations(self) -> None:
        constant_feature = sorted(CONSTANT_TRAINING_FEATURES)[0]
        all_features = ["sensor_a", "sensor_b", "sensor_c", constant_feature]
        scores = {
            "sensor_a": {"average_precision_best_direction": 0.20},
            "sensor_b": {"average_precision_best_direction": 0.35},
            "sensor_c": {"average_precision_best_direction": 0.10},
            constant_feature: {"average_precision_best_direction": 0.90},
        }
        high_corr_pairs = [
            {
                "feature_a": "sensor_a",
                "feature_b": "sensor_b",
                "abs_correlation": 0.991,
            }
        ]

        variants = _feature_variant_columns(
            all_features=all_features,
            scores=scores,
            high_corr_pairs=high_corr_pairs,
        )

        self.assertEqual(variants["full"], all_features)
        self.assertNotIn(constant_feature, variants["no_constants"])
        self.assertNotIn("sensor_a", variants["no_constants_no_corr"])
        self.assertIn("sensor_b", variants["no_constants_no_corr"])
        self.assertEqual(variants["top_ap_40"][0], constant_feature)


if __name__ == "__main__":
    unittest.main()
