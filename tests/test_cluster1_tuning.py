from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tune_cluster1_proposed import build_trials, run_tuning  # noqa: E402


class Cluster1TuningTests(unittest.TestCase):
    def test_tuning_config_defines_exact_required_search_space(self) -> None:
        config = yaml.safe_load((REPO_ROOT / "configs" / "tuning_cluster1.yaml").read_text(encoding="utf-8"))
        trials = build_trials(config["search_space"])

        self.assertEqual(len(trials), 6144)
        self.assertEqual(trials[0].learning_rate, 0.003)
        self.assertEqual(trials[0].batch_size, 128)
        self.assertEqual(trials[0].local_epochs, 1)
        self.assertEqual(trials[0].window_length, 32)
        self.assertEqual(trials[0].stride, 8)
        self.assertEqual(trials[0].block_channels, (32, 64, 64))
        self.assertEqual(trials[0].hidden_dim, 32)
        self.assertEqual(trials[0].dropout, 0.1)
        self.assertEqual(trials[0].positive_class_weight_scale, 0.75)

    def test_dry_run_writes_tuning_summary_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config = yaml.safe_load((REPO_ROOT / "configs" / "tuning_cluster1.yaml").read_text(encoding="utf-8"))
            config["output"]["root"] = str(tmp_path / "tuning")
            config_path = tmp_path / "tuning_cluster1.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            report = run_tuning(
                SimpleNamespace(
                    config=str(config_path),
                    smoke_test=True,
                    rounds=None,
                    seed=None,
                    max_trials=2,
                    start_index=0,
                    max_train_examples_per_client=None,
                    max_eval_examples_per_client=None,
                    force=True,
                    dry_run=True,
                    no_heuristic_order=False,
                )
            )

            self.assertEqual(report["status"], "DRY_RUN")
            self.assertEqual(report["total_trials_in_space"], 6144)
            self.assertEqual(len(report["selected_trials"]), 2)
            self.assertTrue((tmp_path / "tuning" / "best_config.json").exists())
            self.assertTrue((tmp_path / "tuning" / "cluster1_tuning_summary.md").exists())


if __name__ == "__main__":
    unittest.main()
