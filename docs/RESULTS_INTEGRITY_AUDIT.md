# RESULTS INTEGRITY AUDIT

## Scope

This audit inspected the current generated artifacts under:

- `outputs/runs/`
- `outputs/metrics/`
- `outputs/plots/`
- `outputs/ledgers/`
- `outputs/reports/`

It also inspected the checked-in reporting and plotting path in:

- [src/train.py](/Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/src/train.py:723)
- [src/ledger/metadata_schema.py](/Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/src/ledger/metadata_schema.py:13)
- [src/ledger/mock_ledger.py](/Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/src/ledger/mock_ledger.py:28)

No training was rerun for this audit.

## Verdict

- The nine `A_*`, `B_*`, and `P_*` experiment results currently marked `COMPLETE` appear to be experiment-generated, not hardcoded in source code.
- For all nine complete experiments, `run_summary.json`, `round_metrics.csv`, `*_metrics.csv`, `*_ledger.jsonl`, and `convergence_*.svg` all exist.
- For all nine complete experiments, the metrics CSV values match `run_summary.json` and the selected `best_validation_round` exists in `round_metrics.csv`.
- The convergence and comparison plots are generated from actual metric rows in [src/train.py](/Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/src/train.py:723), not from hardcoded static SVG values.
- The three ablation IDs `AB_C1_FEDAVG_TCN`, `AB_C2_FEDAVG_MLP`, and `AB_C3_FEDAVG_CNN1D` are missing because no standalone output directories or files exist for them under `outputs/`.
- One integrity issue was found: [confusion_matrices.json](/Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/reports/confusion_matrices.json) does not match the confusion matrices in the current `outputs/metrics/*.csv` files even though it names those files as its source.

## 1. Complete Experiments Verified

The following experiments satisfy the required artifact set:

- `A_C1`
- `A_C2`
- `A_C3`
- `B_C1`
- `B_C2`
- `B_C3`
- `P_C1`
- `P_C2`
- `P_C3`

## 2. Artifact Presence Check

| experiment_id | run_summary | round_metrics | metrics_csv | ledger | convergence_plot | result |
| --- | --- | --- | --- | --- | --- | --- |
| A_C1 | PASS | PASS | PASS | PASS | PASS | PASS |
| A_C2 | PASS | PASS | PASS | PASS | PASS | PASS |
| A_C3 | PASS | PASS | PASS | PASS | PASS | PASS |
| B_C1 | PASS | PASS | PASS | PASS | PASS | PASS |
| B_C2 | PASS | PASS | PASS | PASS | PASS | PASS |
| B_C3 | PASS | PASS | PASS | PASS | PASS | PASS |
| P_C1 | PASS | PASS | PASS | PASS | PASS | PASS |
| P_C2 | PASS | PASS | PASS | PASS | PASS | PASS |
| P_C3 | PASS | PASS | PASS | PASS | PASS | PASS |

## 3. Metrics Provenance Check

For all nine complete experiments:

- `metrics_csv.best_validation_round == run_summary.best_validation_round`
- `metrics_csv.best_validation_f1 == run_summary.best_validation_f1`
- `metrics_csv.test_*` values match `run_summary.best_round_test_metrics`
- the `best_validation_round` exists exactly once in `round_metrics.csv`
- `round_metrics.csv` at that round matches `run_summary.best_validation_f1`
- `round_metrics.csv` at that round matches `run_summary.test_f1_at_best_validation_round`

Important nuance:

- `run_summary.json` does not duplicate every test metric as top-level `*_at_best_validation_round` fields.
- The full test metric set is stored under `best_round_test_metrics`.
- The metrics CSVs are therefore consistent with `run_summary.json`, but through the nested `best_round_test_metrics` object rather than only top-level keys.

## 4. Best-Round Logic Check

The current outputs follow clear best-round selection logic:

- the chosen round is `run_summary.best_validation_round`
- `round_metrics.csv` contains that round
- the selected row's `validation_f1` equals `run_summary.best_validation_f1`
- the selected row's `test_f1` and `test_accuracy` match the exported test metrics for that experiment

This means the reported test metrics are tied to the best validation round rather than a separate undocumented selection rule.

## 5. Plot Provenance Check

The checked-in plotting path is in [src/train.py](/Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/src/train.py:723):

- `_write_convergence_plot(...)` reads `round_rows` and plots `train_f1`, `validation_f1`, and `test_f1`
- `_write_comparison_plot(...)` reads numeric metric rows from the experiment summary rows
- `_finalize_run_outputs(...)` calls `_load_round_metrics(...)` and then `_write_convergence_plot(...)`
- `run_experiments(...)` builds `summary_rows` from `*_metrics.csv` and then calls `_write_comparison_plot(...)`

Current plot-file verification:

- [convergence_A_C1.svg](/Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/plots/convergence_A_C1.svg)
- [convergence_P_C1.svg](/Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/plots/convergence_P_C1.svg)
- [comparison_test_f1.svg](/Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/plots/comparison_test_f1.svg)

These files contain `Matplotlib v...` metadata and are not placeholder note SVGs. That matches the current checked-in plot-generation code path.

## 6. Hardcoded Metric Search

Searched for:

- `0.5183`
- `0.8696`
- `0.7582`
- `0.9897`
- `0.9550`

Findings:

- No matches were found in `src/`, `scripts/`, `tests/`, or `configs/`.
- The rounded values do appear in [RESULTS_SUMMARY.md](/Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/docs/RESULTS_SUMMARY.md:22), which is a reporting document outside `outputs/`.
- No hardcoded metric values were found in the checked-in training, plotting, or summary-generation source code.

Conclusion for this check:

- No source-code hardcoding of the audited metric values was found.
- The only non-output occurrences are in the generated-style markdown report.

## 7. Ledger Metadata-Only Check

The ledger schema and writer enforce metadata-only records:

- [src/ledger/metadata_schema.py](/Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/src/ledger/metadata_schema.py:13) defines the allowed fields
- [src/ledger/metadata_schema.py](/Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/src/ledger/metadata_schema.py:36) blocks forbidden payload-name patterns such as `raw`, `weight`, `tensor`, `state_dict`, `optimizer`, and `dataset`
- [src/ledger/mock_ledger.py](/Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/src/ledger/mock_ledger.py:32) validates every append and only permits `main_cluster_head`

Actual JSONL audit findings:

- each discovered ledger contains `50` records
- records are flat JSON objects
- no raw rows were found
- no full model weights were found
- no tensors were found
- no nested payloads were found in the current ledger records

## 8. Missing Ablation Outputs

The following IDs are genuinely missing as standalone outputs under `outputs/`:

| experiment_id | outputs/runs | outputs/metrics | outputs/ledgers | outputs/plots | result |
| --- | --- | --- | --- | --- | --- |
| AB_C1_FEDAVG_TCN | absent | absent | absent | absent | MISSING |
| AB_C2_FEDAVG_MLP | absent | absent | absent | absent | MISSING |
| AB_C3_FEDAVG_CNN1D | absent | absent | absent | absent | MISSING |

This confirms that they are missing because no standalone output directories or files exist for those IDs.

## 9. Reporting-Layer Audit

Good:

- [results_master_table.csv](/Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/reports/results_master_table.csv) matches [summary_all_experiments.csv](/Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/metrics/summary_all_experiments.csv) for all nine complete experiments on the audited metric columns.
- [results_master_table.json](/Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/reports/results_master_table.json) contains the same `12` experiment rows as the CSV version.
- [RESULTS_SUMMARY.md](/Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/docs/RESULTS_SUMMARY.md) correctly lists all matrix experiment IDs as `COMPLETE` or `MISSING`.

Gap:

- No checked-in generator script was found for:
  - `docs/RESULTS_SUMMARY.md`
  - `outputs/reports/results_master_table.csv`
  - `outputs/reports/results_master_table.json`
  - `outputs/reports/confusion_matrices.json`

This is a reproducibility gap for the reporting layer, even though the current master-table values themselves match the experiment outputs.

## 10. Suspicious File / Integrity Failure

### `outputs/reports/confusion_matrices.json`

This file is inconsistent with the current experiment metrics.

Examples:

- `A_C1`
  - report file says `TN=45, FP=51, FN=0, TP=0`
  - current metrics CSV says `[[24524, 0], [256, 0]]`

- `P_C1`
  - report file says `TN=702, FP=30, FN=31, TP=5`
  - current metrics CSV says `[[24517, 7], [164, 92]]`

- `P_C3`
  - report file says `TN=835, FP=13, FN=56, TP=56`
  - current metrics CSV says `[[158485, 3159], [4899, 12632]]`

Conclusion:

- [confusion_matrices.json](/Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/reports/confusion_matrices.json) does not appear to be derived from the current `outputs/metrics/*.csv` files.
- It is likely stale or derived from a different output context.
- It should not be trusted as a current-source report artifact without regeneration.

## 11. Tests Run

- `python3 -m unittest tests.test_experiment_matrix tests.test_ledger_metadata_only tests.test_maincluster_only_ledger_access`
- Result: `OK` (`7` tests)

## Final Conclusion

- The current `A_C1` through `P_C3` result metrics appear experiment-generated and internally consistent across `run_summary.json`, `round_metrics.csv`, `*_metrics.csv`, ledgers, and plots.
- No hardcoded metric values were found in checked-in source code.
- The current plots are real matplotlib plots generated from experiment metrics.
- The three standalone ablation experiment IDs are missing because no standalone output artifacts exist for them under `outputs/`.
- The one clear integrity problem is [confusion_matrices.json](/Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/reports/confusion_matrices.json), which does not match the current experiment outputs.
