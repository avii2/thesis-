# Results Diagnosis

This diagnosis uses only existing repository outputs. No training was rerun, no missing results were fabricated, and no source/config files were changed.

## Sources Inspected

- `outputs/metrics/summary_all_experiments.csv`
- `outputs/metrics/*_metrics.csv`
- `outputs/runs/*/round_metrics.csv`
- `outputs/runs/*/run_summary.json`
- `outputs/models/*/model_manifest.json`
- `outputs/clients/cluster*_leaf_clients.json`
- `outputs/clustering/cluster*_memberships.json`
- `outputs/ledgers/*_ledger.jsonl` for run presence only

## Experiments With Real Outputs

`summary_all_experiments.csv` contains 9 completed experiments. Each experiment below has a per-experiment metrics CSV, per-round metrics, run summary, model manifest, ledger, convergence plot, and saved validation/test prediction files.

| Experiment | Cluster | Dataset | Run category | Model | FL method | Best validation round | Test F1 | Test precision | Test recall | Test AUROC | Test PR-AUC | Test FPR |
|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A_C1 | 1 | HAI 21.03 | baseline_flat | cnn1d | FedAvg | 40 | 0.5622 | 0.9123 | 0.4062 | 0.8998 | 0.5194 | 0.0004 |
| B_C1 | 1 | HAI 21.03 | baseline_uniform_hierarchical | cnn1d | FedAvg | 39 | 0.5622 | 0.9123 | 0.4062 | 0.8997 | 0.5197 | 0.0004 |
| P_C1 | 1 | HAI 21.03 | proposed_specialized_hierarchical | tcn | FedBN | 40 | 0.4932 | 0.8257 | 0.3516 | 0.8389 | 0.4198 | 0.0008 |
| A_C2 | 2 | TON IoT combined telemetry | baseline_flat | cnn1d | FedAvg | 1 | 0.9265 | 0.8687 | 0.9926 | 0.7873 | 0.8337 | 0.2999 |
| B_C2 | 2 | TON IoT combined telemetry | baseline_uniform_hierarchical | cnn1d | FedAvg | 1 | 0.9265 | 0.8687 | 0.9926 | 0.7873 | 0.8337 | 0.2999 |
| P_C2 | 2 | TON IoT combined telemetry | proposed_specialized_hierarchical | compact_mlp | FedProx | 50 | 0.9293 | 0.8679 | 1.0000 | 0.8124 | 0.8557 | 0.3045 |
| A_C3 | 3 | WUSTL-IIOT-2021 | baseline_flat | cnn1d | FedAvg | 50 | 0.8702 | 0.7759 | 0.9906 | 0.9784 | 0.6481 | 0.0310 |
| B_C3 | 3 | WUSTL-IIOT-2021 | baseline_uniform_hierarchical | cnn1d | FedAvg | 42 | 0.8710 | 0.7762 | 0.9922 | 0.9780 | 0.6438 | 0.0310 |
| P_C3 | 3 | WUSTL-IIOT-2021 | proposed_specialized_hierarchical | cnn1d | SCAFFOLD | 50 | 0.8697 | 0.7714 | 0.9965 | 0.9813 | 0.6901 | 0.0320 |

Real output paths are present under:

- `outputs/metrics/{experiment_id}_metrics.csv`
- `outputs/runs/{experiment_id}/round_metrics.csv`
- `outputs/runs/{experiment_id}/run_summary.json`
- `outputs/models/{experiment_id}/model_manifest.json`
- `outputs/ledgers/{experiment_id}_ledger.jsonl`
- `outputs/plots/convergence_{experiment_id}.svg`
- `outputs/predictions/{experiment_id}/selected_threshold_seed_42.json`
- `outputs/predictions/{experiment_id}/validation_predictions_seed_42.npz`
- `outputs/predictions/{experiment_id}/test_predictions_seed_42.npz`

## Experiments Missing

The experiment matrix defines 12 experiments. The following 3 ablation experiments have no current result outputs and are absent from `summary_all_experiments.csv`.

| Missing experiment | Expected role | Missing evidence |
|---|---|---|
| AB_C1_FEDAVG_TCN | Cluster 1 ablation control | No metrics CSV, run directory, run summary, round metrics, ledger, model manifest, convergence plot, or prediction files found |
| AB_C2_FEDAVG_MLP | Cluster 2 ablation control | No metrics CSV, run directory, run summary, round metrics, ledger, model manifest, convergence plot, or prediction files found |
| AB_C3_FEDAVG_CNN1D | Cluster 3 ablation control | No metrics CSV, run directory, run summary, round metrics, ledger, model manifest, convergence plot, or prediction files found |

Because these ablations are missing, the current outputs cannot isolate whether the proposed methods themselves caused the observed differences.

## Cluster 1 Diagnosis

Cluster 1 has weak recall and weak F1, especially for `P_C1`.

| Experiment | Threshold | Confusion matrix `[TN, FP; FN, TP]` | Test F1 | Precision | Recall | FPR | AUROC | PR-AUC | Default-threshold F1 | Default recall | Default FPR |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_C1 | 0.9642 | `[[24514, 10], [152, 104]]` | 0.5622 | 0.9123 | 0.4062 | 0.0004 | 0.8998 | 0.5194 | 0.5573 | 0.4844 | 0.0027 |
| B_C1 | 0.9632 | `[[24514, 10], [152, 104]]` | 0.5622 | 0.9123 | 0.4062 | 0.0004 | 0.8997 | 0.5197 | 0.5618 | 0.4883 | 0.0026 |
| P_C1 | 0.9930 | `[[24505, 19], [166, 90]]` | 0.4932 | 0.8257 | 0.3516 | 0.0008 | 0.8389 | 0.4198 | 0.1418 | 0.5547 | 0.0654 |

Findings:

- Low F1 is likely partly caused by thresholding. `P_C1` selected a very high threshold of `0.9929866194725037`, which leaves only 90 true positives and 166 false negatives in the selected test confusion matrix. At the default threshold, recall rises to 0.5547 but precision collapses to 0.0813 and FPR rises to 0.0654, so the available scores show a poor operating tradeoff.
- Class imbalance is clearly present. The selected test confusion matrix for `P_C1` contains 256 positive test cases out of 24,780 total test cases, about 1.03 percent positives. The output metadata also reports `positive_class_weight = 120.23141361256545`.
- Precision/recall tradeoff is poor for `P_C1`. The tuned threshold gives high precision and low recall; the default threshold gives higher recall but extremely low precision. This is not only a threshold issue: `P_C1` also has lower AUROC and PR-AUC than `A_C1` and `B_C1`.
- Tuned FPR is not too high. `P_C1` tuned FPR is 0.0008, but that low FPR is achieved by missing 166 of 256 positives. The default-threshold FPR of 0.0654 is high for this rare-positive cluster.
- Per-round stability is weak for `P_C1`. Its per-round test F1 ranges from 0.0337 to 0.4932, and its final-round test F1 is 0.4350, below the selected best-validation result.

Most likely reason for weak `P_C1`: severe positive-class rarity plus a poor score distribution for the TCN/FedBN run. Threshold tuning protects precision and FPR, but it does so by suppressing recall.

## Cluster 2 Diagnosis

Cluster 2 has high F1 but a high false-positive rate. This means the result is not weak by F1, but it is unstable as an intrusion-detection operating point because many normal samples are flagged as attacks.

| Experiment | Threshold | Confusion matrix `[TN, FP; FN, TP]` | Test F1 | Precision | Recall | FPR | AUROC | PR-AUC | Default-threshold F1 | Default recall | Default FPR |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_C2 | 0.5080 | `[[4148, 1777], [88, 11762]]` | 0.9265 | 0.8687 | 0.9926 | 0.2999 | 0.7873 | 0.8337 | 0.8000 | 1.0000 | 1.0000 |
| B_C2 | 0.5080 | `[[4148, 1777], [88, 11762]]` | 0.9265 | 0.8687 | 0.9926 | 0.2999 | 0.7873 | 0.8337 | 0.8000 | 1.0000 | 1.0000 |
| P_C2 | 0.6379 | `[[4121, 1804], [0, 11850]]` | 0.9293 | 0.8679 | 1.0000 | 0.3045 | 0.8124 | 0.8557 | 0.8696 | 1.0000 | 0.6000 |

Findings:

- Low F1 is not the main problem in Cluster 2. All three real runs have F1 above 0.926.
- FPR is too high. The tuned results incorrectly flag about 30 percent of normal test samples as attacks: `1777 / 5925` for `A_C2` and `B_C2`, and `1804 / 5925` for `P_C2`.
- Thresholding is a major issue. At the default threshold, `A_C2` and `B_C2` classify every normal test sample as attack (`FPR = 1.0`). `P_C2` improves that default-threshold FPR to 0.6000, but its tuned FPR is still 0.3045.
- Class imbalance is present, but in the attack-majority direction. The selected test confusion matrices contain 11,850 positives and 5,925 negatives, so positives are 66.67 percent of the test set. This makes high recall and high F1 easier to obtain while hiding a large false-positive count.
- Precision/recall tradeoff is poor for an IDS deployment target. `P_C2` reaches recall 1.0000, but with 1,804 false positives. `A_C2` and `B_C2` miss only 88 positives, but also produce 1,777 false positives.
- Per-round behavior shows instability. `A_C2` and `B_C2` select round 1 as best validation round, while their final-round test F1 is about 0.876. `P_C2` improves later, but its per-round FPR ranges from 0.3033 to 0.9652.

Most likely reason for weak practical behavior in Cluster 2: the score distributions do not separate normal and attack cleanly enough to keep recall high while reducing false positives. The high attack-positive test ratio also makes F1 look strong despite high FPR.

## Cluster 3 Diagnosis

Cluster 3 has the most stable final F1, but the default decision threshold is unusable for `A_C3` and `B_C3`.

| Experiment | Threshold | Confusion matrix `[TN, FP; FN, TP]` | Test F1 | Precision | Recall | FPR | AUROC | PR-AUC | Default-threshold F1 | Default recall | Default FPR |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_C3 | 0.0075 | `[[156627, 5017], [165, 17366]]` | 0.8702 | 0.7759 | 0.9906 | 0.0310 | 0.9784 | 0.6481 | 0.0061 | 0.0033 | 0.0060 |
| B_C3 | 0.0093 | `[[156628, 5016], [137, 17394]]` | 0.8710 | 0.7762 | 0.9922 | 0.0310 | 0.9780 | 0.6438 | 0.0031 | 0.0017 | 0.0063 |
| P_C3 | 0.1317 | `[[156468, 5176], [61, 17470]]` | 0.8697 | 0.7714 | 0.9965 | 0.0320 | 0.9813 | 0.6901 | 0.8225 | 0.8845 | 0.0289 |

Findings:

- Current tuned F1 is not weak. All three real runs have test F1 around 0.870.
- Thresholding is critical. `A_C3` has AUROC 0.9784 but default-threshold F1 0.0061 and default recall 0.0033. `B_C3` has AUROC 0.9780 but default-threshold F1 0.0031 and default recall 0.0017. This directly indicates a threshold/calibration issue if default threshold results are considered.
- Class imbalance is present. The selected test confusion matrices contain 17,531 positives and 161,644 negatives, so positives are about 9.78 percent of the test set. The metadata reports `positive_class_weight = 14.747824612950616`.
- FPR is moderate by rate but large by count. Tuned FPR is about 0.031 to 0.032, which creates roughly 5,000 false positives because the normal class is large.
- Precision/recall tradeoff is acceptable but not perfect. Recall is very high, from 0.9906 to 0.9965, while precision stays around 0.771 to 0.776 because of the false-positive count.
- Per-round F1 is stable compared with Clusters 1 and 2. Test F1 ranges are narrow: `A_C3` 0.8684-0.8768, `B_C3` 0.8685-0.8773, and `P_C3` 0.8686-0.8769.

Most likely reason for any weak Cluster 3 reporting: not model ranking, but threshold calibration. With tuned thresholds, Cluster 3 is stable; with default thresholds, `A_C3` and `B_C3` have high AUROC but near-zero recall and F1.

## Cross-Cluster Evidence

- The missing ablations prevent method-level attribution. There is no current output for `AB_C1_FEDAVG_TCN`, `AB_C2_FEDAVG_MLP`, or `AB_C3_FEDAVG_CNN1D`.
- Class imbalance is visible in every cluster, but with different direction and severity:
  - Cluster 1: rare positives, about 1.03 percent positives in the selected test confusion matrix.
  - Cluster 2: attack-majority test set, 66.67 percent positives.
  - Cluster 3: minority positives, about 9.78 percent positives.
- Single-class local partitions are present in all clusters:
  - Cluster 1: 27 single-class local split partitions.
  - Cluster 2: 44 single-class local split partitions.
  - Cluster 3: 28 single-class local split partitions.
- The current results are single-seed outputs. The saved threshold files are `selected_threshold_seed_42.json`, and the model manifests report metadata manifests rather than emitted weight files.

## Ranked Changes Most Likely To Improve Results Without Changing Architecture

1. Use cluster-specific threshold objectives instead of relying only on validation F1.
   Evidence: `P_C1` improves precision/FPR only by dropping recall to 0.3516; `A_C2` and `B_C2` default threshold gives FPR 1.0000; `A_C3` and `B_C3` have AUROC near 0.978 but default-threshold F1 below 0.007. A threshold rule with explicit recall or FPR constraints would directly target the observed failure modes.

2. Calibrate scores before selecting thresholds.
   Evidence: `A_C3` and `B_C3` require thresholds around 0.008-0.009 despite high AUROC, while `P_C1` requires threshold 0.993 and still has weak F1. These are signs that raw probabilities are poorly calibrated even when ranking quality is useful.

3. Re-tune existing class-imbalance handling per cluster.
   Evidence: Cluster 1 uses `positive_class_weight = 120.2314` and still misses 166 of 256 positives in `P_C1`; Cluster 3 uses `positive_class_weight = 14.7478`; Cluster 2 has attack-majority labels and `positive_class_weight = 0.6658`. The imbalance profiles are different enough that one generic setting is unlikely to be optimal.

4. Prioritize `P_C1` TCN/FedBN hyperparameter tuning within the fixed architecture and method.
   Evidence: `P_C1` underperforms the Cluster 1 CNN/FedAvg baselines on F1, recall, AUROC, and PR-AUC. `P_C1` also has the widest practical tradeoff: default recall 0.5547 with precision 0.0813, versus tuned recall 0.3516 with precision 0.8257.

5. Preserve and report best-validation checkpoints rather than final-round behavior, and add the missing ablation/multiseed evidence before drawing method conclusions.
   Evidence: `A_C2` and `B_C2` select round 1, but their final-round test F1 is about 0.876 instead of 0.9265. `P_C1` selects round 40, while final-round test F1 falls to 0.4350. The missing ablations mean current outputs cannot prove whether the proposed FL methods or the model choices caused the differences.

## Bottom Line

- Cluster 1 weakness is mainly severe imbalance plus a poor precision/recall threshold tradeoff, strongest in `P_C1`.
- Cluster 2 looks strong by F1, but the high FPR makes the result weak for IDS use.
- Cluster 3 is stable with tuned thresholds; default thresholds would make `A_C3` and `B_C3` appear failed despite high AUROC.
- The current repository has real outputs for 9 experiments and is missing the 3 required ablation experiment outputs.
