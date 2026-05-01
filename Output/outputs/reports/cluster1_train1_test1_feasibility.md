# Cluster 1 Train1/Test1 Feasibility Audit

Generated on: 2026-04-27

## Scope

This audit checks whether Cluster 1 can be temporarily restricted to:

- training/validation source: `data/raw/hai_2103/hai-21.03/train1.csv`
- held-out test source: `data/raw/hai_2103/hai-21.03/test1.csv`

No configs, source code, data files, clustering files, or training outputs were modified.

## Source-of-Truth Rules Applied

- Cluster 1 remains HAI 21.03, TCN, FedBN, weighted non-BN aggregation.
- Binary labels are interpreted as `0 = normal` and `1 = attack`.
- The configured Cluster 1 label column is `attack`.
- A valid supervised binary-classification training source must contain both labels.
- Validation/test data must not be used to fit preprocessing or clustering descriptors.
- Report.pdf was used only for academic context; implementation docs remain the coding source of truth.

## File-Level Label Counts

| File | Rows | Normal (`attack=0`) | Attack (`attack=1`) | Contains Both Labels | Binary Values Only |
| --- | ---: | ---: | ---: | --- | --- |
| `train1.csv` | 216001 | 216001 | 0 | NO | YES |
| `train2.csv` | 226801 | 226801 | 0 | NO | YES |
| `train3.csv` | 478801 | 478801 | 0 | NO | YES |
| `test1.csv` | 43201 | 42572 | 629 | YES | YES |
| `test2.csv` | 118801 | 115352 | 3449 | YES | YES |
| `test3.csv` | 108001 | 106466 | 1535 | YES | YES |
| `test4.csv` | 39601 | 38444 | 1157 | YES | YES |
| `test5.csv` | 92401 | 90224 | 2177 | YES | YES |

## Feasibility Answers

Is `train1.csv` alone feasible for supervised binary classification?

**NO.**

Reason: `train1.csv` contains only normal samples: `216001` rows with `attack=0` and `0` rows with `attack=1`. Restricting Cluster 1 training/validation to train1-only would break the supervised binary-classification requirement because the training/validation source has no positive class.

Is `test1.csv` alone feasible for held-out evaluation?

**YES, as a held-out evaluation file.**

Reason: `test1.csv` contains both labels: `42572` normal rows and `629` attack rows.

Is the proposed train1/test1 restricted-data variant valid overall?

**NO.**

Reason: the proposed training/validation source is single-class. Even though `test1.csv` is usable for held-out evaluation, supervised training and validation cannot be built correctly from `train1.csv` alone.

## Expected Breakage If Forced

Forcing train1-only would conflict with current validation rules and training logic:

- Binary label validation expects both labels for supervised classification.
- Positive-class weighting cannot be computed from train labels because positive count is zero.
- Validation-threshold tuning cannot be meaningful if the validation split has no attack samples.
- The model would receive no supervised attack examples during training.
- Any Cluster 1-specific regenerated memberships from train1-only descriptors would be based only on normal data.

## Required Changes If It Had Been Feasible

If both train1 and test1 had contained both labels, the likely required changes would have been:

- Add explicit Cluster 1 config support for separate train/validation source files and held-out test files.
- Restrict Cluster 1 raw file list to `train1.csv` for training/validation and `test1.csv` for held-out evaluation.
- Regenerate only Cluster 1 preprocessing/client/clustering artifacts for the restricted-data variant, because client partitions and descriptors would change.
- Keep Cluster 2, Cluster 3, FL methods, aggregation rules, and ledger logic unchanged.

These changes should **not** be made for this train1-only proposal because the audit shows the training source is invalid.

## Recommendation

Do not restrict Cluster 1 supervised training/validation to `train1.csv` only.

Use a Cluster 1 training/validation source that contains both normal and attack labels, or design a separate anomaly-detection workflow. The latter would be a different modeling objective and should not be mixed with the current supervised binary FCFL implementation without explicit architecture approval.
