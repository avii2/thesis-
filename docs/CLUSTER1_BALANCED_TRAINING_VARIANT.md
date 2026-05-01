# Cluster 1 Balanced-Training Variant

Cluster 1 balancing was needed because the original HAI 21.03 training files are fully normal and the available attack supervision is sparse. The variant keeps the FCFL architecture, FedBN proposed method, no-cross-cluster averaging rule, and ledger metadata logic unchanged.

## Data Split

- Training source files: `['train1.csv', 'train2.csv', 'train3.csv', 'test1.csv', 'test2.csv']`
- Validation and threshold/ratio-selection file: `['test3.csv']`
- Held-out test files: `['test4.csv', 'test5.csv']`
- Validation windows are not balanced.
- Held-out test windows are not balanced.
- Held-out test predictions are not used for threshold tuning or ratio selection.
- Preprocessing uses `SimpleImputer(strategy="median")` and `StandardScaler` fitted only on balanced training windows.

## Row-Level Audit

- `train1.csv`: rows=216001, attack=0=216001, attack=1=0
- `train2.csv`: rows=226801, attack=0=226801, attack=1=0
- `train3.csv`: rows=478801, attack=0=478801, attack=1=0
- `test1.csv`: rows=43201, attack=0=42572, attack=1=629
- `test2.csv`: rows=118801, attack=0=115352, attack=1=3449
- `test3.csv`: rows=108001, attack=0=106466, attack=1=1535
- `test4.csv`: rows=39601, attack=0=38444, attack=1=1157
- `test5.csv`: rows=92401, attack=0=90224, attack=1=2177

## Window-Level Audit

- `train1.csv`: windows=26997, positive=0, negative=26997
- `train2.csv`: windows=28347, positive=0, negative=28347
- `train3.csv`: windows=59847, positive=0, negative=59847
- `test1.csv`: windows=5397, positive=98, negative=5299
- `test2.csv`: windows=14847, positive=507, negative=14340
- `test3.csv`: windows=13497, positive=224, negative=13273
- `test4.csv`: windows=4947, positive=165, negative=4782
- `test5.csv`: windows=11547, positive=316, negative=11231

## Ratios Tested

- `ratio_1to1`: `{'0': 605, '1': 605}`
- `ratio_2to1`: `{'0': 1210, '1': 605}`
- `ratio_3to1`: `{'0': 1815, '1': 605}`
- `ratio_5to1`: `{'0': 3025, '1': 605}`
- `ratio_8to1`: `{'0': 4840, '1': 605}`
- `ratio_10to1`: `{'0': 6050, '1': 605}`

## Result

- Final selected ratio by validation F1: `ratio_10to1`
- Best proposed validation F1: `0.655827`
- Best proposed held-out test F1 after validation-based selection: `0.421669`
- Same-ratio balanced hierarchical baseline held-out test F1: `0.627936`
- Proposed now beats original `B_C1`: `NO`
- Proposed now beats same-ratio balanced baseline: `NO`
- Balancing reduced false negatives versus original `P_C1`: `NO`
- Balancing created more false positives versus original `P_C1`: `YES`

## Recommendation

Do not replace original `P_C1` with this balanced-training variant for final thesis reporting. If the balanced-training variant is discussed as an ablation or negative result, report `ratio_10to1` because it was selected by validation F1 only. The variant is defensible without test leakage because `test4.csv` and `test5.csv` are held out from training, validation, scaling, imputation, clustering, ratio selection, and threshold tuning.

## Generated Summary

- Ratios generated: `['ratio_10to1', 'ratio_1to1', 'ratio_2to1', 'ratio_3to1', 'ratio_5to1', 'ratio_8to1']`
- Feature count: `79`
- Train pool counts before balancing: `{'0': 134830, '1': 605}`
- Natural validation counts: `{'0': 13273, '1': 224}`
- Natural held-out test counts: `{'0': 16013, '1': 481}`
