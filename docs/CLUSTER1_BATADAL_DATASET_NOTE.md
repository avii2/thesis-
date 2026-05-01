# Cluster 1 BATADAL Dataset Note

Cluster 1 now uses BATADAL water-distribution SCADA telemetry for the active thesis-facing protocol.

## Raw Files

Expected files:

```text
data/raw/batadal/training_dataset_1.csv
data/raw/batadal/training_dataset_2.csv
data/raw/batadal/test_dataset.csv
```

`DATETIME` is parsed as a day-first timestamp using values such as `04/07/16 00`. `ATT_FLAG` is the only label column. `DATETIME` and `ATT_FLAG` are excluded from model features, leaving the 43 telemetry columns with prefixes `L_`, `F_`, `S_`, and `P_`.

## Split Protocol

Training uses all rows from `training_dataset_1.csv` and rows from `training_dataset_2.csv` where `DATETIME < 2016-11-29 05:00`.

Validation uses rows from `training_dataset_2.csv` where `DATETIME >= 2016-11-29 05:00`.

Held-out test uses all rows from `test_dataset.csv`.

Rows are sorted chronologically inside each file. Raw rows are split before windowing, and windows never cross train, validation, or test boundaries. `test_dataset.csv` is never used for training, validation, scaler or imputer fitting, descriptor computation, clustering, threshold tuning, or hyperparameter selection.

## Windowing

Cluster 1 BATADAL uses multivariate time-series windows:

```text
window_length = 48
stride = 1
window_label_rule = last_row
```

For a window `X[t-47:t]`, the label is `ATT_FLAG` at timestamp `t`.

## Federated Emulation

The 12 Cluster 1 leaf clients are deterministic emulated clients named `C1_L001` through `C1_L012`. This is controlled federated emulation over one BATADAL dataset, not a claim that BATADAL contains 12 physical independent water utilities.

Positive training windows are distributed round-robin across the 12 clients. Negative windows fill remaining client quotas while preserving chronological order as much as practical. Validation windows and held-out test windows are partitioned deterministically across the same 12 clients.

## Fixed Sub-Clusters

Cluster 1 keeps exactly two fixed sub-clusters, `H1` and `H2`. Memberships are generated once with `AgglomerativeClustering(n_clusters=2)` from training-only client descriptors and saved to:

```text
outputs_c1_batadal/clustering/cluster1_memberships.json
```

Descriptors use mean/std summaries of `L_*`, `F_*`, and `P_*` telemetry, duty-cycle means of `S_*` statuses, and the training-window attack ratio. H1/H2 names are identifiers only; profile reports should discuss numerical centroid differences without assigning unsupported physical meanings.

## Active Experiments

```text
A_C1_BATADAL: flat CNN1D + FedAvg + weighted arithmetic mean
B_C1_BATADAL: fixed hierarchical CNN1D + FedAvg + weighted arithmetic mean
P_C1_BATADAL: fixed hierarchical CNN1D-BN + FedBN + weighted non-BN mean
```

Primary metric is held-out test F1. Accuracy is secondary. Threshold selection uses validation predictions only and reports both threshold `0.5` and the validation-tuned threshold.
