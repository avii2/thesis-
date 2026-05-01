# Cluster 1 BATADAL Validation Summary

- Train row counts and label counts: `12318`, `{'0': 12030, '1': 288}`
- Validation row counts and label counts: `620`, `{'0': 416, '1': 204}`
- Test row counts and label counts: `2089`, `{'0': 1682, '1': 407}`
- Train window counts and label counts: `12224`, `{'0': 11936, '1': 288}`
- Validation window counts and label counts: `573`, `{'0': 369, '1': 204}`
- Test window counts and label counts: `2042`, `{'0': 1635, '1': 407}`
- Number of leaf clients: `12`
- Number of sub-clusters: `2`
- Every client has positive training windows: `YES`
- Test leakage confirmation: `test_dataset.csv` is never used for training, validation, scaler fitting, imputation fitting, clustering, descriptor computation, threshold tuning, or hyperparameter selection.
- Active model family used for P_C1_BATADAL: `cnn1d_bn`

Primary thesis metric for BATADAL Cluster 1 is held-out test F1. Accuracy is reported only as a secondary metric.
