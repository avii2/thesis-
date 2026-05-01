# Cluster 1 BATADAL File Split Report

Rows are sorted chronologically inside each source file before splitting. Raw train/validation/test boundaries are applied before sliding-window generation.

- Validation cutoff for `training_dataset_2.csv`: `2016-11-29 05:00`
- Training: all `training_dataset_1.csv` rows plus `training_dataset_2.csv` rows before the cutoff.
- Validation: `training_dataset_2.csv` rows at or after the cutoff.
- Held-out test: all `test_dataset.csv` rows.

| split | source segment | rows | ATT_FLAG=0 | ATT_FLAG=1 | start | end |
|---|---|---:|---:|---:|---|---|
| train | training_dataset_1.csv::train | 8761 | 8761 | 0 | 2014-01-06T00:00:00 | 2015-01-06T00:00:00 |
| train | training_dataset_2.csv::train | 3557 | 3269 | 288 | 2016-07-04T00:00:00 | 2016-11-29T04:00:00 |
| validation | training_dataset_2.csv::validation | 620 | 416 | 204 | 2016-11-29T05:00:00 | 2016-12-25T00:00:00 |
| test | test_dataset.csv::test | 2089 | 1682 | 407 | 2017-01-04T00:00:00 | 2017-04-01T00:00:00 |
