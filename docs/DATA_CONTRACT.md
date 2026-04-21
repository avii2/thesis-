# DATA_CONTRACT.md

## 1. Purpose

This document defines the data-layer contract for the simulation-only FCFL CPS/IIoT intrusion-detection project.

It specifies:
- where the raw datasets are expected to live,
- what file types are allowed,
- how labels must be handled,
- which columns are allowed or forbidden as model inputs,
- how preprocessing must be performed,
- what artifacts must be saved,
- and what checks must fail fast.

If the implementation encounters a dataset schema that does not match this contract, it must stop and raise a clear error.

This contract is aligned to the current FCFL report architecture: three independent main clusters, one fixed sub-cluster layer per main cluster, one-time offline Agglomerative Clustering for fixed sub-cluster formation, no cross-cluster averaging, raw-data locality, and metadata-only governance.

---

## 2. Data root and expected folder structure

### 2.1 Expected root

User-provided folder structure:

```text
desktop/thesis/data/
  raw/
    hai_2103/
    ton_iot/
      original_archive/
      combined_telemetry/
    wustl_iiot_2021/
```

### 2.2 Path resolution rule

Recommended implementation choice:

- Default data root:

```text
desktop/thesis/data/
```

- Support override with environment variable:

```text
FCFL_DATA_ROOT
```

Example:

```bash
export FCFL_DATA_ROOT=desktop/thesis/data
```

### 2.3 Path ambiguity

TODO / USER MUST CONFIRM:

- Confirm whether the actual path is literally `desktop/thesis/data/` or a home-expanded path such as `~/Desktop/thesis/data/`.
- Confirm operating system path style before final runs.

---

## 3. Global data-handling rules

These rules apply to all datasets.

1. Only files inside the configured raw dataset folders may be used.
2. Allowed input file type: `.csv`.
3. If multiple CSV files are present for one dataset, they may be concatenated only after schema validation.
4. Raw files must never be modified in place.
5. Timestamp or ordering columns may be retained temporarily only for sorting/partitioning; they must be removed from model features and clustering descriptors.
6. All preprocessing artifacts must be written under `outputs/preprocessing/`.
7. Labels must be converted to binary:
   - `0` = normal / benign
   - `1` = attack / malicious / anomaly
8. Raw data must remain local to leaf-client partitions during FCFL training.
9. Raw records must never be written to the ledger.
10. Full model weights must never be written to the ledger.
11. Validation and test data must never be used to fit imputers, scalers, encoders, or clustering descriptors.
12. If required columns are missing, the implementation must raise an error and print the available columns.
13. No silent guessing of label columns is allowed.

---

## 4. Dataset inventory

| Main cluster | Dataset | Expected raw path | Primary modality | Final task |
|---|---|---|---|---|
| Cluster 1 | HAI 21.03 | `desktop/thesis/data/raw/hai_2103/` | process-control telemetry | binary classification |
| Cluster 2 | TON IoT combined telemetry | `desktop/thesis/data/raw/ton_iot/combined_telemetry/` | IIoT telemetry table | binary classification |
| Cluster 3 | WUSTL-IIOT-2021 | `desktop/thesis/data/raw/wustl_iiot_2021/` | network-flow features | binary classification |

The `original_archive/` under TON IoT must not be used for primary model training unless the combined telemetry dataset is missing or incomplete.

---

## 5. Cluster 1 data contract — HAI 21.03

### 5.1 Expected location

```text
desktop/thesis/data/raw/hai_2103/
```

### 5.2 Allowed files

- `.csv` only

### 5.3 Expected label column

Expected configured label column:

```text
attack
```

TODO / USER MUST CONFIRM:
- Confirm that the HAI files actually contain a column exactly named `attack`.

Permitted candidate label names for data profiling only:

```text
attack
Attack
label
Label
target
Target
```

The implementation may inspect candidates during schema profiling, but final training must use the configured label column only.

### 5.4 Optional process-specific label columns

If present, these are allowed for auxiliary reporting only and must never be used as model input features:

```text
attack_P1
attack_P2
attack_P3
attack_P4
```

### 5.5 Included feature columns

Include:
- all numerical telemetry columns
- after removing excluded columns
- after dropping constant or all-missing columns

### 5.6 Excluded columns

Always exclude from model input:

```text
attack
Attack
label
Label
target
Target
timestamp
Timestamp
time
Time
date
Date
attack_P1
attack_P2
attack_P3
attack_P4
```

If a new identifier-like column appears, stop and log it for user review.

### 5.7 Missing-value handling

Recommended implementation choice:
- numerical missing values: fill with median computed from training data only
- all-missing columns: drop and log
- constant columns: drop and log

### 5.8 Scaling

Recommended implementation choice:
- `StandardScaler`
- fit on Cluster 1 training data only
- apply to validation and test only after fitting

Save to:

```text
outputs/preprocessing/cluster1_hai_scaler.pkl
```

### 5.9 Windowing

Cluster 1 uses a TCN and therefore requires sliding windows.

Recommended implementation choice:

```text
window_length = 32
stride = 8
```

Window labeling rule:

```text
window_label = 1 if any row in the window has label 1
window_label = 0 otherwise
```

If a client partition has fewer than `window_length` rows, skip that partition for window generation and log a warning.

### 5.10 Output tensor shape

```text
batch_size x num_features x window_length
```

---

## 6. Cluster 2 data contract — TON IoT combined telemetry

### 6.1 Expected location

```text
desktop/thesis/data/raw/ton_iot/combined_telemetry/
```

### 6.2 Allowed files

- `.csv` only

### 6.3 Data source rule

Use only the combined telemetry dataset for training.

Do not train directly from:

```text
desktop/thesis/data/raw/ton_iot/original_archive/
```

unless the user explicitly approves a reconstruction step.

### 6.4 Expected label column

TODO / USER MUST CONFIRM:
The exact label column name is not safely inferable from the report alone.

Permitted candidate label names for profiling only:

```text
label
Label
attack
Attack
target
Target
type
Type
```

The implementation must not silently choose among candidates.

### 6.5 Included feature columns

Include:
- all numerical telemetry columns from the combined telemetry table
- after excluding configured forbidden columns
- after dropping constant or all-missing columns

### 6.6 Excluded columns

Always exclude from model input:

```text
<configured label column>
timestamp
Timestamp
time
Time
date
Date
device
Device
device_id
Device_ID
source
Source
src
dst
id
ID
```

Device-origin fields may be preserved in a side metadata file for interpretation only. They must not be used as model features, and they must not manually define sub-cluster memberships.

### 6.7 Missing-value handling

Recommended implementation choice:
- numerical missing values: fill with training-set median
- all-missing columns: drop and log
- constant columns: drop and log

### 6.8 Scaling

Recommended implementation choice:
- `StandardScaler`
- fit on Cluster 2 training data only
- apply to validation and test only after fitting

Save to:

```text
outputs/preprocessing/cluster2_ton_iot_scaler.pkl
```

### 6.9 Windowing

No windowing.

Each row is one sample.

### 6.10 Output tensor shape

```text
batch_size x num_features
```

---

## 7. Cluster 3 data contract — WUSTL-IIOT-2021

### 7.1 Expected location

```text
desktop/thesis/data/raw/wustl_iiot_2021/
```

### 7.2 Allowed files

- `.csv` only

### 7.3 Expected label column

TODO / USER MUST CONFIRM:
The exact binary label column name is not safely inferable from the report alone.

Permitted candidate label names for profiling only:

```text
label
Label
class
Class
target
Target
traffic
Traffic
attack
Attack
```

The implementation must not silently choose among candidates.

### 7.4 Included feature columns

Include:
- all numerical flow-feature columns
- after removing excluded leakage columns and identifiers
- after dropping constant or all-missing columns

### 7.5 Mandatory excluded leakage / identifier columns

Always exclude from model input:

```text
StartTime
LastTime
SrcAddr
DstAddr
sIpId
dIpId
<configured label column>
```

Also exclude if present:

```text
attack_type
AttackType
traffic_class
TrafficClass
timestamp
Timestamp
time
Time
date
Date
```

### 7.6 Missing-value handling

Recommended implementation choice:
- numerical missing values: fill with training-set median
- all-missing columns: drop and log
- constant columns: drop and log

### 7.7 Scaling

Recommended implementation choice:
- `StandardScaler`
- fit on Cluster 3 training data only
- apply to validation and test only after fitting

Save to:

```text
outputs/preprocessing/cluster3_wustl_scaler.pkl
```

### 7.8 Windowing

No time-series windowing.

Each row is one flow sample.

### 7.9 Output tensor shape

```text
batch_size x 1 x num_features
```

---

## 8. Categorical feature handling

If categorical columns remain after explicit exclusions, use the following rule.

Recommended implementation choice:

1. If categorical column cardinality in training data is `<= 20`:
   - one-hot encode using training-fitted categories only.
2. If categorical column cardinality in training data is `> 20`:
   - drop the column and log the reason.
3. Unknown validation/test categories must not crash the pipeline.
4. Label columns and leakage columns must never be one-hot encoded.

Save encoder artifacts under:

```text
outputs/preprocessing/
```

---

## 9. Label contract

### 9.1 Final internal label semantics

All clusters must use:

```text
0 = normal / benign
1 = attack / malicious / anomaly
```

### 9.2 Numeric labels

If labels are already numeric and equal to `{0, 1}`, keep them.

If labels are not binary, stop and ask the user for a mapping.

### 9.3 String labels

Map normal aliases to `0`:

```text
normal
Normal
NORMAL
benign
Benign
BENIGN
0
```

Map attack aliases to `1`:

```text
attack
Attack
ATTACK
malicious
Malicious
MALICIOUS
anomaly
Anomaly
ANOMALY
1
```

If any unknown label values appear, stop and print the unique values.

### 9.4 Label validation

For every dataset after mapping:
- assert labels are only `{0, 1}`
- assert at least one normal sample exists
- assert at least one attack sample exists
- write class counts to JSON summary

Save to:

```text
outputs/reports/label_summary_cluster1.json
outputs/reports/label_summary_cluster2.json
outputs/reports/label_summary_cluster3.json
```

---

## 10. Split and preprocessing contract

### 10.1 Split ratios

Recommended implementation choice:

```text
train = 70%
validation = 15%
test = 15%
```

### 10.2 Split seed

Default seed:

```text
42
```

### 10.3 Split order

1. load raw data
2. validate schema
3. map labels to binary
4. identify timestamp/order columns if available and keep them temporarily for ordering only
5. sort by timestamp if available; otherwise preserve file order
6. create candidate leaf-client partitions
7. split each candidate leaf client into train/validation/test locally
8. construct model/descriptor feature matrices by removing labels, timestamps, identifiers, and leakage-prone fields
9. fit imputers/scalers/encoders using the union of training feature matrices within the same main cluster only
10. apply fitted preprocessing objects to train/validation/test feature matrices
11. compute clustering descriptors using only transformed local training features

### 10.4 Leakage prevention rules

- validation/test data must not be used to compute clustering descriptors
- validation/test data must not be used to fit imputers
- validation/test data must not be used to fit scalers or encoders
- timestamp/order columns may be used only for sorting or partitioning and must not appear in model features, preprocessing-fit inputs, or descriptor features
- leakage-prone columns must be removed before model input creation

---

## 11. Required data artifacts

The implementation must save the following outputs.

### 11.1 Preprocessing artifacts

```text
outputs/preprocessing/cluster1_hai_scaler.pkl
outputs/preprocessing/cluster2_ton_iot_scaler.pkl
outputs/preprocessing/cluster3_wustl_scaler.pkl
```

### 11.2 Data profiling artifacts

```text
outputs/reports/data_profile_cluster1.json
outputs/reports/data_profile_cluster2.json
outputs/reports/data_profile_cluster3.json
```

### 11.3 Client partition artifacts

```text
outputs/clients/cluster1_leaf_clients.json
outputs/clients/cluster2_leaf_clients.json
outputs/clients/cluster3_leaf_clients.json
```

Each client metadata file must include:
- client id
- train/val/test sample counts
- train/val/test label counts

---

## 12. Fail-fast rules

The implementation must stop immediately if any of the following happens:

1. the expected dataset folder does not exist
2. no CSV files are found in a required folder
3. the configured label column is missing
4. the mapped labels are not binary
5. all samples belong to a single class
6. a mandatory leakage column is present in model input after preprocessing
7. preprocessing uses validation/test data for fitting
8. an all-missing or constant column is used as a model feature
9. the same dataset is silently loaded from multiple inconsistent schemas
10. raw data or full model weights are about to be written to the ledger

---

## 13. Items requiring user confirmation

### TODO / USER MUST CONFIRM

1. actual absolute path of the data root
2. actual HAI label column name
3. actual TON IoT combined telemetry label column name
4. actual WUSTL-IIOT-2021 label column name
5. actual CSV filenames inside each raw dataset folder
6. whether any dataset contains high-cardinality categorical fields that should be preserved rather than dropped

Until these are confirmed, the implementation may only perform schema profiling and pipeline scaffolding, not final training.
