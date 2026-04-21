# FCFL_IMPLEMENTATION_SPEC.md

## 0. Document purpose and status

This document is the **implementation source of truth** for the simulation-only implementation of the proposed FCFL intrusion-detection system.

This document is intended for later use with Codex or any coding assistant. It translates the research report into concrete implementation rules, dataset handling rules, module expectations, experiment definitions, output files, and validation checks.

Primary research source: attached report. The report defines the final system as a fixed clustered federated learning architecture with \(N=3\) independent main clusters, one fixed sub-cluster layer inside each main cluster, one-time offline Agglomerative Clustering for fixed sub-cluster formation, no cross-cluster averaging, raw-data locality, and Hyperledger Fabric used only for metadata-level governance.

This document, together with `ARCHITECTURE_CONTRACT.md`, `DATA_CONTRACT.md`, `ALGORITHMS.md`, `EXPERIMENT_MATRIX.csv`, and `VALIDATION_CHECKS.md`, is the coding source of truth. If any implementation task conflicts with these files or with the research report, stop and ask the user before coding. If the PDF report contains older figure labels or drafting artifacts, follow the implementation documents and flag the mismatch.

---

# 1. Project objective

## 1.1 Main objective

Implement a simulation-only **Fixed Clustered Federated Learning (FCFL)** system for **binary intrusion detection** across three heterogeneous CPS/IIoT domains.

The system must implement:

1. Three independent main clusters.
2. One fixed sub-cluster layer inside each main cluster.
3. One-time offline Agglomerative Clustering inside each main cluster to form fixed sub-clusters.
4. Hierarchical federated training:
   - leaf clients \(\rightarrow\) sub-cluster heads
   - sub-cluster heads \(\rightarrow\) main-cluster heads
5. No cross-cluster parameter averaging.
6. Raw data remain inside simulated leaf-client partitions during FCFL training.
7. Main-cluster heads log only metadata to a ledger/Fabric-compatible logging layer.
8. No raw data and no full model weights are written on-chain or into the ledger.

Important scope note: this implementation is **data-locality-preserving**, not a formal privacy mechanism. It does not implement differential privacy, secure aggregation, or Byzantine robustness.

## 1.2 Implementation objective

The implementation must support:

- data loading and preprocessing for three datasets,
- binary label preparation,
- candidate leaf-client partitioning,
- offline Agglomerative Clustering,
- fixed sub-cluster membership storage,
- hierarchical FCFL training,
- FedBN for Cluster 1,
- FedProx for Cluster 2,
- SCAFFOLD for Cluster 3,
- uniform hierarchical FCFL baseline,
- flat per-cluster FL baseline,
- proposed specialized hierarchical FCFL method,
- metrics export,
- metadata ledger logging,
- validation tests.

---

# 2. Source-of-truth constraints from the report

The following constraints are non-negotiable.

## 2.1 Architecture constraints

- There are exactly **three main clusters**.
- Each main cluster is an independent FCFL problem.
- Each main cluster contains exactly one fixed sub-cluster layer.
- All leaf clients must be placed under fixed sub-clusters in the proof-of-concept.
- No cross-cluster averaging is allowed.
- No global central server across the three main clusters is allowed.
- Each main cluster may use a different dataset, model family, FL method, and feature space.
- Inside a single main cluster, all sub-clusters must use aggregation-compatible models.

## 2.2 Clustering constraints

- Agglomerative Clustering is used only as a **one-time offline / pre-training** step.
- Agglomerative Clustering is applied separately inside each main cluster.
- No clustering is performed across main clusters.
- No reclustering is allowed during FCFL training rounds.
- Sub-cluster memberships are frozen before the first training round.
- The same fixed sub-cluster memberships must be reused for:
  - uniform hierarchical FCFL baseline,
  - proposed specialized hierarchical FCFL.

## 2.3 Learning constraints

All clusters solve supervised binary classification:

- label `0` = normal / benign
- label `1` = attack / malicious

Cluster mapping:

| Main cluster | Dataset | Model | FL method | Aggregation |
|---|---|---|---|---|
| Cluster 1 | HAI 21.03 | TCN | FedBN | weighted non-BN mean |
| Cluster 2 | TON IoT combined telemetry | compact MLP | FedProx | weighted arithmetic mean |
| Cluster 3 | WUSTL-IIOT-2021 | 1D-CNN | SCAFFOLD | weighted arithmetic mean |

## 2.4 Blockchain / ledger constraints

- Only main-cluster heads interact with the ledger/Fabric layer.
- Sub-cluster heads do not write on-chain.
- Leaf clients do not write on-chain.
- Ledger stores metadata only.
- Raw data must never be logged.
- Full model weights must never be logged.
- Model hashes may be logged.
- Sub-cluster configuration hashes and membership hashes may be logged as metadata evidence.

---

# 3. Dataset inventory and expected file locations

## 3.1 User-provided folder structure

The current expected dataset root is:

```text
desktop/thesis/data/
```

The expected raw dataset folders are:

```text
desktop/thesis/data/raw/hai_2103/
desktop/thesis/data/raw/ton_iot/original_archive/
desktop/thesis/data/raw/ton_iot/combined_telemetry/
desktop/thesis/data/raw/wustl_iiot_2021/
```

## 3.2 Implementation path rule

Recommended implementation choice:

Use a configurable data root instead of hardcoding absolute paths.

Default:

```text
DATA_ROOT = desktop/thesis/data
RAW_DATA_ROOT = desktop/thesis/data/raw
```

Codex must implement support for an environment variable:

```text
FCFL_DATA_ROOT
```

If `FCFL_DATA_ROOT` is set, use it instead of the default path.

Example:

```text
FCFL_DATA_ROOT=desktop/thesis/data
```

TODO / USER MUST CONFIRM:

- Confirm whether the actual path is `desktop/thesis/data/` or `~/Desktop/thesis/data/`.
- Confirm operating system path style before final run.

## 3.3 Dataset folder usage

### Cluster 1: HAI 21.03

Use:

```text
desktop/thesis/data/raw/hai_2103/
```

Expected file type:

```text
.csv
```

If multiple CSV files exist, concatenate them only after schema validation.

### Cluster 2: TON IoT combined telemetry

Use only:

```text
desktop/thesis/data/raw/ton_iot/combined_telemetry/
```

Do not use this folder for training unless explicitly required:

```text
desktop/thesis/data/raw/ton_iot/original_archive/
```

The `original_archive/` folder may be used only for reference or reconstruction if the combined telemetry files are missing.

If `combined_telemetry/` is empty, stop and ask the user.

### Cluster 3: WUSTL-IIOT-2021

Use:

```text
desktop/thesis/data/raw/wustl_iiot_2021/
```

Expected file type:

```text
.csv
```

If multiple CSV files exist, concatenate them only after schema validation.

---

# 4. Dataset-specific preprocessing rules

## 4.1 Universal preprocessing rules

These rules apply to all clusters.

1. Load CSV files.
2. Validate that all required columns exist.
3. Do not silently infer unknown label columns.
4. Do not silently drop columns unless they are explicitly listed in this specification or are constant/all-missing columns.
5. Convert labels to binary:
   - normal/benign = 0
   - attack/malicious/anomaly = 1
6. Remove label columns from model features.
7. Keep timestamp or ordering columns only temporarily for ordering/partitioning if needed.
8. Remove timestamp, identifier, and leakage-prone fields from model features before model-input creation and descriptor computation.
9. Convert numerical features to `float32`.
10. Handle missing values using training-set median imputation.
11. Fit preprocessing transformations using training data only.
12. Apply the fitted transformations to validation and test data.
13. Save preprocessing artifacts under:

```text
outputs/preprocessing/
```

14. If a required label column is missing, stop and raise a clear error showing available columns.
15. If a dataset contains unknown categorical columns, follow the categorical handling rule in Section 4.5.

## 4.2 Cluster 1 preprocessing: HAI 21.03

Dataset path:

```text
desktop/thesis/data/raw/hai_2103/
```

Task:

```text
binary process-control telemetry intrusion detection
```

Model family:

```text
TCN
```

Input type:

```text
time-windowed multivariate telemetry
```

### 4.2.1 Label column

Expected label column:

```text
attack
```

TODO / USER MUST CONFIRM:

- Confirm that the HAI files contain a column exactly named `attack`.
- If the label column has a different name, update the config before implementation.

Allowed label-column candidates for profiling only:

```text
attack
Attack
label
Label
target
Target
```

Codex must not choose among candidates silently. It may print candidates during data profiling, but final training must use the configured label column.

### 4.2.2 Feature columns

Include:

```text
all numerical process-control telemetry columns except excluded columns
```

Exclude:

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
```

Also exclude any process-specific attack label columns if present, such as:

```text
attack_P1
attack_P2
attack_P3
attack_P4
```

These may be used only for auxiliary analysis and must not be model input features.

### 4.2.3 Missing values

Recommended implementation choice:

- Numerical missing values: fill using median computed from training data only.
- If a column is all missing in training data, drop the column and log it.
- If a column is constant in training data, drop the column and log it.

### 4.2.4 Scaling

Recommended implementation choice:

- Use `StandardScaler`.
- Fit on Cluster 1 training features only.
- Apply to validation and test.
- Save scaler to:

```text
outputs/preprocessing/cluster1_hai_scaler.pkl
```

### 4.2.5 Windowing for TCN

Recommended implementation choice:

Use sliding windows.

```text
window_length = 32
stride = 8
```

Window label rule:

```text
window_label = 1 if any row inside the window has attack label 1
window_label = 0 otherwise
```

Input shape for TCN:

```text
batch_size x num_features x window_length
```

If a client partition has fewer than `window_length` rows, skip that partition for window generation and log a warning.

---

## 4.3 Cluster 2 preprocessing: TON IoT combined telemetry

Dataset path:

```text
desktop/thesis/data/raw/ton_iot/combined_telemetry/
```

Task:

```text
binary heterogeneous IIoT telemetry intrusion detection
```

Model family:

```text
compact MLP
```

Input type:

```text
fixed-length tabular telemetry vector
```

### 4.3.1 Label column

TODO / USER MUST CONFIRM:

The report states that TON IoT combined telemetry is used for binary classification, but the exact label-column name is not safely inferable from the report alone.

Before training, user must confirm the exact label column.

Expected possible candidates for profiling only:

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

Codex must not silently choose the label column.

### 4.3.2 Feature columns

Include:

```text
all numerical columns from the TON IoT combined telemetry file except excluded columns
```

Exclude:

```text
configured label column
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

Device-origin information may be used for interpretation only if present. It must not manually define the fixed sub-cluster membership, because sub-clusters are formed by Agglomerative Clustering.

### 4.3.3 Missing values

Recommended implementation choice:

- Numerical missing values: fill using median computed from training data only.
- If a column is all missing in training data, drop the column and log it.
- If a column is constant in training data, drop the column and log it.

### 4.3.4 Scaling

Recommended implementation choice:

- Use `StandardScaler`.
- Fit on Cluster 2 training features only.
- Apply to validation and test.
- Save scaler to:

```text
outputs/preprocessing/cluster2_ton_iot_scaler.pkl
```

### 4.3.5 Windowing

No sliding windows.

Each row is one sample.

Input shape for compact MLP:

```text
batch_size x num_features
```

---

## 4.4 Cluster 3 preprocessing: WUSTL-IIOT-2021

Dataset path:

```text
desktop/thesis/data/raw/wustl_iiot_2021/
```

Task:

```text
binary IIoT network-flow intrusion detection
```

Model family:

```text
1D-CNN
```

Input type:

```text
fixed-length flow-feature vector
```

### 4.4.1 Label column

TODO / USER MUST CONFIRM:

The report states that WUSTL-IIOT-2021 can be converted to binary classification by labeling all attacks as 1 and normal traffic as 0, but the exact label-column name is not safely inferable from the report alone.

Before training, user must confirm the exact label column.

Expected possible candidates for profiling only:

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

Codex must not silently choose the label column.

### 4.4.2 Feature columns

Include:

```text
all numerical network-flow feature columns except excluded columns
```

Mandatory excluded leakage / identifier columns:

```text
StartTime
LastTime
SrcAddr
DstAddr
sIpId
dIpId
```

Also exclude:

```text
configured label column
attack type column if present
traffic class text column if separate from binary label
timestamp
Timestamp
time
Time
date
Date
```

### 4.4.3 Missing values

Recommended implementation choice:

- Numerical missing values: fill using median computed from training data only.
- If a column is all missing in training data, drop the column and log it.
- If a column is constant in training data, drop the column and log it.

### 4.4.4 Scaling

Recommended implementation choice:

- Use `StandardScaler`.
- Fit on Cluster 3 training features only.
- Apply to validation and test.
- Save scaler to:

```text
outputs/preprocessing/cluster3_wustl_scaler.pkl
```

### 4.4.5 Windowing

No time-series windowing for Cluster 3.

Each row is one flow sample.

Input shape for 1D-CNN:

```text
batch_size x 1 x num_features
```

---

## 4.5 Categorical feature handling

If categorical columns remain after explicit exclusions:

Recommended implementation choice:

1. If the categorical column has at most 20 unique values in training data:
   - apply one-hot encoding,
   - fit categories on training data only,
   - unknown validation/test categories map to all-zero for that feature group.
2. If the categorical column has more than 20 unique values:
   - drop the column,
   - log the column name and reason.
3. Never one-hot encode label columns.
4. Never one-hot encode leakage columns.

Save encoder artifacts to:

```text
outputs/preprocessing/
```

---

# 5. Label definitions

All clusters are supervised binary classification tasks.

## 5.1 Universal label mapping

Final internal labels must be:

```text
normal / benign = 0
attack / malicious / anomaly = 1
```

## 5.2 Numeric labels

If the configured label column is numeric:

- If values are already `{0, 1}`, keep them.
- If values are `{1, 0}`, keep them.
- If values are not binary, user must provide mapping.

TODO / USER MUST CONFIRM:

- For each dataset, confirm whether labels are already binary.

## 5.3 String labels

If the configured label column is string/object:

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

If any unknown label values are found, stop and print all unique label values.

Do not guess.

## 5.4 Binary-label validation

For every dataset, after mapping:

- assert labels are only `{0, 1}`,
- assert at least one normal sample exists,
- assert at least one attack sample exists,
- log class counts.

Output:

```text
outputs/reports/label_summary_cluster1.json
outputs/reports/label_summary_cluster2.json
outputs/reports/label_summary_cluster3.json
```

---

# 6. Feature inclusion / exclusion rules

## 6.1 Universal model-input exclusion

Never include these as model input:

```text
label column
attack type column
timestamp columns
date/time columns
source/destination address columns
unique identifiers
client id
subcluster id
cluster id
raw file path
```

## 6.2 Universal clustering descriptor exclusion

The clustering descriptor must use the same cleaned feature space used for training.

Do not use:

```text
label
attack type
timestamp
client id
subcluster id
cluster id
source/destination address fields
leakage-prone fields
```

## 6.3 Feature consistency requirement

Inside each main cluster:

- all leaf clients must have the same feature dimension,
- all sub-clusters must use the same model architecture,
- all sub-cluster and main-cluster models must have compatible parameters.

Across different main clusters:

- feature spaces may differ,
- model families may differ,
- FL methods may differ,
- no model parameters are averaged across clusters.

---

# 7. Train / validation / test splitting rules

## 7.1 Split ratios

Recommended implementation choice:

Use:

```text
train = 70%
validation = 15%
test = 15%
```

## 7.2 Split seed

Use fixed seed:

```text
seed = 42
```

For final experiments, run seeds:

```text
[42, 123, 2025]
```

## 7.3 Dataset split procedure

Recommended implementation choice:

1. Load each dataset and validate schema.
2. Map labels to binary.
3. Identify timestamp or ordering columns if available; keep them temporarily only for ordering/partitioning.
4. Sort samples using timestamp if a timestamp exists; otherwise preserve file order.
5. Create candidate leaf-client partitions according to Section 8.
6. Inside each candidate leaf client, split local samples into train/validation/test using the 70/15/15 ratio.
7. Construct feature matrices for model input and clustering descriptors by removing labels, timestamps, identifiers, and leakage-prone fields while keeping timestamp/order metadata only outside the feature matrices if needed for auditing.
8. Fit imputers, encoders, and scalers using the union of training feature matrices within the same main cluster only.
9. Apply the fitted preprocessing objects to train/validation/test feature matrices.
10. Compute Agglomerative Clustering descriptors from the transformed local training features only.

Important:

- Agglomerative Clustering descriptors must be computed using only each leaf client's **training** partition after training-only preprocessing has been fitted.
- Validation and test data must never be used to compute clustering descriptors.
- Validation and test data must never be used to fit scalers, imputers, or encoders.
- Timestamp/order columns may be used only to order samples or create contiguous partitions; they must not be used as model features, scaler inputs, imputer inputs, encoder inputs, or descriptor features.
- If a candidate leaf client has too few samples for 70/15/15 splitting, merge it with the nearest adjacent candidate partition before clustering or stop and ask the user.

## 7.4 Class-presence validation

After splitting, for each main cluster:

- the union of all training partitions must contain both labels 0 and 1,
- the union of all validation partitions must contain both labels 0 and 1,
- the union of all test partitions must contain both labels 0 and 1.

If any split is single-class at main-cluster level, stop and ask the user.

At individual leaf-client level, single-class local partitions are allowed but must be logged.

---

# 8. Candidate leaf-client formation rules

## 8.1 Leaf-client counts

Recommended implementation choice:

| Cluster | Dataset | Candidate leaf clients | Fixed sub-clusters |
|---|---:|---:|---:|
| Cluster 1 | HAI 21.03 | 12 | \(K_1=2\) |
| Cluster 2 | TON IoT combined telemetry | 15 | \(K_2=3\) |
| Cluster 3 | WUSTL-IIOT-2021 | 15 | \(K_3=3\) |

These values are implementation defaults. They provide enough candidate leaf clients for Agglomerative Clustering while keeping training manageable.

## 8.2 Leaf-client partition method

Recommended implementation choice:

For each main cluster:

1. Use the cleaned, label-mapped dataset.
2. Sort by timestamp if available.
3. If timestamp is unavailable, preserve file order.
4. Split the ordered samples into `num_leaf_clients` contiguous shards.
5. Each shard becomes one candidate leaf client.
6. Each candidate leaf client is then split locally into train/validation/test.

## 8.3 Leaf-client IDs

Use deterministic IDs:

Cluster 1:

```text
C1_L001
C1_L002
...
C1_L012
```

Cluster 2:

```text
C2_L001
C2_L002
...
C2_L015
```

Cluster 3:

```text
C3_L001
C3_L002
...
C3_L015
```

## 8.4 Leaf-client metadata file

Save candidate client metadata to:

```text
outputs/clients/cluster1_leaf_clients.json
outputs/clients/cluster2_leaf_clients.json
outputs/clients/cluster3_leaf_clients.json
```

Each file must contain:

```json
{
  "cluster_id": 1,
  "dataset": "HAI 21.03",
  "num_leaf_clients": 12,
  "clients": [
    {
      "client_id": "C1_L001",
      "num_train_samples": 0,
      "num_val_samples": 0,
      "num_test_samples": 0,
      "train_label_counts": {"0": 0, "1": 0},
      "val_label_counts": {"0": 0, "1": 0},
      "test_label_counts": {"0": 0, "1": 0}
    }
  ]
}
```

---

# 9. Agglomerative Clustering specification

## 9.1 Purpose

Agglomerative Clustering is used only to form fixed sub-cluster memberships before FCFL training begins.

It must not be used during training rounds.

## 9.2 Clustering scope

Run Agglomerative Clustering separately for each main cluster:

```text
Cluster 1 only uses Cluster 1 leaf-client descriptors.
Cluster 2 only uses Cluster 2 leaf-client descriptors.
Cluster 3 only uses Cluster 3 leaf-client descriptors.
```

No clustering across main clusters is allowed.

## 9.3 Number of fixed sub-clusters

Use:

```text
K1 = 2
K2 = 3
K3 = 3
```

## 9.4 Descriptor definition

For each candidate leaf client \(i\), compute a descriptor using only its local training partition \(X_i\).

Let each \(x \in X_i\) be a cleaned, transformed feature vector after training-only imputation/scaling/encoding in the input space of the corresponding main cluster. For Cluster 1, compute descriptors from row-level transformed telemetry features before TCN window generation.

Compute:

\[
z_i = [\mu_i ; \sigma_i]
\]

where:

\[
\mu_i = \frac{1}{n_i}\sum_{x\in X_i}x
\]

and:

\[
\sigma_i =
\sqrt{
\frac{1}{n_i}\sum_{x\in X_i}(x-\mu_i)^2
}
\]

Here:

- \(\mu_i\) is the feature-wise mean vector.
- \(\sigma_i\) is the feature-wise standard-deviation vector.
- \(z_i\) is the concatenation of the two vectors.

Do not include labels in \(z_i\).

Do not include timestamps or leakage fields in \(z_i\).

Do not include raw rows in the clustering input.

## 9.5 Descriptor standardization

Recommended implementation choice:

Before clustering, standardize descriptors inside each main cluster using `StandardScaler`.

Fit the descriptor scaler only on the descriptors from that main cluster.

Save descriptor scalers to:

```text
outputs/clustering/cluster1_descriptor_scaler.pkl
outputs/clustering/cluster2_descriptor_scaler.pkl
outputs/clustering/cluster3_descriptor_scaler.pkl
```

## 9.6 Agglomerative Clustering parameters

Recommended implementation choice:

Use scikit-learn Agglomerative Clustering with:

```text
n_clusters = K_m
linkage = ward
metric = euclidean
```

If using an older scikit-learn version where the parameter name is `affinity` instead of `metric`, use:

```text
affinity = euclidean
```

Do not change linkage or metric without user confirmation.

## 9.7 Clustering outputs

Save membership files:

```text
outputs/clustering/cluster1_memberships.json
outputs/clustering/cluster2_memberships.json
outputs/clustering/cluster3_memberships.json
```

Required JSON structure:

```json
{
  "cluster_id": 1,
  "dataset": "HAI 21.03",
  "clustering_method": "AgglomerativeClustering",
  "n_subclusters": 2,
  "linkage": "ward",
  "metric": "euclidean",
  "descriptor": "feature_mean_std",
  "memberships": {
    "C1_L001": "H1",
    "C1_L002": "H2"
  }
}
```

## 9.8 Sub-cluster naming

Cluster 1:

```text
H1
H2
```

Cluster 2:

```text
T1
T2
T3
```

Cluster 3:

```text
W1
W2
W3
```

## 9.9 Frozen membership rule

After membership files are created:

- do not modify them during training,
- do not regenerate them automatically,
- all training runs must load the saved memberships,
- if membership files already exist, reuse them unless user explicitly sets `--recluster`.

Default:

```text
recluster = false
```

If `--recluster` is used, require explicit user confirmation.

---

# 10. Fixed sub-cluster definitions and mapping rules

## 10.1 Cluster 1 fixed sub-clusters

| Sub-cluster | Formation |
|---|---|
| H1 | Agglomerative group inside Cluster 1 |
| H2 | Agglomerative group inside Cluster 1 |

Do not interpret H1 or H2 as manually assigned process labels.

## 10.2 Cluster 2 fixed sub-clusters

| Sub-cluster | Formation |
|---|---|
| T1 | Agglomerative group inside Cluster 2 |
| T2 | Agglomerative group inside Cluster 2 |
| T3 | Agglomerative group inside Cluster 2 |

Do not interpret T1/T2/T3 as manually assigned device-origin labels.

Device-origin information may be used only for interpretation if available.

## 10.3 Cluster 3 fixed sub-clusters

| Sub-cluster | Formation |
|---|---|
| W1 | Agglomerative group inside Cluster 3 |
| W2 | Agglomerative group inside Cluster 3 |
| W3 | Agglomerative group inside Cluster 3 |

Do not form W1/W2/W3 using attack labels, attack families, source addresses, destination addresses, or leakage-prone identifiers.

## 10.4 Sub-cluster assignment rule

Each leaf client must belong to exactly one sub-cluster.

Validation:

```text
number of unique assigned clients == number of leaf clients
no missing client IDs
no duplicate client IDs
no empty sub-clusters
```

If any sub-cluster is empty, stop and raise an error.

---

# 11. Cluster-wise model definitions

All models output a single logit for binary classification.

Use:

```text
loss = BCEWithLogitsLoss
```

Prediction rule:

```text
probability = sigmoid(logit)
predicted_label = 1 if probability >= 0.5 else 0
```

Threshold tuning is not part of the first implementation.

## 11.1 Cluster 1 model: TCN classifier

Recommended implementation choice.

Model name:

```text
TCNClassifier
```

Input shape:

```text
batch_size x num_features x window_length
```

Default:

```text
window_length = 32
```

Architecture:

```text
Input: [B, F, 32]

TCN Block 1:
- Conv1d(in_channels=F, out_channels=32, kernel_size=3, dilation=1, padding=same)
- BatchNorm1d(32)
- ReLU
- Dropout(0.1)

TCN Block 2:
- Conv1d(32, 64, kernel_size=3, dilation=2, padding=same)
- BatchNorm1d(64)
- ReLU
- Dropout(0.1)

TCN Block 3:
- Conv1d(64, 64, kernel_size=3, dilation=4, padding=same)
- BatchNorm1d(64)
- ReLU
- Dropout(0.1)

Global average pooling over time
Linear(64, 32)
ReLU
Dropout(0.1)
Linear(32, 1)
Output: binary logit
```

Important for FedBN:

- BatchNorm statistics must remain local.
- BatchNorm running statistics must not be averaged globally.

## 11.2 Cluster 2 model: compact MLP classifier

Recommended implementation choice.

Model name:

```text
CompactMLPClassifier
```

Input shape:

```text
batch_size x num_features
```

Architecture:

```text
Input: [B, F]

Linear(F, 64)
ReLU
Dropout(0.2)

Linear(64, 32)
ReLU
Dropout(0.2)

Linear(32, 1)
Output: binary logit
```

No BatchNorm in the default MLP.

## 11.3 Cluster 3 model: 1D-CNN classifier

Recommended implementation choice.

Model name:

```text
CNN1DClassifier
```

Input shape:

```text
batch_size x 1 x num_features
```

Architecture:

```text
Input: [B, 1, F]

Conv1d(1, 32, kernel_size=3, padding=1)
ReLU
MaxPool1d(kernel_size=2)

Conv1d(32, 64, kernel_size=3, padding=1)
ReLU
AdaptiveAvgPool1d(1)

Flatten
Linear(64, 32)
ReLU
Dropout(0.2)
Linear(32, 1)
Output: binary logit
```

Do not add BatchNorm to Cluster 3 in the first implementation.

Reason:

```text
Cluster 3 uses SCAFFOLD, not FedBN. Avoiding BatchNorm reduces ambiguity about local normalization state.
```

---

# 12. FL method and aggregation rules per cluster

## 12.1 Common hierarchical FCFL structure

For every main cluster \(m\):

1. Main-cluster head holds \(w_m^t\).
2. Main-cluster head sends \(w_m^t\) to all sub-cluster heads.
3. Sub-cluster head initializes:

\[
w_{m,s}^{t} \leftarrow w_m^t
\]

4. Leaf clients train locally from \(w_{m,s}^{t}\).
5. Sub-cluster head aggregates leaf-client updates into \(w_{m,s}^{t+1}\).
6. Main-cluster head aggregates sub-cluster models into \(w_m^{t+1}\).
7. Main-cluster head logs metadata.

## 12.2 Standard weighted aggregation

For sub-cluster \(s\) in main cluster \(m\):

\[
w_{m,s}^{t+1}
=
\sum_{i\in C_{m,s}}
\frac{n_i}{N_{m,s}} w_{m,s,i}^{t+1}
\]

where:

\[
N_{m,s}=\sum_{i\in C_{m,s}}n_i
\]

For main-cluster aggregation:

\[
w_m^{t+1}
=
\sum_{s\in S_m}
\frac{N_{m,s}}{N_m}w_{m,s}^{t+1}
\]

where:

\[
N_m=\sum_{s\in S_m}N_{m,s}
\]

## 12.3 Cluster 1: FedBN

Cluster 1 uses:

```text
FL method: FedBN
Aggregation: weighted non-BN mean
```

Implementation rule:

- Aggregate only non-BatchNorm parameters.
- Keep BatchNorm running statistics local.
- Do not average:
  - `running_mean`
  - `running_var`
  - `num_batches_tracked`
  - BatchNorm affine parameters if configured as local.

Recommended implementation choice:

Treat all BatchNorm-related keys as local.

Exclude keys containing:

```text
bn
batchnorm
running_mean
running_var
num_batches_tracked
```

Case-insensitive matching.

## 12.4 Cluster 2: FedProx

Cluster 2 uses:

```text
FL method: FedProx
Aggregation: weighted arithmetic mean
```

Local objective for client \(i\) in sub-cluster \(s\):

\[
F_{m,s,i}(w) + \frac{\mu}{2}\|w-w_{m,s}^t\|_2^2
\]

Recommended implementation choice:

```text
mu = 0.01
```

FedProx local optimizer:

```text
Adam
learning_rate = 1e-3
local_epochs = 1
batch_size = 128
```

If training is unstable, tune only after the first complete implementation works.

## 12.5 Cluster 3: SCAFFOLD

Cluster 3 uses:

```text
FL method: SCAFFOLD
Aggregation: weighted arithmetic mean
```

Control variates:

- main-cluster server control variate: \(c_m^t\)
- local client control variate: \(c_{m,s,i}^t\)

Recommended implementation choice:

Use full participation in the first implementation.

```text
client_fraction = 1.0
```

Local update:

\[
y_{m,s,i}^{t,k+1}
=
y_{m,s,i}^{t,k}
-
\eta_\ell
\left(
g_{m,s,i}(y_{m,s,i}^{t,k})
-
c_{m,s,i}^t
+
c_m^t
\right)
\]

Use \(E\) for number of local SCAFFOLD steps.

Recommended implementation choice:

```text
optimizer style = SGD-style corrected update
learning_rate = 0.01
local_epochs = 1
batch_size = 128
```

Client control-variate update:

\[
c_{m,s,i}^{t+1}
=
c_{m,s,i}^{t}
-
c_m^t
+
\frac{1}{E\eta_\ell}
\left(
w_{m,s}^{t}-w_{m,s,i}^{t+1}
\right)
\]

Control-variate increment:

\[
\Delta c_{m,s,i}^{t}
=
c_{m,s,i}^{t+1}-c_{m,s,i}^{t}
\]

Server control-variate update:

\[
c_m^{t+1}
=
c_m^t
+
\frac{1}{Q_m}
\sum_{(s,i)\in P_m^t}
\Delta c_{m,s,i}^{t}
\]

where:

```text
P_m^t = participating leaf clients in Cluster 3
Q_m = total number of leaf clients in Cluster 3
```

Sub-cluster heads forward both:

- model updates,
- SCAFFOLD control-variate increments.

Sub-cluster heads do not maintain independent SCAFFOLD server control variates in the first implementation.

---

# 13. FCFL round procedure

## 13.1 Before round 1

Before training starts:

1. Load datasets.
2. Preprocess datasets.
3. Create candidate leaf clients.
4. Compute descriptors using training partitions only.
5. Run Agglomerative Clustering inside each main cluster.
6. Save fixed membership files.
7. Initialize main-cluster global models.
8. Initialize sub-cluster heads.
9. Initialize ledger.

## 13.2 One FCFL round

For each main cluster \(m\), independently:

1. Main-cluster head holds current parent model \(w_m^t\).
2. Main-cluster head sends \(w_m^t\) to each sub-cluster head.
3. Each sub-cluster head initializes \(w_{m,s}^t \leftarrow w_m^t\).
4. Each sub-cluster head sends \(w_{m,s}^t\) to its leaf clients.
5. Each leaf client trains locally.
6. Each leaf client sends model update to its sub-cluster head.
7. For Cluster 3, each leaf client also updates its SCAFFOLD local control variate and sends the control-variate increment.
8. Each sub-cluster head aggregates leaf-client models.
9. Each sub-cluster head sends sub-cluster model to main-cluster head.
10. For Cluster 3, sub-cluster heads also forward control-variate increments.
11. Main-cluster head aggregates sub-cluster models into \(w_m^{t+1}\).
12. Main-cluster head updates method-specific state if needed.
13. Main-cluster head computes model hash.
14. Main-cluster head writes metadata to ledger.
15. Updated \(w_m^{t+1}\) becomes parent initialization for next round.

## 13.3 Cross-cluster rule

Clusters may run sequentially or in parallel.

But:

```text
No model parameters from different main clusters may ever be averaged.
```

---

# 14. Baselines to implement

## 14.1 Baseline A: flat per-cluster FL

Purpose:

```text
Measure performance without sub-cluster hierarchy.
```

Configuration:

| Cluster | Dataset | Model | FL method | Hierarchy |
|---|---|---|---|---|
| C1 | HAI 21.03 | 1D-CNN | FedAvg | flat |
| C2 | TON IoT combined telemetry | 1D-CNN | FedAvg | flat |
| C3 | WUSTL-IIOT-2021 | 1D-CNN | FedAvg | flat |

Rules:

- One main-cluster head per domain.
- No sub-cluster layer.
- No Agglomerative Clustering.
- No cross-cluster averaging.

## 14.2 Baseline B: uniform hierarchical FCFL

Purpose:

```text
Measure effect of hierarchy while removing model/FL specialization.
```

Configuration:

| Cluster | Dataset | Model | FL method | Hierarchy |
|---|---|---|---|---|
| C1 | HAI 21.03 | 1D-CNN | FedAvg | agglomerative hierarchical |
| C2 | TON IoT combined telemetry | 1D-CNN | FedAvg | agglomerative hierarchical |
| C3 | WUSTL-IIOT-2021 | 1D-CNN | FedAvg | agglomerative hierarchical |

Rules:

- Use same Agglomerative Clustering membership files as proposed method.
- Use same leaf-client partitions as proposed method.
- Use weighted arithmetic mean everywhere.
- Cluster 1 baseline does not use FedBN.

## 14.3 Proposed specialized hierarchical FCFL

Purpose:

```text
Evaluate final proposed method.
```

Configuration:

| Cluster | Dataset | Model | FL method | Aggregation |
|---|---|---|---|---|
| C1 | HAI 21.03 | TCN | FedBN | weighted non-BN mean |
| C2 | TON IoT combined telemetry | compact MLP | FedProx | weighted arithmetic mean |
| C3 | WUSTL-IIOT-2021 | 1D-CNN | SCAFFOLD | weighted arithmetic mean |

Rules:

- Use same Agglomerative Clustering membership files as Baseline B.
- Use same leaf-client partitions as Baseline B.
- No cross-cluster averaging.

---

# 15. Experiment matrix

## 15.1 Main experiments

| Experiment ID | Cluster | Dataset | Model | FL method | Hierarchy | Clustering | Aggregation |
|---|---:|---|---|---|---|---|---|
| A_C1 | 1 | HAI 21.03 | 1D-CNN | FedAvg | flat | none | weighted mean |
| A_C2 | 2 | TON IoT combined telemetry | 1D-CNN | FedAvg | flat | none | weighted mean |
| A_C3 | 3 | WUSTL-IIOT-2021 | 1D-CNN | FedAvg | flat | none | weighted mean |
| B_C1 | 1 | HAI 21.03 | 1D-CNN | FedAvg | hierarchical | agglomerative | weighted mean |
| B_C2 | 2 | TON IoT combined telemetry | 1D-CNN | FedAvg | hierarchical | agglomerative | weighted mean |
| B_C3 | 3 | WUSTL-IIOT-2021 | 1D-CNN | FedAvg | hierarchical | agglomerative | weighted mean |
| P_C1 | 1 | HAI 21.03 | TCN | FedBN | hierarchical | agglomerative | weighted non-BN mean |
| P_C2 | 2 | TON IoT combined telemetry | compact MLP | FedProx | hierarchical | agglomerative | weighted mean |
| P_C3 | 3 | WUSTL-IIOT-2021 | 1D-CNN | SCAFFOLD | hierarchical | agglomerative | weighted mean |
| AB_C3_FEDAVG_CNN1D | 3 | WUSTL-IIOT-2021 | 1D-CNN | FedAvg | hierarchical | agglomerative | weighted mean |

## 15.2 Required ablations

| Ablation ID | Purpose |
|---|---|
| HIERARCHY_EFFECT | Compare Baseline A vs Baseline B |
| SPECIALIZATION_EFFECT | Compare Baseline B vs Proposed |
| C1_FEDAVG_VS_FEDBN | Compare Cluster 1 FedAvg vs FedBN |
| C2_FEDAVG_VS_FEDPROX | Compare Cluster 2 FedAvg vs FedProx |
| C3_FEDAVG_VS_SCAFFOLD | Compare Cluster 3 FedAvg vs SCAFFOLD using `AB_C3_FEDAVG_CNN1D` or an equivalent reuse of `B_C3` against `P_C3` |

---

# 16. Metrics and reporting outputs

## 16.1 Classification metrics

Report per cluster:

```text
Accuracy
Precision
Recall
F1-score
AUROC
PR-AUC
False Positive Rate
Confusion matrix
```

Definitions:

```text
False Positive Rate = FP / (FP + TN)
```

If AUROC or PR-AUC cannot be computed because only one class appears in a test split, write:

```text
metric_unavailable_single_class
```

and log a warning.

## 16.2 FL behavior metrics

Report:

```text
rounds to best validation F1
best validation F1
test F1 at best validation round
communication cost per round
total communication cost
wall-clock training time
convergence curve
```

Communication cost approximation:

```text
model_parameter_bytes_transmitted_downward
+
model_parameter_bytes_transmitted_upward
+
optimizer_state_bytes_if_applicable
```

For Cluster 3 SCAFFOLD, separately report:

```text
control_variate_bytes
```

## 16.3 Ledger metrics

If ledger logging is implemented:

```text
number of metadata records
ledger size in bytes
average logging latency
maximum logging latency
percentage overhead relative to training time
```

## 16.4 Output metrics files

Save:

```text
outputs/metrics/A_C1_metrics.csv
outputs/metrics/A_C2_metrics.csv
outputs/metrics/A_C3_metrics.csv
outputs/metrics/B_C1_metrics.csv
outputs/metrics/B_C2_metrics.csv
outputs/metrics/B_C3_metrics.csv
outputs/metrics/P_C1_metrics.csv
outputs/metrics/P_C2_metrics.csv
outputs/metrics/P_C3_metrics.csv
```

Also save summary:

```text
outputs/metrics/summary_all_experiments.csv
```

## 16.5 Plots

Generate:

```text
outputs/plots/convergence_A_C1.png
outputs/plots/convergence_A_C2.png
outputs/plots/convergence_A_C3.png
outputs/plots/convergence_B_C1.png
outputs/plots/convergence_B_C2.png
outputs/plots/convergence_B_C3.png
outputs/plots/convergence_P_C1.png
outputs/plots/convergence_P_C2.png
outputs/plots/convergence_P_C3.png
outputs/plots/f1_comparison.png
outputs/plots/pr_auc_comparison.png
outputs/plots/communication_cost_comparison.png
```

---

# 17. Output directory structure and filenames

Codex must create or use this structure:

```text
desktop/thesis/
  data/
    raw/
      hai_2103/
      ton_iot/
        original_archive/
        combined_telemetry/
      wustl_iiot_2021/
  fcfl-cps-ids/
    configs/
    docs/
    outputs/
      clients/
      clustering/
      preprocessing/
      ledgers/
      metrics/
      models/
      plots/
      reports/
      runs/
    src/
    tests/
```

## 17.1 Required config files

Create:

```text
configs/cluster1_hai.yaml
configs/cluster2_ton_iot.yaml
configs/cluster3_wustl.yaml
configs/baseline_flat.yaml
configs/baseline_hierarchical.yaml
configs/proposed.yaml
```

Each config must include:

```yaml
cluster_id:
dataset_name:
raw_data_path:
label_column:
excluded_columns:
model:
fl_method:
aggregation:
num_leaf_clients:
num_subclusters:
clustering:
  method:
  linkage:
  metric:
  descriptor:
  run_once:
training:
  rounds:
  local_epochs:
  batch_size:
  learning_rate:
  seed:
ledger:
  enabled:
  metadata_only:
```

## 17.2 Required saved artifacts

Preprocessing:

```text
outputs/preprocessing/cluster1_hai_scaler.pkl
outputs/preprocessing/cluster2_ton_iot_scaler.pkl
outputs/preprocessing/cluster3_wustl_scaler.pkl
```

Client partitions:

```text
outputs/clients/cluster1_leaf_clients.json
outputs/clients/cluster2_leaf_clients.json
outputs/clients/cluster3_leaf_clients.json
```

Clustering:

```text
outputs/clustering/cluster1_memberships.json
outputs/clustering/cluster2_memberships.json
outputs/clustering/cluster3_memberships.json
```

Models:

```text
outputs/models/{experiment_id}/best_model.pt
outputs/models/{experiment_id}/final_model.pt
```

Ledger:

```text
outputs/ledgers/{experiment_id}_ledger.jsonl
```

Reports:

```text
outputs/reports/data_profile_cluster1.json
outputs/reports/data_profile_cluster2.json
outputs/reports/data_profile_cluster3.json
outputs/reports/label_summary_cluster1.json
outputs/reports/label_summary_cluster2.json
outputs/reports/label_summary_cluster3.json
```

---

# 18. Validation and acceptance tests

Codex must implement tests before full experiments.

## 18.1 Required tests

Create:

```text
tests/test_data_contract.py
tests/test_binary_labels.py
tests/test_feature_exclusion.py
tests/test_leaf_client_partitioning.py
tests/test_agglomerative_memberships.py
tests/test_fixed_memberships.py
tests/test_no_cross_cluster_averaging.py
tests/test_weighted_aggregation.py
tests/test_fedbn_excludes_bn.py
tests/test_fedprox_loss.py
tests/test_scaffold_state.py
tests/test_ledger_metadata_only.py
tests/test_experiment_matrix.py
```

## 18.2 Acceptance criteria

The implementation is acceptable only if all criteria below pass.

### Data tests

- Each dataset loads successfully.
- Label column is explicitly configured.
- Labels map to only `{0, 1}`.
- Each main cluster has at least one normal and one attack sample.
- Excluded columns are not present in model features.
- WUSTL leakage columns are removed.

### Client tests

- Cluster 1 has exactly 12 candidate leaf clients.
- Cluster 2 has exactly 15 candidate leaf clients.
- Cluster 3 has exactly 15 candidate leaf clients.
- Every leaf client has a train split.
- Every leaf client belongs to exactly one sub-cluster after clustering.

### Clustering tests

- Cluster 1 has exactly 2 sub-clusters.
- Cluster 2 has exactly 3 sub-clusters.
- Cluster 3 has exactly 3 sub-clusters.
- No sub-cluster is empty.
- Membership files are created before training.
- Membership files do not change during training.
- Agglomerative Clustering is not called inside any FL training round.

### Architecture tests

- No cross-cluster model aggregation occurs.
- No model from Cluster 1 is averaged with Cluster 2 or Cluster 3.
- No model from Cluster 2 is averaged with Cluster 1 or Cluster 3.
- No model from Cluster 3 is averaged with Cluster 1 or Cluster 2.

### Aggregation tests

- Weighted aggregation equals manually computed weighted average.
- Cluster 1 FedBN aggregation excludes BatchNorm statistics.
- Cluster 2 FedProx includes proximal penalty.
- Cluster 3 SCAFFOLD maintains and updates control variates.

### Ledger tests

- Only main-cluster heads create ledger records.
- Ledger records contain metadata only.
- Ledger records do not contain raw samples.
- Ledger records do not contain full model weights.
- Ledger records include:
  - round id
  - cluster id
  - cluster head id
  - model version
  - previous main model hash
  - new main model hash
  - clustering method
  - clustering configuration hash
  - subcluster count
  - subcluster membership hash
  - fl method
  - aggregation rule
  - participant count
  - effective sample count
  - timestamp start
  - timestamp end
  - submitter identity

### Experiment tests

- Baseline A runs for all three clusters.
- Baseline B runs for all three clusters.
- Proposed method runs for all three clusters.
- Metrics CSV files are produced.
- No experiment silently skips a cluster.

---

# 19. Known ambiguities / TODO items requiring user confirmation

The following items must be confirmed before final training.

## 19.1 Dataset path confirmation

TODO / USER MUST CONFIRM:

```text
Is the real path exactly desktop/thesis/data/
or is it ~/Desktop/thesis/data/?
```

## 19.2 HAI label column

TODO / USER MUST CONFIRM:

Expected:

```text
attack
```

Confirm actual HAI label column name.

## 19.3 TON IoT label column

TODO / USER MUST CONFIRM:

The exact label column for TON IoT combined telemetry is not safely inferable from the report alone.

User must confirm the exact column name.

Possible candidates for profiling only:

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

## 19.4 WUSTL-IIOT-2021 label column

TODO / USER MUST CONFIRM:

The exact binary label column for WUSTL-IIOT-2021 is not safely inferable from the report alone.

User must confirm the exact column name.

Possible candidates for profiling only:

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

## 19.5 Dataset file names

TODO / USER MUST CONFIRM:

List actual CSV filenames inside:

```text
desktop/thesis/data/raw/hai_2103/
desktop/thesis/data/raw/ton_iot/combined_telemetry/
desktop/thesis/data/raw/wustl_iiot_2021/
```

Codex may implement a profiling script to print filenames and columns, but it must not begin model training until the label columns are confirmed.

## 19.6 Categorical columns

TODO / USER MUST CONFIRM:

If any dataset contains categorical columns that are not labels or excluded identifiers, confirm whether to:

```text
one-hot encode
drop
or manually map
```

Default rule is in Section 4.5.

## 19.7 Training budget

Recommended implementation choice:

```text
rounds = 50
local_epochs = 1
batch_size = 128
```

TODO / USER MUST CONFIRM if hardware is limited.

## 19.8 Full Fabric vs mock ledger

Recommended implementation choice:

Use a JSONL mock ledger first:

```text
outputs/ledgers/{experiment_id}_ledger.jsonl
```

This validates metadata-only governance without requiring full Hyperledger Fabric deployment.

TODO / USER MUST CONFIRM:

Whether actual Hyperledger Fabric integration is required in the first implementation phase.

## 19.9 Leaf-client counts

Recommended implementation choice:

```text
Cluster 1: 12 leaf clients
Cluster 2: 15 leaf clients
Cluster 3: 15 leaf clients
```

TODO / USER MUST CONFIRM if different client counts are required.

---

# 20. Implementation guardrails for Codex

Codex must obey these rules.

## 20.1 Do not change architecture

Do not change:

```text
N = 3
K1 = 2
K2 = 3
K3 = 3
HAI + TCN + FedBN
TON IoT + compact MLP + FedProx
WUSTL + 1D-CNN + SCAFFOLD
metadata-only ledger
one-time offline Agglomerative Clustering
```

## 20.2 Do not hallucinate dataset columns

If label columns or required feature columns are unclear:

```text
STOP and ask the user.
```

Do not guess.

## 20.3 Do not introduce new methods

Do not add:

```text
new datasets
new FL methods
new aggregation methods
dynamic clustering
reclustering
secure aggregation
differential privacy
Byzantine robustness
central global averaging
```

unless explicitly requested by the user.

## 20.4 Do not write raw data to outputs except processed dataset cache

Raw or processed data caches may be saved only under:

```text
outputs/preprocessing/
```

Ledger files must never contain raw records.

## 20.5 Every implementation step must be testable

Every major module must have at least one test.

No full experiment should be run until:

```text
pytest tests/
```

passes.

---

# 21. Recommended implementation order

Codex should implement in this order:

1. Repository skeleton.
2. Config files.
3. Dataset profiling script.
4. Data loaders with strict label-column validation.
5. Preprocessing pipeline.
6. Candidate leaf-client partitioning.
7. Descriptor computation.
8. Offline Agglomerative Clustering.
9. Membership saving and loading.
10. Weighted aggregation utilities.
11. Model definitions.
12. Flat FL baseline.
13. Hierarchical FCFL baseline.
14. FedBN for Cluster 1.
15. FedProx for Cluster 2.
16. SCAFFOLD for Cluster 3.
17. Metadata ledger simulator.
18. Experiment runner.
19. Metrics and plots.
20. Final validation tests.

Do not implement the entire project in one step.
