# FCFL_IMPLEMENTATION_SPEC.md

## 0. Document purpose and status

This document is the **implementation-focused report** for the simulation-only implementation of the proposed Fixed Clustered Federated Learning (FCFL) intrusion-detection system.

This file is intended to be given directly to Codex as the main coding specification. It translates the research report into concrete implementation rules, dataset contracts, module boundaries, experiment definitions, outputs, and validation checks.

This document must be used together with:

```text
ARCHITECTURE_CONTRACT.md
DATA_CONTRACT.md
ALGORITHMS.md
EXPERIMENT_MATRIX.csv
VALIDATION_CHECKS.md
```

The research report remains the academic source. This implementation specification and the contract files are the **coding source of truth**. If the research report contains older figure labels, formatting artifacts, or outdated captions, Codex must follow this implementation specification and flag the mismatch instead of changing the architecture.

---

# 1. Purpose

## 1.1 Project objective

Implement a simulation-only **Fixed Clustered Federated Learning (FCFL)** system for **binary intrusion detection** across three heterogeneous CPS/IIoT domains.

The system must implement:

1. Three independent main clusters.
2. One fixed sub-cluster layer inside each main cluster.
3. One-time offline Agglomerative Clustering inside each main cluster to form fixed sub-clusters.
4. Hierarchical federated training:
   - leaf clients \(\rightarrow\) sub-cluster heads
   - sub-cluster heads \(\rightarrow\) main-cluster heads
5. No cross-cluster parameter averaging.
6. Raw records remain inside simulated leaf-client partitions.
7. Only main-cluster heads write metadata to the ledger / Fabric-compatible logging layer.
8. No raw data and no full model weights are written on-chain or into the ledger.

## 1.2 Scope of implementation

The implementation must support:

- dataset loading and strict data validation,
- dataset-specific preprocessing,
- binary label preparation,
- candidate leaf-client partitioning,
- client-level descriptor computation,
- one-time offline Agglomerative Clustering,
- fixed sub-cluster membership storage,
- flat per-cluster FL baseline,
- uniform hierarchical FCFL baseline,
- proposed specialized hierarchical FCFL method,
- FedBN for Cluster 1,
- FedProx for Cluster 2,
- SCAFFOLD for Cluster 3,
- weighted hierarchical aggregation,
- metadata-only ledger logging,
- metrics export,
- validation tests.

## 1.3 Privacy and security scope

This implementation is **data-locality-preserving**, not a formal privacy mechanism.

It does **not** implement:

- differential privacy,
- secure aggregation,
- encrypted model-update aggregation,
- Byzantine-robust aggregation,
- poisoning defense,
- real deployment security.

The claim is limited to:

```text
raw records remain local to leaf-client partitions during simulated FL training.
```

---

# 2. Non-negotiable architecture rules

## 2.1 Main cluster rules

The implementation must obey the following rules:

```text
N = 3 main clusters
```

The three main clusters are:

| Main cluster | Domain | Dataset | Model | FL method | Aggregation |
|---|---|---|---|---|---|
| Cluster 1 | Process-control telemetry IDS | HAI 21.03 | TCN | FedBN | weighted non-BN mean |
| Cluster 2 | Heterogeneous IIoT telemetry IDS | TON IoT combined telemetry | compact MLP | FedProx | weighted arithmetic mean |
| Cluster 3 | IIoT network-flow IDS | WUSTL-IIOT-2021 | 1D-CNN | SCAFFOLD | weighted arithmetic mean |

## 2.2 Fixed-cluster rules

- Each main cluster contains exactly one fixed sub-cluster layer.
- Agglomerative Clustering is used once before training to form the fixed sub-cluster memberships.
- No reclustering is allowed during training.
- No dynamic clustering is allowed.
- No client reassignment is allowed after membership files are frozen.
- All leaf clients must belong to exactly one sub-cluster.

## 2.3 Cross-cluster rules

The following are forbidden:

```text
Cluster 1 model averaged with Cluster 2 or Cluster 3
Cluster 2 model averaged with Cluster 1 or Cluster 3
Cluster 3 model averaged with Cluster 1 or Cluster 2
any global central server above all clusters
any global cross-cluster model aggregation
```

The three main clusters may run sequentially or in parallel, but their model parameters must remain independent.

## 2.4 Within-cluster compatibility rule

Inside a main cluster:

- all leaf clients use the same feature representation,
- all sub-clusters use the same model family,
- all sub-cluster and main-cluster models must be parameter-compatible,
- aggregation happens only among compatible models inside the same main cluster.

Across main clusters:

- feature spaces may differ,
- model families may differ,
- FL methods may differ,
- aggregation rules may differ,
- no cross-cluster averaging is performed.

## 2.5 Blockchain / ledger rules

- Only main-cluster heads write metadata to the ledger.
- Sub-cluster heads do not write to the ledger.
- Leaf clients do not write to the ledger.
- Ledger stores metadata only.
- Ledger must not store raw records.
- Ledger must not store full model weights.
- Ledger may store model hashes, clustering configuration hashes, and sub-cluster membership hashes.

---

# 3. Cluster configuration

## 3.1 Final cluster configuration

| Cluster | Dataset | Candidate leaf clients | Fixed sub-clusters | Model | FL method | Aggregation |
|---|---:|---:|---:|---|---|---|
| Cluster 1 | HAI 21.03 | 12 | \(K_1=2\) | TCN | FedBN | weighted non-BN mean |
| Cluster 2 | TON IoT combined telemetry | 15 | \(K_2=3\) | compact MLP | FedProx | weighted arithmetic mean |
| Cluster 3 | WUSTL-IIOT-2021 | 15 | \(K_3=3\) | 1D-CNN | SCAFFOLD | weighted arithmetic mean |

## 3.2 Fixed sub-cluster identifiers

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

These are **Agglomerative Clustering group identifiers**, not manually assigned semantic labels.

Do not interpret:

```text
H1/H2 as boiler/turbine labels
T1/T2/T3 as device labels
W1/W2/W3 as attack/time labels
```

unless this is only for post-hoc interpretation.

## 3.3 Default training budget

Recommended implementation choice:

```text
rounds = 50
local_epochs = 1
batch_size = 128
```

Default seeds:

```text
[42, 123, 2025]
```

For initial smoke testing:

```text
rounds = 2
local_epochs = 1
batch_size = 64
seed = 42
```

---

# 4. Dataset contracts

## 4.1 Base dataset path

Current expected folder structure:

```text
desktop/thesis/data/
  raw/
    hai_2103/
    ton_iot/
      original_archive/
      combined_telemetry/
    wustl_iiot_2021/
```

Recommended implementation choice:

Use an environment variable for portability:

```text
FCFL_DATA_ROOT
```

If `FCFL_DATA_ROOT` is not set, use:

```text
desktop/thesis/data
```

TODO / USER MUST CONFIRM:

```text
Confirm whether the actual path is desktop/thesis/data or ~/Desktop/thesis/data.
```

## 4.2 Cluster 1 dataset contract: HAI 21.03

Dataset path:

```text
${FCFL_DATA_ROOT}/raw/hai_2103/
```

Expected file type:

```text
.csv
```

Expected task:

```text
binary process-control telemetry intrusion detection
```

Expected label column:

```text
attack
```

TODO / USER MUST CONFIRM:

```text
Confirm that HAI 21.03 files contain a label column exactly named attack.
```

Label handling:

```text
normal = 0
attack = 1
```

Expected input type:

```text
multivariate time-series telemetry windows
```

Recommended implementation choice:

Use sliding windows:

```text
window_length = 32
stride = 8
```

Window label rule:

```text
window_label = 1 if any row inside the window has attack label 1
window_label = 0 otherwise
```

Columns to exclude from model features:

```text
attack
Attack
label
Label
target
Target
attack_P1
attack_P2
attack_P3
attack_P4
timestamp
Timestamp
time
Time
date
Date
```

Notes:

- Process-specific attack labels may be used only for auxiliary analysis.
- They must not be model input features.
- They must not be used for dynamic clustering.

## 4.3 Cluster 2 dataset contract: TON IoT combined telemetry

Dataset path:

```text
${FCFL_DATA_ROOT}/raw/ton_iot/combined_telemetry/
```

Do not use this folder for training unless explicitly required:

```text
${FCFL_DATA_ROOT}/raw/ton_iot/original_archive/
```

Expected file type:

```text
.csv
```

Expected task:

```text
binary heterogeneous IIoT telemetry intrusion detection
```

Expected input type:

```text
fixed-length tabular telemetry vector
```

Expected label column:

```text
TODO / USER MUST CONFIRM
```

Possible label candidates for profiling only:

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

Codex must not silently choose a label column. It may print candidate columns during profiling, but training must use the configured label column.

Columns to exclude from model features:

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

Device-origin information:

- May be used for interpretation only.
- Must not manually define fixed sub-cluster membership.
- Fixed sub-clusters are formed by offline Agglomerative Clustering.

No sliding windowing is used.

Each row is one sample.

## 4.4 Cluster 3 dataset contract: WUSTL-IIOT-2021

Dataset path:

```text
${FCFL_DATA_ROOT}/raw/wustl_iiot_2021/
```

Expected file type:

```text
.csv
```

Expected task:

```text
binary IIoT network-flow intrusion detection
```

Expected input type:

```text
fixed-length network-flow feature vector
```

Expected label column:

```text
TODO / USER MUST CONFIRM
```

Possible label candidates for profiling only:

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

Mandatory leakage / identifier columns to exclude:

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

No sliding windowing is used.

Each row is one flow sample.

## 4.5 Data profiling requirement

Before implementation proceeds to training, Codex must produce dataset profiles:

```text
outputs/reports/data_profile_cluster1.json
outputs/reports/data_profile_cluster2.json
outputs/reports/data_profile_cluster3.json
```

Each profile must include:

```text
file names
row counts
column names
candidate label columns
missing-value counts
numeric/categorical column summary
class counts after label mapping if label is confirmed
columns excluded
columns retained
```

If label columns are not confirmed, profile only and stop before training.

---

# 5. Preprocessing rules

## 5.1 Universal preprocessing sequence

For each main cluster:

1. Load raw CSV files.
2. Validate required columns.
3. Map labels to binary.
4. Sort by timestamp/order column if available.
5. Create candidate leaf-client partitions.
6. Split each candidate leaf client into train/validation/test.
7. Fit imputer/scaler/encoder using training data only.
8. Transform train/validation/test partitions.
9. Compute clustering descriptors using transformed training features only.
10. Run Agglomerative Clustering.
11. Freeze membership files.

## 5.2 Important timestamp/leakage handling rule

Timestamp or order columns may be used temporarily for ordering or partitioning.

After ordering/partitioning, timestamp, identifier, and leakage-prone columns must be excluded from:

```text
model input features
descriptor features
scaler inputs
imputer inputs
encoder inputs
```

## 5.3 Missing-value handling

Recommended implementation choice:

- Numerical missing values: fill using training-set median.
- Categorical missing values: fill using training-set mode or `UNKNOWN`.
- Columns that are all missing in training data: drop and log.
- Columns that are constant in training data: drop and log.

## 5.4 Scaling

Recommended implementation choice:

Use `StandardScaler` for numerical model features.

Fit scaler on training data only.

Save scalers:

```text
outputs/preprocessing/cluster1_hai_scaler.pkl
outputs/preprocessing/cluster2_ton_iot_scaler.pkl
outputs/preprocessing/cluster3_wustl_scaler.pkl
```

## 5.5 Categorical handling

Recommended implementation choice:

If categorical columns remain after explicit exclusions:

1. If the column has at most 20 unique values in training data:
   - one-hot encode,
   - fit categories on training data only,
   - unknown validation/test categories map to all-zero for that group.
2. If the column has more than 20 unique values:
   - drop it,
   - log the column and reason.

Never one-hot encode label columns, identifier columns, or leakage columns.

## 5.6 Label mapping

Internal labels:

```text
normal / benign = 0
attack / malicious / anomaly = 1
```

For numeric labels:

- If labels are already `{0, 1}`, keep them.
- If labels are not binary, stop and ask for mapping.

For string labels:

Map to `0`:

```text
normal
Normal
NORMAL
benign
Benign
BENIGN
0
```

Map to `1`:

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

If unknown label values appear, stop and print all unique values.

## 5.7 Train / validation / test split

Recommended implementation choice:

```text
train = 70%
validation = 15%
test = 15%
```

Use seed:

```text
42
```

For final experiments, repeat with:

```text
42, 123, 2025
```

Partitioning rule:

- Sort by timestamp if available.
- If no timestamp exists, preserve file order.
- Create contiguous candidate leaf-client partitions.
- Split each candidate leaf client locally into train/validation/test.

Validation rule:

- Main-cluster-level train, validation, and test unions must contain both labels.
- Leaf-client-level single-class partitions are allowed but must be logged.

---

# 6. Leaf-client creation

## 6.1 Candidate leaf-client counts

Recommended implementation choice:

| Cluster | Candidate leaf clients | Fixed sub-clusters |
|---|---:|---:|
| Cluster 1 | 12 | 2 |
| Cluster 2 | 15 | 3 |
| Cluster 3 | 15 | 3 |

## 6.2 Candidate leaf-client creation method

For each main cluster:

1. Use the cleaned, label-mapped dataset.
2. Sort by timestamp/order column if available.
3. If no timestamp exists, preserve file order.
4. Split the ordered samples into the configured number of contiguous shards.
5. Each shard becomes one candidate leaf client.
6. Each candidate leaf client is split into train/validation/test.

## 6.3 Leaf-client IDs

Cluster 1:

```text
C1_L001 ... C1_L012
```

Cluster 2:

```text
C2_L001 ... C2_L015
```

Cluster 3:

```text
C3_L001 ... C3_L015
```

## 6.4 Leaf-client metadata output

Save:

```text
outputs/clients/cluster1_leaf_clients.json
outputs/clients/cluster2_leaf_clients.json
outputs/clients/cluster3_leaf_clients.json
```

Each file must include:

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

# 7. Offline Agglomerative Clustering

## 7.1 Purpose

Agglomerative Clustering is used only for fixed sub-cluster initialization.

It is not part of the training loop.

## 7.2 Descriptor definition

For each candidate leaf client \(i\), use only transformed local training features \(X_i\).

Each \(x \in X_i\) is a feature vector.

Compute:

\[
z_i=[\mu_i;\sigma_i]
\]

where:

\[
\mu_i=\frac{1}{n_i}\sum_{x\in X_i}x
\]

and:

\[
\sigma_i=\sqrt{\frac{1}{n_i}\sum_{x\in X_i}(x-\mu_i)^2}
\]

Do not include:

```text
labels
attack types
timestamps
identifiers
source/destination addresses
raw records
```

## 7.3 Descriptor standardization

Recommended implementation choice:

Use `StandardScaler` on descriptor vectors within each main cluster.

Save:

```text
outputs/clustering/cluster1_descriptor_scaler.pkl
outputs/clustering/cluster2_descriptor_scaler.pkl
outputs/clustering/cluster3_descriptor_scaler.pkl
```

## 7.4 Clustering parameters

Recommended implementation choice:

```text
method = AgglomerativeClustering
linkage = ward
metric = euclidean
```

If scikit-learn version uses `affinity` instead of `metric`, use:

```text
affinity = euclidean
```

Fixed sub-cluster counts:

```text
K1 = 2
K2 = 3
K3 = 3
```

## 7.5 Membership output files

Save:

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

## 7.6 Frozen membership rule

Default:

```text
recluster = false
```

If membership files already exist:

- reuse them,
- do not regenerate automatically,
- do not modify them during training.

If `--recluster` is passed:

- require explicit user confirmation,
- overwrite membership files only after confirmation,
- log old and new membership hashes.

## 7.7 Empty sub-cluster rule

After clustering:

- every leaf client must be assigned exactly once,
- every sub-cluster must contain at least one leaf client.

If any sub-cluster is empty, stop and raise an error.

---

# 8. Model definitions

All models output one binary logit.

Loss:

```text
BCEWithLogitsLoss
```

Prediction:

```text
probability = sigmoid(logit)
predicted_label = 1 if probability >= 0.5 else 0
```

Threshold tuning is not part of the first implementation.

## 8.1 Cluster 1: TCNClassifier

Input shape:

```text
batch_size x num_features x window_length
```

Default:

```text
window_length = 32
```

Recommended architecture:

```text
Input: [B, F, 32]

TCN Block 1:
- Conv1d(F, 32, kernel_size=3, dilation=1, padding=same)
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

FedBN note:

- BatchNorm-related parameters/statistics are local.
- They must not be averaged globally.

## 8.2 Cluster 2: CompactMLPClassifier

Input shape:

```text
batch_size x num_features
```

Recommended architecture:

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

Do not add BatchNorm in the first implementation.

## 8.3 Cluster 3: CNN1DClassifier

Input shape:

```text
batch_size x 1 x num_features
```

Recommended architecture:

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
Cluster 3 uses SCAFFOLD, not FedBN. Avoiding BatchNorm reduces optimizer-state ambiguity.
```

---

# 9. FL algorithms

## 9.1 Shared hierarchical FCFL round

For each main cluster \(m\):

1. Main-cluster head holds \(w_m^t\).
2. Main-cluster head sends \(w_m^t\) to all sub-cluster heads.
3. Sub-cluster head initializes:

\[
w_{m,s}^{t} \leftarrow w_m^t
\]

4. Leaf clients train locally from \(w_{m,s}^{t}\).
5. Sub-cluster head aggregates leaf-client models into \(w_{m,s}^{t+1}\).
6. Main-cluster head aggregates sub-cluster models into \(w_m^{t+1}\).
7. Main-cluster head logs metadata.

## 9.2 Weighted sub-cluster aggregation

\[
w_{m,s}^{t+1}=\sum_{i\in C_{m,s}}\frac{n_i}{N_{m,s}}w_{m,s,i}^{t+1}
\]

\[
N_{m,s}=\sum_{i\in C_{m,s}}n_i
\]

## 9.3 Weighted main-cluster aggregation

\[
w_m^{t+1}=\sum_{s\in S_m}\frac{N_{m,s}}{N_m}w_{m,s}^{t+1}
\]

\[
N_m=\sum_{s\in S_m}N_{m,s}
\]

## 9.4 Cluster 1: FedBN

Cluster 1 uses:

```text
FL method = FedBN
Aggregation = weighted non-BN mean
```

Exclude BatchNorm-related entries from aggregation.

Case-insensitive key exclusion patterns:

```text
bn
batchnorm
running_mean
running_var
num_batches_tracked
```

Recommended implementation choice:

Treat all BatchNorm-related parameters and statistics as local.

## 9.5 Cluster 2: FedProx

Cluster 2 uses:

```text
FL method = FedProx
Aggregation = weighted arithmetic mean
```

Local objective:

\[
F_{m,s,i}(w)+\frac{\mu}{2}\|w-w_{m,s}^t\|_2^2
\]

Recommended implementation choice:

```text
mu = 0.01
optimizer = Adam
learning_rate = 1e-3
local_epochs = 1
batch_size = 128
```

## 9.6 Cluster 3: SCAFFOLD

Cluster 3 uses:

```text
FL method = SCAFFOLD
Aggregation = weighted arithmetic mean
```

Control variates:

```text
main-cluster server control variate: c_m^t
local client control variate: c_{m,s,i}^t
```

Recommended implementation choice:

```text
client_fraction = 1.0
optimizer_style = corrected SGD update
learning_rate = 0.01
local_epochs = 1
batch_size = 128
```

Use \(E\) for local SCAFFOLD steps.

Local update:

\[
y_{m,s,i}^{t,k+1}=y_{m,s,i}^{t,k}-\eta_\ell\left(g_{m,s,i}(y_{m,s,i}^{t,k})-c_{m,s,i}^{t}+c_m^t\right)
\]

After \(E\) local steps:

\[
w_{m,s,i}^{t+1}=y_{m,s,i}^{t,E}
\]

Client control-variate update:

\[
c_{m,s,i}^{t+1}=c_{m,s,i}^{t}-c_m^t+\frac{1}{E\eta_\ell}(w_{m,s}^{t}-w_{m,s,i}^{t+1})
\]

Control-variate increment:

\[
\Delta c_{m,s,i}^{t}=c_{m,s,i}^{t+1}-c_{m,s,i}^{t}
\]

Server control-variate update:

\[
c_m^{t+1}=c_m^t+\frac{1}{Q_m}\sum_{(s,i)\in P_m^t}\Delta c_{m,s,i}^{t}
\]

where:

```text
P_m^t = participating leaf clients in Cluster 3 at round t
Q_m = total number of leaf clients in Cluster 3
```

Sub-cluster heads forward both:

```text
model updates
SCAFFOLD control-variate increments
```

Sub-cluster heads do not maintain independent SCAFFOLD server control variates in the first implementation.

---

# 10. Ledger / blockchain simulation

## 10.1 Implementation choice

Recommended implementation choice:

Use a JSONL mock ledger first.

Reason:

```text
The research report uses Hyperledger Fabric as metadata-level governance. A mock ledger validates metadata schema, hashing, and audit behavior before real Fabric deployment.
```

Output:

```text
outputs/ledgers/{experiment_id}_ledger.jsonl
```

## 10.2 Metadata record schema

Each ledger record must contain:

```json
{
  "round_id": 0,
  "cluster_id": 1,
  "cluster_head_id": "MC1_HEAD",
  "model_version": "C1_R000",
  "previous_main_model_hash": null,
  "new_main_model_hash": "sha256:...",
  "clustering_method": "AgglomerativeClustering",
  "clustering_configuration_hash": "sha256:...",
  "subcluster_count": 2,
  "subcluster_membership_hash": "sha256:...",
  "subcluster_digest": "sha256:...",
  "fl_method": "FedBN",
  "aggregation_rule": "weighted_non_bn_mean",
  "effective_sample_count": 0,
  "participant_count": 0,
  "timestamp_start": "ISO-8601",
  "timestamp_end": "ISO-8601",
  "submitter_identity": "MC1_HEAD"
}
```

## 10.3 Ledger restrictions

Ledger records must not contain:

```text
raw samples
full local datasets
full model weights
full gradients
full model tensors
client raw records
```

## 10.4 Fabric integration phase

TODO / USER MUST CONFIRM:

```text
Should real Hyperledger Fabric be implemented in phase 1?
```

Default:

```text
Use mock JSONL ledger in phase 1.
```

---

# 11. Experiment definitions

## 11.1 Baseline A: flat per-cluster FL

Purpose:

```text
Measure performance without sub-cluster hierarchy.
```

| Cluster | Dataset | Model | FL method | Hierarchy | Clustering |
|---|---|---|---|---|---|
| C1 | HAI 21.03 | 1D-CNN | FedAvg | flat | none |
| C2 | TON IoT combined telemetry | 1D-CNN | FedAvg | flat | none |
| C3 | WUSTL-IIOT-2021 | 1D-CNN | FedAvg | flat | none |

Rules:

- One main-cluster head per domain.
- No sub-cluster layer.
- No Agglomerative Clustering.
- No cross-cluster averaging.

## 11.2 Baseline B: uniform hierarchical FCFL

Purpose:

```text
Measure the effect of hierarchy while removing model/FL specialization.
```

| Cluster | Dataset | Model | FL method | Hierarchy | Clustering |
|---|---|---|---|---|---|
| C1 | HAI 21.03 | 1D-CNN | FedAvg | hierarchical | agglomerative |
| C2 | TON IoT combined telemetry | 1D-CNN | FedAvg | hierarchical | agglomerative |
| C3 | WUSTL-IIOT-2021 | 1D-CNN | FedAvg | hierarchical | agglomerative |

Rules:

- Use same leaf-client partitions as the proposed method.
- Use same agglomerative membership files as the proposed method.
- Use weighted arithmetic mean everywhere.
- Cluster 1 baseline does not use FedBN.

## 11.3 Proposed specialized hierarchical FCFL

Purpose:

```text
Evaluate the final proposed FCFL mechanism.
```

| Cluster | Dataset | Model | FL method | Hierarchy | Clustering | Aggregation |
|---|---|---|---|---|---|---|
| C1 | HAI 21.03 | TCN | FedBN | hierarchical | agglomerative | weighted non-BN mean |
| C2 | TON IoT combined telemetry | compact MLP | FedProx | hierarchical | agglomerative | weighted mean |
| C3 | WUSTL-IIOT-2021 | 1D-CNN | SCAFFOLD | hierarchical | agglomerative | weighted mean |

## 11.4 Ablations

Required ablations:

```text
Baseline A vs Baseline B
Baseline B vs Proposed
Cluster 1 FedAvg vs FedBN
Cluster 2 FedAvg vs FedProx
Cluster 3 FedAvg vs SCAFFOLD
```

---

# 12. Metrics and outputs

## 12.1 Classification metrics

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

False Positive Rate:

```text
FPR = FP / (FP + TN)
```

If AUROC or PR-AUC cannot be computed because only one class appears:

```text
metric_unavailable_single_class
```

Log a warning.

## 12.2 FL behavior metrics

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

For Cluster 3 SCAFFOLD, separately report:

```text
control_variate_bytes
```

## 12.3 Ledger metrics

If ledger logging is enabled, report:

```text
number of metadata records
ledger size in bytes
average logging latency
maximum logging latency
percentage overhead relative to training time
```

## 12.4 Output files

Metrics:

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
outputs/metrics/summary_all_experiments.csv
```

Plots:

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

# 13. Tests and acceptance criteria

## 13.1 Required test files

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

## 13.2 Data acceptance criteria

The implementation passes data validation only if:

- each dataset loads successfully,
- label column is explicitly configured,
- labels map to only `{0, 1}`,
- each main cluster contains at least one normal and one attack sample,
- excluded columns are absent from model features,
- excluded columns are absent from descriptor features,
- WUSTL leakage columns are removed.

## 13.3 Clustering acceptance criteria

The implementation passes clustering validation only if:

- Cluster 1 has exactly 12 candidate leaf clients,
- Cluster 2 has exactly 15 candidate leaf clients,
- Cluster 3 has exactly 15 candidate leaf clients,
- Cluster 1 has exactly 2 sub-clusters,
- Cluster 2 has exactly 3 sub-clusters,
- Cluster 3 has exactly 3 sub-clusters,
- no sub-cluster is empty,
- every leaf client belongs to exactly one sub-cluster,
- membership files are created before training,
- membership files do not change during training,
- Agglomerative Clustering is not called inside any FL training round.

## 13.4 Architecture acceptance criteria

The implementation passes architecture validation only if:

- no cross-cluster aggregation occurs,
- no global central server is created,
- all leaf clients are under sub-clusters in hierarchical experiments,
- only main-cluster heads write ledger metadata,
- raw records are not logged,
- full model weights are not logged.

## 13.5 FL-method acceptance criteria

Cluster 1:

- FedBN aggregation excludes BatchNorm-related state.

Cluster 2:

- FedProx local loss includes the proximal term.

Cluster 3:

- SCAFFOLD local control variates exist,
- main-cluster server control variate exists,
- control-variate increments are updated and forwarded,
- SCAFFOLD state is not used in Cluster 1 or Cluster 2.

## 13.6 Experiment acceptance criteria

The implementation passes experiment validation only if:

- Baseline A runs for all three clusters,
- Baseline B runs for all three clusters,
- Proposed method runs for all three clusters,
- required metrics files are generated,
- no cluster is silently skipped,
- all tests pass before full experiments.

---

# 14. Known out-of-scope items

The following are out of scope for the first implementation:

```text
real-world deployment
dynamic clustering
reclustering during training
cross-cluster model averaging
secure aggregation
differential privacy
Byzantine robustness
poisoning defense
full Hyperledger Fabric deployment in phase 1
multiclass IDS
new datasets beyond HAI 21.03, TON IoT combined telemetry, WUSTL-IIOT-2021
new FL methods beyond FedAvg, FedBN, FedProx, SCAFFOLD
```

The default phase-1 ledger is a JSONL mock ledger. Real Fabric integration can be added later only if explicitly requested.

---

# Appendix A. Recommended repository structure

Recommended structure:

```text
fcfl-cps-ids/
├── AGENTS.md
├── README.md
├── requirements.txt
├── configs/
│   ├── cluster1_hai.yaml
│   ├── cluster2_ton_iot.yaml
│   ├── cluster3_wustl.yaml
│   ├── baseline_flat.yaml
│   ├── baseline_hierarchical.yaml
│   └── proposed.yaml
├── docs/
│   ├── FCFL_IMPLEMENTATION_SPEC.md
│   ├── DATA_CONTRACT.md
│   ├── ARCHITECTURE_CONTRACT.md
│   ├── ALGORITHMS.md
│   ├── EXPERIMENT_MATRIX.csv
│   └── VALIDATION_CHECKS.md
├── src/
│   ├── data/
│   │   ├── loaders.py
│   │   ├── preprocess.py
│   │   ├── partitions.py
│   │   └── descriptors.py
│   ├── clustering/
│   │   └── agglomerative.py
│   ├── models/
│   │   ├── cnn1d.py
│   │   ├── tcn.py
│   │   └── mlp.py
│   ├── fl/
│   │   ├── client.py
│   │   ├── subcluster.py
│   │   ├── maincluster.py
│   │   ├── aggregators.py
│   │   ├── fedbn.py
│   │   ├── fedprox.py
│   │   └── scaffold.py
│   ├── ledger/
│   │   ├── metadata_schema.py
│   │   └── mock_ledger.py
│   ├── metrics/
│   │   └── classification.py
│   └── train.py
├── scripts/
│   ├── prepare_data.py
│   ├── run_clustering.py
│   ├── run_experiment.py
│   └── run_all.sh
├── tests/
└── outputs/
```

---

# Appendix B. Recommended implementation order

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
