# ARCHITECTURE_CONTRACT.md

## 1. Purpose

This document defines the non-negotiable architectural rules for the FCFL CPS/IIoT intrusion-detection implementation.

Any code generation, refactoring, or experiment orchestration must obey these rules. If a requested change conflicts with this contract, implementation must stop and ask the user.

This contract is aligned with the current report architecture: three independent main clusters, one fixed sub-cluster layer per main cluster, one-time offline Agglomerative Clustering, no cross-cluster averaging, raw-data locality, and metadata-only governance.

---

## 2. Non-negotiable top-level rules

1. There are exactly **three** main clusters.
2. The overall system is **Fixed Clustered Federated Learning (FCFL)**, not flat FL and not dynamic clustering.
3. Each main cluster has exactly **one fixed sub-cluster layer**.
4. All leaf clients must be placed under sub-clusters in the proof-of-concept implementation.
5. No cross-cluster parameter averaging is allowed.
6. No central server may aggregate all three main clusters together.
7. Raw data must remain local to leaf-client partitions during FCFL training.
8. Only main-cluster heads may interact with the ledger / Hyperledger-Fabric-compatible governance layer.
9. The ledger stores metadata only.
10. The ledger must never store raw data or full model weights.
11. The implementation is data-locality-preserving only; it must not claim formal privacy mechanisms such as differential privacy or secure aggregation.

---

## 3. Cluster-specific architectural mapping

| Main cluster | Dataset | Model | FL method | Aggregation | Fixed sub-clusters |
|---|---|---|---|---|---|
| Cluster 1 | HAI 21.03 | TCN | FedBN | weighted non-BN mean | H1–H2, K1 = 2 |
| Cluster 2 | TON IoT combined telemetry | compact MLP | FedProx | weighted arithmetic mean | T1–T3, K2 = 3 |
| Cluster 3 | WUSTL-IIOT-2021 | 1D-CNN | SCAFFOLD | weighted arithmetic mean | W1–W3, K3 = 3 |

No implementation may change this mapping without explicit user approval.

---

## 4. Hierarchy rules

## 4.1 Main-cluster level

Each main cluster must contain:
- one main-cluster head,
- one main-cluster global model,
- one fixed sub-cluster layer.

A main cluster is an independent FCFL problem.

## 4.2 Sub-cluster level

Each sub-cluster must contain:
- one sub-cluster head,
- one sub-cluster global model,
- one or more leaf clients.

Each leaf client must belong to exactly one sub-cluster.


## 4.3 Leaf-client level

Each leaf client:
- stores a local data partition,
- performs local supervised binary training,
- sends model updates upward,
- never writes directly to the ledger.

---

## 5. Clustering rules

## 5.1 Agglomerative Clustering role

Agglomerative Clustering is used only as a **one-time offline / pre-training** step.

It is used only to assign candidate leaf clients to fixed sub-clusters.

It is not used:
- during training rounds,
- between rounds,
- for dynamic reclustering,
- across main clusters.

## 5.2 Frozen membership rule

After the one-time Agglomerative Clustering step:
- memberships must be saved,
- memberships must remain fixed during all FCFL rounds,
- memberships must be reused for both the hierarchical baseline and the proposed method.

## 5.3 Clustering scope

Agglomerative Clustering must be run separately inside each main cluster:
- Cluster 1: only Cluster 1 candidate leaf clients
- Cluster 2: only Cluster 2 candidate leaf clients
- Cluster 3: only Cluster 3 candidate leaf clients

No cross-cluster clustering is allowed.

---

## 6. Model-compatibility rules

## 6.1 Within a main cluster

Inside the same main cluster:
- all leaf clients must use the same input representation,
- all sub-clusters must use the same model family,
- the main-cluster model and sub-cluster models must be aggregation-compatible.

## 6.2 Across main clusters

Across different main clusters:
- feature spaces may differ,
- model families may differ,
- FL methods may differ,
- aggregation rules may differ,
- no parameters may be averaged together.

---

## 7. Aggregation rules

## 7.1 Universal hierarchical aggregation

Leaf-client to sub-cluster aggregation:

\[
 w_{m,s}^{t+1}
 =
 \sum_{i\in C_{m,s}}
 \frac{n_i}{N_{m,s}}w_{m,s,i}^{t+1}
\]

Sub-cluster to main-cluster aggregation:

\[
 w_m^{t+1}
 =
 \sum_{s\in S_m}
 \frac{N_{m,s}}{N_m}w_{m,s}^{t+1}
\]

## 7.2 Cluster-specific aggregation

### Cluster 1

- Aggregate only non-BatchNorm parameters.
- Keep BN state local.

### Cluster 2

- Standard weighted arithmetic mean.

### Cluster 3

- Standard weighted arithmetic mean for model parameters.
- Maintain SCAFFOLD control-variate state separately.

---

## 8. FL-method rules

## 8.1 Cluster 1 — FedBN

Required behavior:
- local BatchNorm statistics must not be globally averaged,
- non-BN parameters must be aggregated by weighted mean,
- main-cluster head and sub-cluster heads must both obey the non-BN aggregation rule.

## 8.2 Cluster 2 — FedProx

Required behavior:
- local loss must include the FedProx proximal penalty,
- server-side aggregation remains weighted arithmetic mean,
- no change to hierarchy.

## 8.3 Cluster 3 — SCAFFOLD

Required behavior:
- Cluster 3 must maintain main-cluster control variate state,
- each participating leaf client must maintain its local control variate,
- sub-cluster heads must forward both model updates and control-variate increments,
- SCAFFOLD must not be implemented as a flat FL system ignoring the sub-cluster layer.

---

## 9. Round semantics

At round `t`:

1. Main-cluster head holds parent model `w_m^t`.
2. Main-cluster head sends `w_m^t` to all sub-cluster heads.
3. Each sub-cluster head initializes `w_{m,s}^t <- w_m^t`.
4. Sub-cluster head sends the initialized sub-cluster model to its leaf clients.
5. Leaf clients train locally.
6. Sub-cluster heads aggregate leaf-client updates.
7. Main-cluster heads aggregate sub-cluster models.
8. Main-cluster heads log metadata.
9. Updated `w_m^{t+1}` becomes next-round parent initialization.

No round may violate this order.

---

## 10. Blockchain / ledger rules

## 10.1 Allowed ledger writers

Only main-cluster heads may create ledger records.

Sub-cluster heads and leaf clients are never ledger writers.

## 10.2 Allowed ledger content

Allowed metadata examples:
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
- subcluster child-model digest
- FL method
- aggregation rule
- participant count
- effective sample count
- timestamps
- submitter identity

## 10.3 Forbidden ledger content

The following must never be written to the ledger:
- raw records
- local client datasets
- full model tensors / state_dict dumps
- optimizer checkpoints
- BatchNorm local statistics as raw payload
- feature descriptors `z_i` as raw values

---

## 11. Baseline rules

## 11.1 Flat baseline (Baseline A)

- one main-cluster head per domain,
- no sub-cluster layer,
- no Agglomerative Clustering,
- 1D-CNN + FedAvg for all three clusters.

## 11.2 Uniform hierarchical baseline (Baseline B)

- same fixed agglomerative hierarchy as proposal,
- same membership files as proposal,
- 1D-CNN + FedAvg for all three clusters.

## 11.3 Proposed method

- same fixed agglomerative hierarchy as Baseline B,
- specialized cluster-specific models and FL methods.

---

## 12. Prohibited changes

The implementation must not introduce any of the following without explicit user approval:

- dynamic clustering
- reclustering during training
- cross-cluster averaging
- extra central server above the three main clusters
- new datasets
- new model families
- new FL methods
- new aggregation rules
- secure aggregation
- differential privacy
- Byzantine-robust aggregation
- blockchain-stored model weights
- blockchain-stored raw data

---

## 13. Failure policy

If any requested coding task conflicts with this architecture contract, the implementation must stop and return a message stating:

```text
ARCHITECTURE_CONTRACT_VIOLATION
```

followed by a short explanation of the violated rule.
