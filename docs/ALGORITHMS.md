# ALGORITHMS.md

## 1. Purpose

This document defines the implementation-level algorithms for the FCFL CPS/IIoT intrusion-detection project.

It contains:
- notation,
- the one-time offline Agglomerative Clustering procedure,
- candidate leaf-client formation,
- one complete hierarchical FCFL round,
- method-specific rules for FedBN, FedProx, and SCAFFOLD,
- metadata logging logic.

This document is aligned with the current report architecture and does not introduce dynamic clustering or cross-cluster averaging. fileciteturn3file0

---

## 2. Notation

- `m` = main-cluster index
- `I_m` = set of candidate leaf clients inside main cluster `m` before sub-cluster formation
- `z_i` = client-level descriptor for candidate leaf client `i`
- `K_m` = fixed number of sub-clusters in main cluster `m`
- `a_m(i)` = agglomerative assignment of client `i` to sub-cluster index in main cluster `m`
- `S_m` = set of fixed sub-clusters inside main cluster `m`
- `C_{m,s}` = set of leaf clients assigned to sub-cluster `s` inside main cluster `m`
- `n_i` = local sample count of leaf client `i`
- `N_{m,s}` = total sample count inside sub-cluster `s` of main cluster `m`
- `N_m` = total sample count inside main cluster `m`
- `w_m^t` = main-cluster parent model at round `t`
- `w_{m,s}^t` = sub-cluster model at round `t`
- `w_{m,s,i}^{t+1}` = locally updated model of leaf client `i`
- `c_m^t` = Cluster 3 main-cluster SCAFFOLD server control variate at round `t`
- `c_{m,s,i}^t` = Cluster 3 local client control variate
- `E` = number of local SCAFFOLD steps

---

## 3. Algorithm 1 — Candidate leaf-client formation

### Objective

Create candidate leaf clients from each main-cluster dataset before offline sub-cluster formation.

### Input

- cleaned dataset `D_m`
- target number of candidate leaf clients `L_m`
- timestamp column if available

### Output

- candidate leaf-client partitions `I_m`
- per-client train/validation/test splits

### Procedure

1. Load the cleaned dataset for main cluster `m`.
2. Sort samples by timestamp if a valid timestamp column exists.
3. If no timestamp exists, preserve file order.
4. Split the ordered samples into `L_m` contiguous shards.
5. Create one candidate leaf client per shard.
6. For each candidate leaf client, split local samples into train / validation / test.
7. Save client metadata and split sizes.

### Constraints

- Candidate leaf-client creation must occur before Agglomerative Clustering.
- Validation and test samples must not be used for descriptor computation.
- Client IDs must be deterministic.

---

## 4. Algorithm 2 — Offline Agglomerative sub-cluster formation

### Objective

Form one fixed sub-cluster layer inside each main cluster before FCFL training.

### Input

- candidate leaf clients `I_m`
- local training partitions `X_i`
- target sub-cluster count `K_m`

### Output

- fixed membership mapping `a_m(i)`
- fixed sub-cluster client sets `C_{m,s}`

### Descriptor definition

For each candidate leaf client `i`, compute:

\[
 z_i = [\mu_i ; \sigma_i]
\]

where

\[
 \mu_i = \frac{1}{n_i}\sum_{x\in X_i}x
\]

and

\[
 \sigma_i =
 \sqrt{
 \frac{1}{n_i}\sum_{x\in X_i}(x-\mu_i)^2
 }
\]

Here:
- `x` is a feature vector from the local training partition,
- `\mu_i` and `\sigma_i` are feature-wise statistics.

### Procedure

1. For main cluster `m`, build candidate leaf clients using Algorithm 1.
2. For each candidate leaf client `i`, compute descriptor `z_i` from training data only.
3. Standardize all descriptors inside the same main cluster.
4. Apply Agglomerative Clustering inside that main cluster only.
5. Use the fixed target number of clusters `K_m`.
6. Assign each candidate leaf client to exactly one sub-cluster.
7. Save the membership mapping to disk.
8. Freeze the memberships.
9. Reuse the same membership file for all hierarchical experiments.

### Default clustering configuration

Recommended implementation choice:

```text
linkage = ward
metric = euclidean
```

### Constraints

- No clustering across main clusters.
- No reclustering during training.
- No label-based clustering.
- No leakage-prone identifier fields in descriptors.

---

## 5. Algorithm 3 — One complete hierarchical FCFL round

### Objective

Execute one training round for a single main cluster.

### Input

- current parent model `w_m^t`
- fixed sub-cluster memberships `C_{m,s}`
- cluster-specific FL method
- local client data partitions

### Output

- updated parent model `w_m^{t+1}`
- method-specific state updates
- metadata record

### Procedure

1. Main-cluster head `m` holds the parent model `w_m^t`.
2. Main-cluster head sends `w_m^t` to all sub-cluster heads.
3. Each sub-cluster head initializes:

\[
 w_{m,s}^{t} \leftarrow w_m^t
\]

4. Each sub-cluster head sends the initialized sub-cluster model to its assigned leaf clients.
5. Each leaf client trains locally on its own partition.
6. Each leaf client returns its updated model to the sub-cluster head.
7. Each sub-cluster head performs leaf-to-sub-cluster aggregation:

\[
 w_{m,s}^{t+1}
 =
 \sum_{i\in C_{m,s}}
 \frac{n_i}{N_{m,s}}w_{m,s,i}^{t+1}
\]

where

\[
 N_{m,s}=\sum_{i\in C_{m,s}}n_i
\]

8. Each main-cluster head performs sub-cluster-to-main-cluster aggregation:

\[
 w_m^{t+1}
 =
 \sum_{s\in S_m}
 \frac{N_{m,s}}{N_m}w_{m,s}^{t+1}
\]

where

\[
 N_m=\sum_{s\in S_m}N_{m,s}
\]

9. Main-cluster head computes a model hash for `w_m^{t+1}`.
10. Main-cluster head writes metadata to the ledger.
11. Updated `w_m^{t+1}` becomes the next-round parent model.

### Constraints

- No direct leaf-client writes to the ledger.
- No sub-cluster-head writes to the ledger.
- No cross-cluster parameter sharing.

---

## 6. Cluster-specific method rules

## 6.1 Cluster 1 — FedBN special case

### Purpose

Handle feature-shift non-IID behavior by keeping BatchNorm statistics local.

### Parameter partition

Let the model parameters be partitioned into:
- `θ` = shared non-BN parameters
- `β_{m,s}` = BN statistics and local BN state

### Hierarchical aggregation rule

Sub-cluster level:

\[
 \theta_{m,s}^{t+1}
 =
 \sum_{i\in C_{m,s}}
 \frac{n_i}{N_{m,s}}\theta_{m,s,i}^{t+1}
\]

Main-cluster level:

\[
 \theta_m^{t+1}
 =
 \sum_{s\in S_m}
 \frac{N_{m,s}}{N_m}\theta_{m,s}^{t+1}
\]

BN state `β_{m,s}` remains local and is not globally averaged.

### Implementation rule

Exclude all BN-related keys from global aggregation.

Case-insensitive exclusion patterns:

```text
bn
batchnorm
running_mean
running_var
num_batches_tracked
```

---

## 6.2 Cluster 2 — FedProx special case

### Purpose

Stabilize local optimization under statistical heterogeneity.

### Local objective

For leaf client `i` in sub-cluster `s`:

\[
 \min_w\;F_{m,s,i}(w)+\frac{\mu}{2}\|w-w_{m,s}^t\|_2^2
\]

### Recommended default

```text
mu = 0.01
```

### Aggregation rule

Use standard weighted arithmetic mean at both hierarchy levels.

FedProx changes the local objective but not the server-side weighted aggregation logic.

---

## 6.3 Cluster 3 — SCAFFOLD special case

### Purpose

Correct client drift under heterogeneous data using control variates.

### State variables

- main-cluster server control variate: `c_m^t`
- local client control variate: `c_{m,s,i}^t`

### Local SCAFFOLD update

Let `y_{m,s,i}^{t,k}` denote the local model after `k` SCAFFOLD steps, with:

\[
 y_{m,s,i}^{t,0}=w_{m,s}^t
\]

The corrected local update for `k = 0, ..., E-1` is:

\[
 y_{m,s,i}^{t,k+1}
 =
 y_{m,s,i}^{t,k}
 -
 \eta_\ell
 \Bigl(
 g_{m,s,i}(y_{m,s,i}^{t,k})
 -
 c_{m,s,i}^t
 +
 c_m^t
 \Bigr)
\]

After `E` local steps:

\[
 w_{m,s,i}^{t+1}=y_{m,s,i}^{t,E}
\]

### Local control-variate update

\[
 c_{m,s,i}^{t+1}
 =
 c_{m,s,i}^{t}
 -
 c_m^{t}
 +
 \frac{1}{E\eta_\ell}
 (w_{m,s}^{t}-w_{m,s,i}^{t+1})
\]

Define increment:

\[
 \Delta c_{m,s,i}^{t}=c_{m,s,i}^{t+1}-c_{m,s,i}^{t}
\]

### Main-cluster server control-variate update

Let `P_m^t` denote participating Cluster 3 leaf clients and let:

\[
 Q_m=\sum_{s\in S_m}|C_{m,s}|
\]

Then:

\[
 c_m^{t+1}
 =
 c_m^t
 +
 \frac{1}{Q_m}
 \sum_{(s,i)\in P_m^t}\Delta c_{m,s,i}^{t}
\]

with `Δc = 0` for non-participating clients.

### Hierarchical implementation rule

Sub-cluster heads do not maintain independent server control variates.

Sub-cluster heads must:
- aggregate model updates upward,
- forward control-variate increments upward.

Main-cluster head updates the Cluster 3 server control variate.

---

## 7. Algorithm 4 — Metadata logging

### Objective

Write metadata-level governance records after a main-cluster model update is accepted.

### Allowed writer

- main-cluster head only

### Input

- round id
- cluster id
- cluster head id
- previous main model hash
- new main model hash
- clustering method
- clustering configuration hash
- subcluster membership hash
- subcluster count
- FL method
- aggregation rule
- effective sample count
- participant count
- timestamps
- submitter identity

### Procedure

1. After main-cluster aggregation completes, compute the new main model hash.
2. Collect round-level metadata.
3. Validate that no forbidden payload fields are present.
4. Append one metadata record to the ledger.
5. Continue to the next round.

### Forbidden payload

Never write:
- raw data
- full model weights
- optimizer checkpoints
- raw feature descriptors `z_i`

---

## 8. Algorithm 5 — Experiment execution order

### Objective

Run experiments in a consistent order that matches the report.

### Procedure

1. Profile datasets and validate schemas.
2. Preprocess all three datasets.
3. Form candidate leaf clients.
4. Run and save Agglomerative Clustering memberships.
5. Execute Baseline A (flat per-cluster FL).
6. Execute Baseline B (uniform hierarchical FCFL).
7. Execute proposed specialized hierarchical FCFL.
8. Execute required method ablations.
9. Compute metrics and generate outputs.
10. Validate ledger logs.

### Constraints

- The same frozen membership files must be used for Baseline B and the proposed method.
- No experiment may regenerate memberships unless the user explicitly requests reclustering.
