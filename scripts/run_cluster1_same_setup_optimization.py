from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from src.data import cluster1_balanced as c1_balanced  # noqa: E402
from src.data.schema_validation import DatasetSchemaError  # noqa: E402
from src.train_cluster1_proposed import run_cluster1_proposed  # noqa: E402


OUTPUT_ROOT = Path("outputs_c1_same_setup_optimization")
VARIANT_ROOT = Path("data/variants/hai_2103_same_setup_optimization")
CONFIG_DIR = Path("configs/same_setup_optimization")
DOC_PATH = Path("docs/CLUSTER1_SAME_SETUP_OPTIMIZATION.md")

FEATURE_VARIANTS = ("full", "no_constants", "no_constants_no_corr", "top_ap_40")
WINDOW_SETTINGS = ((32, 8), (64, 8), (64, 16))
BALANCE_RATIOS = (5, 8, 10, 15, 20)
POSITIVE_CLASS_WEIGHT_SCALES = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
LEARNING_RATES = (0.001, 0.003, 0.005)
DROPOUTS = (0.0, 0.05, 0.1)
SEED = 42
NUM_CLIENTS = 12

CONSTANT_TRAINING_FEATURES = {
    "P1_PP01AD",
    "P1_PP01AR",
    "P1_PP01BD",
    "P1_PP01BR",
    "P1_PP02D",
    "P1_PP02R",
    "P1_STSP",
    "P2_ASD",
    "P2_AutoGO",
    "P2_MSD",
    "P2_ManualGO",
    "P2_RTR",
    "P2_TripEx",
    "P2_VTR01",
    "P2_VTR02",
    "P2_VTR03",
    "P2_VTR04",
    "P3_LH",
    "P3_LL",
}


@dataclass(frozen=True)
class DataVariantSpec:
    feature_variant: str
    window_length: int
    stride: int
    ratio: int

    @property
    def key(self) -> str:
        return f"{self.feature_variant}_w{self.window_length}_s{self.stride}_r{self.ratio}to1"


@dataclass(frozen=True)
class Candidate:
    data: DataVariantSpec
    positive_class_weight_scale: float
    learning_rate: float
    dropout: float

    @property
    def key(self) -> str:
        scale = str(self.positive_class_weight_scale).replace(".", "p")
        lr = str(self.learning_rate).replace(".", "p")
        dropout = str(self.dropout).replace(".", "p")
        return f"{self.data.key}_pcw{scale}_lr{lr}_do{dropout}"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _ensure_membership_reuse_groups(membership_path: Path) -> None:
    if not membership_path.exists():
        return
    payload = json.loads(membership_path.read_text(encoding="utf-8"))
    required = ["baseline_uniform_hierarchical", "proposed_specialized_hierarchical"]
    observed = list(payload.get("reuse_for_experiment_groups", []))
    changed = False
    for group in required:
        if group not in observed:
            observed.append(group)
            changed = True
    if changed:
        payload["reuse_for_experiment_groups"] = observed
        _write_json(membership_path, payload)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_one_row_csv(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else None


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _format(value: Any) -> str:
    parsed = _float(value)
    if parsed is not None:
        return f"{parsed:.6f}"
    if value in (None, ""):
        return "MISSING"
    return str(value)


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


def _metric_sort_key(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    validation_f1 = _float(row.get("validation_f1")) or _float(row.get("best_validation_f1")) or -1.0
    validation_pr_auc = _float(row.get("validation_pr_auc")) or -1.0
    validation_fpr = _float(row.get("validation_fpr"))
    wall_clock = _float(row.get("wall_clock_training_seconds"))
    return (
        validation_f1,
        validation_pr_auc,
        -(validation_fpr if validation_fpr is not None else float("inf")),
        -(wall_clock if wall_clock is not None else float("inf")),
    )


def _all_data_specs() -> list[DataVariantSpec]:
    return [
        DataVariantSpec(feature_variant, window_length, stride, ratio)
        for feature_variant, (window_length, stride), ratio in product(
            FEATURE_VARIANTS,
            WINDOW_SETTINGS,
            BALANCE_RATIOS,
        )
    ]


def _all_candidate_space() -> list[Candidate]:
    return [
        Candidate(data, scale, learning_rate, dropout)
        for data, scale, learning_rate, dropout in product(
            _all_data_specs(),
            POSITIVE_CLASS_WEIGHT_SCALES,
            LEARNING_RATES,
            DROPOUTS,
        )
    ]


def _load_train_validation_feature_matrix(
    *,
    raw_files: Mapping[str, c1_balanced.ResolvedRawFile],
    headers: Mapping[str, Sequence[str]],
    feature_columns: Sequence[str],
    label_arrays: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for file_name in (*c1_balanced.TRAIN_SOURCE_FILES, *c1_balanced.VALIDATION_SOURCE_FILES):
        header = list(headers[file_name])
        indices = [header.index(feature) for feature in feature_columns]
        matrices.append(c1_balanced._load_feature_matrix(raw_files[file_name].path, indices))
        labels.append(label_arrays[file_name])
    return np.vstack(matrices), np.concatenate(labels).astype(np.int8, copy=False)


def _average_precision_scores_by_feature(
    matrix: np.ndarray,
    labels: np.ndarray,
    feature_columns: Sequence[str],
) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    both_classes = set(labels.astype(int).tolist()) == {0, 1}
    for index, feature in enumerate(feature_columns):
        values = matrix[:, index]
        finite = np.isfinite(values)
        if not both_classes or np.unique(values[finite]).size <= 1:
            direct = 0.0
            inverse = 0.0
        else:
            direct = float(average_precision_score(labels[finite], values[finite]))
            inverse = float(average_precision_score(labels[finite], -values[finite]))
        scores[feature] = {
            "average_precision_direct": direct,
            "average_precision_inverse": inverse,
            "average_precision_best_direction": max(direct, inverse),
        }
    return scores


def _high_correlation_pairs_train_validation(
    matrix: np.ndarray,
    feature_columns: Sequence[str],
    *,
    threshold: float = 0.98,
) -> list[dict[str, Any]]:
    std = np.nanstd(matrix, axis=0)
    valid = np.flatnonzero(std > 0.0)
    pairs: list[dict[str, Any]] = []
    if valid.size < 2:
        return pairs
    corr = np.corrcoef(matrix[:, valid].astype(np.float64, copy=False), rowvar=False)
    for left in range(valid.size):
        for right in range(left + 1, valid.size):
            value = float(corr[left, right])
            if np.isfinite(value) and abs(value) >= threshold:
                pairs.append(
                    {
                        "feature_a": feature_columns[int(valid[left])],
                        "feature_b": feature_columns[int(valid[right])],
                        "correlation": value,
                        "abs_correlation": abs(value),
                    }
                )
    pairs.sort(key=lambda row: row["abs_correlation"], reverse=True)
    return pairs


def _feature_variant_columns(
    *,
    all_features: Sequence[str],
    scores: Mapping[str, Mapping[str, float]],
    high_corr_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    no_constants = [feature for feature in all_features if feature not in CONSTANT_TRAINING_FEATURES]
    keep_corr = set(no_constants)
    original_index = {feature: index for index, feature in enumerate(all_features)}
    for pair in high_corr_pairs:
        feature_a = str(pair["feature_a"])
        feature_b = str(pair["feature_b"])
        if feature_a not in keep_corr or feature_b not in keep_corr:
            continue
        score_a = scores.get(feature_a, {}).get("average_precision_best_direction", 0.0)
        score_b = scores.get(feature_b, {}).get("average_precision_best_direction", 0.0)
        if score_a > score_b:
            keep_corr.remove(feature_b)
        elif score_b > score_a:
            keep_corr.remove(feature_a)
        elif original_index[feature_a] <= original_index[feature_b]:
            keep_corr.remove(feature_b)
        else:
            keep_corr.remove(feature_a)

    ranked = sorted(
        all_features,
        key=lambda feature: (
            scores.get(feature, {}).get("average_precision_best_direction", 0.0),
            -original_index[feature],
        ),
        reverse=True,
    )
    return {
        "full": list(all_features),
        "no_constants": no_constants,
        "no_constants_no_corr": [feature for feature in all_features if feature in keep_corr],
        "top_ap_40": ranked[:40],
    }


def _write_feature_selection_report(
    *,
    output_root: Path,
    feature_columns: Sequence[str],
    scores: Mapping[str, Mapping[str, float]],
    high_corr_pairs: Sequence[Mapping[str, Any]],
    feature_variants: Mapping[str, Sequence[str]],
) -> None:
    report = {
        "selection_scope": "train_source_plus_validation_only",
        "heldout_test_used_for_feature_selection": False,
        "all_feature_count": len(feature_columns),
        "constant_training_features": sorted(CONSTANT_TRAINING_FEATURES),
        "high_correlation_threshold": 0.98,
        "high_correlation_pair_count": len(high_corr_pairs),
        "high_correlation_pairs": list(high_corr_pairs),
        "feature_variants": {
            name: {
                "feature_count": len(columns),
                "feature_columns": list(columns),
                "dropped_features": [feature for feature in feature_columns if feature not in columns],
            }
            for name, columns in feature_variants.items()
        },
    }
    _write_json(output_root / "reports" / "feature_selection_report.json", report)
    rows = []
    for feature in feature_columns:
        rows.append(
            {
                "feature": feature,
                **scores[feature],
                "constant_training_source": feature in CONSTANT_TRAINING_FEATURES,
                "in_full": feature in feature_variants["full"],
                "in_no_constants": feature in feature_variants["no_constants"],
                "in_no_constants_no_corr": feature in feature_variants["no_constants_no_corr"],
                "in_top_ap_40": feature in feature_variants["top_ap_40"],
            }
        )
    _write_rows(output_root / "reports" / "feature_selection_scores.csv", rows)


def _build_windows(labels: np.ndarray, source_file: str, *, window_length: int, stride: int, start_sequence: int) -> list[c1_balanced.WindowRecord]:
    records: list[c1_balanced.WindowRecord] = []
    sequence = start_sequence
    if labels.shape[0] < window_length:
        return records
    for start in range(0, labels.shape[0] - window_length + 1, stride):
        stop = start + window_length
        attack_count = int(np.sum(labels[start:stop] == 1))
        records.append(
            c1_balanced.WindowRecord(
                source_file=source_file,
                start_row=int(start),
                end_row=int(stop - 1),
                label=int(attack_count > 0),
                attack_row_count_inside_window=attack_count,
                sequence=sequence,
            )
        )
        sequence += 1
    return records


def _window_counts(records: Sequence[c1_balanced.WindowRecord]) -> dict[str, int]:
    counts = Counter(str(int(record.label)) for record in records)
    return {key: int(counts[key]) for key in sorted(counts)}


def _split_records(
    windows_by_file: Mapping[str, Sequence[c1_balanced.WindowRecord]],
) -> tuple[list[c1_balanced.WindowRecord], list[c1_balanced.WindowRecord], list[c1_balanced.WindowRecord]]:
    train_records = [record for name in c1_balanced.TRAIN_SOURCE_FILES for record in windows_by_file[name]]
    val_records = [record for name in c1_balanced.VALIDATION_SOURCE_FILES for record in windows_by_file[name]]
    test_records = [record for name in c1_balanced.HELDOUT_TEST_FILES for record in windows_by_file[name]]
    return train_records, val_records, test_records


def _assemble_inputs(
    records: Sequence[c1_balanced.WindowRecord],
    *,
    raw_files: Mapping[str, c1_balanced.ResolvedRawFile],
    headers: Mapping[str, Sequence[str]],
    feature_columns: Sequence[str],
    window_length: int,
) -> np.ndarray:
    if not records:
        return np.empty((0, len(feature_columns), window_length), dtype=np.float32)
    output = np.empty((len(records), len(feature_columns), window_length), dtype=np.float32)
    by_file: dict[str, list[tuple[int, c1_balanced.WindowRecord]]] = {}
    for output_index, record in enumerate(records):
        by_file.setdefault(record.source_file, []).append((output_index, record))
    for source_file, indexed_records in by_file.items():
        header = list(headers[source_file])
        indices = [header.index(column) for column in feature_columns]
        matrix = c1_balanced._load_feature_matrix(raw_files[source_file].path, indices)
        for output_index, record in indexed_records:
            output[output_index] = matrix[record.start_row : record.end_row + 1].T
    return output


def _flatten_inputs(inputs: np.ndarray) -> np.ndarray:
    if inputs.size == 0:
        return np.empty((0, inputs.shape[1] if inputs.ndim == 3 else 0), dtype=np.float32)
    return inputs.transpose(0, 2, 1).reshape(-1, inputs.shape[1])


def _transform_inputs(
    inputs: np.ndarray,
    *,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    window_length: int,
) -> np.ndarray:
    if inputs.size == 0:
        return inputs.astype(np.float32, copy=True)
    flat = _flatten_inputs(inputs)
    transformed = scaler.transform(imputer.transform(flat)).astype(np.float32, copy=False)
    return transformed.reshape(inputs.shape[0], window_length, inputs.shape[1]).transpose(0, 2, 1)


def _client_ids() -> list[str]:
    return [f"C1_L{index + 1:03d}" for index in range(NUM_CLIENTS)]


def _write_client_metadata(
    *,
    output_root: Path,
    variant_id: str,
    data_spec: DataVariantSpec,
    train_records: Sequence[c1_balanced.WindowRecord],
    val_records: Sequence[c1_balanced.WindowRecord],
    test_records: Sequence[c1_balanced.WindowRecord],
) -> dict[str, Any]:
    clients: list[dict[str, Any]] = []
    for client_id in _client_ids():
        train = [record for record in train_records if record.client_id == client_id]
        validation = [record for record in val_records if record.client_id == client_id]
        test = [record for record in test_records if record.client_id == client_id]
        train_counts = _window_counts(train)
        val_counts = _window_counts(validation)
        test_counts = _window_counts(test)
        clients.append(
            {
                "client_id": client_id,
                "train_window_count": len(train),
                "train_positive_count": int(train_counts.get("1", 0)),
                "train_negative_count": int(train_counts.get("0", 0)),
                "val_window_count": len(validation),
                "val_positive_count": int(val_counts.get("1", 0)),
                "val_negative_count": int(val_counts.get("0", 0)),
                "test_window_count": len(test),
                "test_positive_count": int(test_counts.get("1", 0)),
                "test_negative_count": int(test_counts.get("0", 0)),
                "source_files_represented": sorted({record.source_file for record in train + validation + test}),
                "warning": "attack-free training client" if int(train_counts.get("1", 0)) == 0 else "",
            }
        )
    payload = {
        "cluster_id": 1,
        "dataset": "HAI 21.03",
        "variant": "cluster1_same_setup_optimization",
        "variant_id": variant_id,
        "feature_variant": data_spec.feature_variant,
        "window_length": data_spec.window_length,
        "stride": data_spec.stride,
        "negative_to_positive_ratio": data_spec.ratio,
        "num_leaf_clients": NUM_CLIENTS,
        "clients": clients,
        "cluster_split_label_counts": {
            "train": _window_counts(train_records),
            "validation": _window_counts(val_records),
            "test": _window_counts(test_records),
        },
    }
    _write_json(output_root / "data_variants" / variant_id / "clients" / "cluster1_leaf_clients.json", payload)
    return payload


def _agglomerative_model(n_subclusters: int) -> AgglomerativeClustering:
    try:
        return AgglomerativeClustering(n_clusters=n_subclusters, linkage="ward", metric="euclidean")
    except TypeError:
        return AgglomerativeClustering(n_clusters=n_subclusters, linkage="ward", affinity="euclidean")


def _client_sequence(client_id: str) -> int:
    return int(client_id.rsplit("L", 1)[1])


def _normalize_memberships(client_ids: Sequence[str], raw_labels: Sequence[int], fixed_ids: Sequence[str]) -> dict[str, str]:
    by_label: dict[int, list[str]] = {}
    for client_id, raw_label in zip(client_ids, raw_labels, strict=True):
        by_label.setdefault(int(raw_label), []).append(client_id)
    sorted_labels = sorted(by_label, key=lambda label: min(_client_sequence(client_id) for client_id in by_label[label]))
    return {
        client_id: fixed_ids[sorted_labels.index(int(raw_label))]
        for client_id, raw_label in zip(client_ids, raw_labels, strict=True)
    }


def _write_clustering(
    *,
    output_root: Path,
    variant_id: str,
    data_spec: DataVariantSpec,
    train_inputs: np.ndarray,
    train_records: Sequence[c1_balanced.WindowRecord],
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    client_ids = _client_ids()
    record_client_ids = np.asarray([record.client_id for record in train_records])
    descriptors: list[np.ndarray] = []
    descriptor_payload: dict[str, Any] = {
        "cluster_id": 1,
        "dataset": "HAI 21.03",
        "variant": "cluster1_same_setup_optimization",
        "variant_id": variant_id,
        "descriptor": "feature_wise_mean_std",
        "descriptor_source_split": "train",
        "feature_columns": list(feature_columns),
        "clients": [],
    }
    for client_id in client_ids:
        mask = record_client_ids == client_id
        if not bool(np.any(mask)):
            raise DatasetSchemaError(f"{variant_id}: {client_id} has no training windows for clustering.")
        client_inputs = train_inputs[mask]
        mean = client_inputs.mean(axis=(0, 2), dtype=np.float64)
        std = client_inputs.std(axis=(0, 2), dtype=np.float64)
        descriptor = np.concatenate([mean, std]).astype(np.float32, copy=False)
        descriptors.append(descriptor)
        descriptor_payload["clients"].append(
            {
                "client_id": client_id,
                "train_window_count": int(mask.sum()),
                "descriptor": descriptor.astype(float).tolist(),
            }
        )
    descriptor_matrix = np.vstack(descriptors)
    raw_labels = _agglomerative_model(2).fit_predict(descriptor_matrix).tolist()
    assignments = _normalize_memberships(client_ids, raw_labels, ("H1", "H2"))
    clients_for_membership = [
        {"client_id": client_id, "subcluster_id": assignments[client_id]}
        for client_id in client_ids
    ]
    membership_hash = hashlib.sha256(json.dumps(clients_for_membership, sort_keys=True).encode("utf-8")).hexdigest()
    clustering_dir = output_root / "data_variants" / variant_id / "clustering"
    membership = {
        "cluster_id": 1,
        "dataset": "HAI 21.03",
        "variant": "cluster1_same_setup_optimization",
        "variant_id": variant_id,
        "feature_variant": data_spec.feature_variant,
        "window_length": data_spec.window_length,
        "stride": data_spec.stride,
        "negative_to_positive_ratio": data_spec.ratio,
        "status": "ok",
        "clustering_method": "AgglomerativeClustering",
        "linkage": "ward",
        "metric": "euclidean",
        "descriptor": "feature_wise_mean_std",
        "descriptor_dim": int(descriptor_matrix.shape[1]),
        "descriptor_source_split": "train",
        "n_subclusters": 2,
        "fixed_subcluster_ids": ["H1", "H2"],
        "frozen": True,
        "membership_hash": membership_hash,
        "membership_file": str(clustering_dir / "cluster1_memberships.json"),
        "reuse_for_experiment_groups": [
            "baseline_uniform_hierarchical",
            "proposed_specialized_hierarchical",
        ],
        "subclusters": [
            {
                "subcluster_id": subcluster_id,
                "client_ids": [client_id for client_id in client_ids if assignments[client_id] == subcluster_id],
            }
            for subcluster_id in ("H1", "H2")
        ],
        "clients": clients_for_membership,
    }
    _write_json(clustering_dir / "client_descriptors.json", descriptor_payload)
    _write_json(clustering_dir / "cluster1_memberships.json", membership)
    return membership


def _empty_records() -> list[c1_balanced.WindowRecord]:
    return []


def prepare_data_variant(
    *,
    data_spec: DataVariantSpec,
    feature_columns: Sequence[str],
    raw_files: Mapping[str, c1_balanced.ResolvedRawFile],
    headers: Mapping[str, Sequence[str]],
    labels_by_file: Mapping[str, np.ndarray],
    output_root: Path,
    variant_root: Path,
    include_heldout_test: bool,
    seed: int,
) -> dict[str, Any]:
    variant_id = f"{data_spec.key}_{'with_test' if include_heldout_test else 'selection_no_test'}"
    variant_dir = variant_root / variant_id
    metadata_path = variant_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _ensure_membership_reuse_groups(
            output_root / "data_variants" / variant_id / "clustering" / "cluster1_memberships.json"
        )
        return metadata
    print(f"Preparing data variant {variant_id}", flush=True)

    sequence = 0
    windows_by_file: dict[str, list[c1_balanced.WindowRecord]] = {}
    for file_name in c1_balanced.EXPECTED_FILE_ORDER:
        records = _build_windows(
            labels_by_file[file_name],
            file_name,
            window_length=data_spec.window_length,
            stride=data_spec.stride,
            start_sequence=sequence,
        )
        sequence += len(records)
        windows_by_file[file_name] = records
    train_pool, val_pool, test_pool = _split_records(windows_by_file)
    train_positive = [record for record in train_pool if record.label == 1]
    train_negative = [record for record in train_pool if record.label == 0]
    if not train_positive:
        raise DatasetSchemaError(f"{data_spec.key}: no positive training windows.")
    needed_negatives = int(data_spec.ratio * len(train_positive))
    if needed_negatives > len(train_negative):
        raise DatasetSchemaError(
            f"{data_spec.key}: ratio requires {needed_negatives} negative windows but only "
            f"{len(train_negative)} are available."
        )
    rng = np.random.default_rng(seed)
    sampled_negative_indices = rng.choice(len(train_negative), size=needed_negatives, replace=False)
    sampled_negatives = [train_negative[int(index)] for index in sorted(sampled_negative_indices.tolist())]
    selected_train = train_positive + sampled_negatives
    train_records = c1_balanced._assign_clients(
        selected_train,
        num_clients=NUM_CLIENTS,
        seed=seed,
        shuffle_each_client=True,
    )
    val_records = c1_balanced._assign_clients(
        val_pool,
        num_clients=NUM_CLIENTS,
        seed=seed,
        shuffle_each_client=False,
    )
    test_records = (
        c1_balanced._assign_clients(
            test_pool,
            num_clients=NUM_CLIENTS,
            seed=seed,
            shuffle_each_client=False,
        )
        if include_heldout_test
        else _empty_records()
    )

    raw_train = _assemble_inputs(
        train_records,
        raw_files=raw_files,
        headers=headers,
        feature_columns=feature_columns,
        window_length=data_spec.window_length,
    )
    raw_val = _assemble_inputs(
        val_records,
        raw_files=raw_files,
        headers=headers,
        feature_columns=feature_columns,
        window_length=data_spec.window_length,
    )
    raw_test = _assemble_inputs(
        test_records,
        raw_files=raw_files,
        headers=headers,
        feature_columns=feature_columns,
        window_length=data_spec.window_length,
    )
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_flat = _flatten_inputs(raw_train)
    imputed_train = imputer.fit_transform(train_flat)
    scaler.fit(imputed_train)
    train_inputs = scaler.transform(imputed_train).astype(np.float32, copy=False).reshape(
        raw_train.shape[0],
        data_spec.window_length,
        raw_train.shape[1],
    ).transpose(0, 2, 1)
    val_inputs = _transform_inputs(raw_val, imputer=imputer, scaler=scaler, window_length=data_spec.window_length)
    test_inputs = _transform_inputs(raw_test, imputer=imputer, scaler=scaler, window_length=data_spec.window_length)

    c1_balanced._save_window_npz(variant_dir / "train_windows.npz", inputs=train_inputs, records=train_records)
    c1_balanced._save_window_npz(variant_dir / "val_windows.npz", inputs=val_inputs, records=val_records)
    c1_balanced._save_window_npz(variant_dir / "test_windows.npz", inputs=test_inputs, records=test_records)

    preprocessing_dir = output_root / "data_variants" / variant_id / "preprocessing"
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    imputer_path = preprocessing_dir / "imputer.joblib"
    scaler_path = preprocessing_dir / "scaler.joblib"
    joblib.dump(imputer, imputer_path)
    joblib.dump(scaler, scaler_path)

    train_counts = _window_counts(train_records)
    val_counts = _window_counts(val_records)
    test_counts = _window_counts(test_records)
    positive_class_weight = float(train_counts.get("0", 0) / train_counts.get("1", 1))
    preprocessing_summary = {
        "cluster_id": 1,
        "dataset": "HAI 21.03",
        "variant": "cluster1_same_setup_optimization",
        "variant_id": variant_id,
        "fit_scope": "balanced_training_windows_only",
        "imputer": "SimpleImputer(strategy='median')",
        "scaler": "StandardScaler",
        "fit_shape": [int(train_flat.shape[0]), int(train_flat.shape[1])],
        "input_tensor_layout": "N,C,T",
        "window_length": data_spec.window_length,
        "stride": data_spec.stride,
        "feature_count": len(feature_columns),
        "feature_columns": list(feature_columns),
        "positive_class_weight": positive_class_weight,
        "heldout_test_windows_included_for_evaluation": include_heldout_test,
        "artifact_paths": {"imputer": str(imputer_path), "scaler": str(scaler_path)},
    }
    _write_json(preprocessing_dir / "preprocessing_summary.json", preprocessing_summary)
    _write_client_metadata(
        output_root=output_root,
        variant_id=variant_id,
        data_spec=data_spec,
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
    )
    membership = _write_clustering(
        output_root=output_root,
        variant_id=variant_id,
        data_spec=data_spec,
        train_inputs=train_inputs,
        train_records=train_records,
        feature_columns=feature_columns,
    )
    metadata = {
        "cluster_id": 1,
        "dataset": "HAI 21.03",
        "variant": "cluster1_same_setup_optimization",
        "variant_id": variant_id,
        "feature_variant": data_spec.feature_variant,
        "negative_to_positive_ratio": data_spec.ratio,
        "seed": seed,
        "window_length": data_spec.window_length,
        "stride": data_spec.stride,
        "label_rule": "any_positive_row",
        "input_tensor_layout": "N,C,T",
        "preprocessing_applied_to_npz_inputs": True,
        "train_source_files": list(c1_balanced.TRAIN_SOURCE_FILES),
        "validation_source_files": list(c1_balanced.VALIDATION_SOURCE_FILES),
        "heldout_test_source_files": list(c1_balanced.HELDOUT_TEST_FILES),
        "heldout_test_windows_included_for_evaluation": include_heldout_test,
        "class_counts": {"train": train_counts, "validation": val_counts, "test": test_counts},
        "natural_window_counts_before_balancing": {
            "train": _window_counts(train_pool),
            "validation": _window_counts(val_pool),
            "heldout_test": _window_counts(test_pool),
        },
        "positive_class_weight": positive_class_weight,
        "feature_count": len(feature_columns),
        "feature_columns": list(feature_columns),
        "paths": {
            "train_windows": str(variant_dir / "train_windows.npz"),
            "val_windows": str(variant_dir / "val_windows.npz"),
            "test_windows": str(variant_dir / "test_windows.npz"),
            "membership": str(output_root / "data_variants" / variant_id / "clustering" / "cluster1_memberships.json"),
            "preprocessing_summary": str(preprocessing_dir / "preprocessing_summary.json"),
        },
        "leakage_prevention": {
            "test3_used_for_training": False,
            "test4_test5_used_for_training": False,
            "test4_test5_used_for_validation": False,
            "test4_test5_used_for_scaling_or_imputation": False,
            "test4_test5_used_for_clustering": False,
            "test4_test5_used_for_feature_pruning": False,
            "test4_test5_used_for_ratio_selection": False,
            "test4_test5_used_for_hyperparameter_selection": False,
            "validation_balanced": False,
            "heldout_test_balanced": False,
            "heldout_test_attached_only_for_final_evaluation": include_heldout_test,
        },
        "membership_hash": membership["membership_hash"],
    }
    _write_json(metadata_path, metadata)
    return metadata


def _cluster_config_payload(
    *,
    data_spec: DataVariantSpec,
    variant_metadata: Mapping[str, Any],
    cluster_config_path: Path,
) -> dict[str, Any]:
    variant_dir = Path(str(variant_metadata["paths"]["train_windows"])).parent.resolve()
    return {
        "config_version": 1,
        "cluster": {
            "id": 1,
            "key": "C1",
            "dataset_key": f"HAI_2103_SAME_SETUP_OPT_{_short_hash(data_spec.key).upper()}",
            "dataset_name": f"HAI 21.03 same-setup optimized Cluster 1 ({data_spec.key})",
            "audit_report": str(OUTPUT_ROOT / "reports" / "hai_file_audit.json"),
        },
        "data": {
            "data_root_env_var": "FCFL_DATA_ROOT",
            "default_data_root": "data",
            "current_raw_input_dir": "data/raw/hai_2103/hai-21.03",
            "current_raw_files": list(c1_balanced.EXPECTED_FILE_ORDER),
            "balanced_variant_dir": str(variant_dir),
            "balanced_output_root": str((OUTPUT_ROOT / "data_variants" / str(variant_metadata["variant_id"])).resolve()),
            "training_input_mode": "raw_csv_glob",
            "training_input_glob": "data/raw/hai_2103/hai-21.03/*.csv",
            "schema_consistent_across_files": True,
            "label_column": c1_balanced.LABEL_COLUMN,
            "label_column_confirmed_from_audit": True,
            "candidate_label_columns_present": [c1_balanced.LABEL_COLUMN],
            "timestamp_or_order_columns": ["time"],
            "excluded_columns": ["attack", "time", "attack_P1", "attack_P2", "attack_P3"],
            "exclude_if_present": ["attack_P4", "Attack", "label", "Label", "target", "Target", "timestamp", "Timestamp", "Time", "date", "Date"],
        },
        "partitioning": {
            "candidate_leaf_clients": NUM_CLIENTS,
            "strategy": "balanced_window_npz",
            "train_source_files": list(c1_balanced.TRAIN_SOURCE_FILES),
            "validation_source_files": list(c1_balanced.VALIDATION_SOURCE_FILES),
            "heldout_test_files": list(c1_balanced.HELDOUT_TEST_FILES),
            "negative_to_positive_ratio": data_spec.ratio,
        },
        "clustering": {
            "fixed_subclusters": 2,
            "fixed_subcluster_ids": ["H1", "H2"],
            "membership_file": str((OUTPUT_ROOT / "data_variants" / str(variant_metadata["variant_id"]) / "clustering" / "cluster1_memberships.json").resolve()),
        },
        "preprocessing": {
            "input_type": "multivariate_time_series",
            "window_length": data_spec.window_length,
            "stride": data_spec.stride,
            "window_label_rule": "any_positive_row",
            "preprocessed_npz": True,
        },
        "runtime_validation": {
            "require_training_input_to_exist": True,
            "require_schema_consistency_across_files": True,
            "require_label_column_to_exist": True,
            "heldout_test_files_must_not_be_used_for_training": True,
            "heldout_test_files_must_not_be_used_for_validation": True,
            "fit_preprocessing_on_training_windows_only": True,
        },
    }


def _proposed_config_payload(
    *,
    experiment_id: str,
    candidate: Candidate,
    cluster_config_path: Path,
    membership_file: Path,
    output_root: Path,
) -> dict[str, Any]:
    return {
        "config_version": 1,
        "experiment_group": "proposed_specialized_hierarchical",
        "description": f"Same-setup optimized Cluster 1 candidate {candidate.key}.",
        "source_of_truth": {
            "architecture_contract": "docs/ARCHITECTURE_CONTRACT.md",
            "data_contract": "docs/DATA_CONTRACT.md",
            "same_setup_optimization": str(DOC_PATH),
        },
        "ledger": {"metadata_only": True},
        "training_defaults": {"rounds": 50, "local_epochs": 1, "batch_size": 128, "seeds": [SEED]},
        "smoke_test_defaults": {"rounds": 2, "local_epochs": 1, "batch_size": 64, "seed": SEED},
        "clustering_defaults": {
            "method": "AgglomerativeClustering",
            "linkage": "ward",
            "metric": "euclidean",
            "descriptor": "feature_wise_mean_std",
            "reuse_frozen_memberships": True,
        },
        "clusters": [
            {
                "experiment_id": experiment_id,
                "cluster_config": str(cluster_config_path),
                "hierarchy": "hierarchical_fixed",
                "clustering_method": "agglomerative",
                "n_subclusters": 2,
                "descriptor": "feature_wise_mean_std",
                "membership_file": str(membership_file),
                "model_family": "cnn1d_bn",
                "fl_method": "FedBN",
                "aggregation": "weighted_non_bn_mean",
                "model_hyperparameters": {
                    "channels": [32, 64, 64],
                    "kernel_sizes": [5, 3, 3],
                    "hidden_dim": 32,
                    "dropout": candidate.dropout,
                },
                "training_hyperparameters": {
                    "learning_rate": candidate.learning_rate,
                    "positive_class_weight_scale": candidate.positive_class_weight_scale,
                },
                "output_root": str(output_root),
            }
        ],
        "runtime_validation": {
            "require_cluster_configs_to_exist": True,
            "require_frozen_membership_files_before_training": True,
            "require_ledger_metadata_only": True,
            "require_same_setup_optimized_cluster1_outputs": True,
        },
    }


def _write_candidate_configs(
    *,
    phase: str,
    rank: int,
    candidate: Candidate,
    variant_metadata: Mapping[str, Any],
    run_output_root: Path,
) -> tuple[str, Path]:
    candidate_slug = f"{rank:04d}_{_short_hash(candidate.key)}"
    experiment_id = f"P_C1_OPT_{phase.upper()}_{rank:04d}"
    cluster_config_path = CONFIG_DIR / phase / f"cluster1_{candidate_slug}.yaml"
    proposed_config_path = CONFIG_DIR / phase / f"proposed_{candidate_slug}.yaml"
    membership_file = OUTPUT_ROOT / "data_variants" / str(variant_metadata["variant_id"]) / "clustering" / "cluster1_memberships.json"
    cluster_config = _cluster_config_payload(
        data_spec=candidate.data,
        variant_metadata=variant_metadata,
        cluster_config_path=cluster_config_path,
    )
    cluster_config_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_config_path.write_text(yaml.safe_dump(cluster_config, sort_keys=False), encoding="utf-8")
    proposed_config = _proposed_config_payload(
        experiment_id=experiment_id,
        candidate=candidate,
        cluster_config_path=cluster_config_path,
        membership_file=membership_file,
        output_root=run_output_root,
    )
    proposed_config_path.parent.mkdir(parents=True, exist_ok=True)
    proposed_config_path.write_text(yaml.safe_dump(proposed_config, sort_keys=False), encoding="utf-8")
    return experiment_id, proposed_config_path


def _result_row(
    *,
    phase: str,
    rank: int,
    candidate: Candidate,
    experiment_id: str,
    metrics: Mapping[str, Any],
    summary: Mapping[str, Any],
    config_path: Path,
    run_output_root: Path,
    include_heldout_test: bool,
) -> dict[str, Any]:
    validation = summary.get("best_round_validation_metrics", {})
    if not isinstance(validation, Mapping):
        validation = {}
    test = summary.get("best_round_test_metrics", {})
    if not isinstance(test, Mapping):
        test = {}
    return {
        "phase": phase,
        "rank": rank,
        "candidate_key": candidate.key,
        "experiment_id": experiment_id,
        "feature_variant": candidate.data.feature_variant,
        "window_length": candidate.data.window_length,
        "stride": candidate.data.stride,
        "negative_to_positive_ratio": candidate.data.ratio,
        "positive_class_weight_scale": candidate.positive_class_weight_scale,
        "learning_rate": candidate.learning_rate,
        "dropout": candidate.dropout,
        "rounds": summary.get("rounds"),
        "max_train_examples_per_client": summary.get("data_summary", {}).get("max_train_examples_per_client"),
        "heldout_test_attached": include_heldout_test,
        "validation_precision": validation.get("precision"),
        "validation_recall": validation.get("recall"),
        "validation_f1": validation.get("f1", metrics.get("best_validation_f1")),
        "validation_pr_auc": validation.get("pr_auc"),
        "validation_fpr": validation.get("fpr"),
        "threshold_used": metrics.get("threshold_used"),
        "test_accuracy": test.get("accuracy", metrics.get("test_accuracy")),
        "test_precision": test.get("precision", metrics.get("test_precision")),
        "test_recall": test.get("recall", metrics.get("test_recall")),
        "test_f1": test.get("f1", metrics.get("test_f1")),
        "test_auroc": test.get("auroc", metrics.get("test_auroc")),
        "test_pr_auc": test.get("pr_auc", metrics.get("test_pr_auc")),
        "test_fpr": test.get("fpr", metrics.get("test_fpr")),
        "test_confusion_matrix": json.dumps(test.get("confusion_matrix", metrics.get("test_confusion_matrix"))),
        "wall_clock_training_seconds": metrics.get("wall_clock_training_seconds"),
        "config_path": str(config_path),
        "run_output_root": str(run_output_root),
    }


def _run_candidate(
    *,
    phase: str,
    rank: int,
    candidate: Candidate,
    variant_metadata: Mapping[str, Any],
    rounds: int,
    max_train_examples_per_client: int | None,
    include_heldout_test: bool,
) -> dict[str, Any]:
    run_output_root = OUTPUT_ROOT / phase / f"{rank:04d}_{_short_hash(candidate.key)}"
    print(
        f"Running {phase} candidate {rank}: {candidate.key} "
        f"(rounds={rounds}, max_train_per_client={max_train_examples_per_client}, "
        f"heldout_test={include_heldout_test})",
        flush=True,
    )
    experiment_id, proposed_config_path = _write_candidate_configs(
        phase=phase,
        rank=rank,
        candidate=candidate,
        variant_metadata=variant_metadata,
        run_output_root=run_output_root,
    )
    result = run_cluster1_proposed(
        proposed_config_path=proposed_config_path,
        rounds=rounds,
        local_epochs=1,
        batch_size=128,
        learning_rate=candidate.learning_rate,
        seed=SEED,
        max_train_examples_per_client=max_train_examples_per_client,
        output_root=run_output_root,
        cnn_bn_dropout=candidate.dropout,
        positive_class_weight_scale=candidate.positive_class_weight_scale,
    )
    metrics = _load_one_row_csv(result.metrics_csv_path) or {}
    return _result_row(
        phase=phase,
        rank=rank,
        candidate=candidate,
        experiment_id=experiment_id,
        metrics=metrics,
        summary=result.summary,
        config_path=proposed_config_path,
        run_output_root=run_output_root,
        include_heldout_test=include_heldout_test,
    )


def _dedupe_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    deduped: list[Candidate] = []
    for candidate in candidates:
        if candidate.key in seen:
            continue
        deduped.append(candidate)
        seen.add(candidate.key)
    return deduped


def _candidate_from_row(row: Mapping[str, Any]) -> Candidate:
    data = DataVariantSpec(
        feature_variant=str(row["feature_variant"]),
        window_length=int(row["window_length"]),
        stride=int(row["stride"]),
        ratio=int(row["negative_to_positive_ratio"]),
    )
    return Candidate(
        data=data,
        positive_class_weight_scale=float(row["positive_class_weight_scale"]),
        learning_rate=float(row["learning_rate"]),
        dropout=float(row["dropout"]),
    )


def _read_search_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _comparison_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment_id in ("A_C1", "B_C1", "P_C1"):
        metrics = _load_one_row_csv(Path("outputs") / "metrics" / f"{experiment_id}_metrics.csv")
        if metrics:
            rows.append({"source": "original", **metrics})
    balanced_path = Path("outputs_c1_balanced_train/reports/cluster1_balanced_training_comparison.csv")
    if balanced_path.exists():
        with balanced_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("experiment_id", "")).startswith(("P_C1_BAL_", "B_C1_BAL_")):
                    rows.append({"source": "balanced_cluster1", **row})
    return rows


def _write_final_comparison(
    *,
    search_rows: Sequence[Mapping[str, Any]],
    best_config: Mapping[str, Any],
    final_row: Mapping[str, Any],
) -> None:
    comparison_rows = _comparison_rows()
    original_p = next((row for row in comparison_rows if row.get("experiment_id") == "P_C1"), None)
    original_b = next((row for row in comparison_rows if row.get("experiment_id") == "B_C1"), None)
    balanced_b_rows = [row for row in comparison_rows if str(row.get("experiment_id", "")).startswith("B_C1_BAL_")]
    balanced_p_rows = [row for row in comparison_rows if str(row.get("experiment_id", "")).startswith("P_C1_BAL_")]
    best_balanced_b = max(balanced_b_rows, key=_metric_sort_key) if balanced_b_rows else None
    best_balanced_p = max(balanced_p_rows, key=_metric_sort_key) if balanced_p_rows else None

    final_f1 = _float(final_row.get("test_f1"))
    final_val_f1 = _float(final_row.get("validation_f1"))
    p_f1 = _float(original_p.get("test_f1")) if original_p else None
    b_f1 = _float(original_b.get("test_f1")) if original_b else None
    bb_f1 = _float(best_balanced_b.get("test_f1")) if best_balanced_b else None
    original_p_fpr = _float(original_p.get("test_fpr")) if original_p else None
    final_fpr = _float(final_row.get("test_fpr"))
    original_p_recall = _float(original_p.get("test_recall")) if original_p else None
    final_recall = _float(final_row.get("test_recall"))

    beats_p = final_f1 is not None and p_f1 is not None and final_f1 > p_f1
    beats_b = final_f1 is not None and b_f1 is not None and final_f1 > b_f1
    beats_balanced_b = final_f1 is not None and bb_f1 is not None and final_f1 > bb_f1
    recall_improved = final_recall is not None and original_p_recall is not None and final_recall > original_p_recall
    fpr_exploded = final_fpr is not None and original_p_fpr is not None and final_fpr > max(0.05, original_p_fpr * 10.0)

    lines = [
        "# Cluster 1 Same-Setup Optimization Final Comparison",
        "",
        "This is a same-setup optimized Cluster 1 run. It keeps the current proposed Cluster 1 model family, FedBN method, weighted non-BN aggregation, fixed hierarchy, agglomerative membership role, no-cross-cluster averaging, and ledger logic unchanged.",
        "",
        "Selection used validation F1 only. Held-out `test4.csv` and `test5.csv` were attached only for the final selected candidate evaluation.",
        "",
        "## Selected Candidate",
        "",
        f"- Candidate: `{best_config.get('candidate_key')}`",
        f"- Feature variant: `{best_config.get('feature_variant')}`",
        f"- Balance ratio: `{best_config.get('negative_to_positive_ratio')}:1`",
        f"- Positive class weight scale: `{best_config.get('positive_class_weight_scale')}`",
        f"- Window: `{best_config.get('window_length')}` rows, stride `{best_config.get('stride')}`",
        f"- Learning rate/dropout: `{best_config.get('learning_rate')}` / `{best_config.get('dropout')}`",
        f"- Validation F1 used for selection: `{_format(final_val_f1)}`",
        "",
        "## Final Held-Out Test Metrics",
        "",
        f"- test_accuracy: `{_format(final_row.get('test_accuracy'))}`",
        f"- test_precision: `{_format(final_row.get('test_precision'))}`",
        f"- test_recall: `{_format(final_row.get('test_recall'))}`",
        f"- test_f1: `{_format(final_row.get('test_f1'))}`",
        f"- test_auroc: `{_format(final_row.get('test_auroc'))}`",
        f"- test_pr_auc: `{_format(final_row.get('test_pr_auc'))}`",
        f"- test_fpr: `{_format(final_row.get('test_fpr'))}`",
        f"- confusion_matrix: `{final_row.get('test_confusion_matrix')}`",
        f"- threshold_used: `{_format(final_row.get('threshold_used'))}`",
        "",
        "## Comparisons",
        "",
        "| row | validation_f1 | test_f1 | test_recall | test_precision | test_fpr | confusion_matrix |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| `same_setup_optimized_P_C1` | {_format(final_row.get('validation_f1'))} | {_format(final_row.get('test_f1'))} | {_format(final_row.get('test_recall'))} | {_format(final_row.get('test_precision'))} | {_format(final_row.get('test_fpr'))} | `{final_row.get('test_confusion_matrix')}` |",
    ]
    for label, row in (("original_A_C1", next((r for r in comparison_rows if r.get("experiment_id") == "A_C1"), None)), ("original_B_C1", original_b), ("original_P_C1", original_p), ("best_balanced_B_C1", best_balanced_b), ("best_balanced_P_C1", best_balanced_p)):
        if not row:
            continue
        lines.append(
            f"| `{label}` | {_format(row.get('validation_f1') or row.get('best_validation_f1'))} | "
            f"{_format(row.get('test_f1'))} | {_format(row.get('test_recall'))} | "
            f"{_format(row.get('test_precision'))} | {_format(row.get('test_fpr'))} | "
            f"`{row.get('test_confusion_matrix')}` |"
        )
    lines.extend(
        [
            "",
            "## Required Answers",
            "",
            f"1. Did optimized same-setup `P_C1` beat original `P_C1`? `{'YES' if beats_p else 'NO'}`.",
            f"2. Did it beat original `B_C1`? `{'YES' if beats_b else 'NO'}`.",
            f"3. Did it beat balanced `B_C1` if available? `{'YES' if beats_balanced_b else 'NO' if best_balanced_b else 'MISSING BALANCED B_C1'}`.",
            f"4. Did it improve recall without exploding FPR? `{'YES' if recall_improved and not fpr_exploded else 'NO'}`. Recall improved=`{recall_improved}`, FPR exploded=`{fpr_exploded}`.",
            f"5. Is this strong enough to keep as final Cluster 1 proposed method? `{'YES' if beats_p and beats_b and not fpr_exploded else 'NO'}`.",
            "",
            "## Search Scope",
            "",
            f"- Search rows written: `{len(search_rows)}`",
            f"- Full-search rows considered for selection: `{sum(1 for row in search_rows if row.get('phase') == 'full')}`",
            "- Selection rule: validation F1, then validation PR-AUC, then lower validation FPR, then lower wall-clock time.",
        ]
    )
    path = OUTPUT_ROOT / "reports" / "final_comparison.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_doc() -> None:
    lines = [
        "# Cluster 1 Same-Setup Optimization",
        "",
        "This document tracks a same-setup optimization sweep for proposed Cluster 1. The sweep changes only leakage-safe dataset construction, feature selection, window setting, training-window balance, positive-class weighting scale, learning rate, and dropout.",
        "",
        "Unchanged architecture and protocol:",
        "- Exactly the current proposed Cluster 1 model family from `configs/proposed.yaml` is used: CNN1D-BN.",
        "- FL method remains FedBN.",
        "- Aggregation remains weighted non-BN mean.",
        "- The fixed sub-cluster hierarchy, agglomerative membership role, no-cross-cluster averaging, and ledger metadata-only rules are unchanged.",
        "- Cluster 2 and Cluster 3 are not touched.",
        "",
        "Leakage controls:",
        "- Training source: `train1.csv`, `train2.csv`, `train3.csv`, `test1.csv`, `test2.csv`.",
        "- Validation and selection: `test3.csv`.",
        "- Held-out test: `test4.csv`, `test5.csv`.",
        "- Held-out test windows are omitted during fast and full candidate selection and attached only to the final selected rerun.",
        "- Feature pruning and ranking use training source plus validation only.",
        "- Imputation and scaling are fitted on balanced training windows only.",
        "- Validation and held-out test windows are not balanced.",
        "",
        "Generated reports:",
        "- `outputs_c1_same_setup_optimization/reports/search_results.csv`",
        "- `outputs_c1_same_setup_optimization/reports/best_config.json`",
        "- `outputs_c1_same_setup_optimization/reports/final_comparison.md`",
    ]
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_optimization(
    *,
    fast_rounds: int = 10,
    full_rounds: int = 50,
    fast_max_train_examples_per_client: int | None = 256,
    top_data_variants_for_hyperparams: int = 5,
    top_full_candidates: int = 10,
    seed: int = SEED,
) -> dict[str, Path]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    VARIANT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _write_doc()

    raw_audit, raw_files, headers, labels_by_file = c1_balanced._audit_raw_files(
        raw_dir=c1_balanced.DEFAULT_RAW_DIR,
        output_root=OUTPUT_ROOT,
        expected_counts=c1_balanced.EXPECTED_HAI_COUNTS,
    )
    if raw_audit["status"] != "ok":
        raise RuntimeError(f"Raw HAI audit failed: {raw_audit['status']}")
    all_features, _ = c1_balanced._numeric_feature_columns(
        headers=headers,
        raw_files=raw_files,
        label_column=c1_balanced.LABEL_COLUMN,
    )
    train_val_matrix, train_val_labels = _load_train_validation_feature_matrix(
        raw_files=raw_files,
        headers=headers,
        feature_columns=all_features,
        label_arrays=labels_by_file,
    )
    feature_scores = _average_precision_scores_by_feature(train_val_matrix, train_val_labels, all_features)
    high_corr_pairs = _high_correlation_pairs_train_validation(train_val_matrix, all_features)
    feature_variants = _feature_variant_columns(
        all_features=all_features,
        scores=feature_scores,
        high_corr_pairs=high_corr_pairs,
    )
    _write_feature_selection_report(
        output_root=OUTPUT_ROOT,
        feature_columns=all_features,
        scores=feature_scores,
        high_corr_pairs=high_corr_pairs,
        feature_variants=feature_variants,
    )

    all_space = _all_candidate_space()
    _write_json(
        OUTPUT_ROOT / "reports" / "candidate_space.json",
        {
            "candidate_count": len(all_space),
            "feature_variants": list(FEATURE_VARIANTS),
            "window_settings": [{"window_length": w, "stride": s} for w, s in WINDOW_SETTINGS],
            "ratios": list(BALANCE_RATIOS),
            "positive_class_weight_scales": list(POSITIVE_CLASS_WEIGHT_SCALES),
            "learning_rates": list(LEARNING_RATES),
            "dropouts": list(DROPOUTS),
            "full_cartesian_not_run_blindly": True,
        },
    )

    search_results_path = OUTPUT_ROOT / "reports" / "search_results.csv"
    search_rows = _read_search_rows(search_results_path)
    completed_keys = {f"{row.get('phase')}::{row.get('candidate_key')}" for row in search_rows}

    data_screen_candidates = [
        Candidate(data, 0.0, 0.003, 0.05)
        for data in _all_data_specs()
    ]
    fast_rank = 1 + sum(1 for row in search_rows if row.get("phase") == "fast")
    for candidate in data_screen_candidates:
        if f"fast::{candidate.key}" in completed_keys:
            continue
        metadata = prepare_data_variant(
            data_spec=candidate.data,
            feature_columns=feature_variants[candidate.data.feature_variant],
            raw_files=raw_files,
            headers=headers,
            labels_by_file=labels_by_file,
            output_root=OUTPUT_ROOT,
            variant_root=VARIANT_ROOT,
            include_heldout_test=False,
            seed=seed,
        )
        row = _run_candidate(
            phase="fast",
            rank=fast_rank,
            candidate=candidate,
            variant_metadata=metadata,
            rounds=fast_rounds,
            max_train_examples_per_client=fast_max_train_examples_per_client,
            include_heldout_test=False,
        )
        search_rows.append(row)
        _write_rows(search_results_path, search_rows)
        completed_keys.add(f"fast::{candidate.key}")
        fast_rank += 1

    data_screen_rows = [row for row in search_rows if row.get("phase") == "fast" and float(row.get("positive_class_weight_scale", -1)) == 0.0 and float(row.get("learning_rate", -1)) == 0.003 and float(row.get("dropout", -1)) == 0.05]
    top_data_rows = sorted(data_screen_rows, key=_metric_sort_key, reverse=True)[:top_data_variants_for_hyperparams]

    hyperparam_candidates: list[Candidate] = []
    for row in top_data_rows:
        base = _candidate_from_row(row)
        for scale in POSITIVE_CLASS_WEIGHT_SCALES:
            hyperparam_candidates.append(replace(base, positive_class_weight_scale=scale))
        best_scale_candidates = [candidate for candidate in hyperparam_candidates if candidate.data == base.data]
        for learning_rate in LEARNING_RATES:
            hyperparam_candidates.append(replace(base, learning_rate=learning_rate))
        for dropout in DROPOUTS:
            hyperparam_candidates.append(replace(base, dropout=dropout))
    for candidate in _dedupe_candidates(hyperparam_candidates):
        if f"fast::{candidate.key}" in completed_keys:
            continue
        metadata = prepare_data_variant(
            data_spec=candidate.data,
            feature_columns=feature_variants[candidate.data.feature_variant],
            raw_files=raw_files,
            headers=headers,
            labels_by_file=labels_by_file,
            output_root=OUTPUT_ROOT,
            variant_root=VARIANT_ROOT,
            include_heldout_test=False,
            seed=seed,
        )
        row = _run_candidate(
            phase="fast",
            rank=fast_rank,
            candidate=candidate,
            variant_metadata=metadata,
            rounds=fast_rounds,
            max_train_examples_per_client=fast_max_train_examples_per_client,
            include_heldout_test=False,
        )
        search_rows.append(row)
        _write_rows(search_results_path, search_rows)
        completed_keys.add(f"fast::{candidate.key}")
        fast_rank += 1

    fast_rows = [row for row in search_rows if row.get("phase") == "fast"]
    top_fast_rows = sorted(fast_rows, key=_metric_sort_key, reverse=True)[:top_full_candidates]
    full_candidates = [_candidate_from_row(row) for row in top_fast_rows]
    full_rank = 1 + sum(1 for row in search_rows if row.get("phase") == "full")
    for candidate in full_candidates:
        if f"full::{candidate.key}" in completed_keys:
            continue
        metadata = prepare_data_variant(
            data_spec=candidate.data,
            feature_columns=feature_variants[candidate.data.feature_variant],
            raw_files=raw_files,
            headers=headers,
            labels_by_file=labels_by_file,
            output_root=OUTPUT_ROOT,
            variant_root=VARIANT_ROOT,
            include_heldout_test=False,
            seed=seed,
        )
        row = _run_candidate(
            phase="full",
            rank=full_rank,
            candidate=candidate,
            variant_metadata=metadata,
            rounds=full_rounds,
            max_train_examples_per_client=None,
            include_heldout_test=False,
        )
        search_rows.append(row)
        _write_rows(search_results_path, search_rows)
        completed_keys.add(f"full::{candidate.key}")
        full_rank += 1

    full_rows = [row for row in search_rows if row.get("phase") == "full"]
    if not full_rows:
        raise RuntimeError("No full-search candidates completed.")
    best_full = sorted(full_rows, key=_metric_sort_key, reverse=True)[0]
    best_candidate = _candidate_from_row(best_full)
    final_metadata = prepare_data_variant(
        data_spec=best_candidate.data,
        feature_columns=feature_variants[best_candidate.data.feature_variant],
        raw_files=raw_files,
        headers=headers,
        labels_by_file=labels_by_file,
        output_root=OUTPUT_ROOT,
        variant_root=VARIANT_ROOT,
        include_heldout_test=True,
        seed=seed,
    )
    final_row = _run_candidate(
        phase="final",
        rank=1,
        candidate=best_candidate,
        variant_metadata=final_metadata,
        rounds=full_rounds,
        max_train_examples_per_client=None,
        include_heldout_test=True,
    )
    search_rows = [row for row in search_rows if row.get("phase") != "final"] + [final_row]
    _write_rows(search_results_path, search_rows)

    best_config = {
        **{key: best_full[key] for key in best_full if key not in {"test_accuracy", "test_precision", "test_recall", "test_f1", "test_auroc", "test_pr_auc", "test_fpr", "test_confusion_matrix"}},
        "selection_source": "full_search_validation_only",
        "final_evaluation_row": final_row,
        "heldout_test_used_for_selection": False,
    }
    _write_json(OUTPUT_ROOT / "reports" / "best_config.json", best_config)
    _write_final_comparison(search_rows=search_rows, best_config=best_config, final_row=final_row)
    return {
        "search_results": search_results_path,
        "best_config": OUTPUT_ROOT / "reports" / "best_config.json",
        "final_comparison": OUTPUT_ROOT / "reports" / "final_comparison.md",
        "doc": DOC_PATH,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run same-setup optimized proposed Cluster 1 sweep.")
    parser.add_argument("--fast-rounds", type=int, default=10)
    parser.add_argument("--full-rounds", type=int, default=50)
    parser.add_argument("--fast-max-train-examples-per-client", type=int, default=256)
    parser.add_argument("--top-data-variants-for-hyperparams", type=int, default=5)
    parser.add_argument("--top-full-candidates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    paths = run_optimization(
        fast_rounds=args.fast_rounds,
        full_rounds=args.full_rounds,
        fast_max_train_examples_per_client=args.fast_max_train_examples_per_client,
        top_data_variants_for_hyperparams=args.top_data_variants_for_hyperparams,
        top_full_candidates=args.top_full_candidates,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - started
    for name, path in paths.items():
        print(f"{name}: {path}")
    print(f"elapsed_seconds: {elapsed:.3f}")


if __name__ == "__main__":
    main()
