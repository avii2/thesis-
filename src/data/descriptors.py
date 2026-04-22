from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

from .partitions import PartitionBuildResult, build_candidate_leaf_clients
from .preprocess import RawPreparedDataset, prepare_training_dataset
from .transforms import CategoricalTransformArtifacts, NumericTransformArtifacts


DESCRIPTOR_SCALER_PATHS = {
    1: Path("outputs/clustering/cluster1_descriptor_scaler.pkl"),
    2: Path("outputs/clustering/cluster2_descriptor_scaler.pkl"),
    3: Path("outputs/clustering/cluster3_descriptor_scaler.pkl"),
}


@dataclass(frozen=True)
class ClusterDescriptorResult:
    cluster_id: int
    dataset: str
    client_ids: tuple[str, ...]
    raw_descriptor_matrix: np.ndarray
    standardized_descriptor_matrix: np.ndarray
    descriptor_scaler: StandardScaler
    descriptor_scaler_path: Path
    output_feature_names: tuple[str, ...]
    descriptor_dim: int
    client_metadata_path: Path
    summary: Mapping[str, Any]


def _default_scaler_path(cluster_id: int) -> Path:
    if cluster_id in DESCRIPTOR_SCALER_PATHS:
        return DESCRIPTOR_SCALER_PATHS[cluster_id]
    return Path(f"outputs/clustering/cluster{cluster_id}_descriptor_scaler.pkl")


def _fit_feature_artifacts(
    prepared: RawPreparedDataset,
    partition_result: PartitionBuildResult,
) -> tuple[NumericTransformArtifacts, CategoricalTransformArtifacts]:
    ordered_numeric = prepared.numeric_matrix[partition_result.ordered_row_indices]
    ordered_categorical = prepared.categorical_matrix[partition_result.ordered_row_indices]

    numeric_train_chunks: list[np.ndarray] = []
    categorical_train_chunks: list[np.ndarray] = []
    for client in partition_result.metadata.clients:
        train_range = client.train
        numeric_train_chunks.append(ordered_numeric[train_range.start_index:train_range.end_index])
        categorical_train_chunks.append(ordered_categorical[train_range.start_index:train_range.end_index])

    if prepared.numeric_input_feature_names:
        cluster_train_numeric = np.vstack(numeric_train_chunks)
        numeric_artifacts = NumericTransformArtifacts.fit(
            prepared.numeric_input_feature_names,
            cluster_train_numeric,
        )
    else:
        numeric_artifacts = NumericTransformArtifacts.fit(
            prepared.numeric_input_feature_names,
            np.empty((partition_result.metadata.total_ordered_samples, 0), dtype=np.float32),
        )

    if prepared.categorical_input_feature_names:
        cluster_train_categorical = np.vstack(categorical_train_chunks)
        categorical_artifacts = CategoricalTransformArtifacts.fit(
            prepared.categorical_input_feature_names,
            cluster_train_categorical,
        )
    else:
        categorical_artifacts = CategoricalTransformArtifacts.fit(
            prepared.categorical_input_feature_names,
            np.empty((partition_result.metadata.total_ordered_samples, 0), dtype=object),
        )

    return numeric_artifacts, categorical_artifacts


def _transform_training_slice(
    numeric_artifacts: NumericTransformArtifacts,
    categorical_artifacts: CategoricalTransformArtifacts,
    numeric_slice: np.ndarray,
    categorical_slice: np.ndarray,
) -> np.ndarray:
    numeric_output = numeric_artifacts.transform(numeric_slice)
    categorical_output = categorical_artifacts.transform(categorical_slice)
    if numeric_output.size and categorical_output.size:
        return np.hstack([numeric_output, categorical_output])
    if numeric_output.size:
        return numeric_output
    if categorical_output.size:
        return categorical_output
    return np.empty((numeric_slice.shape[0], 0), dtype=np.float32)


def build_cluster_descriptors(
    config_path: str | Path,
    *,
    client_metadata_path: str | Path | None = None,
    descriptor_scaler_path: str | Path | None = None,
) -> ClusterDescriptorResult:
    partition_result = build_candidate_leaf_clients(config_path, output_path=client_metadata_path)
    prepared = prepare_training_dataset(config_path)
    numeric_artifacts, categorical_artifacts = _fit_feature_artifacts(prepared, partition_result)

    output_feature_names = numeric_artifacts.kept_features + categorical_artifacts.output_features
    if not output_feature_names:
        raise ValueError(f"{prepared.inspection.config.dataset_name}: no usable descriptor features remain.")

    ordered_numeric = prepared.numeric_matrix[partition_result.ordered_row_indices]
    ordered_categorical = prepared.categorical_matrix[partition_result.ordered_row_indices]
    descriptor_rows: list[np.ndarray] = []
    client_ids: list[str] = []
    client_train_sizes: dict[str, int] = {}

    for client in partition_result.metadata.clients:
        train_range = client.train
        numeric_slice = ordered_numeric[train_range.start_index:train_range.end_index]
        categorical_slice = ordered_categorical[train_range.start_index:train_range.end_index]
        transformed_training = _transform_training_slice(
            numeric_artifacts,
            categorical_artifacts,
            numeric_slice,
            categorical_slice,
        )
        if transformed_training.shape[0] == 0:
            raise ValueError(
                f"{prepared.inspection.config.dataset_name}: client {client.client_id} has an empty transformed training matrix."
            )

        descriptor = np.concatenate(
            [
                transformed_training.mean(axis=0, dtype=np.float64),
                transformed_training.std(axis=0, dtype=np.float64),
            ]
        ).astype(np.float32, copy=False)
        descriptor_rows.append(descriptor)
        client_ids.append(client.client_id)
        client_train_sizes[client.client_id] = client.train.num_samples

    raw_descriptor_matrix = np.vstack(descriptor_rows)
    descriptor_scaler = StandardScaler()
    standardized_descriptor_matrix = descriptor_scaler.fit_transform(raw_descriptor_matrix).astype(np.float32, copy=False)

    scaler_path = Path(descriptor_scaler_path) if descriptor_scaler_path is not None else _default_scaler_path(prepared.inspection.config.cluster_id)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with scaler_path.open("wb") as handle:
        pickle.dump(descriptor_scaler, handle)

    summary = {
        "cluster_id": prepared.inspection.config.cluster_id,
        "dataset": prepared.inspection.config.dataset_name,
        "descriptor_source_split": "train",
        "descriptor_type": "feature_mean_std",
        "client_ids": client_ids,
        "num_clients": len(client_ids),
        "descriptor_dim": int(raw_descriptor_matrix.shape[1]),
        "output_feature_names": list(output_feature_names),
        "client_train_sizes": client_train_sizes,
        "dropped_all_missing_feature_columns": list(numeric_artifacts.dropped_all_missing_features),
        "dropped_constant_feature_columns": list(
            numeric_artifacts.dropped_constant_features + categorical_artifacts.dropped_constant_features
        ),
        "dropped_high_cardinality_categorical_columns": list(categorical_artifacts.dropped_high_cardinality_features),
        "client_metadata_path": str(partition_result.output_path),
        "descriptor_scaler_path": str(scaler_path),
    }

    return ClusterDescriptorResult(
        cluster_id=prepared.inspection.config.cluster_id,
        dataset=prepared.inspection.config.dataset_name,
        client_ids=tuple(client_ids),
        raw_descriptor_matrix=raw_descriptor_matrix,
        standardized_descriptor_matrix=standardized_descriptor_matrix,
        descriptor_scaler=descriptor_scaler,
        descriptor_scaler_path=scaler_path,
        output_feature_names=output_feature_names,
        descriptor_dim=int(raw_descriptor_matrix.shape[1]),
        client_metadata_path=partition_result.output_path,
        summary=summary,
    )
