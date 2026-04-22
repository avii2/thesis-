from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .loaders import DEFAULT_CLUSTER_CONFIG_PATHS, load_dataset_for_training
from .schema_validation import (
    DatasetConfig,
    DatasetInspection,
    DatasetSchemaError,
    label_value_to_binary,
    load_cluster_config,
    normalize_header,
    normalize_value,
)
from .transforms import CategoricalTransformArtifacts, NumericTransformArtifacts


REPORT_PATHS = {
    1: {
        "summary": Path("outputs/reports/preprocessing_summary_cluster1.json"),
        "label_summary": Path("outputs/reports/label_summary_cluster1.json"),
        "imputer": Path("outputs/preprocessing/cluster1_hai_imputer.pkl"),
        "scaler": Path("outputs/preprocessing/cluster1_hai_scaler.pkl"),
        "preprocessor": Path("outputs/preprocessing/cluster1_hai_preprocessor.pkl"),
    },
    2: {
        "summary": Path("outputs/reports/preprocessing_summary_cluster2.json"),
        "label_summary": Path("outputs/reports/label_summary_cluster2.json"),
        "imputer": Path("outputs/preprocessing/cluster2_ton_iot_imputer.pkl"),
        "scaler": Path("outputs/preprocessing/cluster2_ton_iot_scaler.pkl"),
        "preprocessor": Path("outputs/preprocessing/cluster2_ton_iot_preprocessor.pkl"),
    },
    3: {
        "summary": Path("outputs/reports/preprocessing_summary_cluster3.json"),
        "label_summary": Path("outputs/reports/label_summary_cluster3.json"),
        "imputer": Path("outputs/preprocessing/cluster3_wustl_imputer.pkl"),
        "scaler": Path("outputs/preprocessing/cluster3_wustl_scaler.pkl"),
        "preprocessor": Path("outputs/preprocessing/cluster3_wustl_preprocessor.pkl"),
    },
}


@dataclass(frozen=True)
class RawPreparedDataset:
    inspection: DatasetInspection
    retained_input_feature_names: tuple[str, ...]
    numeric_input_feature_names: tuple[str, ...]
    categorical_input_feature_names: tuple[str, ...]
    label_column: str
    labels: np.ndarray
    numeric_matrix: np.ndarray
    categorical_matrix: np.ndarray


@dataclass(frozen=True)
class ClusterPreprocessor:
    cluster_id: int
    dataset_name: str
    label_column: str
    retained_input_feature_names: tuple[str, ...]
    numeric_input_feature_names: tuple[str, ...]
    categorical_input_feature_names: tuple[str, ...]
    numeric_artifacts: NumericTransformArtifacts
    categorical_artifacts: CategoricalTransformArtifacts

    @property
    def output_feature_names(self) -> tuple[str, ...]:
        return self.numeric_artifacts.kept_features + self.categorical_artifacts.output_features

    def transform_matrices(
        self,
        numeric_matrix: np.ndarray,
        categorical_matrix: np.ndarray,
    ) -> np.ndarray:
        numeric_output = self.numeric_artifacts.transform(numeric_matrix)
        categorical_output = self.categorical_artifacts.transform(categorical_matrix)
        if numeric_output.size and categorical_output.size:
            return np.hstack([numeric_output, categorical_output])
        if numeric_output.size:
            return numeric_output
        if categorical_output.size:
            return categorical_output
        return np.empty((numeric_matrix.shape[0], 0), dtype=np.float32)

    def transform_records(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        numeric_matrix, categorical_matrix = _records_to_feature_matrices(
            records,
            self.numeric_input_feature_names,
            self.categorical_input_feature_names,
        )
        return self.transform_matrices(numeric_matrix, categorical_matrix)


@dataclass(frozen=True)
class PreprocessingResult:
    preprocessor: ClusterPreprocessor
    transformed_training_matrix: np.ndarray
    labels: np.ndarray
    summary: Mapping[str, Any]


def _artifact_paths_for_config(config: DatasetConfig) -> Mapping[str, Path]:
    if config.cluster_id in REPORT_PATHS:
        return REPORT_PATHS[config.cluster_id]

    cluster_stub = f"cluster{config.cluster_id}_{config.dataset_key.lower()}"
    return {
        "summary": Path(f"outputs/reports/preprocessing_summary_{cluster_stub}.json"),
        "label_summary": Path(f"outputs/reports/label_summary_{cluster_stub}.json"),
        "imputer": Path(f"outputs/preprocessing/{cluster_stub}_imputer.pkl"),
        "scaler": Path(f"outputs/preprocessing/{cluster_stub}_scaler.pkl"),
        "preprocessor": Path(f"outputs/preprocessing/{cluster_stub}_preprocessor.pkl"),
    }


def _retained_feature_names(inspection: DatasetInspection) -> tuple[str, ...]:
    return inspection.file_inspections[0].retained_feature_columns


def _numeric_and_categorical_feature_names(
    inspection: DatasetInspection,
    retained_feature_names: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    first_file = inspection.file_inspections[0]
    numeric_lookup = set(first_file.numeric_columns)
    categorical_lookup = set(first_file.categorical_columns)
    numeric_features = tuple(feature for feature in retained_feature_names if feature in numeric_lookup)
    categorical_features = tuple(feature for feature in retained_feature_names if feature in categorical_lookup)
    return numeric_features, categorical_features


def _feature_indices(normalized_columns: Sequence[str], selected_columns: Sequence[str]) -> list[int]:
    return [normalized_columns.index(column) for column in selected_columns]


def _load_numeric_training_matrix(
    inspection: DatasetInspection,
    numeric_feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    if not numeric_feature_names:
        return np.empty((sum(file.row_count for file in inspection.file_inspections), 0), dtype=np.float32), np.empty(0, dtype=np.int8)

    first_file = inspection.file_inspections[0]
    feature_indices = _feature_indices(first_file.normalized_columns, numeric_feature_names)
    label_index = first_file.normalized_columns.index(inspection.config.label_column)

    feature_arrays: list[np.ndarray] = []
    label_arrays: list[np.ndarray] = []

    for path in inspection.file_paths:
        feature_array = np.genfromtxt(
            path,
            delimiter=",",
            skip_header=1,
            usecols=feature_indices,
            dtype=np.float32,
            encoding="utf-8",
            autostrip=True,
            missing_values=["", "na", "n/a", "nan", "null", "none"],
            filling_values=np.nan,
            invalid_raise=True,
        )
        if feature_array.size == 0:
            feature_array = np.empty((0, len(feature_indices)), dtype=np.float32)
        elif feature_array.ndim == 1:
            feature_array = feature_array.reshape(-1, len(feature_indices))

        raw_labels = np.genfromtxt(
            path,
            delimiter=",",
            skip_header=1,
            usecols=(label_index,),
            dtype=str,
            encoding="utf-8",
            autostrip=True,
        )
        if raw_labels.size == 0:
            mapped_labels = np.empty(0, dtype=np.int8)
        else:
            raw_labels = np.atleast_1d(raw_labels)
            mapped = []
            for raw_label in raw_labels:
                mapped_label = label_value_to_binary(str(raw_label))
                if mapped_label is None:
                    raise DatasetSchemaError(
                        f"{inspection.config.dataset_name}: unexpected label value {raw_label!r} while building preprocessing matrices."
                    )
                mapped.append(int(mapped_label))
            mapped_labels = np.asarray(mapped, dtype=np.int8)

        feature_arrays.append(feature_array)
        label_arrays.append(mapped_labels)

    return np.vstack(feature_arrays), np.concatenate(label_arrays)


def _load_object_training_matrices(
    inspection: DatasetInspection,
    numeric_feature_names: Sequence[str],
    categorical_feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_file = inspection.file_inspections[0]
    numeric_indices = _feature_indices(first_file.normalized_columns, numeric_feature_names)
    categorical_indices = _feature_indices(first_file.normalized_columns, categorical_feature_names)
    label_index = first_file.normalized_columns.index(inspection.config.label_column)

    numeric_rows: list[list[float]] = []
    categorical_rows: list[list[str]] = []
    labels: list[int] = []

    for path in inspection.file_paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            raw_header = next(reader)
            normalized_columns = [normalize_header(column) for column in raw_header]
            if inspection.config.label_column not in normalized_columns:
                raise DatasetSchemaError(
                    f"{inspection.config.dataset_name}: label column {inspection.config.label_column!r} disappeared while reading {path}."
                )

            for row in reader:
                label_value = label_value_to_binary(row[label_index])
                if label_value is None:
                    raise DatasetSchemaError(
                        f"{inspection.config.dataset_name}: unexpected label value {row[label_index]!r} while building preprocessing matrices."
                    )
                labels.append(int(label_value))

                numeric_row: list[float] = []
                for index in numeric_indices:
                    value = normalize_value(row[index])
                    numeric_row.append(np.nan if value.lower() in {"", "na", "n/a", "nan", "null", "none"} else float(value))
                numeric_rows.append(numeric_row)

                categorical_row: list[str] = []
                for index in categorical_indices:
                    categorical_row.append(row[index])
                categorical_rows.append(categorical_row)

    numeric_matrix = np.asarray(numeric_rows, dtype=np.float32) if numeric_feature_names else np.empty((len(labels), 0), dtype=np.float32)
    categorical_matrix = np.asarray(categorical_rows, dtype=object) if categorical_feature_names else np.empty((len(labels), 0), dtype=object)
    return numeric_matrix, categorical_matrix, np.asarray(labels, dtype=np.int8)


def _records_to_feature_matrices(
    records: Sequence[Mapping[str, Any]],
    numeric_feature_names: Sequence[str],
    categorical_feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    numeric_rows: list[list[float]] = []
    categorical_rows: list[list[str]] = []

    for record in records:
        numeric_row: list[float] = []
        for feature_name in numeric_feature_names:
            raw_value = normalize_value("" if record.get(feature_name) is None else str(record.get(feature_name)))
            numeric_row.append(np.nan if raw_value.lower() in {"", "na", "n/a", "nan", "null", "none"} else float(raw_value))
        numeric_rows.append(numeric_row)

        categorical_row: list[str] = []
        for feature_name in categorical_feature_names:
            categorical_row.append("" if record.get(feature_name) is None else str(record.get(feature_name)))
        categorical_rows.append(categorical_row)

    numeric_matrix = np.asarray(numeric_rows, dtype=np.float32) if numeric_feature_names else np.empty((len(records), 0), dtype=np.float32)
    categorical_matrix = np.asarray(categorical_rows, dtype=object) if categorical_feature_names else np.empty((len(records), 0), dtype=object)
    return numeric_matrix, categorical_matrix


def prepare_training_dataset(config_path: str | Path) -> RawPreparedDataset:
    inspection = load_dataset_for_training(config_path)
    retained_feature_names = _retained_feature_names(inspection)
    numeric_feature_names, categorical_feature_names = _numeric_and_categorical_feature_names(
        inspection,
        retained_feature_names,
    )

    if categorical_feature_names:
        numeric_matrix, categorical_matrix, labels = _load_object_training_matrices(
            inspection,
            numeric_feature_names,
            categorical_feature_names,
        )
    else:
        numeric_matrix, labels = _load_numeric_training_matrix(inspection, numeric_feature_names)
        categorical_matrix = np.empty((numeric_matrix.shape[0], 0), dtype=object)

    return RawPreparedDataset(
        inspection=inspection,
        retained_input_feature_names=retained_feature_names,
        numeric_input_feature_names=numeric_feature_names,
        categorical_input_feature_names=categorical_feature_names,
        label_column=inspection.config.label_column,
        labels=labels,
        numeric_matrix=numeric_matrix,
        categorical_matrix=categorical_matrix,
    )


def _empty_categorical_artifacts(feature_names: Sequence[str]) -> CategoricalTransformArtifacts:
    return CategoricalTransformArtifacts(
        input_features=tuple(feature_names),
        kept_features=(),
        kept_indices=(),
        dropped_constant_features=(),
        dropped_high_cardinality_features=(),
        categories_by_feature={},
        output_features=(),
    )


def fit_preprocessor_for_training(config_path: str | Path) -> PreprocessingResult:
    prepared = prepare_training_dataset(config_path)
    numeric_artifacts = NumericTransformArtifacts.fit(
        prepared.numeric_input_feature_names,
        prepared.numeric_matrix,
    )
    categorical_artifacts = (
        CategoricalTransformArtifacts.fit(
            prepared.categorical_input_feature_names,
            prepared.categorical_matrix,
        )
        if prepared.categorical_input_feature_names
        else _empty_categorical_artifacts(prepared.categorical_input_feature_names)
    )

    preprocessor = ClusterPreprocessor(
        cluster_id=prepared.inspection.config.cluster_id,
        dataset_name=prepared.inspection.config.dataset_name,
        label_column=prepared.label_column,
        retained_input_feature_names=prepared.retained_input_feature_names,
        numeric_input_feature_names=prepared.numeric_input_feature_names,
        categorical_input_feature_names=prepared.categorical_input_feature_names,
        numeric_artifacts=numeric_artifacts,
        categorical_artifacts=categorical_artifacts,
    )

    transformed_training_matrix = preprocessor.transform_matrices(
        prepared.numeric_matrix,
        prepared.categorical_matrix,
    )

    if transformed_training_matrix.shape[1] == 0:
        raise DatasetSchemaError(
            f"{prepared.inspection.config.dataset_name}: no usable model features remain after preprocessing."
        )

    artifact_paths = _artifact_paths_for_config(prepared.inspection.config)
    summary = {
        "cluster_id": preprocessor.cluster_id,
        "dataset": preprocessor.dataset_name,
        "status": "ok",
        "source_kind": prepared.inspection.source_kind,
        "source_paths": [str(path) for path in prepared.inspection.file_paths],
        "fit_scope": "full_loaded_training_input_pre_split",
        "label_column": preprocessor.label_column,
        "label_counts": dict(sorted(Counter(prepared.labels.astype(str)).items())),
        "retained_input_feature_columns": list(preprocessor.retained_input_feature_names),
        "numeric_input_feature_columns": list(preprocessor.numeric_input_feature_names),
        "categorical_input_feature_columns": list(preprocessor.categorical_input_feature_names),
        "dropped_all_missing_feature_columns": list(preprocessor.numeric_artifacts.dropped_all_missing_features),
        "dropped_constant_feature_columns": list(
            preprocessor.numeric_artifacts.dropped_constant_features
            + preprocessor.categorical_artifacts.dropped_constant_features
        ),
        "dropped_high_cardinality_categorical_columns": list(
            preprocessor.categorical_artifacts.dropped_high_cardinality_features
        ),
        "output_feature_columns": list(preprocessor.output_feature_names),
        "output_feature_count": len(preprocessor.output_feature_names),
        "row_count": int(prepared.labels.shape[0]),
        "transformed_shape": list(transformed_training_matrix.shape),
        "windowing_deferred": preprocessor.cluster_id == 1,
        "descriptor_computation_deferred": True,
        "client_partitioning_deferred": True,
        "artifact_paths": {key: str(path) for key, path in artifact_paths.items() if key in {"imputer", "scaler", "preprocessor"}},
    }

    return PreprocessingResult(
        preprocessor=preprocessor,
        transformed_training_matrix=transformed_training_matrix,
        labels=prepared.labels,
        summary=summary,
    )


def _save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_preprocessing(config_path: str | Path, *, continue_on_blocked: bool = False) -> Mapping[str, Any]:
    config = load_cluster_config(config_path)
    artifact_paths = _artifact_paths_for_config(config)

    try:
        result = fit_preprocessor_for_training(config_path)
    except DatasetSchemaError as exc:
        if not continue_on_blocked:
            raise

        blocked_summary = {
            "cluster_id": config.cluster_id,
            "dataset": config.dataset_name,
            "status": "blocked",
            "label_column": config.label_column,
            "error": str(exc),
            "expected_processed_input_path": config.expected_processed_input_path,
            "windowing_deferred": config.cluster_id == 1,
            "descriptor_computation_deferred": True,
            "client_partitioning_deferred": True,
        }
        _save_json(artifact_paths["summary"], blocked_summary)
        return blocked_summary

    _save_pickle(
        artifact_paths["imputer"],
        {
            "input_features": result.preprocessor.numeric_artifacts.input_features,
            "kept_features": result.preprocessor.numeric_artifacts.kept_features,
            "medians": result.preprocessor.numeric_artifacts.medians.tolist(),
        },
    )
    _save_pickle(
        artifact_paths["scaler"],
        {
            "input_features": result.preprocessor.numeric_artifacts.input_features,
            "kept_features": result.preprocessor.numeric_artifacts.kept_features,
            "means": result.preprocessor.numeric_artifacts.means.tolist(),
            "scales": result.preprocessor.numeric_artifacts.scales.tolist(),
        },
    )
    _save_pickle(artifact_paths["preprocessor"], result.preprocessor)
    _save_json(artifact_paths["summary"], result.summary)
    _save_json(
        artifact_paths["label_summary"],
        {
            "cluster_id": result.preprocessor.cluster_id,
            "dataset": result.preprocessor.dataset_name,
            "label_column": result.preprocessor.label_column,
            "label_counts": result.summary["label_counts"],
        },
    )
    return result.summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and save preprocessing artifacts for cluster configs.")
    parser.add_argument(
        "--config",
        dest="config_paths",
        action="append",
        help="Optional cluster config path. If omitted, all three default cluster configs are processed.",
    )
    parser.add_argument(
        "--continue-on-blocked",
        action="store_true",
        help="Write blocked summaries instead of failing fast when a configured training input is intentionally unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = args.config_paths or [str(path) for path in DEFAULT_CLUSTER_CONFIG_PATHS.values()]
    for config_path in config_paths:
        summary = run_preprocessing(config_path, continue_on_blocked=args.continue_on_blocked)
        print(f"{summary['dataset']}: {summary['status']}")


if __name__ == "__main__":
    main()
