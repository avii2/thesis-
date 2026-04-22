from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .schema_validation import is_missing, normalize_value


UNKNOWN_CATEGORY = "UNKNOWN"
MAX_CATEGORICAL_CARDINALITY = 20


@dataclass(frozen=True)
class NumericTransformArtifacts:
    input_features: tuple[str, ...]
    kept_features: tuple[str, ...]
    kept_indices: tuple[int, ...]
    dropped_all_missing_features: tuple[str, ...]
    dropped_constant_features: tuple[str, ...]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(
        cls,
        feature_names: Sequence[str],
        matrix: np.ndarray,
    ) -> "NumericTransformArtifacts":
        feature_names = tuple(feature_names)
        if matrix.ndim != 2:
            raise ValueError("Numeric matrix must be 2-dimensional.")
        if matrix.shape[1] != len(feature_names):
            raise ValueError("Numeric matrix width must match feature name count.")

        kept_features: list[str] = []
        kept_indices: list[int] = []
        dropped_all_missing: list[str] = []
        dropped_constant: list[str] = []
        medians: list[float] = []
        means: list[float] = []
        scales: list[float] = []

        for index, feature_name in enumerate(feature_names):
            column = matrix[:, index].astype(np.float64, copy=False)
            if np.isnan(column).all():
                dropped_all_missing.append(feature_name)
                continue

            median = float(np.nanmedian(column))
            imputed = np.where(np.isnan(column), median, column)
            if np.allclose(imputed, imputed[0]):
                dropped_constant.append(feature_name)
                continue

            mean = float(imputed.mean())
            scale = float(imputed.std())
            if scale == 0.0:
                dropped_constant.append(feature_name)
                continue

            kept_features.append(feature_name)
            kept_indices.append(index)
            medians.append(median)
            means.append(mean)
            scales.append(scale)

        return cls(
            input_features=feature_names,
            kept_features=tuple(kept_features),
            kept_indices=tuple(kept_indices),
            dropped_all_missing_features=tuple(dropped_all_missing),
            dropped_constant_features=tuple(dropped_constant),
            medians=np.asarray(medians, dtype=np.float64),
            means=np.asarray(means, dtype=np.float64),
            scales=np.asarray(scales, dtype=np.float64),
        )

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if not self.kept_indices:
            return np.empty((matrix.shape[0], 0), dtype=np.float32)

        selected = matrix[:, self.kept_indices].astype(np.float64, copy=True)
        missing_mask = np.isnan(selected)
        if missing_mask.any():
            selected[missing_mask] = np.take(self.medians, np.where(missing_mask)[1])
        transformed = (selected - self.means) / self.scales
        return transformed.astype(np.float32, copy=False)


@dataclass(frozen=True)
class CategoricalTransformArtifacts:
    input_features: tuple[str, ...]
    kept_features: tuple[str, ...]
    kept_indices: tuple[int, ...]
    dropped_constant_features: tuple[str, ...]
    dropped_high_cardinality_features: tuple[str, ...]
    categories_by_feature: Mapping[str, tuple[str, ...]]
    output_features: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        feature_names: Sequence[str],
        matrix: np.ndarray,
        *,
        max_categories: int = MAX_CATEGORICAL_CARDINALITY,
    ) -> "CategoricalTransformArtifacts":
        feature_names = tuple(feature_names)
        if matrix.ndim != 2:
            raise ValueError("Categorical matrix must be 2-dimensional.")
        if matrix.shape[1] != len(feature_names):
            raise ValueError("Categorical matrix width must match feature name count.")

        kept_features: list[str] = []
        kept_indices: list[int] = []
        dropped_constant: list[str] = []
        dropped_high_cardinality: list[str] = []
        categories_by_feature: dict[str, tuple[str, ...]] = {}
        output_features: list[str] = []

        for index, feature_name in enumerate(feature_names):
            normalized_values = tuple(_normalize_categorical_value(value) for value in matrix[:, index])
            categories = tuple(sorted(set(normalized_values)))

            if len(categories) <= 1:
                dropped_constant.append(feature_name)
                continue

            if len(categories) > max_categories:
                dropped_high_cardinality.append(feature_name)
                continue

            kept_features.append(feature_name)
            kept_indices.append(index)
            categories_by_feature[feature_name] = categories
            output_features.extend(f"{feature_name}={category}" for category in categories)

        return cls(
            input_features=feature_names,
            kept_features=tuple(kept_features),
            kept_indices=tuple(kept_indices),
            dropped_constant_features=tuple(dropped_constant),
            dropped_high_cardinality_features=tuple(dropped_high_cardinality),
            categories_by_feature=categories_by_feature,
            output_features=tuple(output_features),
        )

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if not self.kept_indices:
            return np.empty((matrix.shape[0], 0), dtype=np.float32)

        transformed_blocks: list[np.ndarray] = []
        for feature_name, index in zip(self.kept_features, self.kept_indices, strict=True):
            categories = self.categories_by_feature[feature_name]
            block = np.zeros((matrix.shape[0], len(categories)), dtype=np.float32)
            normalized_values = [_normalize_categorical_value(value) for value in matrix[:, index]]
            for row_index, value in enumerate(normalized_values):
                if value in categories:
                    category_index = categories.index(value)
                    block[row_index, category_index] = 1.0
            transformed_blocks.append(block)
        return np.hstack(transformed_blocks) if transformed_blocks else np.empty((matrix.shape[0], 0), dtype=np.float32)


def _normalize_categorical_value(value: object) -> str:
    text = normalize_value("" if value is None else str(value))
    return UNKNOWN_CATEGORY if is_missing(text) else text
