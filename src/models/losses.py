from __future__ import annotations

import numpy as np


def sigmoid_from_logits(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))
    return np.clip(probabilities, 1e-6, 1.0 - 1e-6).astype(np.float32, copy=False)


def softplus(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (np.maximum(values, 0.0) + np.log1p(np.exp(-np.abs(values)))).astype(
        np.float32,
        copy=False,
    )


def binary_cross_entropy_with_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    positive_class_weight: float = 1.0,
) -> float:
    label_values = np.asarray(labels, dtype=np.float32).reshape(-1)
    logit_values = np.asarray(logits, dtype=np.float32).reshape(-1)
    if label_values.shape != logit_values.shape:
        raise ValueError(
            "BCEWithLogits labels/logits shape mismatch. "
            f"Observed labels={label_values.shape}, logits={logit_values.shape}."
        )
    if positive_class_weight <= 0.0:
        raise ValueError(f"positive_class_weight must be positive. Observed {positive_class_weight}.")

    positive_loss = positive_class_weight * label_values * softplus(-logit_values)
    negative_loss = (1.0 - label_values) * softplus(logit_values)
    return float(np.mean(positive_loss + negative_loss))


def binary_cross_entropy_with_logits_gradient(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    positive_class_weight: float = 1.0,
) -> np.ndarray:
    label_values = np.asarray(labels, dtype=np.float32).reshape(-1)
    logit_values = np.asarray(logits, dtype=np.float32).reshape(-1)
    if label_values.shape != logit_values.shape:
        raise ValueError(
            "BCEWithLogits gradient labels/logits shape mismatch. "
            f"Observed labels={label_values.shape}, logits={logit_values.shape}."
        )
    if positive_class_weight <= 0.0:
        raise ValueError(f"positive_class_weight must be positive. Observed {positive_class_weight}.")

    probabilities = sigmoid_from_logits(logit_values)
    batch_size = max(1, label_values.shape[0])
    gradient = (
        probabilities * (1.0 - label_values)
        - positive_class_weight * label_values * (1.0 - probabilities)
    ) / batch_size
    return gradient.astype(np.float32, copy=False)


def binary_cross_entropy_with_logits_loss_and_gradient(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    positive_class_weight: float = 1.0,
) -> tuple[float, np.ndarray]:
    return (
        binary_cross_entropy_with_logits(
            logits,
            labels,
            positive_class_weight=positive_class_weight,
        ),
        binary_cross_entropy_with_logits_gradient(
            logits,
            labels,
            positive_class_weight=positive_class_weight,
        ),
    )
