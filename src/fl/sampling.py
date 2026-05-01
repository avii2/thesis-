from __future__ import annotations

import numpy as np


def positive_window_oversample_indices(
    labels: np.ndarray,
    *,
    rng: np.random.Generator,
    target_positive_fraction: float = 0.5,
) -> np.ndarray:
    """Return training-only indices with positive windows oversampled.

    All original negative indices are retained. Positive indices are sampled
    with replacement until the requested positive fraction is reached.
    """
    label_values = np.asarray(labels, dtype=np.int8).reshape(-1)
    if label_values.size == 0:
        return np.empty(0, dtype=np.int64)
    if not 0.0 < target_positive_fraction < 1.0:
        raise ValueError("target_positive_fraction must be in (0, 1).")

    positive_indices = np.flatnonzero(label_values == 1)
    negative_indices = np.flatnonzero(label_values == 0)
    if positive_indices.size == 0 or negative_indices.size == 0:
        return rng.permutation(label_values.size).astype(np.int64, copy=False)

    desired_positive_count = int(
        np.ceil((target_positive_fraction / (1.0 - target_positive_fraction)) * negative_indices.size)
    )
    desired_positive_count = max(desired_positive_count, positive_indices.size)
    sampled_positive = rng.choice(
        positive_indices,
        size=desired_positive_count,
        replace=desired_positive_count > positive_indices.size,
    )
    combined = np.concatenate([negative_indices, sampled_positive]).astype(np.int64, copy=False)
    return rng.permutation(combined)

