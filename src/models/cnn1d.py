from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


STATE_KEYS = ("conv_weight", "conv_bias", "dense_weight", "dense_bias")


@dataclass(frozen=True)
class CNN1DConfig:
    input_channels: int
    input_length: int
    num_filters: int = 8
    kernel_size: int = 3
    weight_scale: float = 0.05

    def validate(self) -> None:
        if self.input_channels <= 0:
            raise ValueError("CNN1DConfig.input_channels must be positive.")
        if self.input_length <= 0:
            raise ValueError("CNN1DConfig.input_length must be positive.")
        if self.num_filters <= 0:
            raise ValueError("CNN1DConfig.num_filters must be positive.")
        if self.kernel_size <= 0:
            raise ValueError("CNN1DConfig.kernel_size must be positive.")

    @property
    def resolved_kernel_size(self) -> int:
        self.validate()
        return min(self.kernel_size, self.input_length)

    @property
    def parameter_shapes(self) -> dict[str, tuple[int, ...]]:
        kernel_size = self.resolved_kernel_size
        return {
            "conv_weight": (self.num_filters, self.input_channels, kernel_size),
            "conv_bias": (self.num_filters,),
            "dense_weight": (self.num_filters,),
            "dense_bias": (),
        }


class CNN1DClassifier:
    def __init__(self, config: CNN1DConfig, *, seed: int = 42) -> None:
        self.config = config
        self.config.validate()
        self._rng = np.random.default_rng(seed)
        self._state = self._initialize_state()

    def _initialize_state(self) -> dict[str, np.ndarray]:
        shapes = self.config.parameter_shapes
        return {
            "conv_weight": (
                self._rng.normal(
                    loc=0.0,
                    scale=self.config.weight_scale,
                    size=shapes["conv_weight"],
                ).astype(np.float32)
            ),
            "conv_bias": np.zeros(shapes["conv_bias"], dtype=np.float32),
            "dense_weight": (
                self._rng.normal(
                    loc=0.0,
                    scale=self.config.weight_scale,
                    size=shapes["dense_weight"],
                ).astype(np.float32)
            ),
            "dense_bias": np.asarray(0.0, dtype=np.float32),
        }

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            key: value.copy() if isinstance(value, np.ndarray) else np.asarray(value, dtype=np.float32)
            for key, value in self._state.items()
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        missing = [key for key in STATE_KEYS if key not in state]
        extra = [key for key in state.keys() if key not in STATE_KEYS]
        if missing or extra:
            raise ValueError(
                f"CNN1DClassifier state keys mismatch. Missing={missing or '[]'} Extra={extra or '[]'}."
            )

        shapes = self.config.parameter_shapes
        loaded_state: dict[str, np.ndarray] = {}
        for key in STATE_KEYS:
            value = np.asarray(state[key], dtype=np.float32)
            if value.shape != shapes[key]:
                raise ValueError(
                    f"CNN1DClassifier state {key!r} has shape {value.shape}, expected {shapes[key]}."
                )
            loaded_state[key] = value.copy()
        self._state = loaded_state

    @classmethod
    def from_state(
        cls,
        config: CNN1DConfig,
        state: Mapping[str, Any],
        *,
        seed: int = 42,
    ) -> "CNN1DClassifier":
        model = cls(config, seed=seed)
        model.load_state_dict(state)
        return model

    def parameter_bytes(self) -> int:
        return int(sum(np.asarray(value).nbytes for value in self._state.values()))

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        cache = self._forward(inputs.astype(np.float32, copy=False))
        return cache["probs"]

    def predict(self, inputs: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(inputs) >= threshold).astype(np.int8, copy=False)

    def binary_cross_entropy(self, inputs: np.ndarray, labels: np.ndarray) -> float:
        labels = labels.astype(np.float32, copy=False)
        probs = self.predict_proba(inputs)
        return float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))

    def train_epoch(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        *,
        batch_size: int,
        learning_rate: float,
        rng: np.random.Generator,
    ) -> float:
        if inputs.shape[0] == 0:
            raise ValueError("CNN1DClassifier cannot train on an empty batch.")

        indices = rng.permutation(inputs.shape[0])
        batch_losses: list[float] = []
        for start in range(0, inputs.shape[0], batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_inputs = inputs[batch_indices]
            batch_labels = labels[batch_indices].astype(np.float32, copy=False)
            loss, gradients = self._loss_and_gradients(batch_inputs, batch_labels)
            batch_losses.append(loss)
            for key in STATE_KEYS:
                self._state[key] = self._state[key] - gradients[key] * learning_rate
        return float(np.mean(batch_losses))

    def _forward(self, inputs: np.ndarray) -> dict[str, np.ndarray]:
        if inputs.ndim != 3:
            raise ValueError(
                f"CNN1DClassifier expects inputs with shape (batch, channels, length). Observed {inputs.shape}."
            )
        if inputs.shape[1] != self.config.input_channels or inputs.shape[2] != self.config.input_length:
            raise ValueError(
                "CNN1DClassifier input shape mismatch. "
                f"Observed {inputs.shape[1:]} expected {(self.config.input_channels, self.config.input_length)}."
            )

        windows = np.lib.stride_tricks.sliding_window_view(
            inputs,
            window_shape=self.config.resolved_kernel_size,
            axis=2,
        )
        conv_linear = np.einsum(
            "bcwk,ock->bow",
            windows,
            self._state["conv_weight"],
            optimize=True,
        )
        conv_linear = conv_linear + self._state["conv_bias"][None, :, None]
        conv_activated = np.maximum(conv_linear, 0.0)
        pooled = conv_activated.mean(axis=2)
        logits = pooled @ self._state["dense_weight"] + float(self._state["dense_bias"])
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6).astype(np.float32, copy=False)
        return {
            "windows": windows,
            "conv_linear": conv_linear,
            "conv_activated": conv_activated,
            "pooled": pooled,
            "logits": logits,
            "probs": probs,
        }

    def _loss_and_gradients(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[float, dict[str, np.ndarray]]:
        cache = self._forward(inputs)
        probs = cache["probs"]
        batch_size = max(1, inputs.shape[0])
        loss = float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))

        dlogits = ((probs - labels) / batch_size).astype(np.float32, copy=False)
        grad_dense_weight = cache["pooled"].T @ dlogits
        grad_dense_bias = np.asarray(dlogits.sum(), dtype=np.float32)

        dpooled = dlogits[:, None] * self._state["dense_weight"][None, :]
        out_length = cache["conv_activated"].shape[2]
        dactivated = np.broadcast_to(
            (dpooled[:, :, None] / out_length).astype(np.float32, copy=False),
            cache["conv_activated"].shape,
        ).copy()
        dconv = dactivated * (cache["conv_linear"] > 0.0)

        grad_conv_bias = dconv.sum(axis=(0, 2)).astype(np.float32, copy=False)
        grad_conv_weight = np.einsum(
            "bcwk,bow->ock",
            cache["windows"],
            dconv,
            optimize=True,
        ).astype(np.float32, copy=False)

        gradients = {
            "conv_weight": grad_conv_weight,
            "conv_bias": grad_conv_bias,
            "dense_weight": grad_dense_weight.astype(np.float32, copy=False),
            "dense_bias": grad_dense_bias,
        }
        return loss, gradients
