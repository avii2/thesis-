from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.models.losses import (
    binary_cross_entropy_with_logits,
    binary_cross_entropy_with_logits_loss_and_gradient,
    sigmoid_from_logits,
)


STATE_KEYS = (
    "linear1_weight",
    "linear1_bias",
    "linear2_weight",
    "linear2_bias",
    "linear3_weight",
    "linear3_bias",
)


@dataclass(frozen=True)
class MLPConfig:
    input_dim: int
    hidden_dim1: int = 64
    hidden_dim2: int = 32
    dropout: float = 0.2
    weight_scale: float = 0.05

    def validate(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("MLPConfig.input_dim must be positive.")
        if self.hidden_dim1 <= 0 or self.hidden_dim2 <= 0:
            raise ValueError("MLPConfig hidden dimensions must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("MLPConfig.dropout must be in [0, 1).")

    @property
    def parameter_shapes(self) -> dict[str, tuple[int, ...]]:
        self.validate()
        return {
            "linear1_weight": (self.input_dim, self.hidden_dim1),
            "linear1_bias": (self.hidden_dim1,),
            "linear2_weight": (self.hidden_dim1, self.hidden_dim2),
            "linear2_bias": (self.hidden_dim2,),
            "linear3_weight": (self.hidden_dim2,),
            "linear3_bias": (),
        }


class CompactMLPClassifier:
    def __init__(self, config: MLPConfig, *, seed: int = 42) -> None:
        self.config = config
        self.config.validate()
        self._rng = np.random.default_rng(seed)
        self._state = self._initialize_state()

    def _initialize_state(self) -> dict[str, np.ndarray]:
        shapes = self.config.parameter_shapes
        return {
            "linear1_weight": self._rng.normal(
                loc=0.0,
                scale=self.config.weight_scale,
                size=shapes["linear1_weight"],
            ).astype(np.float32),
            "linear1_bias": np.zeros(shapes["linear1_bias"], dtype=np.float32),
            "linear2_weight": self._rng.normal(
                loc=0.0,
                scale=self.config.weight_scale,
                size=shapes["linear2_weight"],
            ).astype(np.float32),
            "linear2_bias": np.zeros(shapes["linear2_bias"], dtype=np.float32),
            "linear3_weight": self._rng.normal(
                loc=0.0,
                scale=self.config.weight_scale,
                size=shapes["linear3_weight"],
            ).astype(np.float32),
            "linear3_bias": np.asarray(0.0, dtype=np.float32),
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
                f"CompactMLPClassifier state keys mismatch. Missing={missing or '[]'} Extra={extra or '[]'}."
            )

        shapes = self.config.parameter_shapes
        loaded_state: dict[str, np.ndarray] = {}
        for key in STATE_KEYS:
            value = np.asarray(state[key], dtype=np.float32)
            if value.shape != shapes[key]:
                raise ValueError(
                    f"CompactMLPClassifier state {key!r} has shape {value.shape}, expected {shapes[key]}."
                )
            loaded_state[key] = value.copy()
        self._state = loaded_state

    @classmethod
    def from_state(
        cls,
        config: MLPConfig,
        state: Mapping[str, Any],
        *,
        seed: int = 42,
    ) -> "CompactMLPClassifier":
        model = cls(config, seed=seed)
        model.load_state_dict(state)
        return model

    def parameter_bytes(self) -> int:
        return int(sum(np.asarray(value).nbytes for value in self._state.values()))

    def predict_logits(self, inputs: np.ndarray) -> np.ndarray:
        cache = self._forward(inputs.astype(np.float32, copy=False), training=False, rng=None)
        return cache["logits"]

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(inputs)
        return sigmoid_from_logits(logits)

    def predict(self, inputs: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(inputs) >= threshold).astype(np.int8, copy=False)

    def binary_cross_entropy(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        *,
        positive_class_weight: float = 1.0,
    ) -> float:
        logits = self.predict_logits(inputs)
        return binary_cross_entropy_with_logits(
            logits,
            labels,
            positive_class_weight=positive_class_weight,
        )

    def loss_and_gradients(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        *,
        rng: np.random.Generator,
        positive_class_weight: float = 1.0,
    ) -> tuple[float, dict[str, np.ndarray]]:
        cache = self._forward(inputs, training=True, rng=rng)
        loss, dlogits = binary_cross_entropy_with_logits_loss_and_gradient(
            cache["logits"],
            labels,
            positive_class_weight=positive_class_weight,
        )
        gradients: dict[str, np.ndarray] = {}
        gradients["linear3_weight"] = (cache["hidden2_dropout"].T @ dlogits).astype(np.float32, copy=False)
        gradients["linear3_bias"] = np.asarray(dlogits.sum(), dtype=np.float32)

        dhidden2_dropout = dlogits[:, None] * self._state["linear3_weight"][None, :]
        dhidden2_relu = dhidden2_dropout * cache["hidden2_dropout_mask"]
        dhidden2_linear = dhidden2_relu * (cache["hidden2_linear"] > 0.0)

        gradients["linear2_weight"] = (cache["hidden1_dropout"].T @ dhidden2_linear).astype(np.float32, copy=False)
        gradients["linear2_bias"] = dhidden2_linear.sum(axis=0).astype(np.float32, copy=False)

        dhidden1_dropout = dhidden2_linear @ self._state["linear2_weight"].T
        dhidden1_relu = dhidden1_dropout * cache["hidden1_dropout_mask"]
        dhidden1_linear = dhidden1_relu * (cache["hidden1_linear"] > 0.0)

        gradients["linear1_weight"] = (inputs.T @ dhidden1_linear).astype(np.float32, copy=False)
        gradients["linear1_bias"] = dhidden1_linear.sum(axis=0).astype(np.float32, copy=False)
        return loss, gradients

    def apply_adam_gradients(
        self,
        gradients: Mapping[str, np.ndarray],
        optimizer_state: dict[str, dict[str, np.ndarray] | int],
        *,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        timestep = int(optimizer_state.get("t", 0)) + 1
        optimizer_state["t"] = timestep
        moment1 = optimizer_state.setdefault("m", {})
        moment2 = optimizer_state.setdefault("v", {})

        for key in STATE_KEYS:
            gradient = np.asarray(gradients[key], dtype=np.float32)
            current_m = np.asarray(moment1.get(key, np.zeros_like(self._state[key])), dtype=np.float32)
            current_v = np.asarray(moment2.get(key, np.zeros_like(self._state[key])), dtype=np.float32)

            current_m = beta1 * current_m + (1.0 - beta1) * gradient
            current_v = beta2 * current_v + (1.0 - beta2) * np.square(gradient)
            moment1[key] = current_m
            moment2[key] = current_v

            m_hat = current_m / (1.0 - beta1 ** timestep)
            v_hat = current_v / (1.0 - beta2 ** timestep)
            self._state[key] = (
                self._state[key] - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            ).astype(np.float32, copy=False)

    def _forward(
        self,
        inputs: np.ndarray,
        *,
        training: bool,
        rng: np.random.Generator | None,
    ) -> dict[str, np.ndarray]:
        if inputs.ndim != 2:
            raise ValueError(
                f"CompactMLPClassifier expects inputs with shape (batch, features). Observed {inputs.shape}."
            )
        if inputs.shape[1] != self.config.input_dim:
            raise ValueError(
                "CompactMLPClassifier input dimension mismatch. "
                f"Observed {inputs.shape[1]} expected {self.config.input_dim}."
            )

        hidden1_linear = inputs @ self._state["linear1_weight"] + self._state["linear1_bias"][None, :]
        hidden1_relu = np.maximum(hidden1_linear, 0.0)
        hidden1_dropout, hidden1_dropout_mask = self._dropout_forward(hidden1_relu, training=training, rng=rng)

        hidden2_linear = hidden1_dropout @ self._state["linear2_weight"] + self._state["linear2_bias"][None, :]
        hidden2_relu = np.maximum(hidden2_linear, 0.0)
        hidden2_dropout, hidden2_dropout_mask = self._dropout_forward(hidden2_relu, training=training, rng=rng)

        logits = hidden2_dropout @ self._state["linear3_weight"] + float(self._state["linear3_bias"])
        return {
            "hidden1_linear": hidden1_linear,
            "hidden1_dropout": hidden1_dropout,
            "hidden1_dropout_mask": hidden1_dropout_mask,
            "hidden2_linear": hidden2_linear,
            "hidden2_dropout": hidden2_dropout,
            "hidden2_dropout_mask": hidden2_dropout_mask,
            "logits": logits.astype(np.float32, copy=False),
        }

    def _dropout_forward(
        self,
        activations: np.ndarray,
        *,
        training: bool,
        rng: np.random.Generator | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not training or self.config.dropout <= 0.0:
            ones = np.ones_like(activations, dtype=np.float32)
            return activations.astype(np.float32, copy=False), ones
        if rng is None:
            raise ValueError("CompactMLPClassifier dropout requires an RNG during training.")

        keep_prob = 1.0 - self.config.dropout
        mask = (rng.random(activations.shape) < keep_prob).astype(np.float32) / keep_prob
        return (activations * mask).astype(np.float32, copy=False), mask
