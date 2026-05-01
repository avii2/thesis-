from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.models.losses import (
    binary_cross_entropy_with_logits,
    binary_cross_entropy_with_logits_loss_and_gradient,
    sigmoid_from_logits,
)


@dataclass(frozen=True)
class TCNGNConfig:
    input_channels: int
    input_length: int
    channels: int = 64
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    kernel_size: int = 3
    groups: int = 8
    hidden_dim: int = 32
    dropout: float = 0.15
    gn_eps: float = 1e-5
    weight_scale: float = 0.04

    def validate(self) -> None:
        if self.input_channels <= 0:
            raise ValueError("TCNGNConfig.input_channels must be positive.")
        if self.input_length <= 0:
            raise ValueError("TCNGNConfig.input_length must be positive.")
        if self.channels <= 0:
            raise ValueError("TCNGNConfig.channels must be positive.")
        if self.channels % self.groups != 0:
            raise ValueError("TCNGNConfig.channels must be divisible by groups.")
        if not self.dilations:
            raise ValueError("TCNGNConfig.dilations must not be empty.")
        if any(dilation <= 0 for dilation in self.dilations):
            raise ValueError("TCNGNConfig.dilations values must be positive.")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("TCNGNConfig.kernel_size must be a positive odd integer.")
        if self.hidden_dim <= 0:
            raise ValueError("TCNGNConfig.hidden_dim must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("TCNGNConfig.dropout must be in [0, 1).")


def _state_keys(config: TCNGNConfig) -> tuple[str, ...]:
    keys = ["input_projection_weight", "input_projection_bias"]
    for index, _ in enumerate(config.dilations, start=1):
        prefix = f"tcn_block{index}"
        keys.extend(
            [
                f"{prefix}_conv_weight",
                f"{prefix}_conv_bias",
                f"{prefix}_gn_weight",
                f"{prefix}_gn_bias",
            ]
        )
    keys.extend(["linear1_weight", "linear1_bias", "linear2_weight", "linear2_bias"])
    return tuple(keys)


class TCNGNClassifier:
    def __init__(self, config: TCNGNConfig, *, seed: int = 42) -> None:
        self.config = config
        self.config.validate()
        self._rng = np.random.default_rng(seed)
        self._state = self._initialize_state()

    @property
    def state_keys(self) -> tuple[str, ...]:
        return _state_keys(self.config)

    @property
    def parameter_shapes(self) -> dict[str, tuple[int, ...]]:
        shapes: dict[str, tuple[int, ...]] = {
            "input_projection_weight": (self.config.channels, self.config.input_channels, 1),
            "input_projection_bias": (self.config.channels,),
        }
        for index, _ in enumerate(self.config.dilations, start=1):
            prefix = f"tcn_block{index}"
            shapes[f"{prefix}_conv_weight"] = (
                self.config.channels,
                self.config.channels,
                self.config.kernel_size,
            )
            shapes[f"{prefix}_conv_bias"] = (self.config.channels,)
            shapes[f"{prefix}_gn_weight"] = (self.config.channels,)
            shapes[f"{prefix}_gn_bias"] = (self.config.channels,)
        shapes["linear1_weight"] = (self.config.channels, self.config.hidden_dim)
        shapes["linear1_bias"] = (self.config.hidden_dim,)
        shapes["linear2_weight"] = (self.config.hidden_dim,)
        shapes["linear2_bias"] = ()
        return shapes

    def _initialize_state(self) -> dict[str, np.ndarray]:
        state: dict[str, np.ndarray] = {}
        shapes = self.parameter_shapes
        state["input_projection_weight"] = self._rng.normal(
            loc=0.0,
            scale=self.config.weight_scale,
            size=shapes["input_projection_weight"],
        ).astype(np.float32)
        state["input_projection_bias"] = np.zeros(shapes["input_projection_bias"], dtype=np.float32)
        for index, _ in enumerate(self.config.dilations, start=1):
            prefix = f"tcn_block{index}"
            state[f"{prefix}_conv_weight"] = self._rng.normal(
                loc=0.0,
                scale=self.config.weight_scale,
                size=shapes[f"{prefix}_conv_weight"],
            ).astype(np.float32)
            state[f"{prefix}_conv_bias"] = np.zeros(shapes[f"{prefix}_conv_bias"], dtype=np.float32)
            state[f"{prefix}_gn_weight"] = np.ones(shapes[f"{prefix}_gn_weight"], dtype=np.float32)
            state[f"{prefix}_gn_bias"] = np.zeros(shapes[f"{prefix}_gn_bias"], dtype=np.float32)
        state["linear1_weight"] = self._rng.normal(
            loc=0.0,
            scale=self.config.weight_scale,
            size=shapes["linear1_weight"],
        ).astype(np.float32)
        state["linear1_bias"] = np.zeros(shapes["linear1_bias"], dtype=np.float32)
        state["linear2_weight"] = self._rng.normal(
            loc=0.0,
            scale=self.config.weight_scale,
            size=shapes["linear2_weight"],
        ).astype(np.float32)
        state["linear2_bias"] = np.asarray(0.0, dtype=np.float32)
        return state

    def state_dict(self) -> dict[str, np.ndarray]:
        return {key: np.asarray(value, dtype=np.float32).copy() for key, value in self._state.items()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        missing = [key for key in self.state_keys if key not in state]
        extra = [key for key in state.keys() if key not in self.state_keys]
        if missing or extra:
            raise ValueError(f"TCNGNClassifier state mismatch. Missing={missing or '[]'} Extra={extra or '[]'}.")
        shapes = self.parameter_shapes
        loaded: dict[str, np.ndarray] = {}
        for key in self.state_keys:
            value = np.asarray(state[key], dtype=np.float32)
            if value.shape != shapes[key]:
                raise ValueError(f"TCNGNClassifier state {key!r} has shape {value.shape}, expected {shapes[key]}.")
            loaded[key] = value.copy()
        self._state = loaded

    @classmethod
    def from_state(
        cls,
        config: TCNGNConfig,
        state: Mapping[str, Any],
        *,
        seed: int = 42,
    ) -> "TCNGNClassifier":
        model = cls(config, seed=seed)
        model.load_state_dict(state)
        return model

    def parameter_bytes(self) -> int:
        return int(sum(np.asarray(value, dtype=np.float32).nbytes for value in self._state.values()))

    def predict_logits(self, inputs: np.ndarray) -> np.ndarray:
        return self._forward(inputs.astype(np.float32, copy=False), training=False, rng=None)["logits"]

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        return sigmoid_from_logits(self.predict_logits(inputs))

    def binary_cross_entropy(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        *,
        positive_class_weight: float = 1.0,
    ) -> float:
        return binary_cross_entropy_with_logits(
            self.predict_logits(inputs),
            labels,
            positive_class_weight=positive_class_weight,
        )

    def train_epoch(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        *,
        batch_size: int,
        learning_rate: float,
        rng: np.random.Generator,
        positive_class_weight: float = 1.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
    ) -> float:
        if inputs.shape[0] == 0:
            raise ValueError("TCNGNClassifier cannot train on an empty batch.")
        indices = rng.permutation(inputs.shape[0])
        adam_m = {key: np.zeros_like(value, dtype=np.float32) for key, value in self._state.items()}
        adam_v = {key: np.zeros_like(value, dtype=np.float32) for key, value in self._state.items()}
        step = 0
        losses: list[float] = []
        for start in range(0, inputs.shape[0], batch_size):
            step += 1
            batch_indices = indices[start : start + batch_size]
            loss, gradients = self._loss_and_gradients(
                inputs[batch_indices],
                labels[batch_indices].astype(np.float32, copy=False),
                rng,
                positive_class_weight=positive_class_weight,
            )
            losses.append(loss)
            for key in self.state_keys:
                grad = gradients[key].astype(np.float32, copy=False)
                adam_m[key] = adam_beta1 * adam_m[key] + (1.0 - adam_beta1) * grad
                adam_v[key] = adam_beta2 * adam_v[key] + (1.0 - adam_beta2) * (grad * grad)
                m_hat = adam_m[key] / (1.0 - adam_beta1**step)
                v_hat = adam_v[key] / (1.0 - adam_beta2**step)
                self._state[key] = (
                    self._state[key] - learning_rate * m_hat / (np.sqrt(v_hat) + adam_eps)
                ).astype(np.float32, copy=False)
        return float(np.mean(losses))

    def _loss_and_gradients(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        rng: np.random.Generator,
        *,
        positive_class_weight: float,
    ) -> tuple[float, dict[str, np.ndarray]]:
        cache = self._forward(inputs.astype(np.float32, copy=False), training=True, rng=rng)
        loss, dlogits = binary_cross_entropy_with_logits_loss_and_gradient(
            cache["logits"],
            labels,
            positive_class_weight=positive_class_weight,
        )

        gradients: dict[str, np.ndarray] = {}
        gradients["linear2_weight"] = (cache["hidden_dropout"].T @ dlogits).astype(np.float32, copy=False)
        gradients["linear2_bias"] = np.asarray(dlogits.sum(), dtype=np.float32)
        dhidden_dropout = dlogits[:, None] * self._state["linear2_weight"][None, :]
        dhidden_relu = dhidden_dropout * cache["hidden_dropout_mask"]
        dhidden = dhidden_relu * (cache["hidden_linear"] > 0.0)
        gradients["linear1_weight"] = (cache["pooled"].T @ dhidden).astype(np.float32, copy=False)
        gradients["linear1_bias"] = dhidden.sum(axis=0).astype(np.float32, copy=False)
        dpooled = dhidden @ self._state["linear1_weight"].T
        dblock = np.broadcast_to(
            (dpooled[:, :, None] / cache["final_tcn_output"].shape[2]).astype(np.float32, copy=False),
            cache["final_tcn_output"].shape,
        ).copy()

        for index in range(len(self.config.dilations), 0, -1):
            prefix = f"tcn_block{index}"
            didentity = dblock
            ddropout = dblock * cache[f"{prefix}_dropout_mask"]
            dactivation = ddropout * (cache[f"{prefix}_gn_out"] > 0.0)
            dgn, gn_grads = self._group_norm_backward(prefix, dactivation, cache)
            gradients.update(gn_grads)
            dconv_input, conv_grads = self._conv1d_backward(
                f"{prefix}_conv",
                dgn,
                cache[f"{prefix}_input_shape"],
                cache,
            )
            gradients.update(conv_grads)
            dblock = didentity + dconv_input

        _, projection_grads = self._conv1d_backward(
            "input_projection",
            dblock,
            cache["input_projection_input_shape"],
            cache,
        )
        gradients.update(projection_grads)
        return loss, gradients

    def _forward(
        self,
        inputs: np.ndarray,
        *,
        training: bool,
        rng: np.random.Generator | None,
    ) -> dict[str, Any]:
        if inputs.ndim != 3:
            raise ValueError(f"TCNGNClassifier expects inputs with shape (batch, channels, length). Observed {inputs.shape}.")
        if inputs.shape[1] != self.config.input_channels or inputs.shape[2] != self.config.input_length:
            raise ValueError(
                "TCNGNClassifier input shape mismatch. "
                f"Observed {inputs.shape[1:]} expected {(self.config.input_channels, self.config.input_length)}."
            )
        cache: dict[str, Any] = {"input_projection_input_shape": inputs.shape}
        x, projection_cache = self._conv1d_forward("input_projection", inputs, dilation=1)
        cache.update(projection_cache)

        for index, dilation in enumerate(self.config.dilations, start=1):
            prefix = f"tcn_block{index}"
            residual = x
            cache[f"{prefix}_input_shape"] = x.shape
            conv_out, conv_cache = self._conv1d_forward(f"{prefix}_conv", x, dilation=dilation)
            cache.update(conv_cache)
            gn_out, gn_cache = self._group_norm_forward(prefix, conv_out)
            cache.update(gn_cache)
            activation = np.maximum(gn_out, 0.0)
            dropout_out, dropout_mask = self._dropout_forward(activation, training=training, rng=rng)
            x = (residual + dropout_out).astype(np.float32, copy=False)
            cache[f"{prefix}_gn_out"] = gn_out
            cache[f"{prefix}_dropout_mask"] = dropout_mask
        cache["final_tcn_output"] = x
        pooled = x.mean(axis=2)
        hidden_linear = pooled @ self._state["linear1_weight"] + self._state["linear1_bias"][None, :]
        hidden_relu = np.maximum(hidden_linear, 0.0)
        hidden_dropout, hidden_dropout_mask = self._dropout_forward(hidden_relu, training=training, rng=rng)
        logits = hidden_dropout @ self._state["linear2_weight"] + float(self._state["linear2_bias"])
        cache["pooled"] = pooled
        cache["hidden_linear"] = hidden_linear
        cache["hidden_dropout"] = hidden_dropout
        cache["hidden_dropout_mask"] = hidden_dropout_mask
        cache["logits"] = logits.astype(np.float32, copy=False)
        return cache

    def _conv1d_forward(
        self,
        prefix: str,
        inputs: np.ndarray,
        *,
        dilation: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        weight = self._state[f"{prefix}_weight"]
        bias = self._state[f"{prefix}_bias"]
        kernel_size = int(weight.shape[2])
        effective_kernel = dilation * (kernel_size - 1) + 1
        pad = effective_kernel // 2
        padded = np.pad(inputs, ((0, 0), (0, 0), (pad, pad)), mode="constant")
        windows = np.lib.stride_tricks.sliding_window_view(
            padded,
            window_shape=effective_kernel,
            axis=2,
        )[..., ::dilation]
        conv_out = np.einsum("bclk,ock->bol", windows, weight, optimize=True) + bias[None, :, None]
        return conv_out.astype(np.float32, copy=False), {
            f"{prefix}_conv_windows": windows,
            f"{prefix}_conv_dilation": dilation,
            f"{prefix}_conv_pad": pad,
        }

    def _conv1d_backward(
        self,
        prefix: str,
        dconv: np.ndarray,
        input_shape: Sequence[int],
        cache: Mapping[str, Any],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        weight = self._state[f"{prefix}_weight"]
        windows = cache[f"{prefix}_conv_windows"]
        dilation = int(cache[f"{prefix}_conv_dilation"])
        pad = int(cache[f"{prefix}_conv_pad"])
        batch_size, in_channels, input_length = input_shape
        kernel_size = int(weight.shape[2])
        grad_weight = np.einsum("bclk,bol->ock", windows, dconv, optimize=True).astype(np.float32, copy=False)
        grad_bias = dconv.sum(axis=(0, 2)).astype(np.float32, copy=False)
        padded_length = input_length + 2 * pad
        grad_padded = np.zeros((batch_size, in_channels, padded_length), dtype=np.float32)
        for kernel_index in range(kernel_size):
            grad_slice = np.einsum("bol,oc->bcl", dconv, weight[:, :, kernel_index], optimize=True)
            start = kernel_index * dilation
            grad_padded[:, :, start : start + input_length] += grad_slice.astype(np.float32, copy=False)
        return grad_padded[:, :, pad : pad + input_length].astype(np.float32, copy=False), {
            f"{prefix}_weight": grad_weight,
            f"{prefix}_bias": grad_bias,
        }

    def _group_norm_forward(self, prefix: str, inputs: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        gamma = self._state[f"{prefix}_gn_weight"]
        beta = self._state[f"{prefix}_gn_bias"]
        batch_size, channels, length = inputs.shape
        groups = self.config.groups
        grouped = inputs.reshape(batch_size, groups, channels // groups, length)
        mean = grouped.mean(axis=(2, 3), keepdims=True)
        var = grouped.var(axis=(2, 3), keepdims=True)
        inv_std = (1.0 / np.sqrt(var + self.config.gn_eps)).astype(np.float32, copy=False)
        x_hat_grouped = (grouped - mean) * inv_std
        x_hat = x_hat_grouped.reshape(inputs.shape).astype(np.float32, copy=False)
        out = gamma[None, :, None] * x_hat + beta[None, :, None]
        return out.astype(np.float32, copy=False), {
            f"{prefix}_gn_x_hat": x_hat,
            f"{prefix}_gn_inv_std": inv_std,
        }

    def _group_norm_backward(
        self,
        prefix: str,
        dout: np.ndarray,
        cache: Mapping[str, Any],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        gamma = self._state[f"{prefix}_gn_weight"]
        x_hat = cache[f"{prefix}_gn_x_hat"]
        inv_std = cache[f"{prefix}_gn_inv_std"]
        batch_size, channels, length = dout.shape
        groups = self.config.groups
        channels_per_group = channels // groups
        dgamma = (dout * x_hat).sum(axis=(0, 2)).astype(np.float32, copy=False)
        dbeta = dout.sum(axis=(0, 2)).astype(np.float32, copy=False)
        dxhat = (dout * gamma[None, :, None]).reshape(batch_size, groups, channels_per_group, length)
        xhat_grouped = x_hat.reshape(batch_size, groups, channels_per_group, length)
        population = float(channels_per_group * length)
        sum_dxhat = dxhat.sum(axis=(2, 3), keepdims=True)
        sum_dxhat_xhat = (dxhat * xhat_grouped).sum(axis=(2, 3), keepdims=True)
        dx_grouped = (
            inv_std
            / population
            * (population * dxhat - sum_dxhat - xhat_grouped * sum_dxhat_xhat)
        )
        return dx_grouped.reshape(dout.shape).astype(np.float32, copy=False), {
            f"{prefix}_gn_weight": dgamma,
            f"{prefix}_gn_bias": dbeta,
        }

    def _dropout_forward(
        self,
        inputs: np.ndarray,
        *,
        training: bool,
        rng: np.random.Generator | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not training or self.config.dropout == 0.0:
            return inputs.astype(np.float32, copy=False), np.ones_like(inputs, dtype=np.float32)
        if rng is None:
            raise ValueError("TCNGNClassifier dropout requires an RNG during training.")
        keep_probability = 1.0 - self.config.dropout
        mask = (rng.random(inputs.shape) < keep_probability).astype(np.float32) / keep_probability
        return (inputs * mask).astype(np.float32, copy=False), mask.astype(np.float32, copy=False)
