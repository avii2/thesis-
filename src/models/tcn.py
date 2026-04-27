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
class TCNConfig:
    input_channels: int
    input_length: int
    block_channels: tuple[int, int, int] = (32, 64, 64)
    dilations: tuple[int, int, int] = (1, 2, 4)
    kernel_size: int = 3
    hidden_dim: int = 32
    dropout: float = 0.1
    bn_eps: float = 1e-5
    bn_momentum: float = 0.1
    weight_scale: float = 0.05

    def validate(self) -> None:
        if self.input_channels <= 0:
            raise ValueError("TCNConfig.input_channels must be positive.")
        if self.input_length <= 0:
            raise ValueError("TCNConfig.input_length must be positive.")
        if len(self.block_channels) != 3 or len(self.dilations) != 3:
            raise ValueError("TCNConfig requires exactly three TCN blocks.")
        if any(channel <= 0 for channel in self.block_channels):
            raise ValueError("TCNConfig.block_channels values must be positive.")
        if any(dilation <= 0 for dilation in self.dilations):
            raise ValueError("TCNConfig.dilations values must be positive.")
        if self.kernel_size <= 0:
            raise ValueError("TCNConfig.kernel_size must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("TCNConfig.hidden_dim must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("TCNConfig.dropout must be in [0, 1).")


def _state_keys(config: TCNConfig) -> tuple[str, ...]:
    keys: list[str] = []
    in_channels = config.input_channels
    for block_index, out_channels in enumerate(config.block_channels, start=1):
        prefix = f"tcn_block{block_index}"
        keys.extend(
            [
                f"{prefix}_conv_weight",
                f"{prefix}_conv_bias",
                f"{prefix}_bn_weight",
                f"{prefix}_bn_bias",
                f"{prefix}_bn_running_mean",
                f"{prefix}_bn_running_var",
                f"{prefix}_bn_num_batches_tracked",
            ]
        )
        in_channels = out_channels

    keys.extend(
        [
            "linear1_weight",
            "linear1_bias",
            "linear2_weight",
            "linear2_bias",
        ]
    )
    return tuple(keys)


class TCNClassifier:
    def __init__(self, config: TCNConfig, *, seed: int = 42) -> None:
        self.config = config
        self.config.validate()
        self._rng = np.random.default_rng(seed)
        self._state = self._initialize_state()

    @property
    def state_keys(self) -> tuple[str, ...]:
        return _state_keys(self.config)

    @property
    def parameter_shapes(self) -> dict[str, tuple[int, ...]]:
        shapes: dict[str, tuple[int, ...]] = {}
        in_channels = self.config.input_channels
        for block_index, out_channels in enumerate(self.config.block_channels, start=1):
            prefix = f"tcn_block{block_index}"
            shapes[f"{prefix}_conv_weight"] = (out_channels, in_channels, self.config.kernel_size)
            shapes[f"{prefix}_conv_bias"] = (out_channels,)
            shapes[f"{prefix}_bn_weight"] = (out_channels,)
            shapes[f"{prefix}_bn_bias"] = (out_channels,)
            shapes[f"{prefix}_bn_running_mean"] = (out_channels,)
            shapes[f"{prefix}_bn_running_var"] = (out_channels,)
            shapes[f"{prefix}_bn_num_batches_tracked"] = ()
            in_channels = out_channels

        shapes["linear1_weight"] = (self.config.block_channels[-1], self.config.hidden_dim)
        shapes["linear1_bias"] = (self.config.hidden_dim,)
        shapes["linear2_weight"] = (self.config.hidden_dim,)
        shapes["linear2_bias"] = ()
        return shapes

    def _initialize_state(self) -> dict[str, np.ndarray]:
        state: dict[str, np.ndarray] = {}
        in_channels = self.config.input_channels
        for block_index, out_channels in enumerate(self.config.block_channels, start=1):
            prefix = f"tcn_block{block_index}"
            state[f"{prefix}_conv_weight"] = self._rng.normal(
                loc=0.0,
                scale=self.config.weight_scale,
                size=(out_channels, in_channels, self.config.kernel_size),
            ).astype(np.float32)
            state[f"{prefix}_conv_bias"] = np.zeros((out_channels,), dtype=np.float32)
            state[f"{prefix}_bn_weight"] = np.ones((out_channels,), dtype=np.float32)
            state[f"{prefix}_bn_bias"] = np.zeros((out_channels,), dtype=np.float32)
            state[f"{prefix}_bn_running_mean"] = np.zeros((out_channels,), dtype=np.float32)
            state[f"{prefix}_bn_running_var"] = np.ones((out_channels,), dtype=np.float32)
            state[f"{prefix}_bn_num_batches_tracked"] = np.asarray(0.0, dtype=np.float32)
            in_channels = out_channels

        state["linear1_weight"] = self._rng.normal(
            loc=0.0,
            scale=self.config.weight_scale,
            size=(self.config.block_channels[-1], self.config.hidden_dim),
        ).astype(np.float32)
        state["linear1_bias"] = np.zeros((self.config.hidden_dim,), dtype=np.float32)
        state["linear2_weight"] = self._rng.normal(
            loc=0.0,
            scale=self.config.weight_scale,
            size=(self.config.hidden_dim,),
        ).astype(np.float32)
        state["linear2_bias"] = np.asarray(0.0, dtype=np.float32)
        return state

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            key: value.copy() if isinstance(value, np.ndarray) else np.asarray(value, dtype=np.float32)
            for key, value in self._state.items()
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        missing = [key for key in self.state_keys if key not in state]
        extra = [key for key in state.keys() if key not in self.state_keys]
        if missing or extra:
            raise ValueError(
                f"TCNClassifier state keys mismatch. Missing={missing or '[]'} Extra={extra or '[]'}."
            )

        shapes = self.parameter_shapes
        loaded: dict[str, np.ndarray] = {}
        for key in self.state_keys:
            value = np.asarray(state[key], dtype=np.float32)
            if value.shape != shapes[key]:
                raise ValueError(
                    f"TCNClassifier state {key!r} has shape {value.shape}, expected {shapes[key]}."
                )
            loaded[key] = value.copy()
        self._state = loaded

    @classmethod
    def from_state(
        cls,
        config: TCNConfig,
        state: Mapping[str, Any],
        *,
        seed: int = 42,
    ) -> "TCNClassifier":
        model = cls(config, seed=seed)
        model.load_state_dict(state)
        return model

    def parameter_bytes(self) -> int:
        return int(sum(np.asarray(value).nbytes for value in self._state.values()))

    def predict_logits(self, inputs: np.ndarray) -> np.ndarray:
        return self._forward(inputs.astype(np.float32, copy=False), training=False, rng=None)["logits"]

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

    def train_epoch(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        *,
        batch_size: int,
        learning_rate: float,
        rng: np.random.Generator,
        positive_class_weight: float = 1.0,
    ) -> float:
        if inputs.shape[0] == 0:
            raise ValueError("TCNClassifier cannot train on an empty batch.")

        indices = rng.permutation(inputs.shape[0])
        losses: list[float] = []
        for start in range(0, inputs.shape[0], batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_inputs = inputs[batch_indices]
            batch_labels = labels[batch_indices].astype(np.float32, copy=False)
            loss, gradients = self._loss_and_gradients(
                batch_inputs,
                batch_labels,
                rng,
                positive_class_weight=positive_class_weight,
            )
            losses.append(loss)
            for key in self.state_keys:
                if key.endswith("running_mean") or key.endswith("running_var") or key.endswith("num_batches_tracked"):
                    continue
                self._state[key] = self._state[key] - gradients[key] * learning_rate
        return float(np.mean(losses))

    def _loss_and_gradients(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        rng: np.random.Generator,
        *,
        positive_class_weight: float = 1.0,
    ) -> tuple[float, dict[str, np.ndarray]]:
        cache = self._forward(inputs, training=True, rng=rng)
        loss, dlogits = binary_cross_entropy_with_logits_loss_and_gradient(
            cache["logits"],
            labels,
            positive_class_weight=positive_class_weight,
        )

        gradients: dict[str, np.ndarray] = {}
        linear2_weight = self._state["linear2_weight"]
        hidden_dropout = cache["hidden_dropout"]
        gradients["linear2_weight"] = (hidden_dropout.T @ dlogits).astype(np.float32, copy=False)
        gradients["linear2_bias"] = np.asarray(dlogits.sum(), dtype=np.float32)

        dhidden_dropout = dlogits[:, None] * linear2_weight[None, :]
        dhidden_relu = dhidden_dropout * cache["hidden_dropout_mask"]
        dhidden_linear = dhidden_relu * (cache["hidden_linear"] > 0.0)

        pooled = cache["pooled"]
        gradients["linear1_weight"] = (pooled.T @ dhidden_linear).astype(np.float32, copy=False)
        gradients["linear1_bias"] = dhidden_linear.sum(axis=0).astype(np.float32, copy=False)
        dpooled = dhidden_linear @ self._state["linear1_weight"].T

        last_time_length = cache["block3_output"].shape[2]
        dblock = np.broadcast_to(
            (dpooled[:, :, None] / last_time_length).astype(np.float32, copy=False),
            cache["block3_output"].shape,
        ).copy()

        for block_index in (3, 2, 1):
            prefix = f"tcn_block{block_index}"
            dpost_dropout = dblock * cache[f"{prefix}_dropout_mask"]
            drelu = dpost_dropout * (cache[f"{prefix}_bn_out"] > 0.0)
            dbn, bn_grads = self._batch_norm_backward(prefix, drelu, cache)
            gradients.update(bn_grads)
            dblock, conv_grads = self._conv1d_backward(
                prefix,
                dbn,
                cache[f"{prefix}_input_shape"],
                cache,
            )
            gradients.update(conv_grads)

        return loss, gradients

    def _forward(
        self,
        inputs: np.ndarray,
        *,
        training: bool,
        rng: np.random.Generator | None,
    ) -> dict[str, Any]:
        if inputs.ndim != 3:
            raise ValueError(f"TCNClassifier expects inputs with shape (batch, channels, length). Observed {inputs.shape}.")
        if inputs.shape[1] != self.config.input_channels or inputs.shape[2] != self.config.input_length:
            raise ValueError(
                "TCNClassifier input shape mismatch. "
                f"Observed {inputs.shape[1:]} expected {(self.config.input_channels, self.config.input_length)}."
            )

        cache: dict[str, Any] = {}
        x = inputs.astype(np.float32, copy=False)
        for block_index, dilation in enumerate(self.config.dilations, start=1):
            prefix = f"tcn_block{block_index}"
            cache[f"{prefix}_input_shape"] = x.shape
            cache[f"{prefix}_input"] = x
            conv_out, conv_cache = self._conv1d_forward(prefix, x, dilation=dilation)
            cache.update(conv_cache)
            bn_out, bn_cache = self._batch_norm_forward(prefix, conv_out, training=training)
            cache.update(bn_cache)
            relu_out = np.maximum(bn_out, 0.0)
            dropout_out, dropout_mask = self._dropout_forward(relu_out, training=training, rng=rng)
            cache[f"{prefix}_bn_out"] = bn_out
            cache[f"{prefix}_dropout_mask"] = dropout_mask
            x = dropout_out
            cache[f"block{block_index}_output"] = x

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
        weight = self._state[f"{prefix}_conv_weight"]
        bias = self._state[f"{prefix}_conv_bias"]
        effective_kernel = dilation * (self.config.kernel_size - 1) + 1
        pad = effective_kernel // 2
        padded = np.pad(inputs, ((0, 0), (0, 0), (pad, pad)), mode="constant")
        windows = np.lib.stride_tricks.sliding_window_view(
            padded,
            window_shape=effective_kernel,
            axis=2,
        )[..., ::dilation]
        conv_out = np.einsum("bclk,ock->bol", windows, weight, optimize=True)
        conv_out = conv_out + bias[None, :, None]
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
        weight = self._state[f"{prefix}_conv_weight"]
        windows = cache[f"{prefix}_conv_windows"]
        dilation = int(cache[f"{prefix}_conv_dilation"])
        pad = int(cache[f"{prefix}_conv_pad"])
        batch_size, in_channels, input_length = input_shape

        grad_weight = np.einsum("bclk,bol->ock", windows, dconv, optimize=True).astype(np.float32, copy=False)
        grad_bias = dconv.sum(axis=(0, 2)).astype(np.float32, copy=False)

        padded_length = input_length + 2 * pad
        grad_padded = np.zeros((batch_size, in_channels, padded_length), dtype=np.float32)
        for kernel_index in range(self.config.kernel_size):
            grad_slice = np.einsum(
                "bol,oc->bcl",
                dconv,
                weight[:, :, kernel_index],
                optimize=True,
            ).astype(np.float32, copy=False)
            start = kernel_index * dilation
            grad_padded[:, :, start : start + input_length] += grad_slice

        grad_input = grad_padded[:, :, pad : pad + input_length]
        return grad_input.astype(np.float32, copy=False), {
            f"{prefix}_conv_weight": grad_weight,
            f"{prefix}_conv_bias": grad_bias,
        }

    def _batch_norm_forward(
        self,
        prefix: str,
        inputs: np.ndarray,
        *,
        training: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        gamma = self._state[f"{prefix}_bn_weight"]
        beta = self._state[f"{prefix}_bn_bias"]
        running_mean = self._state[f"{prefix}_bn_running_mean"]
        running_var = self._state[f"{prefix}_bn_running_var"]

        if training:
            mean = inputs.mean(axis=(0, 2)).astype(np.float32, copy=False)
            var = inputs.var(axis=(0, 2)).astype(np.float32, copy=False)
            self._state[f"{prefix}_bn_running_mean"] = (
                (1.0 - self.config.bn_momentum) * running_mean + self.config.bn_momentum * mean
            ).astype(np.float32, copy=False)
            self._state[f"{prefix}_bn_running_var"] = (
                (1.0 - self.config.bn_momentum) * running_var + self.config.bn_momentum * var
            ).astype(np.float32, copy=False)
            self._state[f"{prefix}_bn_num_batches_tracked"] = np.asarray(
                float(self._state[f"{prefix}_bn_num_batches_tracked"]) + 1.0,
                dtype=np.float32,
            )
        else:
            mean = running_mean
            var = running_var

        inv_std = (1.0 / np.sqrt(var + self.config.bn_eps)).astype(np.float32, copy=False)
        x_hat = (inputs - mean[None, :, None]) * inv_std[None, :, None]
        bn_out = gamma[None, :, None] * x_hat + beta[None, :, None]
        return bn_out.astype(np.float32, copy=False), {
            f"{prefix}_bn_mean": mean,
            f"{prefix}_bn_var": var,
            f"{prefix}_bn_inv_std": inv_std,
            f"{prefix}_bn_x_hat": x_hat.astype(np.float32, copy=False),
        }

    def _batch_norm_backward(
        self,
        prefix: str,
        dout: np.ndarray,
        cache: Mapping[str, Any],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        gamma = self._state[f"{prefix}_bn_weight"]
        x_hat = cache[f"{prefix}_bn_x_hat"]
        inv_std = cache[f"{prefix}_bn_inv_std"]
        channel_population = float(dout.shape[0] * dout.shape[2])

        dgamma = (dout * x_hat).sum(axis=(0, 2)).astype(np.float32, copy=False)
        dbeta = dout.sum(axis=(0, 2)).astype(np.float32, copy=False)
        dxhat = dout * gamma[None, :, None]
        sum_dxhat = dxhat.sum(axis=(0, 2))
        sum_dxhat_xhat = (dxhat * x_hat).sum(axis=(0, 2))
        dx = (
            inv_std[None, :, None]
            / channel_population
            * (
                channel_population * dxhat
                - sum_dxhat[None, :, None]
                - x_hat * sum_dxhat_xhat[None, :, None]
            )
        )
        gradients = {
            f"{prefix}_bn_weight": dgamma,
            f"{prefix}_bn_bias": dbeta,
            f"{prefix}_bn_running_mean": np.zeros_like(self._state[f"{prefix}_bn_running_mean"]),
            f"{prefix}_bn_running_var": np.zeros_like(self._state[f"{prefix}_bn_running_var"]),
            f"{prefix}_bn_num_batches_tracked": np.asarray(0.0, dtype=np.float32),
        }
        return dx.astype(np.float32, copy=False), gradients

    def _dropout_forward(
        self,
        inputs: np.ndarray,
        *,
        training: bool,
        rng: np.random.Generator | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not training or self.config.dropout == 0.0:
            mask = np.ones_like(inputs, dtype=np.float32)
            return inputs.astype(np.float32, copy=False), mask
        if rng is None:
            raise ValueError("TCNClassifier dropout requires an RNG during training.")
        keep_probability = 1.0 - self.config.dropout
        mask = (rng.random(inputs.shape) < keep_probability).astype(np.float32) / keep_probability
        return (inputs * mask).astype(np.float32, copy=False), mask.astype(np.float32, copy=False)
