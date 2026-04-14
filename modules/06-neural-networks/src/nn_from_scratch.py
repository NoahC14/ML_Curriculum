from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray


def relu(z: Array) -> Array:
    return np.maximum(z, 0.0)


def relu_backward(z: Array, _: Array) -> Array:
    return (z > 0.0).astype(float)


def sigmoid(z: Array) -> Array:
    clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def sigmoid_backward(_: Array, a: Array) -> Array:
    return a * (1.0 - a)


def tanh(z: Array) -> Array:
    return np.tanh(z)


def tanh_backward(_: Array, a: Array) -> Array:
    return 1.0 - np.square(a)


def identity(z: Array) -> Array:
    return z


def identity_backward(z: Array, _: Array) -> Array:
    return np.ones_like(z)


def softmax(z: Array) -> Array:
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


@dataclass(frozen=True)
class ActivationSpec:
    forward: Callable[[Array], Array]
    backward: Callable[[Array, Array], Array]


ACTIVATIONS: dict[str, ActivationSpec] = {
    "relu": ActivationSpec(relu, relu_backward),
    "sigmoid": ActivationSpec(sigmoid, sigmoid_backward),
    "tanh": ActivationSpec(tanh, tanh_backward),
    "identity": ActivationSpec(identity, identity_backward),
}


def one_hot_encode(y: Array, num_classes: int | None = None) -> Array:
    labels = np.asarray(y, dtype=int).reshape(-1)
    classes = int(labels.max() + 1 if num_classes is None else num_classes)
    encoded = np.zeros((labels.shape[0], classes), dtype=float)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def train_test_split_indices(
    sample_count: int,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[Array, Array]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(sample_count)
    test_size = max(1, int(round(sample_count * test_fraction)))
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return train_idx, test_idx


class ScratchMLP:
    def __init__(
        self,
        layer_dims: list[int],
        activations: list[str],
        loss: str = "binary_cross_entropy",
        seed: int = 0,
    ) -> None:
        if len(layer_dims) < 2:
            raise ValueError("layer_dims must contain input and output widths.")
        if len(activations) != len(layer_dims) - 1:
            raise ValueError("Need one activation name per affine layer.")
        if loss not in {"binary_cross_entropy", "cross_entropy", "mse"}:
            raise ValueError(f"Unsupported loss: {loss}")

        self.layer_dims = layer_dims
        self.activations = activations
        self.loss = loss
        self.rng = np.random.default_rng(seed)
        self.parameters = self._initialize_parameters()

    def _initialize_parameters(self) -> dict[str, Array]:
        parameters: dict[str, Array] = {}
        for layer_index, (fan_in, fan_out, activation_name) in enumerate(
            zip(self.layer_dims[:-1], self.layer_dims[1:], self.activations),
            start=1,
        ):
            if activation_name == "relu":
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)

            # W[l] has shape (d_l, d_{l-1}); b[l] has shape (1, d_l).
            parameters[f"W{layer_index}"] = self.rng.normal(
                loc=0.0,
                scale=scale,
                size=(fan_out, fan_in),
            )
            parameters[f"b{layer_index}"] = np.zeros((1, fan_out), dtype=float)
        return parameters

    def _activation_forward(self, z: Array, activation_name: str) -> Array:
        if activation_name == "softmax":
            return softmax(z)
        try:
            return ACTIVATIONS[activation_name].forward(z)
        except KeyError as exc:
            raise ValueError(f"Unsupported activation: {activation_name}") from exc

    def _activation_backward(self, z: Array, a: Array, activation_name: str) -> Array:
        if activation_name == "softmax":
            raise ValueError("Use the paired softmax-cross-entropy shortcut at the output layer.")
        return ACTIVATIONS[activation_name].backward(z, a)

    def forward(self, x: Array) -> tuple[Array, list[dict[str, Array]]]:
        a_prev = np.asarray(x, dtype=float)
        caches: list[dict[str, Array]] = []

        for layer_index, activation_name in enumerate(self.activations, start=1):
            w = self.parameters[f"W{layer_index}"]
            b = self.parameters[f"b{layer_index}"]

            # A_prev: (n, d_{l-1}), W: (d_l, d_{l-1}), b: (1, d_l), Z: (n, d_l)
            z = a_prev @ w.T + b
            a = self._activation_forward(z, activation_name)
            caches.append(
                {
                    "A_prev": a_prev,
                    "Z": z,
                    "A": a,
                    "W": w,
                    "b": b,
                    "activation": np.array([activation_name], dtype=object),
                }
            )
            a_prev = a

        return a_prev, caches

    def compute_loss(self, y_pred: Array, y_true: Array) -> float:
        eps = 1e-12
        if self.loss == "binary_cross_entropy":
            y = np.asarray(y_true, dtype=float).reshape(-1, 1)
            p = np.clip(y_pred, eps, 1.0 - eps)
            return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

        if self.loss == "cross_entropy":
            y = np.asarray(y_true, dtype=float)
            p = np.clip(y_pred, eps, 1.0)
            return float(-np.mean(np.sum(y * np.log(p), axis=1)))

        residual = y_pred - np.asarray(y_true, dtype=float)
        return float(0.5 * np.mean(np.sum(np.square(residual), axis=1)))

    def backward(self, y_true: Array, caches: list[dict[str, Array]]) -> dict[str, Array]:
        grads: dict[str, Array] = {}
        batch_size = caches[-1]["A"].shape[0]
        final_cache = caches[-1]
        y_pred = final_cache["A"]

        if self.loss == "binary_cross_entropy":
            y = np.asarray(y_true, dtype=float).reshape(-1, 1)
            delta = y_pred - y
        elif self.loss == "cross_entropy":
            y = np.asarray(y_true, dtype=float)
            delta = y_pred - y
        else:
            y = np.asarray(y_true, dtype=float)
            output_activation = str(final_cache["activation"][0])
            activation_grad = self._activation_backward(final_cache["Z"], y_pred, output_activation)
            delta = (y_pred - y) * activation_grad

        for reverse_index, cache in enumerate(reversed(caches), start=1):
            layer_index = len(caches) - reverse_index + 1
            a_prev = cache["A_prev"]
            w = cache["W"]

            # dW: (d_l, d_{l-1}) from delta^T @ A_prev; db: (1, d_l) from row sum.
            grads[f"dW{layer_index}"] = delta.T @ a_prev / batch_size
            grads[f"db{layer_index}"] = np.sum(delta, axis=0, keepdims=True) / batch_size

            if layer_index == 1:
                continue

            prev_cache = caches[layer_index - 2]
            prev_activation = str(prev_cache["activation"][0])

            # delta_prev_linear has shape (n, d_{l-1}) because delta: (n, d_l), W: (d_l, d_{l-1}).
            delta_prev_linear = delta @ w
            delta = delta_prev_linear * self._activation_backward(
                prev_cache["Z"],
                prev_cache["A"],
                prev_activation,
            )

        return grads

    def update_parameters(self, grads: dict[str, Array], learning_rate: float) -> None:
        for layer_index in range(1, len(self.layer_dims)):
            self.parameters[f"W{layer_index}"] -= learning_rate * grads[f"dW{layer_index}"]
            self.parameters[f"b{layer_index}"] -= learning_rate * grads[f"db{layer_index}"]

    def predict_proba(self, x: Array) -> Array:
        y_pred, _ = self.forward(x)
        return y_pred

    def predict(self, x: Array) -> Array:
        y_pred = self.predict_proba(x)
        if self.loss == "binary_cross_entropy":
            return (y_pred >= 0.5).astype(int).reshape(-1)
        if self.loss == "cross_entropy":
            return np.argmax(y_pred, axis=1)
        return y_pred

    def accuracy(self, x: Array, y_true: Array) -> float:
        preds = self.predict(x)
        truth = np.asarray(y_true).reshape(-1)
        return float(np.mean(preds == truth))

    def fit(
        self,
        x: Array,
        y_true: Array,
        epochs: int = 2000,
        learning_rate: float = 0.1,
        record_every: int = 50,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        history = {"epoch": [], "loss": [], "accuracy": []}

        for epoch in range(1, epochs + 1):
            y_pred, caches = self.forward(x)
            loss = self.compute_loss(y_pred, y_true)
            grads = self.backward(y_true, caches)
            self.update_parameters(grads, learning_rate)

            if epoch == 1 or epoch % record_every == 0 or epoch == epochs:
                history["epoch"].append(epoch)
                history["loss"].append(loss)
                if self.loss in {"binary_cross_entropy", "cross_entropy"}:
                    history["accuracy"].append(self.accuracy(x, y_true))
                else:
                    history["accuracy"].append(float("nan"))
                if verbose:
                    print(
                        f"epoch={epoch:4d} loss={history['loss'][-1]:.4f} "
                        f"accuracy={history['accuracy'][-1]:.3f}"
                    )

        return history

    def loss_for_current_parameters(self, x: Array, y_true: Array) -> float:
        y_pred, _ = self.forward(x)
        return self.compute_loss(y_pred, y_true)

    def gradient_check(
        self,
        x: Array,
        y_true: Array,
        epsilon: float = 1e-6,
        atol: float = 1e-6,
        rtol: float = 1e-4,
        max_checks_per_array: int = 8,
    ) -> dict[str, float]:
        y_pred, caches = self.forward(x)
        _ = y_pred
        analytic_grads = self.backward(y_true, caches)
        report: dict[str, float] = {}

        for layer_index in range(1, len(self.layer_dims)):
            for param_name, grad_name in (
                (f"W{layer_index}", f"dW{layer_index}"),
                (f"b{layer_index}", f"db{layer_index}"),
            ):
                parameter = self.parameters[param_name]
                analytic = analytic_grads[grad_name]
                numeric = np.zeros_like(parameter)
                flat_indices = list(np.ndindex(parameter.shape))[:max_checks_per_array]

                for index in flat_indices:
                    original = parameter[index]
                    parameter[index] = original + epsilon
                    loss_plus = self.loss_for_current_parameters(x, y_true)
                    parameter[index] = original - epsilon
                    loss_minus = self.loss_for_current_parameters(x, y_true)
                    parameter[index] = original
                    numeric[index] = (loss_plus - loss_minus) / (2.0 * epsilon)

                mask = np.zeros_like(parameter, dtype=bool)
                for index in flat_indices:
                    mask[index] = True

                numerator = np.linalg.norm((analytic - numeric)[mask])
                denominator = np.linalg.norm(analytic[mask]) + np.linalg.norm(numeric[mask]) + 1e-12
                relative_error = float(numerator / denominator)
                report[param_name] = relative_error

                if not np.allclose(analytic[mask], numeric[mask], atol=atol, rtol=rtol):
                    raise AssertionError(
                        "Gradient check failed for "
                        f"{param_name}: relative_error={relative_error:.3e}"
                    )

        return report
