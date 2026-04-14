"""Reusable training diagnostics helpers for deep-learning labs and notebooks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .evaluation import confusion_matrix_report, plot_confusion_matrix

ArrayLike = Any


@dataclass(frozen=True)
class GradientSnapshot:
    """Gradient norms captured at one training step."""

    step: float
    total_norm: float
    layer_norms: dict[str, float]
    status: str


def _to_numpy(array: ArrayLike) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    if hasattr(array, "detach"):
        array = array.detach()
    if hasattr(array, "cpu"):
        array = array.cpu()
    return np.asarray(array)


def _resolve_axis(ax: plt.Axes | None, *, figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    if ax is not None:
        return ax.figure, ax
    figure, axis = plt.subplots(figsize=figsize)
    return figure, axis


def _named_arrays(named_values: Mapping[str, ArrayLike] | Iterable[tuple[str, ArrayLike]]) -> dict[str, np.ndarray]:
    values = dict(named_values.items() if isinstance(named_values, Mapping) else named_values)
    return {name: _to_numpy(value) for name, value in values.items()}


def _extract_gradients(
    named_values: Mapping[str, ArrayLike] | Iterable[tuple[str, ArrayLike]],
) -> dict[str, np.ndarray]:
    extracted: dict[str, np.ndarray] = {}
    iterator = named_values.items() if isinstance(named_values, Mapping) else named_values
    for name, value in iterator:
        grad = getattr(value, "grad", value)
        if grad is None:
            continue
        extracted[name] = _to_numpy(grad)
    return extracted


def _infer_activation_saturation(values: np.ndarray, saturation_threshold: float) -> float | None:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None
    value_min = float(np.min(finite_values))
    value_max = float(np.max(finite_values))

    if value_min >= -1.0 - 1e-6 and value_max <= 1.0 + 1e-6:
        if value_min >= -1e-6:
            return float(np.mean((finite_values <= 1.0 - saturation_threshold) | (finite_values >= saturation_threshold)))
        return float(np.mean(np.abs(finite_values) >= saturation_threshold))
    return None


def plot_loss_curves(
    train_loss: Sequence[float],
    valid_loss: Sequence[float] | None = None,
    *,
    steps: Sequence[float] | None = None,
    ax: plt.Axes | None = None,
    title: str = "Loss Curves",
) -> plt.Axes:
    """Plot train and optional validation loss against epochs or steps."""

    train_array = _to_numpy(train_loss).reshape(-1)
    if steps is None:
        x_values = np.arange(1, train_array.shape[0] + 1)
    else:
        x_values = _to_numpy(steps).reshape(-1)
    if train_array.shape[0] != x_values.shape[0]:
        raise ValueError("train_loss and steps must have the same length.")

    _, axis = _resolve_axis(ax, figsize=(6.0, 4.0))
    axis.plot(x_values, train_array, label="train loss", linewidth=2.0, color="tab:blue")

    if valid_loss is not None:
        valid_array = _to_numpy(valid_loss).reshape(-1)
        if valid_array.shape[0] != x_values.shape[0]:
            raise ValueError("valid_loss must have the same length as train_loss.")
        axis.plot(x_values, valid_array, label="validation loss", linewidth=2.0, color="tab:orange")

    axis.set_title(title)
    axis.set_xlabel("step" if steps is not None else "epoch")
    axis.set_ylabel("loss")
    axis.grid(alpha=0.3, linestyle="--")
    axis.legend()
    axis.figure.tight_layout()
    return axis


def gradient_norms(
    named_values: Mapping[str, ArrayLike] | Iterable[tuple[str, ArrayLike]],
    *,
    norm_order: float = 2.0,
) -> dict[str, float]:
    """Return the norm of each supplied gradient array or parameter gradient."""

    gradients = _extract_gradients(named_values)
    return {
        name: float(np.linalg.norm(np.ravel(values), ord=norm_order))
        for name, values in gradients.items()
    }


def gradient_diagnostics(
    named_values: Mapping[str, ArrayLike] | Iterable[tuple[str, ArrayLike]],
    *,
    vanish_threshold: float = 1e-7,
    explode_threshold: float = 10.0,
) -> GradientSnapshot:
    """Summarize layerwise gradient norms and flag vanishing or exploding regimes."""

    layer_norms = gradient_norms(named_values)
    norm_values = np.fromiter(layer_norms.values(), dtype=float) if layer_norms else np.array([], dtype=float)
    total_norm = float(np.linalg.norm(norm_values, ord=2)) if norm_values.size else 0.0

    if norm_values.size == 0:
        status = "missing"
    elif np.any(norm_values >= explode_threshold):
        status = "exploding"
    elif np.all(norm_values <= vanish_threshold):
        status = "vanishing"
    else:
        status = "healthy"

    return GradientSnapshot(
        step=float("nan"),
        total_norm=total_norm,
        layer_norms=layer_norms,
        status=status,
    )


def append_gradient_snapshot(
    history: dict[str, list[float]],
    named_values: Mapping[str, ArrayLike] | Iterable[tuple[str, ArrayLike]],
    *,
    step: float,
    vanish_threshold: float = 1e-7,
    explode_threshold: float = 10.0,
) -> dict[str, list[float]]:
    """Append one gradient-norm snapshot to a mutable history dictionary."""

    snapshot = gradient_diagnostics(
        named_values,
        vanish_threshold=vanish_threshold,
        explode_threshold=explode_threshold,
    )
    history.setdefault("step", []).append(float(step))
    history.setdefault("total_norm", []).append(snapshot.total_norm)
    for name, norm_value in snapshot.layer_norms.items():
        history.setdefault(name, []).append(norm_value)
    return history


def plot_gradient_norms(
    history: Mapping[str, Sequence[float]],
    *,
    ax: plt.Axes | None = None,
    title: str = "Gradient Norms",
    log_scale: bool = True,
) -> plt.Axes:
    """Plot tracked gradient norms over training."""

    steps = _to_numpy(history["step"]).reshape(-1)
    _, axis = _resolve_axis(ax, figsize=(6.0, 4.0))

    for name, series in history.items():
        if name == "step":
            continue
        values = _to_numpy(series).reshape(-1)
        if values.shape[0] != steps.shape[0]:
            raise ValueError(f"Gradient history for {name!r} does not match step count.")
        axis.plot(steps, values, label=name, linewidth=1.8)

    axis.set_title(title)
    axis.set_xlabel("step")
    axis.set_ylabel("gradient norm")
    if log_scale:
        axis.set_yscale("log")
    axis.grid(alpha=0.3, linestyle="--")
    axis.legend(loc="best", fontsize=8)
    axis.figure.tight_layout()
    return axis


def activation_summary(
    activations: Mapping[str, ArrayLike] | Iterable[tuple[str, ArrayLike]],
    *,
    saturation_threshold: float = 0.95,
) -> dict[str, dict[str, float | None]]:
    """Return summary statistics for hidden activations."""

    activation_arrays = _named_arrays(activations)
    summary: dict[str, dict[str, float | None]] = {}
    for name, values in activation_arrays.items():
        flattened = np.ravel(values.astype(float))
        finite_values = flattened[np.isfinite(flattened)]
        if finite_values.size == 0:
            summary[name] = {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "saturation_fraction": None,
            }
            continue
        summary[name] = {
            "mean": float(np.mean(finite_values)),
            "std": float(np.std(finite_values)),
            "min": float(np.min(finite_values)),
            "max": float(np.max(finite_values)),
            "saturation_fraction": _infer_activation_saturation(finite_values, saturation_threshold),
        }
    return summary


def plot_activation_histograms(
    activations: Mapping[str, ArrayLike] | Iterable[tuple[str, ArrayLike]],
    *,
    bins: int = 40,
    title: str = "Activation Distributions",
    saturation_threshold: float = 0.95,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot one histogram per activation tensor and annotate saturation when detectable."""

    activation_arrays = _named_arrays(activations)
    summary = activation_summary(activation_arrays, saturation_threshold=saturation_threshold)
    num_plots = max(1, len(activation_arrays))
    cols = min(2, num_plots)
    rows = ceil(num_plots / cols)
    figure, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.8 * rows), squeeze=False)
    flat_axes = axes.ravel()

    for axis, (name, values) in zip(flat_axes, activation_arrays.items(), strict=False):
        flattened = np.ravel(values.astype(float))
        finite_values = flattened[np.isfinite(flattened)]
        axis.hist(finite_values, bins=bins, color="tab:blue", alpha=0.8, edgecolor="white")
        saturation = summary[name]["saturation_fraction"]
        suffix = ""
        if saturation is not None:
            suffix = f"\n saturation={saturation:.2%}"
        axis.set_title(f"{name}{suffix}")
        axis.set_xlabel("activation value")
        axis.set_ylabel("count")
        axis.grid(alpha=0.2, linestyle="--")

    for axis in flat_axes[len(activation_arrays) :]:
        axis.set_visible(False)

    figure.suptitle(title, fontsize=13)
    figure.tight_layout()
    return figure, axes


def plot_learning_rates(
    learning_rates: Sequence[float],
    *,
    steps: Sequence[float] | None = None,
    ax: plt.Axes | None = None,
    title: str = "Learning Rate Schedule",
) -> plt.Axes:
    """Visualize the learning-rate trajectory used during training."""

    lr_values = _to_numpy(learning_rates).reshape(-1)
    x_values = np.arange(1, lr_values.shape[0] + 1) if steps is None else _to_numpy(steps).reshape(-1)
    if lr_values.shape[0] != x_values.shape[0]:
        raise ValueError("learning_rates and steps must have the same length.")

    _, axis = _resolve_axis(ax, figsize=(6.0, 3.6))
    axis.plot(x_values, lr_values, color="tab:green", linewidth=2.0)
    axis.set_title(title)
    axis.set_xlabel("step" if steps is not None else "epoch")
    axis.set_ylabel("learning rate")
    axis.grid(alpha=0.3, linestyle="--")
    axis.figure.tight_layout()
    return axis


def plot_confusion_diagnostics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: ArrayLike | None = None,
    normalize: str | None = None,
    ax: plt.Axes | None = None,
    title: str = "Confusion Matrix",
) -> plt.Axes:
    """Render a confusion matrix using the shared evaluation plotting style."""

    report = confusion_matrix_report(y_true, y_pred, labels=labels, normalize=normalize)
    return plot_confusion_matrix(report, ax=ax, title=title)


__all__ = [
    "GradientSnapshot",
    "activation_summary",
    "append_gradient_snapshot",
    "gradient_diagnostics",
    "gradient_norms",
    "plot_activation_histograms",
    "plot_confusion_diagnostics",
    "plot_gradient_norms",
    "plot_learning_rates",
    "plot_loss_curves",
]
