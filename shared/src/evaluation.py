"""Reusable evaluation helpers for curriculum labs and notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

ArrayLike = Any


@dataclass(frozen=True)
class CurveData:
    """Container for threshold curve data and its area metric."""

    x: np.ndarray
    y: np.ndarray
    thresholds: np.ndarray
    area: float


def _to_numpy(array: ArrayLike) -> np.ndarray:
    return np.asarray(array)


def train_valid_test_split(
    X: ArrayLike,
    y: ArrayLike,
    *,
    train_size: float = 0.6,
    valid_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int | None = None,
    stratify: ArrayLike | None = None,
) -> dict[str, np.ndarray]:
    """Split arrays into train, validation, and test partitions."""

    total = train_size + valid_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError("train_size, valid_size, and test_size must sum to 1.0.")

    X_array = _to_numpy(X)
    y_array = _to_numpy(y)
    stratify_array = None if stratify is None else _to_numpy(stratify)

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_array,
        y_array,
        test_size=(valid_size + test_size),
        random_state=random_state,
        stratify=stratify_array,
    )

    holdout_ratio = test_size / (valid_size + test_size)
    holdout_stratify = None
    if stratify_array is not None:
        _, holdout_stratify = train_test_split(
            stratify_array,
            test_size=(valid_size + test_size),
            random_state=random_state,
            stratify=stratify_array,
        )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=holdout_ratio,
        random_state=random_state,
        stratify=holdout_stratify,
    )

    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
    }


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Return standard regression metrics."""

    y_true_array = _to_numpy(y_true)
    y_pred_array = _to_numpy(y_pred)
    mse = mean_squared_error(y_true_array, y_pred_array)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "r2": float(r2_score(y_true_array, y_pred_array)),
    }


def classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    y_score: ArrayLike | None = None,
    positive_label: int | str = 1,
) -> dict[str, float]:
    """Return standard binary classification metrics."""

    y_true_array = _to_numpy(y_true)
    y_pred_array = _to_numpy(y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision": float(
            precision_score(y_true_array, y_pred_array, pos_label=positive_label, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true_array, y_pred_array, pos_label=positive_label, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true_array, y_pred_array, pos_label=positive_label, zero_division=0)
        ),
    }

    if y_score is not None:
        y_score_array = _to_numpy(y_score)
        metrics["roc_auc"] = float(roc_auc_score(y_true_array, y_score_array))
        precision, recall, _ = precision_recall_curve(
            y_true_array,
            y_score_array,
            pos_label=positive_label,
        )
        metrics["pr_auc"] = float(auc(recall[::-1], precision[::-1]))
        metrics["brier_score"] = float(brier_score_loss(y_true_array, y_score_array))

    return metrics


def confusion_matrix_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: ArrayLike | None = None,
    normalize: str | None = None,
) -> dict[str, np.ndarray]:
    """Return a confusion matrix and axis labels."""

    label_array = None if labels is None else _to_numpy(labels)
    matrix = confusion_matrix(y_true, y_pred, labels=label_array, normalize=normalize)
    if label_array is None:
        label_array = np.unique(np.concatenate([_to_numpy(y_true), _to_numpy(y_pred)]))
    return {"matrix": matrix, "labels": label_array}


def roc_curve_data(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    positive_label: int | str = 1,
) -> CurveData:
    """Return ROC curve coordinates and AUC."""

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=positive_label)
    return CurveData(x=fpr, y=tpr, thresholds=thresholds, area=float(auc(fpr, tpr)))


def precision_recall_curve_data(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    positive_label: int | str = 1,
) -> CurveData:
    """Return precision-recall curve coordinates and area."""

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        y_score,
        pos_label=positive_label,
    )
    return CurveData(
        x=recall,
        y=precision,
        thresholds=thresholds,
        area=float(auc(recall[::-1], precision[::-1])),
    )


def calibration_curve_data(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict[str, np.ndarray | float]:
    """Return reliability-curve data and the Brier score."""

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy=strategy,
    )
    return {
        "fraction_of_positives": fraction_of_positives,
        "mean_predicted_value": mean_predicted_value,
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def _resolve_axis(ax: plt.Axes | None) -> tuple[plt.Figure, plt.Axes]:
    if ax is not None:
        return ax.figure, ax
    figure, axis = plt.subplots(figsize=(5.5, 4.0))
    return figure, axis


def plot_roc_curve(
    curve: CurveData,
    *,
    ax: plt.Axes | None = None,
    title: str = "ROC Curve",
) -> plt.Axes:
    """Plot ROC data on a matplotlib axis."""

    _, axis = _resolve_axis(ax)
    axis.plot(curve.x, curve.y, label=f"AUC = {curve.area:.3f}")
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="tab:gray", label="random baseline")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title(title)
    axis.legend()
    return axis


def plot_precision_recall_curve(
    curve: CurveData,
    *,
    baseline: float | None = None,
    ax: plt.Axes | None = None,
    title: str = "Precision-Recall Curve",
) -> plt.Axes:
    """Plot precision-recall data on a matplotlib axis."""

    _, axis = _resolve_axis(ax)
    axis.plot(curve.x, curve.y, label=f"AUC = {curve.area:.3f}")
    if baseline is not None:
        axis.axhline(baseline, linestyle="--", color="tab:gray", label="prevalence baseline")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(title)
    axis.legend()
    return axis


def plot_calibration_curve(
    calibration: dict[str, np.ndarray | float],
    *,
    ax: plt.Axes | None = None,
    title: str = "Calibration Plot",
) -> plt.Axes:
    """Plot a reliability diagram."""

    _, axis = _resolve_axis(ax)
    axis.plot(
        calibration["mean_predicted_value"],
        calibration["fraction_of_positives"],
        marker="o",
        label=f"Brier = {calibration['brier_score']:.3f}",
    )
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="tab:gray", label="perfect calibration")
    axis.set_xlabel("Predicted probability")
    axis.set_ylabel("Observed frequency")
    axis.set_title(title)
    axis.legend()
    return axis


def plot_confusion_matrix(
    report: dict[str, np.ndarray],
    *,
    ax: plt.Axes | None = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
) -> plt.Axes:
    """Plot a confusion matrix heatmap."""

    _, axis = _resolve_axis(ax)
    matrix = report["matrix"]
    labels = report["labels"]
    image = axis.imshow(matrix, cmap=cmap)
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(len(labels)), labels=labels)
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            is_float_like = np.issubdtype(type(value), np.floating)
            text = (
                f"{value:.2f}"
                if is_float_like and not float(value).is_integer()
                else f"{float(value):g}"
            )
            axis.text(col, row, text, ha="center", va="center", color="black")

    return axis
