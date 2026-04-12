from __future__ import annotations

import matplotlib
import numpy as np

from shared.src.evaluation import (
    calibration_curve_data,
    classification_metrics,
    confusion_matrix_report,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    precision_recall_curve_data,
    regression_metrics,
    roc_curve_data,
    train_valid_test_split,
)

matplotlib.use("Agg")


def test_train_valid_test_split_respects_sizes() -> None:
    X = np.arange(30).reshape(15, 2)
    y = np.array([0, 1, 0, 1, 0] * 3)

    split = train_valid_test_split(X, y, random_state=5, stratify=y)

    assert split["X_train"].shape == (9, 2)
    assert split["X_valid"].shape == (3, 2)
    assert split["X_test"].shape == (3, 2)
    assert split["y_train"].shape == (9,)
    assert split["y_valid"].shape == (3,)
    assert split["y_test"].shape == (3,)


def test_regression_metrics_match_known_values() -> None:
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    metrics = regression_metrics(y_true, y_pred)

    assert metrics["mse"] == 0.375
    assert metrics["rmse"] == np.sqrt(0.375)
    assert metrics["mae"] == 0.5
    assert np.isclose(metrics["r2"], 0.9486081370449679)


def test_classification_metrics_include_auc_scores() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_score = np.array([0.1, 0.6, 0.7, 0.9])

    metrics = classification_metrics(y_true, y_pred, y_score=y_score)

    assert metrics["accuracy"] == 0.75
    assert np.isclose(metrics["precision"], 2.0 / 3.0)
    assert metrics["recall"] == 1.0
    assert np.isclose(metrics["f1"], 0.8)
    assert metrics["roc_auc"] == 1.0
    assert np.isclose(metrics["pr_auc"], 1.0)
    assert np.isclose(metrics["brier_score"], 0.1175)


def test_curve_and_calibration_helpers_return_consistent_shapes() -> None:
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_score = np.array([0.05, 0.2, 0.55, 0.7, 0.95, 0.4])

    roc = roc_curve_data(y_true, y_score)
    pr = precision_recall_curve_data(y_true, y_score)
    calibration = calibration_curve_data(y_true, y_score, n_bins=3)

    assert roc.x.shape == roc.y.shape == roc.thresholds.shape
    assert pr.x.shape == pr.y.shape
    assert pr.thresholds.shape[0] == pr.x.shape[0] - 1
    assert 0.0 <= roc.area <= 1.0
    assert 0.0 <= pr.area <= 1.0
    assert calibration["fraction_of_positives"].shape == calibration["mean_predicted_value"].shape
    assert 0.0 <= calibration["brier_score"] <= 1.0


def test_confusion_matrix_report_and_plot_helpers_are_usable() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_score = np.array([0.1, 0.6, 0.7, 0.9])

    report = confusion_matrix_report(y_true, y_pred)
    roc = roc_curve_data(y_true, y_score)
    pr = precision_recall_curve_data(y_true, y_score)
    calibration = calibration_curve_data(y_true, y_score, n_bins=2)

    assert report["matrix"].shape == (2, 2)
    assert report["labels"].tolist() == [0, 1]

    roc_ax = plot_roc_curve(roc)
    pr_ax = plot_precision_recall_curve(pr, baseline=float(np.mean(y_true)))
    calibration_ax = plot_calibration_curve(calibration)
    confusion_ax = plot_confusion_matrix(report)

    assert roc_ax.get_title() == "ROC Curve"
    assert pr_ax.get_title() == "Precision-Recall Curve"
    assert calibration_ax.get_title() == "Calibration Plot"
    assert confusion_ax.get_title() == "Confusion Matrix"
