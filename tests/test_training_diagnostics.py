from __future__ import annotations

import matplotlib
import numpy as np

from shared.src.training_diagnostics import (
    activation_summary,
    append_gradient_snapshot,
    gradient_diagnostics,
    gradient_norms,
    plot_activation_histograms,
    plot_confusion_diagnostics,
    plot_gradient_norms,
    plot_learning_rates,
    plot_loss_curves,
)

matplotlib.use("Agg")


def test_gradient_norm_helpers_track_values_and_status() -> None:
    healthy = {"layer1": np.array([0.4, -0.2]), "layer2": np.array([0.1, 0.3])}
    exploding = {"layer1": np.array([12.0, -1.0]), "layer2": np.array([0.5])}
    vanishing = {"layer1": np.array([1e-9, -2e-9]), "layer2": np.array([4e-10])}

    healthy_norms = gradient_norms(healthy)
    healthy_snapshot = gradient_diagnostics(healthy)
    exploding_snapshot = gradient_diagnostics(exploding, explode_threshold=10.0)
    vanishing_snapshot = gradient_diagnostics(vanishing, vanish_threshold=1e-7)

    assert set(healthy_norms) == {"layer1", "layer2"}
    assert healthy_snapshot.status == "healthy"
    assert exploding_snapshot.status == "exploding"
    assert vanishing_snapshot.status == "vanishing"
    assert healthy_snapshot.total_norm > 0.0


def test_append_gradient_snapshot_and_plot_gradient_norms_are_usable() -> None:
    history: dict[str, list[float]] = {}
    append_gradient_snapshot(history, {"encoder": np.array([0.5, 0.25])}, step=1)
    append_gradient_snapshot(history, {"encoder": np.array([0.2, 0.1])}, step=2)

    axis = plot_gradient_norms(history)

    assert history["step"] == [1.0, 2.0]
    assert len(axis.lines) == 2
    assert axis.get_title() == "Gradient Norms"


def test_activation_summary_detects_saturation_for_bounded_activations() -> None:
    activations = {
        "sigmoid_head": np.array([0.001, 0.02, 0.95, 0.998]),
        "tanh_hidden": np.array([-0.99, -0.97, 0.02, 0.98]),
    }

    summary = activation_summary(activations, saturation_threshold=0.95)
    figure, axes = plot_activation_histograms(activations, saturation_threshold=0.95)

    assert summary["sigmoid_head"]["saturation_fraction"] == 1.0
    assert summary["tanh_hidden"]["saturation_fraction"] == 0.75
    assert figure._suptitle.get_text() == "Activation Distributions"
    assert axes.shape == (1, 2)


def test_plot_loss_learning_rate_and_confusion_helpers_return_axes() -> None:
    loss_axis = plot_loss_curves([1.2, 0.8, 0.5], [1.3, 0.9, 0.6])
    lr_axis = plot_learning_rates([1e-3, 8e-4, 5e-4])
    confusion_axis = plot_confusion_diagnostics(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
    )

    assert loss_axis.get_title() == "Loss Curves"
    assert lr_axis.get_title() == "Learning Rate Schedule"
    assert confusion_axis.get_title() == "Confusion Matrix"
