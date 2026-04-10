from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class Objective2D:
    name: str
    function: Callable[[Array], float]
    gradient: Callable[[Array], Array]
    hessian: Callable[[Array], Array] | None = None
    sample_gradients: Callable[[Array], Array] | None = None
    minimizer: Array | None = None

    def value(self, point: Array) -> float:
        return float(self.function(np.asarray(point, dtype=float)))

    def grad(self, point: Array) -> Array:
        return np.asarray(self.gradient(np.asarray(point, dtype=float)), dtype=float)

    def hess(self, point: Array) -> Array:
        if self.hessian is None:
            raise ValueError(f"{self.name} does not expose a Hessian.")
        return np.asarray(self.hessian(np.asarray(point, dtype=float)), dtype=float)

    def sample_grads(self, point: Array) -> Array:
        if self.sample_gradients is None:
            raise ValueError(f"{self.name} does not expose sample gradients.")
        return np.asarray(self.sample_gradients(np.asarray(point, dtype=float)), dtype=float)


def make_quadratic(
    matrix: Array,
    center: Array | None = None,
    offset: float = 0.0,
    name: str = "quadratic",
) -> Objective2D:
    matrix = np.asarray(matrix, dtype=float)
    center = np.zeros(2, dtype=float) if center is None else np.asarray(center, dtype=float)

    def function(point: Array) -> float:
        delta = point - center
        return 0.5 * float(delta.T @ matrix @ delta) + offset

    def gradient(point: Array) -> Array:
        return matrix @ (point - center)

    def hessian(_: Array) -> Array:
        return matrix

    return Objective2D(
        name=name,
        function=function,
        gradient=gradient,
        hessian=hessian,
        minimizer=center,
    )


def make_shifted_least_squares(
    centers: Array,
    weights: Array | None = None,
    name: str = "finite_sum_quadratic",
) -> Objective2D:
    centers = np.asarray(centers, dtype=float)
    weights = (
        np.ones(centers.shape[0], dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    normalized_weights = weights / weights.sum()
    minimizer = np.average(centers, axis=0, weights=normalized_weights)

    def function(point: Array) -> float:
        deltas = point - centers
        sample_losses = 0.5 * np.sum(deltas * deltas, axis=1)
        return float(normalized_weights @ sample_losses)

    def gradient(point: Array) -> Array:
        deltas = point - centers
        return normalized_weights @ deltas

    def hessian(_: Array) -> Array:
        return np.eye(2, dtype=float)

    def sample_gradients(point: Array) -> Array:
        return point - centers

    return Objective2D(
        name=name,
        function=function,
        gradient=gradient,
        hessian=hessian,
        sample_gradients=sample_gradients,
        minimizer=minimizer,
    )


def saddle_objective() -> Objective2D:
    def function(point: Array) -> float:
        x, y = point
        return x**2 - y**2

    def gradient(point: Array) -> Array:
        x, y = point
        return np.array([2.0 * x, -2.0 * y], dtype=float)

    def hessian(_: Array) -> Array:
        return np.array([[2.0, 0.0], [0.0, -2.0]], dtype=float)

    return Objective2D(
        name="saddle",
        function=function,
        gradient=gradient,
        hessian=hessian,
        minimizer=np.zeros(2, dtype=float),
    )


def rosenbrock_objective(a: float = 1.0, b: float = 15.0) -> Objective2D:
    def function(point: Array) -> float:
        x, y = point
        return (a - x) ** 2 + b * (y - x**2) ** 2

    def gradient(point: Array) -> Array:
        x, y = point
        return np.array(
            [
                -2.0 * (a - x) - 4.0 * b * x * (y - x**2),
                2.0 * b * (y - x**2),
            ],
            dtype=float,
        )

    def hessian(point: Array) -> Array:
        x, y = point
        return np.array(
            [
                [2.0 - 4.0 * b * y + 12.0 * b * x**2, -4.0 * b * x],
                [-4.0 * b * x, 2.0 * b],
            ],
            dtype=float,
        )

    return Objective2D(
        name="rosenbrock",
        function=function,
        gradient=gradient,
        hessian=hessian,
        minimizer=np.array([a, a**2], dtype=float),
    )


def gradient_descent(
    objective: Objective2D,
    start: Array,
    learning_rate: float,
    steps: int,
) -> dict[str, Array | str]:
    point = np.asarray(start, dtype=float).copy()
    points = [point.copy()]
    values = [objective.value(point)]
    gradients = [objective.grad(point)]

    for _ in range(steps):
        point = point - learning_rate * objective.grad(point)
        points.append(point.copy())
        values.append(objective.value(point))
        gradients.append(objective.grad(point))

    return pack_history("Gradient descent", points, values, gradients)


def momentum_descent(
    objective: Objective2D,
    start: Array,
    learning_rate: float,
    momentum: float,
    steps: int,
    nesterov: bool = False,
) -> dict[str, Array | str]:
    point = np.asarray(start, dtype=float).copy()
    velocity = np.zeros_like(point)
    points = [point.copy()]
    values = [objective.value(point)]
    gradients = [objective.grad(point)]

    for _ in range(steps):
        gradient_point = point + momentum * velocity if nesterov else point
        gradient = objective.grad(gradient_point)
        velocity = momentum * velocity - learning_rate * gradient
        point = point + velocity
        points.append(point.copy())
        values.append(objective.value(point))
        gradients.append(objective.grad(point))

    name = "Nesterov momentum" if nesterov else "Heavy-ball momentum"
    return pack_history(name, points, values, gradients)


def stochastic_gradient_descent(
    objective: Objective2D,
    start: Array,
    learning_rate: float,
    steps: int,
    seed: int = 0,
) -> dict[str, Array | str]:
    point = np.asarray(start, dtype=float).copy()
    rng = np.random.default_rng(seed)
    sample_gradients = objective.sample_grads(point)
    sample_count = sample_gradients.shape[0]
    points = [point.copy()]
    values = [objective.value(point)]
    gradients = [objective.grad(point)]
    sample_indices = []

    for _ in range(steps):
        index = int(rng.integers(sample_count))
        sample_indices.append(index)
        gradient = objective.sample_grads(point)[index]
        point = point - learning_rate * gradient
        points.append(point.copy())
        values.append(objective.value(point))
        gradients.append(objective.grad(point))

    history = pack_history("Stochastic gradient descent", points, values, gradients)
    history["sample_indices"] = np.asarray(sample_indices, dtype=int)
    return history


def newton_method(
    objective: Objective2D,
    start: Array,
    steps: int,
    damping: float = 1.0,
    regularization: float = 0.0,
) -> dict[str, Array | str]:
    point = np.asarray(start, dtype=float).copy()
    points = [point.copy()]
    values = [objective.value(point)]
    gradients = [objective.grad(point)]

    for _ in range(steps):
        hessian = objective.hess(point) + regularization * np.eye(2, dtype=float)
        direction = np.linalg.solve(hessian, objective.grad(point))
        point = point - damping * direction
        points.append(point.copy())
        values.append(objective.value(point))
        gradients.append(objective.grad(point))

    return pack_history("Newton method", points, values, gradients)


def adagrad(
    objective: Objective2D,
    start: Array,
    learning_rate: float,
    steps: int,
    epsilon: float = 1e-8,
) -> dict[str, Array | str]:
    point = np.asarray(start, dtype=float).copy()
    accumulator = np.zeros_like(point)
    points = [point.copy()]
    values = [objective.value(point)]
    gradients = [objective.grad(point)]

    for _ in range(steps):
        gradient = objective.grad(point)
        accumulator = accumulator + gradient**2
        point = point - learning_rate * gradient / (np.sqrt(accumulator) + epsilon)
        points.append(point.copy())
        values.append(objective.value(point))
        gradients.append(objective.grad(point))

    return pack_history("AdaGrad", points, values, gradients)


def rmsprop(
    objective: Objective2D,
    start: Array,
    learning_rate: float,
    steps: int,
    decay: float = 0.9,
    epsilon: float = 1e-8,
) -> dict[str, Array | str]:
    point = np.asarray(start, dtype=float).copy()
    second_moment = np.zeros_like(point)
    points = [point.copy()]
    values = [objective.value(point)]
    gradients = [objective.grad(point)]

    for _ in range(steps):
        gradient = objective.grad(point)
        second_moment = decay * second_moment + (1.0 - decay) * gradient**2
        point = point - learning_rate * gradient / (np.sqrt(second_moment) + epsilon)
        points.append(point.copy())
        values.append(objective.value(point))
        gradients.append(objective.grad(point))

    return pack_history("RMSProp", points, values, gradients)


def adam(
    objective: Objective2D,
    start: Array,
    learning_rate: float,
    steps: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> dict[str, Array | str]:
    point = np.asarray(start, dtype=float).copy()
    first_moment = np.zeros_like(point)
    second_moment = np.zeros_like(point)
    points = [point.copy()]
    values = [objective.value(point)]
    gradients = [objective.grad(point)]

    for step in range(1, steps + 1):
        gradient = objective.grad(point)
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * gradient**2
        first_unbiased = first_moment / (1.0 - beta1**step)
        second_unbiased = second_moment / (1.0 - beta2**step)
        point = point - learning_rate * first_unbiased / (np.sqrt(second_unbiased) + epsilon)
        points.append(point.copy())
        values.append(objective.value(point))
        gradients.append(objective.grad(point))

    return pack_history("Adam", points, values, gradients)


def pack_history(
    name: str,
    points: list[Array],
    values: list[float],
    gradients: list[Array],
) -> dict[str, Array | str]:
    gradient_norms = np.linalg.norm(np.vstack(gradients), axis=1)
    return {
        "name": name,
        "points": np.vstack(points),
        "values": np.asarray(values, dtype=float),
        "gradients": np.vstack(gradients),
        "gradient_norms": gradient_norms,
    }


def contour_grid(
    objective: Objective2D,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: int = 220,
) -> tuple[Array, Array, Array]:
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.zeros_like(xx)

    for i in range(resolution):
        for j in range(resolution):
            zz[i, j] = objective.value(np.array([xx[i, j], yy[i, j]], dtype=float))

    return xx, yy, zz


def plot_trajectories(
    objective: Objective2D,
    histories: list[dict[str, Array | str]],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    title: str,
    levels: int = 20,
) -> tuple[plt.Figure, Array]:
    xx, yy, zz = contour_grid(objective, x_range, y_range)
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    contours = ax.contour(xx, yy, zz, levels=levels, cmap="viridis")
    ax.clabel(contours, inline=True, fontsize=7, fmt="%.2f")

    for history in histories:
        points = np.asarray(history["points"])
        ax.plot(
            points[:, 0],
            points[:, 1],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=history["name"],
        )
        ax.scatter(points[0, 0], points[0, 1], color=ax.lines[-1].get_color(), marker="x", s=60)

    if objective.minimizer is not None:
        ax.scatter(
            objective.minimizer[0],
            objective.minimizer[1],
            color="black",
            marker="*",
            s=100,
            label="target",
        )

    ax.set_title(title)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def plot_convergence(
    histories: list[dict[str, Array | str]],
    title: str,
    value_label: str = "objective",
) -> tuple[plt.Figure, Array]:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

    for history in histories:
        iterations = np.arange(len(np.asarray(history["values"])))
        axes[0].plot(iterations, history["values"], linewidth=2, label=history["name"])
        axes[1].plot(iterations, history["gradient_norms"], linewidth=2, label=history["name"])

    axes[0].set_title(f"{title}: {value_label}")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel(value_label)
    axes[0].set_yscale("log")
    axes[1].set_title(f"{title}: gradient norm")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel(r"$||\\nabla f(x_k)||_2$")
    axes[1].set_yscale("log")
    axes[1].legend(loc="best")
    fig.tight_layout()
    return fig, axes


def summarize_history(
    history: dict[str, Array | str],
    objective: Objective2D,
) -> dict[str, float | int]:
    points = np.asarray(history["points"])
    values = np.asarray(history["values"])
    summary = {
        "steps": int(len(points) - 1),
        "final_x1": float(points[-1, 0]),
        "final_x2": float(points[-1, 1]),
        "final_value": float(values[-1]),
        "initial_value": float(values[0]),
        "improvement": float(values[0] - values[-1]),
        "final_grad_norm": float(np.linalg.norm(objective.grad(points[-1]))),
    }
    if objective.minimizer is not None:
        summary["distance_to_target"] = float(np.linalg.norm(points[-1] - objective.minimizer))
    return summary
