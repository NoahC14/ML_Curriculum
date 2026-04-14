from __future__ import annotations

import numpy as np

KARATE_EDGES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 10),
    (0, 11),
    (0, 12),
    (0, 13),
    (0, 17),
    (0, 19),
    (0, 21),
    (0, 31),
    (1, 2),
    (1, 3),
    (1, 7),
    (1, 13),
    (1, 17),
    (1, 19),
    (1, 21),
    (1, 30),
    (2, 3),
    (2, 7),
    (2, 8),
    (2, 9),
    (2, 13),
    (2, 27),
    (2, 28),
    (2, 32),
    (3, 7),
    (3, 12),
    (3, 13),
    (4, 6),
    (4, 10),
    (5, 6),
    (5, 10),
    (5, 16),
    (6, 16),
    (8, 30),
    (8, 32),
    (8, 33),
    (9, 33),
    (13, 33),
    (14, 32),
    (14, 33),
    (15, 32),
    (15, 33),
    (18, 32),
    (18, 33),
    (19, 33),
    (20, 32),
    (20, 33),
    (22, 32),
    (22, 33),
    (23, 25),
    (23, 27),
    (23, 29),
    (23, 32),
    (23, 33),
    (24, 25),
    (24, 27),
    (24, 31),
    (25, 31),
    (26, 29),
    (26, 33),
    (27, 33),
    (28, 31),
    (28, 33),
    (29, 32),
    (29, 33),
    (30, 32),
    (30, 33),
    (31, 32),
    (31, 33),
    (32, 33),
]

KARATE_LABELS = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ],
    dtype=np.int64,
)


def build_adjacency(
    num_nodes: int,
    edges: list[tuple[int, int]],
    add_self_loops: bool = False,
) -> np.ndarray:
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for source, target in edges:
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0
    if add_self_loops:
        adjacency = adjacency + np.eye(num_nodes, dtype=np.float64)
    return adjacency


def degree_matrix(adjacency: np.ndarray) -> np.ndarray:
    return np.diag(adjacency.sum(axis=1))


def normalized_adjacency(
    adjacency: np.ndarray,
    add_self_loops: bool = True,
) -> np.ndarray:
    if add_self_loops:
        propagated = adjacency + np.eye(adjacency.shape[0], dtype=np.float64)
    else:
        propagated = adjacency.copy()
    degrees = propagated.sum(axis=1)
    inv_sqrt = np.zeros_like(degrees)
    nonzero = degrees > 0
    inv_sqrt[nonzero] = degrees[nonzero] ** -0.5
    return inv_sqrt[:, None] * propagated * inv_sqrt[None, :]


def normalized_laplacian(adjacency: np.ndarray) -> np.ndarray:
    return np.eye(adjacency.shape[0], dtype=np.float64) - normalized_adjacency(
        adjacency,
        add_self_loops=False,
    )


def row_mean_aggregate(
    features: np.ndarray,
    adjacency: np.ndarray,
    include_self: bool = True,
) -> np.ndarray:
    if include_self:
        propagated = adjacency + np.eye(adjacency.shape[0], dtype=np.float64)
    else:
        propagated = adjacency.copy()
    degrees = propagated.sum(axis=1, keepdims=True)
    safe_degrees = np.where(degrees == 0.0, 1.0, degrees)
    return propagated @ features / safe_degrees


def karate_club_data() -> dict[str, np.ndarray]:
    num_nodes = int(KARATE_LABELS.shape[0])
    adjacency = build_adjacency(num_nodes=num_nodes, edges=KARATE_EDGES)
    labels = KARATE_LABELS.copy()
    features = np.eye(num_nodes, dtype=np.float64)

    train_nodes = np.array([0, 1, 2, 3, 30, 31, 32, 33], dtype=np.int64)
    val_nodes = np.array([8, 13, 19, 23, 24, 25], dtype=np.int64)
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[train_nodes] = True
    val_mask[val_nodes] = True
    test_mask = ~(train_mask | val_mask)

    return {
        "adjacency": adjacency,
        "features": features,
        "labels": labels,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }


def relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / exp_shifted.sum(axis=1, keepdims=True)


def accuracy(predictions: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    return float((predictions[mask] == labels[mask]).mean())


def train_two_layer_gcn(
    adjacency: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    hidden_dim: int = 16,
    learning_rate: float = 0.6,
    weight_decay: float = 5e-4,
    epochs: int = 400,
    seed: int = 7,
) -> dict[str, np.ndarray | list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    _, input_dim = features.shape
    num_classes = int(labels.max()) + 1
    propagation = normalized_adjacency(adjacency, add_self_loops=True)

    weights_0 = rng.normal(0.0, 0.2, size=(input_dim, hidden_dim))
    weights_1 = rng.normal(0.0, 0.2, size=(hidden_dim, num_classes))
    history: list[dict[str, float]] = []

    feature_mix = propagation @ features
    train_count = int(train_mask.sum())

    for epoch in range(epochs):
        hidden_pre = feature_mix @ weights_0
        hidden = relu(hidden_pre)
        smoothed_hidden = propagation @ hidden
        logits = smoothed_hidden @ weights_1
        probabilities = softmax(logits)

        loss = -np.log(probabilities[train_mask, labels[train_mask]] + 1e-12).mean()
        loss += 0.5 * weight_decay * (np.sum(weights_0 * weights_0) + np.sum(weights_1 * weights_1))

        grad_logits = np.zeros_like(probabilities)
        grad_logits[train_mask] = probabilities[train_mask]
        grad_logits[train_mask, labels[train_mask]] -= 1.0
        grad_logits[train_mask] /= train_count

        grad_w1 = smoothed_hidden.T @ grad_logits + weight_decay * weights_1
        grad_hidden = propagation @ (grad_logits @ weights_1.T)
        grad_hidden_pre = grad_hidden * (hidden_pre > 0.0)
        grad_w0 = feature_mix.T @ grad_hidden_pre + weight_decay * weights_0

        weights_0 -= learning_rate * grad_w0
        weights_1 -= learning_rate * grad_w1

        if epoch in {0, 9, 49, 99, 199, epochs - 1}:
            predictions = probabilities.argmax(axis=1)
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": float(loss),
                    "train_accuracy": accuracy(predictions, labels, train_mask),
                    "val_accuracy": accuracy(predictions, labels, val_mask),
                }
            )

    hidden_pre = feature_mix @ weights_0
    hidden = relu(hidden_pre)
    logits = (propagation @ hidden) @ weights_1
    probabilities = softmax(logits)
    predictions = probabilities.argmax(axis=1)

    return {
        "propagation": propagation,
        "weights_0": weights_0,
        "weights_1": weights_1,
        "logits": logits,
        "probabilities": probabilities,
        "predictions": predictions,
        "history": history,
    }
