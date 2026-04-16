---
title: "Neural Network Exercises"
module: "06-neural-networks"
lesson: "neural-network-exercises"
doc_type: "exercise"
topic: "mlps-backprop-initialization"
status: "draft"
prerequisites:
  - "00-math-toolkit/multivariable-calculus"
  - "00-math-toolkit/chain-rule-matrices"
  - "01-optimization/convexity-and-optimization"
  - "02-statistical-learning/statistical-learning-foundations"
  - "06-neural-networks/neural-networks-first-principles"
  - "06-neural-networks/backpropagation"
  - "06-neural-networks/initialization-and-normalization"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "neural-networks"
  - "backpropagation"
  - "activation-functions"
  - "loss-functions"
  - "initialization"
  - "normalization"
---

## Purpose

These exercises reinforce the conceptual and algebraic core of the module:
perceptrons, multilayer perceptrons, activation functions, backpropagation, initialization, normalization, and loss-landscape reasoning.

## Exercise 1: perceptron geometry

**Taxonomy**

- `difficulty`: `foundational`
- `type`: `analysis`
- `tags`: `perceptron`, `decision-boundary`, `geometry`

Let

$$
f(x) = \mathrm{sign}(w^\top x + b)
$$

with $x \in \mathbb{R}^2$.

1. Show that the decision boundary is a line.
2. Explain how the vector $w$ determines the orientation of that line.
3. Explain how the bias $b$ shifts the line.
4. Give a short argument for why a single perceptron cannot represent XOR.

## Exercise 2: why stacked linear layers collapse

Consider a two-layer network with no nonlinearities:

$$
z^{[1]} = W^{[1]}x + b^{[1]},
\qquad
z^{[2]} = W^{[2]}z^{[1]} + b^{[2]}.
$$

1. Rewrite $z^{[2]}$ as a single affine function of $x$.
2. Identify the effective weight matrix and effective bias.
3. Explain why this shows that nonlinear activations are necessary.

## Exercise 3: forward pass with shapes

Let

- $x \in \mathbb{R}^5$,
- $W^{[1]} \in \mathbb{R}^{4 \times 5}$,
- $W^{[2]} \in \mathbb{R}^{3 \times 4}$,
- $W^{[3]} \in \mathbb{R}^{2 \times 3}$.

Assume corresponding bias vectors and coordinatewise activations.

1. Write the forward pass equations for a two-hidden-layer MLP.
2. State the shape of every $z^{[\ell]}$ and $a^{[\ell]}$.
3. Explain why the matrix multiplications are dimensionally valid.

## Exercise 4: activation comparison

Compare sigmoid, tanh, ReLU, and leaky ReLU.

For each activation:

1. write the formula;
2. write or describe its derivative;
3. state whether it saturates;
4. explain one optimization advantage and one limitation.

## Exercise 5: universal approximation interpretation

Answer the following in complete sentences.

1. State the universal approximation theorem informally.
2. Why does the theorem not imply that one-hidden-layer networks are always easy to train?
3. Why does the theorem not imply that more depth is useless?
4. Give a geometric intuition for how ReLU networks partition input space.

## Exercise 6: scalar backpropagation step

Consider one hidden unit

$$
a = \phi(z),
\qquad
z = w^\top x + b,
$$

and scalar loss $\mathcal{L}(a)$.

1. Compute $\frac{\partial \mathcal{L}}{\partial z}$ using the chain rule.
2. Compute $\frac{\partial \mathcal{L}}{\partial w}$.
3. Compute $\frac{\partial \mathcal{L}}{\partial b}$.
4. Explain in words what information is "local" and what information is "upstream."

## Exercise 7: backpropagation on a two-hidden-layer network

Use the notation from the derivation note:

$$
\begin{aligned}
a^{[0]} &= x, \\
z^{[1]} &= W^{[1]}a^{[0]} + b^{[1]}, \qquad a^{[1]} = \phi^{[1]}(z^{[1]}), \\
z^{[2]} &= W^{[2]}a^{[1]} + b^{[2]}, \qquad a^{[2]} = \phi^{[2]}(z^{[2]}), \\
z^{[3]} &= W^{[3]}a^{[2]} + b^{[3]}, \qquad \hat{y} = \psi(z^{[3]}).
\end{aligned}
$$

1. Define $\delta^{[3]}$, $\delta^{[2]}$, and $\delta^{[1]}$.
2. Derive $\frac{\partial \mathcal{L}}{\partial W^{[3]}}$ and $\frac{\partial \mathcal{L}}{\partial b^{[3]}}$.
3. Derive the recursion for $\delta^{[2]}$.
4. Derive the recursion for $\delta^{[1]}$.
5. Write the final formulas for $\frac{\partial \mathcal{L}}{\partial W^{[2]}}$ and $\frac{\partial \mathcal{L}}{\partial W^{[1]}}$.

Do not skip intermediate chain-rule steps.

## Exercise 8: softmax and cross-entropy

Let $z \in \mathbb{R}^C$ be logits, $\hat{y} = \mathrm{softmax}(z)$, and let $y$ be a one-hot target vector.

1. Write the multiclass cross-entropy loss.
2. Show that $\nabla_z \mathcal{L} = \hat{y} - y$.
3. Explain why this simplification is computationally convenient.
4. State one reason it is numerically better to compute this pairing using a stable log-sum-exp implementation.

## Exercise 9: empirical risk and optimization

Let

$$
J(\theta) = \frac{1}{n}\sum_{i=1}^n \mathcal{L}(f_\theta(x_i), y_i).
$$

1. Explain why $J(\theta)$ is generally nonconvex for a neural network.
2. Explain why nonconvexity does not automatically imply failure of gradient-based learning.
3. Describe one role of mini-batching in practical optimization.

## Exercise 10: zero initialization failure

Suppose all weights in the first hidden layer are initialized to zero.

1. Show that all hidden units in that layer produce the same output on every input.
2. Show that they receive the same gradient update.
3. Explain why this symmetry prevents the layer from learning diverse features.

## Exercise 11: Xavier and He scaling

Answer each prompt briefly but precisely.

1. Derive the heuristic condition $d_{\mathrm{in}}\operatorname{Var}(W_{ji}) \approx 1$ from variance propagation.
2. State the Xavier variance rule.
3. State the He variance rule.
4. Explain why He initialization is usually preferred for ReLU-like activations.

## Exercise 12: normalization choices

Compare batch normalization and layer normalization.

1. Along which dimensions is normalization performed in each case?
2. Which method depends directly on batch statistics?
3. Why is layer normalization natural in transformer architectures?
4. Give one situation where batch normalization may be awkward.

## Exercise 13: loss landscapes

Write a short response to each prompt.

1. What is meant by a loss landscape?
2. Why can two-dimensional visualizations be useful even though the true parameter space is high-dimensional?
3. Why should such visualizations be interpreted cautiously?
4. How do initialization and normalization affect the local geometry explored by optimization?

## Exercise 14: computational graph reasoning

Consider a feedforward network as a computational graph.

1. What quantities must be cached in the forward pass to make backpropagation efficient?
2. Why is the loss required to be scalar in the standard training setup?
3. Explain in words why reverse-mode differentiation is preferable to forward-mode differentiation when the number of parameters is very large and the loss is scalar.

## Exercise 15: small implementation exercise

Implement a NumPy function for one hidden layer:

$$
f_\theta(x) = W^{[2]}\phi(W^{[1]}x + b^{[1]}) + b^{[2]}.
$$

1. Write the forward pass.
2. Derive and implement the backward pass by hand.
3. Numerically check one parameter gradient using finite differences.
4. Report the relative error between analytical and numerical gradients.

## Exercise 16: reflection

Write one short paragraph answering both questions.

1. Why is backpropagation the central derivation of deep learning?
2. Which part of the derivation still feels least intuitive, and what additional example would clarify it?
