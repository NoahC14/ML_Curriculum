---
title: "Backpropagation Derivation"
module: "06-neural-networks"
lesson: "backpropagation"
doc_type: "derivation"
topic: "chain-rule-for-mlps"
status: "draft"
prerequisites:
  - "00-math-toolkit/multivariable-calculus"
  - "00-math-toolkit/chain-rule-matrices"
  - "01-optimization/convexity-and-optimization"
  - "06-neural-networks/neural-networks-first-principles"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "neural-networks"
  - "backpropagation"
  - "chain-rule"
  - "computational-graph"
---

## Goal

Derive backpropagation first on a concrete two-hidden-layer multilayer perceptron, then generalize to an arbitrary feedforward network.
The key point is that backpropagation is an organized application of the chain rule to a scalar loss on a computational graph.

## 1. Network setup

Consider one training example $(x,y)$.
Let

$$
x \in \mathbb{R}^{d_0},
\qquad
y \in \mathcal{Y}.
$$

Take a network with two hidden layers of widths $d_1$ and $d_2$, and output width $d_3$.
Define

$$
\begin{aligned}
a^{[0]} &= x, \\
z^{[1]} &= W^{[1]}a^{[0]} + b^{[1]}, \qquad W^{[1]} \in \mathbb{R}^{d_1 \times d_0}, \quad b^{[1]} \in \mathbb{R}^{d_1}, \\
a^{[1]} &= \phi^{[1]}(z^{[1]}), \\
z^{[2]} &= W^{[2]}a^{[1]} + b^{[2]}, \qquad W^{[2]} \in \mathbb{R}^{d_2 \times d_1}, \quad b^{[2]} \in \mathbb{R}^{d_2}, \\
a^{[2]} &= \phi^{[2]}(z^{[2]}), \\
z^{[3]} &= W^{[3]}a^{[2]} + b^{[3]}, \qquad W^{[3]} \in \mathbb{R}^{d_3 \times d_2}, \quad b^{[3]} \in \mathbb{R}^{d_3}, \\
\hat{y} &= \psi(z^{[3]}), \\
\mathcal{L} &= \mathcal{L}(\hat{y}, y).
\end{aligned}
$$

Assume $\phi^{[1]}$ and $\phi^{[2]}$ act coordinatewise.
The output map $\psi$ may be the identity, sigmoid, or softmax depending on the task.

## 2. What must be computed

Training by gradient descent requires

$$
\frac{\partial \mathcal{L}}{\partial W^{[\ell]}}
\qquad \text{and} \qquad
\frac{\partial \mathcal{L}}{\partial b^{[\ell]}}
\quad \text{for each layer } \ell.
$$

Because $\mathcal{L}$ depends on early-layer parameters only through many nested intermediate variables, the chain rule is unavoidable.

The efficient idea is to compute sensitivities with respect to pre-activations:

$$
\delta^{[\ell]} := \frac{\partial \mathcal{L}}{\partial z^{[\ell]}}.
$$

Once $\delta^{[\ell]}$ is known, parameter gradients at layer $\ell$ become simple.

## 3. Output-layer derivative

Start from the last layer.
By the chain rule,

$$
\delta^{[3]}
=
\frac{\partial \mathcal{L}}{\partial z^{[3]}}
=
\frac{\partial \mathcal{L}}{\partial \hat{y}}
\frac{\partial \hat{y}}{\partial z^{[3]}}.
$$

In vector notation, for coordinatewise output activation this is

$$
\delta^{[3]}
=
\nabla_{\hat{y}} \mathcal{L} \odot \psi'(z^{[3]}),
$$

where $\odot$ denotes elementwise multiplication.

Two important special cases simplify further:

- regression with identity output:
  $$
  \delta^{[3]} = \nabla_{\hat{y}} \mathcal{L};
  $$
- softmax with cross-entropy or sigmoid with binary cross-entropy:
  $$
  \delta^{[3]} = \hat{y} - y.
  $$

The last identity is not magic.
It comes from differentiating the paired output activation and log-likelihood loss together.

## 4. Gradients for the final affine map

Now use

$$
z^{[3]} = W^{[3]}a^{[2]} + b^{[3]}.
$$

Coordinatewise,

$$
z^{[3]}_k = \sum_{j=1}^{d_2} W^{[3]}_{kj} a^{[2]}_j + b^{[3]}_k.
$$

For a single weight $W^{[3]}_{kj}$,

$$
\frac{\partial z^{[3]}_r}{\partial W^{[3]}_{kj}}
=
\begin{cases}
a^{[2]}_j, & r = k, \\
0, & r \neq k.
\end{cases}
$$

Therefore

$$
\frac{\partial \mathcal{L}}{\partial W^{[3]}_{kj}}
=
\sum_{r=1}^{d_3}
\frac{\partial \mathcal{L}}{\partial z^{[3]}_r}
\frac{\partial z^{[3]}_r}{\partial W^{[3]}_{kj}}
=
\delta^{[3]}_k a^{[2]}_j.
$$

Collecting all coordinates,

$$
\boxed{
\frac{\partial \mathcal{L}}{\partial W^{[3]}}
=
\delta^{[3]} (a^{[2]})^\top
}
\qquad
\text{with shape } (d_3 \times d_2).
$$

Similarly,

$$
\frac{\partial z^{[3]}_r}{\partial b^{[3]}_k}
=
\begin{cases}
1, & r = k, \\
0, & r \neq k,
\end{cases}
$$

so

$$
\boxed{
\frac{\partial \mathcal{L}}{\partial b^{[3]}}
=
\delta^{[3]}
}.
$$

## 5. Hidden-layer recursion: second hidden layer

We now compute $\delta^{[2]} = \frac{\partial \mathcal{L}}{\partial z^{[2]}}$.

Fix one coordinate $j \in \{1,\dots,d_2\}$.
The variable $z^{[2]}_j$ affects the loss through

$$
z^{[2]}_j \to a^{[2]}_j \to z^{[3]}_1,\dots,z^{[3]}_{d_3} \to \mathcal{L}.
$$

Applying the scalar chain rule carefully,

$$
\frac{\partial \mathcal{L}}{\partial z^{[2]}_j}
=
\sum_{k=1}^{d_3}
\frac{\partial \mathcal{L}}{\partial z^{[3]}_k}
\frac{\partial z^{[3]}_k}{\partial a^{[2]}_j}
\frac{\partial a^{[2]}_j}{\partial z^{[2]}_j}.
$$

Now:

- $\frac{\partial \mathcal{L}}{\partial z^{[3]}_k} = \delta^{[3]}_k$;
- since $z^{[3]}_k = \sum_{m=1}^{d_2} W^{[3]}_{km} a^{[2]}_m + b^{[3]}_k$,
  $$
  \frac{\partial z^{[3]}_k}{\partial a^{[2]}_j} = W^{[3]}_{kj};
  $$
- since $a^{[2]}_j = \phi^{[2]}(z^{[2]}_j)$,
  $$
  \frac{\partial a^{[2]}_j}{\partial z^{[2]}_j} = \phi^{[2]\prime}(z^{[2]}_j).
  $$

Substitute:

$$
\frac{\partial \mathcal{L}}{\partial z^{[2]}_j}
=
\left(
\sum_{k=1}^{d_3} W^{[3]}_{kj}\delta^{[3]}_k
\right)
\phi^{[2]\prime}(z^{[2]}_j).
$$

In vector form,

$$
\boxed{
\delta^{[2]}
=
\left(W^{[3]}\right)^\top \delta^{[3]}
\odot
\phi^{[2]\prime}(z^{[2]})
}.
$$

This is the first true "backpropagation" step:
the downstream error signal is pulled back through the transpose of the linear map and then modulated by the local derivative.

### Parameter gradients at layer 2

Since

$$
z^{[2]} = W^{[2]}a^{[1]} + b^{[2]},
$$

the same affine differentiation pattern gives

$$
\boxed{
\frac{\partial \mathcal{L}}{\partial W^{[2]}}
=
\delta^{[2]} (a^{[1]})^\top
},
\qquad
\boxed{
\frac{\partial \mathcal{L}}{\partial b^{[2]}}
=
\delta^{[2]}
}.
$$

## 6. Hidden-layer recursion: first hidden layer

Now propagate one layer further.
For coordinate $i \in \{1,\dots,d_1\}$,

$$
\frac{\partial \mathcal{L}}{\partial z^{[1]}_i}
=
\sum_{j=1}^{d_2}
\frac{\partial \mathcal{L}}{\partial z^{[2]}_j}
\frac{\partial z^{[2]}_j}{\partial a^{[1]}_i}
\frac{\partial a^{[1]}_i}{\partial z^{[1]}_i}.
$$

Here:

- $\frac{\partial \mathcal{L}}{\partial z^{[2]}_j} = \delta^{[2]}_j$;
- $\frac{\partial z^{[2]}_j}{\partial a^{[1]}_i} = W^{[2]}_{ji}$;
- $\frac{\partial a^{[1]}_i}{\partial z^{[1]}_i} = \phi^{[1]\prime}(z^{[1]}_i)$.

Thus

$$
\frac{\partial \mathcal{L}}{\partial z^{[1]}_i}
=
\left(
\sum_{j=1}^{d_2} W^{[2]}_{ji}\delta^{[2]}_j
\right)
\phi^{[1]\prime}(z^{[1]}_i).
$$

So in vector form,

$$
\boxed{
\delta^{[1]}
=
\left(W^{[2]}\right)^\top \delta^{[2]}
\odot
\phi^{[1]\prime}(z^{[1]})
}.
$$

The corresponding parameter gradients are

$$
\boxed{
\frac{\partial \mathcal{L}}{\partial W^{[1]}}
=
\delta^{[1]} (a^{[0]})^\top
=
\delta^{[1]} x^\top
},
\qquad
\boxed{
\frac{\partial \mathcal{L}}{\partial b^{[1]}}
=
\delta^{[1]}
}.
$$

## 7. Compact summary for the two-hidden-layer network

For one example, the complete backward pass is:

$$
\begin{aligned}
\delta^{[3]}
&=
\frac{\partial \mathcal{L}}{\partial z^{[3]}}, \\
\frac{\partial \mathcal{L}}{\partial W^{[3]}}
&=
\delta^{[3]} (a^{[2]})^\top, \qquad
\frac{\partial \mathcal{L}}{\partial b^{[3]}} = \delta^{[3]}, \\
\delta^{[2]}
&=
\left(W^{[3]}\right)^\top \delta^{[3]} \odot \phi^{[2]\prime}(z^{[2]}), \\
\frac{\partial \mathcal{L}}{\partial W^{[2]}}
&=
\delta^{[2]} (a^{[1]})^\top, \qquad
\frac{\partial \mathcal{L}}{\partial b^{[2]}} = \delta^{[2]}, \\
\delta^{[1]}
&=
\left(W^{[2]}\right)^\top \delta^{[2]} \odot \phi^{[1]\prime}(z^{[1]}), \\
\frac{\partial \mathcal{L}}{\partial W^{[1]}}
&=
\delta^{[1]} x^\top, \qquad
\frac{\partial \mathcal{L}}{\partial b^{[1]}} = \delta^{[1]}.
\end{aligned}
$$

The structure repeats because each layer is affine-plus-nonlinearity.

## 8. General feedforward network

Now consider a depth-$L$ feedforward network:

$$
a^{[0]} = x,
\qquad
z^{[\ell]} = W^{[\ell]}a^{[\ell-1]} + b^{[\ell]},
\qquad
a^{[\ell]} = \phi^{[\ell]}(z^{[\ell]})
\quad \text{for } \ell=1,\dots,L.
$$

Let the scalar loss be $\mathcal{L}(a^{[L]}, y)$.
Define

$$
\delta^{[\ell]} := \frac{\partial \mathcal{L}}{\partial z^{[\ell]}}.
$$

Then:

### Final layer

$$
\delta^{[L]}
=
\frac{\partial \mathcal{L}}{\partial a^{[L]}}
\odot
\phi^{[L]\prime}(z^{[L]}),
$$

or the appropriate Jacobian form if the output activation couples coordinates, as softmax does.

### Recursive hidden-layer rule

For $\ell=L-1,L-2,\dots,1$,

$$
\boxed{
\delta^{[\ell]}
=
\left(W^{[\ell+1]}\right)^\top \delta^{[\ell+1]}
\odot
\phi^{[\ell]\prime}(z^{[\ell]})
}.
$$

### Parameter gradients

For each layer $\ell$,

$$
\boxed{
\frac{\partial \mathcal{L}}{\partial W^{[\ell]}}
=
\delta^{[\ell]} (a^{[\ell-1]})^\top
},
\qquad
\boxed{
\frac{\partial \mathcal{L}}{\partial b^{[\ell]}}
=
\delta^{[\ell]}
}.
$$

This is the core backpropagation algorithm.

## 9. Mini-batches and matrix form

Suppose we process a batch of $B$ examples at once.
Stack activations columnwise:

$$
A^{[\ell]} \in \mathbb{R}^{d_\ell \times B},
\qquad
Z^{[\ell]} \in \mathbb{R}^{d_\ell \times B},
\qquad
\Delta^{[\ell]} \in \mathbb{R}^{d_\ell \times B}.
$$

Then

$$
Z^{[\ell]} = W^{[\ell]}A^{[\ell-1]} + b^{[\ell]}\mathbf{1}^\top,
\qquad
A^{[\ell]} = \phi^{[\ell]}(Z^{[\ell]}).
$$

For a batch-averaged loss, the gradients become

$$
\frac{\partial \mathcal{L}_{\mathrm{batch}}}{\partial W^{[\ell]}}
=
\frac{1}{B}\Delta^{[\ell]} (A^{[\ell-1]})^\top,
$$

and

$$
\frac{\partial \mathcal{L}_{\mathrm{batch}}}{\partial b^{[\ell]}}
=
\frac{1}{B}\Delta^{[\ell]}\mathbf{1},
$$

where the vector $\mathbf{1} \in \mathbb{R}^B$ sums over batch columns.

This is the form used in efficient linear-algebra implementations.

## 10. Computational graph interpretation

Backpropagation can be stated independently of neural-network language.

If a scalar loss $\mathcal{L}$ is computed from a directed acyclic graph of intermediate variables, reverse-mode differentiation propagates adjoints

$$
\bar{v} := \frac{\partial \mathcal{L}}{\partial v}
$$

from outputs backward to inputs.

Each node only needs:

- the upstream adjoint from nodes that depend on it;
- the local derivative of its output with respect to its input.

Neural networks are especially suited to this because the same module pattern repeats layer after layer.

## 11. Cost advantage over naive differentiation

A naive approach might differentiate the loss separately with respect to each parameter path, repeatedly recomputing the same partial derivatives.
Backpropagation stores forward-pass intermediates and reuses them.

For dense feedforward networks, the backward pass has the same order of computational cost as the forward pass up to constant factors.
That efficiency is what makes gradient-based training practical at scale.

## 12. Practical remarks and caveats

### 12.1 Vanishing and exploding gradients

The recursion

$$
\delta^{[\ell]}
=
\left(W^{[\ell+1]}\right)^\top \delta^{[\ell+1]}
\odot
\phi^{[\ell]\prime}(z^{[\ell]})
$$

multiplies together many Jacobian-like factors.
If these factors usually shrink norms, gradients vanish.
If they usually enlarge norms, gradients explode.

This is why activation choice, initialization, residual structure, and normalization matter.

### 12.2 Nondifferentiable points

Functions like ReLU are not differentiable at $0$.
In practice this is not a major obstacle.
Subgradient conventions suffice, and exact hitting of nondifferentiable points is not the dominant issue in floating-point training.

### 12.3 Backpropagation is exact calculus, not a heuristic

The approximation in learning comes from optimization choices, stochastic sampling, finite precision, and model misspecification.
The derivative identities themselves are exact consequences of the chain rule whenever the assumed derivatives exist.

## 13. Summary

Backpropagation is the repeated application of three ideas:

1. compute pre-activations and activations in a forward pass;
2. define the output error signal $\delta^{[L]}$;
3. propagate that signal backward using transpose linear maps and local activation derivatives.

For feedforward networks, the reusable formulas are

$$
\delta^{[\ell]}
=
\left(W^{[\ell+1]}\right)^\top \delta^{[\ell+1]}
\odot
\phi^{[\ell]\prime}(z^{[\ell]}),
$$

and

$$
\frac{\partial \mathcal{L}}{\partial W^{[\ell]}}
=
\delta^{[\ell]}(a^{[\ell-1]})^\top,
\qquad
\frac{\partial \mathcal{L}}{\partial b^{[\ell]}}
=
\delta^{[\ell]}.
$$

These formulas are the foundation for all later deep learning modules, including convolutional networks, recurrent networks, transformers, and modern autodiff systems.
