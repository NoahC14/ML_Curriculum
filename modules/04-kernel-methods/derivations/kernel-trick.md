---
title: "Derivation of the Kernel Trick"
module: "04-kernel-methods"
lesson: "kernel-trick"
doc_type: "derivation"
topic: "feature-space-inner-products"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
  - "03-linear-models/linear-regression"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "kernel-methods"
  - "kernel-trick"
  - "feature-maps"
  - "representer-form"
---

## Purpose

This derivation shows precisely how a linear predictor in feature space becomes a kernelized predictor that depends only on inner products of mapped examples.
The derivation is elementary, but it is the algebraic heart of kernel methods.

## 1. Setup and notation

Let

$$
\phi : \mathcal{X} \to \mathcal{H}
$$

be a feature map into an inner-product space $\mathcal{H}$.
Assume we want to learn a predictor of the form

$$
f(x) = \langle w, \phi(x) \rangle_{\mathcal{H}} + b,
$$

where $w \in \mathcal{H}$ and $b \in \mathbb{R}$.

Suppose we have training inputs $x_1,\dots,x_n$.
The key claim is that for many objectives, an optimal $w$ can be chosen inside the span of the mapped training examples:

$$
\operatorname{span}\{\phi(x_1),\dots,\phi(x_n)\}.
$$

This is the finite-dimensional subspace actually touched by the data.

## 2. Decompose the parameter vector

Write $w$ as

$$
w = w_\parallel + w_\perp,
$$

where:

- $w_\parallel \in \operatorname{span}\{\phi(x_1),\dots,\phi(x_n)\}$;
- $w_\perp$ is orthogonal to that span.

Then for every training example $x_i$,

$$
\langle w, \phi(x_i) \rangle
= \langle w_\parallel, \phi(x_i) \rangle + \langle w_\perp, \phi(x_i) \rangle
= \langle w_\parallel, \phi(x_i) \rangle,
$$

because $w_\perp$ is orthogonal to every vector in the span, in particular to each $\phi(x_i)$.

Therefore the fitted values on the training set depend only on $w_\parallel$.

If the objective is of the form

$$
\mathcal{L}(w,b)
= \sum_{i=1}^n \ell\bigl(y_i, \langle w,\phi(x_i)\rangle + b\bigr) + \Omega(\|w\|_{\mathcal{H}}),
$$

with $\Omega$ nondecreasing, then replacing $w$ by $w_\parallel$ leaves the loss term unchanged while reducing or preserving the norm:

$$
\|w\|_{\mathcal{H}}^2
= \|w_\parallel\|_{\mathcal{H}}^2 + \|w_\perp\|_{\mathcal{H}}^2
\geq \|w_\parallel\|_{\mathcal{H}}^2.
$$

Hence an optimizer can always be chosen with $w_\perp = 0$.

So we may write

$$
w = \sum_{i=1}^n \alpha_i \phi(x_i)
$$

for some coefficients $\alpha_1,\dots,\alpha_n \in \mathbb{R}$.

## 3. Substitute into the predictor

Now evaluate the predictor at a new point $x$:

$$
f(x)
= \left\langle \sum_{i=1}^n \alpha_i \phi(x_i), \phi(x) \right\rangle + b.
$$

Use linearity of the inner product in the first argument:

$$
f(x)
= \sum_{i=1}^n \alpha_i \langle \phi(x_i), \phi(x) \rangle + b.
$$

Define the kernel

$$
K(x_i, x) = \langle \phi(x_i), \phi(x) \rangle.
$$

Then

$$
f(x) = \sum_{i=1}^n \alpha_i K(x_i, x) + b.
$$

This is the kernelized prediction rule.

## 4. Training predictions and the Gram matrix

For a training point $x_j$,

$$
f(x_j) = \sum_{i=1}^n \alpha_i K(x_i, x_j) + b.
$$

If we define the Gram matrix

$$
G \in \mathbb{R}^{n \times n},
\qquad
G_{ij} = K(x_i, x_j),
$$

and the coefficient vector

$$
\alpha =
\begin{bmatrix}
\alpha_1 \\
\vdots \\
\alpha_n
\end{bmatrix},
$$

then the vector of training predictions is

$$
f_{\text{train}} = G\alpha + b\mathbf{1}.
$$

So the entire learning problem can be expressed using only the Gram matrix rather than explicit feature coordinates.

## 5. Worked example: quadratic polynomial kernel

Consider

$$
\phi(x) =
\begin{bmatrix}
1 \\
\sqrt{2}x_1 \\
\sqrt{2}x_2 \\
x_1^2 \\
\sqrt{2}x_1x_2 \\
x_2^2
\end{bmatrix}
\in \mathbb{R}^6
$$

for $x=(x_1,x_2)^\top$.

Then

$$
\langle \phi(x), \phi(z) \rangle
= 1 + 2x_1z_1 + 2x_2z_2 + x_1^2 z_1^2 + 2x_1x_2z_1z_2 + x_2^2 z_2^2.
$$

Group the quadratic terms:

$$
\langle \phi(x), \phi(z) \rangle
= 1 + 2x^\top z + (x^\top z)^2.
$$

But

$$
(x^\top z + 1)^2
= (x^\top z)^2 + 2x^\top z + 1.
$$

Therefore

$$
\langle \phi(x), \phi(z) \rangle = (x^\top z + 1)^2.
$$

This shows that the degree-two polynomial kernel computes the same inner product as an explicit quadratic feature expansion.

## 6. What the trick does and does not do

The kernel trick does:

- replace explicit feature vectors by pairwise similarities;
- permit very high-dimensional or infinite-dimensional feature spaces; and
- preserve linear optimization structure when the algorithm depends only on inner products.

The kernel trick does not:

- make every learning algorithm kernelizable;
- remove the need for regularization or hyperparameter tuning; or
- guarantee computational savings for very large datasets, since the Gram matrix has size $n \times n$.

## 7. ML interpretation

The derivation explains why kernels are the natural nonlinear extension of several linear methods.
Once the solution lies in the span of the training examples, feature-space inner products are enough.
The geometry of the feature map is therefore encoded by the kernel, and the coefficients $\alpha_i$ determine how each training example contributes to predictions.
