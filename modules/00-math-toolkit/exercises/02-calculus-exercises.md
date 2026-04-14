---
title: "Multivariable Calculus Exercises"
module: "00-math-toolkit"
lesson: "calculus-exercises"
doc_type: "exercise"
topic: "multivariable-calculus"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
  - "00-math-toolkit/multivariable-calculus"
updated: "2026-04-09"
owner: "curriculum-team"
tags:
  - "calculus"
  - "optimization"
  - "matrix-calculus"
  - "backpropagation"
---

## Purpose

These exercises move from derivative fluency to optimization interpretation and backpropagation prerequisites.

- Tier 1: coordinate derivatives and local approximations.
- Tier 2: Jacobians, Hessians, and chain-rule derivations.
- Tier 3: ML-facing matrix calculus and constrained optimization.

## Exercise 1: Partial derivatives and gradient

> **Problem.** Let
> $$
> f(x_1, x_2, x_3) = x_1^2x_2 + e^{x_3} - x_1x_3.
> $$
> Compute all first partial derivatives and write the gradient $\nabla f(\mathbf{x})$.

**Deliverables**

- The three partial derivatives.
- The gradient in column-vector form.

## Exercise 2: Directional derivative

> **Problem.** Let
> $$
> f(x_1, x_2) = x_1^2 + 2x_1x_2 + 3x_2^2.
> $$
> Compute the directional derivative at $\mathbf{x} = (1,-1)^\top$ in the direction $\mathbf{u} = (2,1)^\top$.

**Hints**

- First compute the gradient.
- Then use $D_{\mathbf{u}}f(\mathbf{x}) = \nabla f(\mathbf{x})^\top \mathbf{u}$.

**Deliverables**

- The gradient at the point.
- The directional derivative.

## Exercise 3: Jacobian of a vector-valued map

> **Problem.** Let $\mathbf{F} : \mathbb{R}^3 \to \mathbb{R}^2$ be
> $$
> \mathbf{F}(x_1, x_2, x_3)
> =
> \begin{bmatrix}
> x_1x_2 + x_3 \\
> x_1^2 - \sin(x_2x_3)
> \end{bmatrix}.
> $$
> Compute the Jacobian matrix $\mathbf{J}_{\mathbf{F}}(\mathbf{x})$.

**Deliverables**

- The full $2 \times 3$ Jacobian.

## Exercise 4: Hessian and critical point classification

> **Problem.** Let
> $$
> f(x_1, x_2) = x_1^2 + x_1x_2 + x_2^2 - 4x_1.
> $$
> Find the critical point and determine whether it is a local minimum, local maximum, or saddle point.

**Hints**

- Set the gradient equal to zero.
- Use the Hessian.

**Deliverables**

- The critical point.
- The Hessian.
- The classification with one sentence of justification.

## Exercise 5: Second-order Taylor approximation

> **Problem.** Let
> $$
> f(x_1, x_2) = \log(1 + x_1 + x_2)
> $$
> near $\mathbf{0}$. Compute the second-order Taylor approximation at $\mathbf{0}$.

**Deliverables**

- $f(\mathbf{0})$.
- $\nabla f(\mathbf{0})$.
- $\nabla^2 f(\mathbf{0})$.
- The quadratic approximation.

## Exercise 6: Vector chain rule in coordinates

> **Problem.** Let
> $$
> \mathbf{g}(x_1, x_2)
> =
> \begin{bmatrix}
> x_1 + x_2^2 \\
> x_1x_2
> \end{bmatrix},
> \qquad
> f(y_1, y_2) = y_1^2 + 3y_2.
> $$
> Define $h = f \circ \mathbf{g}$. Compute $\nabla h(\mathbf{x})$ in two ways:
>
> 1. by expanding $h(x_1,x_2)$ directly;
> 2. by using $\nabla h(\mathbf{x}) = \mathbf{J}_{\mathbf{g}}(\mathbf{x})^\top \nabla f(\mathbf{g}(\mathbf{x}))$.
>
> Verify that the answers agree.

**Deliverables**

- The direct derivative.
- The Jacobian-based derivative.
- A brief consistency check.

## Exercise 7: Backprop prerequisite I, affine map plus squared loss

> **Problem.** Let
> $$
> \mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b},
> \qquad
> L(\mathbf{W}, \mathbf{b}) = \frac{1}{2}\|\mathbf{z} - \mathbf{y}\|_2^2,
> $$
> where $\mathbf{x} \in \mathbb{R}^d$, $\mathbf{W} \in \mathbb{R}^{m \times d}$, and $\mathbf{b}, \mathbf{y}, \mathbf{z} \in \mathbb{R}^m$.
> Derive $\nabla_{\mathbf{z}}L$, $\nabla_{\mathbf{b}}L$, and $\nabla_{\mathbf{W}}L$.

**Hints**

- First differentiate with respect to $\mathbf{z}$.
- Then use the chain rule and the fact that $z_i = \sum_{j=1}^d W_{ij}x_j + b_i$.

**Deliverables**

- $\nabla_{\mathbf{z}}L$.
- $\nabla_{\mathbf{b}}L$.
- $\nabla_{\mathbf{W}}L$.

## Exercise 8: Backprop prerequisite II, affine map plus elementwise nonlinearity

> **Problem.** Let
> $$
> \mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b},
> \qquad
> \mathbf{a} = \tanh(\mathbf{z}),
> \qquad
> L = \mathbf{c}^\top \mathbf{a},
> $$
> where $\mathbf{c} \in \mathbb{R}^m$ is fixed and $\tanh$ acts coordinatewise.
> Derive $\nabla_{\mathbf{z}}L$ and $\nabla_{\mathbf{W}}L$.

**Hints**

- Use $\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z)$.
- Express the derivative with respect to $\mathbf{z}$ using coordinatewise multiplication.

**Deliverables**

- $\nabla_{\mathbf{z}}L$.
- $\nabla_{\mathbf{W}}L$.
- A sentence explaining why this is a backpropagation pattern.

## Exercise 9: Hessian of least squares

> **Problem.** Let
> $$
> f(\mathbf{w}) = \frac{1}{2}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2
> $$
> with $\mathbf{X} \in \mathbb{R}^{n \times d}$ and $\mathbf{w} \in \mathbb{R}^d$.
> Derive $\nabla f(\mathbf{w})$ and $\nabla^2 f(\mathbf{w})$.

**Hints**

- Expand the quadratic or differentiate in matrix form.
- Observe that the Hessian does not depend on $\mathbf{w}$.

**Deliverables**

- The gradient.
- The Hessian.
- One sentence explaining why the Hessian is positive semidefinite.

## Exercise 10: Newton step for a quadratic objective

> **Problem.** Let
> $$
> f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top \mathbf{Q}\mathbf{x} - \mathbf{b}^\top \mathbf{x},
> $$
> where $\mathbf{Q} \in \mathbb{R}^{d \times d}$ is symmetric positive definite.
> Show that one Newton step from any starting point reaches the unique minimizer.

**Hints**

- Compute the gradient and Hessian.
- Use the Newton update formula.

**Deliverables**

- The Newton update.
- The minimizer.
- A short explanation of why convergence occurs in one step.

## Exercise 11: Lagrange multiplier derivation for the sphere constraint

> **Problem.** Let $\mathbf{a} \in \mathbb{R}^d$ be fixed. Solve
> $$
> \max_{\|\mathbf{x}\|_2 = 1} \mathbf{a}^\top \mathbf{x}
> $$
> using Lagrange multipliers.

**Hints**

- Use the constraint $g(\mathbf{x}) = \mathbf{x}^\top \mathbf{x} - 1 = 0$.
- Compare the optimizer to the direction of $\mathbf{a}$.

**Deliverables**

- The Lagrangian.
- The stationarity equations.
- The optimizer and optimal value.

## Exercise 12: PCA-style constrained optimization

> **Problem.** Let $\mathbf{S} \in \mathbb{R}^{d \times d}$ be symmetric. Show that maximizing
> $$
> \mathbf{v}^\top \mathbf{S}\mathbf{v}
> \quad \text{subject to} \quad \|\mathbf{v}\|_2 = 1
> $$
> leads to the eigenvalue equation $\mathbf{S}\mathbf{v} = \lambda \mathbf{v}$.

**Deliverables**

- The Lagrangian setup.
- The stationarity condition.
- One sentence connecting the result to PCA.
