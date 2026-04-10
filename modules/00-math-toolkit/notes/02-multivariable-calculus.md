---
title: "Multivariable Calculus Primer for Machine Learning"
module: "00-math-toolkit"
lesson: "multivariable-calculus-primer"
doc_type: "notes"
topic: "multivariable-calculus"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
updated: "2026-04-09"
owner: "curriculum-team"
tags:
  - "calculus"
  - "optimization"
  - "matrix-calculus"
  - "backpropagation"
---

## Motivation

Optimization in machine learning is multivariable calculus in action. A loss function depends on many parameters, often arranged into vectors or matrices. Training requires a disciplined language for sensitivity: which direction increases the loss, how local curvature changes the step size, and how derivatives propagate through compositions of maps.

This note builds that language in a progression:

- scalar-valued functions of several variables;
- vector-valued maps and their Jacobians;
- second-order structure through Hessians and Taylor expansions; and
- the multivariate chain rule in the matrix form used later for backpropagation.

The goal is not full real analysis. The goal is a clean toolkit for Modules 01 and 06.

## Assumptions and Notation

Let $f : \mathbb{R}^d \to \mathbb{R}$ be a scalar-valued function, and let $\mathbf{F} : \mathbb{R}^d \to \mathbb{R}^m$ be a vector-valued map. We write vectors in bold lowercase and matrices in bold uppercase.

If $\mathbf{x} = (x_1, \ldots, x_d)^\top \in \mathbb{R}^d$, the $i$th standard basis vector is $\mathbf{e}_i$. When a derivative exists, we use the denominator-layout convention:

- gradients are column vectors in $\mathbb{R}^d$;
- Jacobians of $\mathbf{F} : \mathbb{R}^d \to \mathbb{R}^m$ are matrices in $\mathbb{R}^{m \times d}$;
- Hessians of $f : \mathbb{R}^d \to \mathbb{R}$ are matrices in $\mathbb{R}^{d \times d}$.

For a perturbation $\mathbf{h} \in \mathbb{R}^d$, $\|\mathbf{h}\|_2$ denotes the Euclidean norm.

## Partial Derivatives and Local Variation

> **Definition.** Let $f : \mathbb{R}^d \to \mathbb{R}$ and let $\mathbf{x} \in \mathbb{R}^d$. The partial derivative of $f$ with respect to $x_i$ at $\mathbf{x}$ is
>
> $$
> \frac{\partial f}{\partial x_i}(\mathbf{x})
> =
> \lim_{t \to 0}
> \frac{f(\mathbf{x} + t\mathbf{e}_i) - f(\mathbf{x})}{t},
> $$
>
> provided the limit exists.

The partial derivative measures local sensitivity when only one coordinate is perturbed and the others are held fixed.

> **Example.** Let
> $$
> f(x_1, x_2) = x_1^2 x_2 + \sin(x_2).
> $$
> Then
> $$
> \frac{\partial f}{\partial x_1} = 2x_1 x_2,
> \qquad
> \frac{\partial f}{\partial x_2} = x_1^2 + \cos(x_2).
> $$
>
> The first derivative measures how the polynomial interaction changes with $x_1$, while the second combines algebraic and oscillatory effects.

Partial derivatives alone do not yet give the best linear approximation to the function. For that we need differentiability.

## Gradient and Differentiability

> **Definition.** Suppose $f : \mathbb{R}^d \to \mathbb{R}$. We say $f$ is differentiable at $\mathbf{x}$ if there exists a vector $\nabla f(\mathbf{x}) \in \mathbb{R}^d$ such that
>
> $$
> f(\mathbf{x} + \mathbf{h})
> =
> f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \mathbf{h} + r(\mathbf{h}),
> $$
>
> where
> $$
> \frac{r(\mathbf{h})}{\|\mathbf{h}\|_2} \to 0
> \quad \text{as } \mathbf{h} \to \mathbf{0}.
> $$

The gradient is the coefficient vector of the best local linear approximation.

If the partial derivatives exist and are continuous in a neighborhood of $\mathbf{x}$, then $f$ is differentiable at $\mathbf{x}$ and

$$
\nabla f(\mathbf{x})
=
\begin{bmatrix}
\frac{\partial f}{\partial x_1}(\mathbf{x}) \\
\vdots \\
\frac{\partial f}{\partial x_d}(\mathbf{x})
\end{bmatrix}.
$$

### Why the gradient gives steepest local increase

For a unit vector $\mathbf{u}$, the linear approximation predicts change

$$
f(\mathbf{x} + t\mathbf{u}) - f(\mathbf{x})
\approx
t \nabla f(\mathbf{x})^\top \mathbf{u}.
$$

By Cauchy-Schwarz,

$$
\nabla f(\mathbf{x})^\top \mathbf{u}
\leq
\|\nabla f(\mathbf{x})\|_2 \|\mathbf{u}\|_2
=
\|\nabla f(\mathbf{x})\|_2.
$$

Equality holds when $\mathbf{u}$ points in the gradient direction. So the gradient points toward steepest first-order increase, and $-\nabla f(\mathbf{x})$ points toward steepest first-order decrease.

> **ML Interpretation.** Gradient descent updates
> $$
> \mathbf{x}_{k+1} = \mathbf{x}_k - \eta_k \nabla f(\mathbf{x}_k)
> $$
> because the negative gradient is the local direction that most rapidly decreases the objective to first order.

## Directional Derivatives

> **Definition.** Let $\mathbf{u} \in \mathbb{R}^d$. The directional derivative of $f$ at $\mathbf{x}$ along $\mathbf{u}$ is
>
> $$
> D_{\mathbf{u}}f(\mathbf{x})
> =
> \lim_{t \to 0}
> \frac{f(\mathbf{x} + t\mathbf{u}) - f(\mathbf{x})}{t},
> $$
>
> provided the limit exists.

If $f$ is differentiable at $\mathbf{x}$, then

$$
D_{\mathbf{u}}f(\mathbf{x}) = \nabla f(\mathbf{x})^\top \mathbf{u}.
$$

This identity is fundamental: once the gradient is known, every directional derivative follows by an inner product.

> **Example.** If $f(x_1, x_2) = x_1^2 + 3x_1x_2$, then
> $$
> \nabla f(x_1, x_2)
> =
> \begin{bmatrix}
> 2x_1 + 3x_2 \\
> 3x_1
> \end{bmatrix}.
> $$
> At $\mathbf{x} = (1, -1)^\top$ and $\mathbf{u} = (2, 1)^\top$,
> $$
> D_{\mathbf{u}}f(\mathbf{x})
> =
> \nabla f(\mathbf{x})^\top \mathbf{u}
> =
> \begin{bmatrix}
> -1 \\
> 3
> \end{bmatrix}^\top
> \begin{bmatrix}
> 2 \\
> 1
> \end{bmatrix}
> = 1.
> $$

## Jacobians: Derivatives of Vector-Valued Maps

When the output has multiple coordinates, the derivative is a linear map represented by the Jacobian matrix.

> **Definition.** Let $\mathbf{F} = (F_1, \ldots, F_m)^\top : \mathbb{R}^d \to \mathbb{R}^m$. The Jacobian of $\mathbf{F}$ at $\mathbf{x}$ is
>
> $$
> \mathbf{J}_{\mathbf{F}}(\mathbf{x})
> =
> \begin{bmatrix}
> \frac{\partial F_1}{\partial x_1} & \cdots & \frac{\partial F_1}{\partial x_d} \\
> \vdots & \ddots & \vdots \\
> \frac{\partial F_m}{\partial x_1} & \cdots & \frac{\partial F_m}{\partial x_d}
> \end{bmatrix}
> \in \mathbb{R}^{m \times d}.
> $$

If $\mathbf{F}$ is differentiable at $\mathbf{x}$, then

$$
\mathbf{F}(\mathbf{x} + \mathbf{h})
=
\mathbf{F}(\mathbf{x}) + \mathbf{J}_{\mathbf{F}}(\mathbf{x})\mathbf{h} + \mathbf{r}(\mathbf{h}),
$$

where $\|\mathbf{r}(\mathbf{h})\|_2 / \|\mathbf{h}\|_2 \to 0$ as $\mathbf{h} \to \mathbf{0}$.

So the Jacobian is the best linear map approximating $\mathbf{F}$ near $\mathbf{x}$.

> **Example.** Let $\mathbf{F} : \mathbb{R}^2 \to \mathbb{R}^2$ be
> $$
> \mathbf{F}(x_1, x_2)
> =
> \begin{bmatrix}
> x_1x_2 \\
> x_1^2 + e^{x_2}
> \end{bmatrix}.
> $$
> Then
> $$
> \mathbf{J}_{\mathbf{F}}(x_1, x_2)
> =
> \begin{bmatrix}
> x_2 & x_1 \\
> 2x_1 & e^{x_2}
> \end{bmatrix}.
> $$

### Linear maps as the cleanest case

If $\mathbf{F}(\mathbf{x}) = \mathbf{A}\mathbf{x} + \mathbf{b}$ with $\mathbf{A} \in \mathbb{R}^{m \times d}$, then

$$
\mathbf{J}_{\mathbf{F}}(\mathbf{x}) = \mathbf{A}
$$

for every $\mathbf{x}$. This is why linear algebra and multivariable calculus fit together so naturally.

## Hessians and Second-Order Structure

The gradient captures first-order sensitivity. The Hessian captures second-order curvature.

> **Definition.** Let $f : \mathbb{R}^d \to \mathbb{R}$ be twice differentiable. Its Hessian at $\mathbf{x}$ is
>
> $$
> \nabla^2 f(\mathbf{x})
> =
> \begin{bmatrix}
> \frac{\partial^2 f}{\partial x_1 \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_d} \\
> \vdots & \ddots & \vdots \\
> \frac{\partial^2 f}{\partial x_d \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_d \partial x_d}
> \end{bmatrix}.
> $$

If the second partial derivatives are continuous, then mixed partials commute:

$$
\frac{\partial^2 f}{\partial x_i \partial x_j}
=
\frac{\partial^2 f}{\partial x_j \partial x_i},
$$

so $\nabla^2 f(\mathbf{x})$ is symmetric.

> **Example.** Let
> $$
> f(x_1, x_2) = x_1^2 + 4x_1x_2 + x_2^2.
> $$
> Then
> $$
> \nabla f(x_1, x_2)
> =
> \begin{bmatrix}
> 2x_1 + 4x_2 \\
> 4x_1 + 2x_2
> \end{bmatrix},
> \qquad
> \nabla^2 f(x_1, x_2)
> =
> \begin{bmatrix}
> 2 & 4 \\
> 4 & 2
> \end{bmatrix}.
> $$

### Second-order test and optimization meaning

If $\nabla f(\mathbf{x}_\star) = \mathbf{0}$, then $\mathbf{x}_\star$ is a critical point. The Hessian helps classify it locally:

- if $\nabla^2 f(\mathbf{x}_\star)$ is positive definite, then $\mathbf{x}_\star$ is a strict local minimum;
- if $\nabla^2 f(\mathbf{x}_\star)$ is negative definite, then $\mathbf{x}_\star$ is a strict local maximum;
- if $\nabla^2 f(\mathbf{x}_\star)$ is indefinite, then $\mathbf{x}_\star$ is a saddle point.

In high-dimensional ML objectives, saddle structure is common, which is one reason curvature information matters.

> **ML Interpretation.** Newton-type methods approximate a loss near $\mathbf{x}$ by a quadratic model and solve that model to propose an update. This requires gradients and Hessians, or approximations to them.

## Taylor Approximation in Several Variables

Suppose $f : \mathbb{R}^d \to \mathbb{R}$ is twice continuously differentiable near $\mathbf{x}$. Then for small $\mathbf{h}$,

$$
f(\mathbf{x} + \mathbf{h})
\approx
f(\mathbf{x})
+ \nabla f(\mathbf{x})^\top \mathbf{h}
+ \frac{1}{2}\mathbf{h}^\top \nabla^2 f(\mathbf{x}) \mathbf{h}.
$$

This is the second-order Taylor approximation.

The approximation separates local behavior into:

- a constant term $f(\mathbf{x})$;
- a linear term determined by the gradient; and
- a quadratic curvature term determined by the Hessian.

> **Example.** Let
> $$
> f(x_1, x_2) = e^{x_1 + x_2}.
> $$
> At $\mathbf{x} = \mathbf{0}$,
> $$
> f(\mathbf{0}) = 1,
> \qquad
> \nabla f(\mathbf{0}) =
> \begin{bmatrix}
> 1 \\
> 1
> \end{bmatrix},
> \qquad
> \nabla^2 f(\mathbf{0}) =
> \begin{bmatrix}
> 1 & 1 \\
> 1 & 1
> \end{bmatrix}.
> $$
> Therefore
> $$
> e^{x_1 + x_2}
> \approx
> 1 + x_1 + x_2 + \frac{1}{2}(x_1 + x_2)^2
> $$
> near the origin.

### Quadratic models and Newton's method

For an objective $f$, the quadratic Taylor model around $\mathbf{x}_k$ is

$$
m_k(\mathbf{p})
=
f(\mathbf{x}_k)
+ \nabla f(\mathbf{x}_k)^\top \mathbf{p}
+ \frac{1}{2}\mathbf{p}^\top \nabla^2 f(\mathbf{x}_k)\mathbf{p}.
$$

Minimizing this model formally gives the Newton step

$$
\mathbf{p}_k = -\nabla^2 f(\mathbf{x}_k)^{-1}\nabla f(\mathbf{x}_k),
$$

provided the Hessian is invertible and the local model is trustworthy.

This formula is one reason Module 01 needs a usable Hessian concept, not only symbolic derivatives.

## Chain Rule for Scalar and Vector Compositions

The single-variable chain rule says that derivatives multiply through composition. In multiple dimensions, the derivative of a composition is the composition of the derivative maps.

### Scalar outer function

Let $\mathbf{g} : \mathbb{R}^d \to \mathbb{R}^m$ and $f : \mathbb{R}^m \to \mathbb{R}$. Define

$$
h(\mathbf{x}) = f(\mathbf{g}(\mathbf{x})).
$$

Then

$$
\nabla h(\mathbf{x})
=
\mathbf{J}_{\mathbf{g}}(\mathbf{x})^\top \nabla f(\mathbf{g}(\mathbf{x})).
$$

Dimensions matter:

- $\mathbf{J}_{\mathbf{g}}(\mathbf{x}) \in \mathbb{R}^{m \times d}$;
- $\nabla f(\mathbf{g}(\mathbf{x})) \in \mathbb{R}^{m}$;
- $\mathbf{J}_{\mathbf{g}}(\mathbf{x})^\top \nabla f(\mathbf{g}(\mathbf{x})) \in \mathbb{R}^{d}$.

This transpose is not cosmetic. It is exactly what converts output-space sensitivities into input-space sensitivities.

### Fully vector-valued form

If $\mathbf{g} : \mathbb{R}^d \to \mathbb{R}^m$ and $\mathbf{f} : \mathbb{R}^m \to \mathbb{R}^p$, then

$$
\mathbf{J}_{\mathbf{f} \circ \mathbf{g}}(\mathbf{x})
=
\mathbf{J}_{\mathbf{f}}(\mathbf{g}(\mathbf{x})) \mathbf{J}_{\mathbf{g}}(\mathbf{x}).
$$

This is the matrix product version of the chain rule. A separate derivation is given in `derivations/chain-rule-matrices.md`.

> **Backpropagation preview.** A neural network is a repeated composition
> $$
> \mathbf{x}
> \mapsto
> \mathbf{a}^{(1)}
> \mapsto
> \mathbf{a}^{(2)}
> \mapsto \cdots \mapsto
> \ell.
> $$
> Backpropagation works because the chain rule converts the derivative of the whole composition into a product of local Jacobians, evaluated in reverse when propagated back to parameters.

## Matrix Calculus Example: Affine Map Followed by a Scalar Loss

Let $\mathbf{x} \in \mathbb{R}^d$, $\mathbf{W} \in \mathbb{R}^{m \times d}$, $\mathbf{b} \in \mathbb{R}^m$, and define

$$
\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}.
$$

Let $\ell : \mathbb{R}^m \to \mathbb{R}$ be a scalar loss, and define

$$
L(\mathbf{W}, \mathbf{b})
=
\ell(\mathbf{W}\mathbf{x} + \mathbf{b}).
$$

If $\boldsymbol{\delta} = \nabla_{\mathbf{z}} \ell(\mathbf{z}) \in \mathbb{R}^m$, then:

$$
\nabla_{\mathbf{b}} L = \boldsymbol{\delta},
\qquad
\nabla_{\mathbf{x}} L = \mathbf{W}^\top \boldsymbol{\delta}.
$$

For the matrix parameter,

$$
\frac{\partial L}{\partial W_{ij}}
=
\delta_i x_j.
$$

Therefore

$$
\nabla_{\mathbf{W}} L = \boldsymbol{\delta}\mathbf{x}^\top.
$$

This outer-product form reappears constantly in neural network training.

> **Common misconception.** Students often know how to differentiate with respect to scalars but lose track of shape when differentiating with respect to matrices. Checking dimensions is one of the most reliable correctness tests in matrix calculus.

## Constrained Optimization Preview: Lagrange Multipliers

Optimization problems in ML often include constraints: normalized vectors, probability simplices, norm budgets, or orthogonality conditions.

Consider the problem

$$
\min_{\mathbf{x} \in \mathbb{R}^d} f(\mathbf{x})
\quad \text{subject to} \quad
g(\mathbf{x}) = 0,
$$

where $f, g : \mathbb{R}^d \to \mathbb{R}$ are differentiable.

> **Lagrange multiplier principle.** At a regular constrained optimum $\mathbf{x}_\star$, there exists a scalar $\lambda_\star$ such that
>
> $$
> \nabla f(\mathbf{x}_\star) = \lambda_\star \nabla g(\mathbf{x}_\star).
> $$

The geometric idea is that at the optimum, the gradient of the objective cannot have a component tangent to the constraint surface. It must lie in the normal direction, and $\nabla g$ is that normal.

### Example: maximize a quadratic form on the unit sphere

Let $\mathbf{A} \in \mathbb{R}^{d \times d}$ be symmetric and consider

$$
\max_{\|\mathbf{v}\|_2 = 1} \mathbf{v}^\top \mathbf{A}\mathbf{v}.
$$

Write the constraint as

$$
g(\mathbf{v}) = \mathbf{v}^\top \mathbf{v} - 1 = 0.
$$

The Lagrangian is

$$
\mathcal{L}(\mathbf{v}, \lambda)
=
\mathbf{v}^\top \mathbf{A}\mathbf{v}
- \lambda(\mathbf{v}^\top \mathbf{v} - 1).
$$

Differentiating with respect to $\mathbf{v}$ gives

$$
2\mathbf{A}\mathbf{v} - 2\lambda \mathbf{v} = \mathbf{0},
$$

so

$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}.
$$

Thus constrained quadratic optimization leads directly to an eigenvalue problem. This is the mechanism behind PCA and many related ML constructions.

## Summary

The main objects of this primer are:

- the gradient, which encodes first-order sensitivity of scalar objectives;
- the Jacobian, which encodes the local linearization of vector maps;
- the Hessian, which captures curvature and second-order optimization structure;
- Taylor approximations, which turn derivatives into local models; and
- the multivariate chain rule, which makes backpropagation possible.

Together these ideas form the calculus backbone of optimization and deep learning.

## References

- Boyd, S., and Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
- Rudin, W. (1976). *Principles of Mathematical Analysis*. McGraw-Hill.
- Magnus, J. R., and Neudecker, H. (1999). *Matrix Differential Calculus with Applications in Statistics and Econometrics*. Wiley.
