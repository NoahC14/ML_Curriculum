---
title: "Ridge Regression and Lasso"
module: "03-linear-models"
lesson: "ridge-lasso"
doc_type: "derivation"
topic: "regularized-linear-regression"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
  - "00-math-toolkit/probability"
  - "01-optimization/convexity-and-optimization"
  - "03-linear-models/linear-regression"
updated: "2026-04-11"
owner: "curriculum-team"
tags:
  - "linear-models"
  - "ridge-regression"
  - "lasso"
  - "regularization"
  - "map-estimation"
  - "sparsity"
---

## Purpose

This note derives the basic optimization and probabilistic interpretations of ridge regression and lasso.
The key outcomes are:

- ridge regression has a closed-form solution and equals MAP estimation under a Gaussian prior; and
- lasso is convex but non-smooth, and its geometry explains why sparse solutions appear.

## Setup

Let

$$
L(w) = \frac{1}{n}\|y - Xw\|_2^2
$$

denote the ordinary least-squares objective.
Regularization augments this objective by a penalty term.

## 1. Ridge regression

Ridge solves

$$
\widehat{w}_{\mathrm{ridge}}
\in
\arg\min_w
\frac{1}{n}\|y - Xw\|_2^2 + \lambda \|w\|_2^2,
$$

where $\lambda \geq 0$.

### Step 1. Differentiate the objective

Define

$$
J_{\mathrm{ridge}}(w) = \frac{1}{n}\|y - Xw\|_2^2 + \lambda w^\top w.
$$

Using the least-squares gradient and $\nabla_w(w^\top w) = 2w$, we get

$$
\nabla_w J_{\mathrm{ridge}}(w)
= \frac{2}{n}(X^\top X w - X^\top y) + 2\lambda w.
$$

Setting this to zero gives

$$
\frac{1}{n}(X^\top X w - X^\top y) + \lambda w = 0.
$$

Multiply by $n$:

$$
X^\top X w - X^\top y + n\lambda w = 0.
$$

Therefore the ridge normal equations are

$$
(X^\top X + n\lambda I)w = X^\top y.
$$

### Step 2. Solve the linear system

For any $\lambda > 0$, the matrix

$$
X^\top X + n\lambda I
$$

is positive definite, so it is invertible.
Hence

$$
\widehat{w}_{\mathrm{ridge}}
= (X^\top X + n\lambda I)^{-1}X^\top y.
$$

This is true even if $X^\top X$ itself is singular.

### Step 3. Interpret the shrinkage

Suppose $X^\top X = Q\Lambda Q^\top$ is an eigendecomposition with orthogonal $Q$ and diagonal $\Lambda$.
Then

$$
\widehat{w}_{\mathrm{ridge}}
= Q(\Lambda + n\lambda I)^{-1}Q^\top X^\top y.
$$

Each eigendirection is scaled by

$$
\frac{1}{\lambda_j + n\lambda}.
$$

Small-eigenvalue directions, which are typically unstable directions, are shrunk most strongly.
This is the spectral reason ridge improves conditioning.

## 2. Ridge as constrained optimization

Ridge can also be written as

$$
\min_w \frac{1}{n}\|y - Xw\|_2^2
\quad
\text{subject to}
\quad
\|w\|_2^2 \leq c
$$

for a suitable radius $c$.
The constrained and penalized forms are equivalent at the level of optimal solutions for appropriate parameter matching.

This matters geometrically.
The loss contours are ellipsoids.
The feasible region is an $\ell_2$ ball.
Tangency usually occurs at a smooth point, which is why ridge shrinks coefficients continuously rather than setting many exactly to zero.

## 3. Ridge as MAP estimation

Assume the Gaussian observation model

$$
y \mid X, w, \sigma^2 \sim \mathcal{N}(Xw, \sigma^2 I_n).
$$

Now place a Gaussian prior on the coefficients:

$$
w \sim \mathcal{N}(0, \tau^2 I_d).
$$

Then

$$
p(w)
=
(2\pi\tau^2)^{-d/2}
\exp\left(
-\frac{1}{2\tau^2}\|w\|_2^2
\right).
$$

The posterior satisfies

$$
p(w \mid X, y, \sigma^2)
\propto
p(y \mid X, w, \sigma^2)p(w).
$$

Take negative logs and discard constants independent of $w$:

$$
-\log p(w \mid X, y, \sigma^2)
=
\frac{1}{2\sigma^2}\|y - Xw\|_2^2
\frac{1}{2\tau^2}\|w\|_2^2
 \,+\, \text{constant}.
$$

Therefore the MAP estimator solves

$$
\widehat{w}_{\mathrm{MAP}}
\in
\arg\min_w
\frac{1}{2\sigma^2}\|y - Xw\|_2^2
\frac{1}{2\tau^2}\|w\|_2^2.
$$

Multiplying by $2\sigma^2$ does not change the minimizer, so this is equivalent to

$$
\arg\min_w
\|y - Xw\|_2^2
\frac{\sigma^2}{\tau^2}\|w\|_2^2.
$$

Thus ridge is MAP estimation with penalty weight

$$
\lambda_{\mathrm{eff}} = \frac{\sigma^2}{\tau^2}
$$

up to whichever normalization convention is used in front of the loss.

Interpretation:

- smaller $\tau^2$ means a stronger prior belief that coefficients should be near zero;
- larger $\lambda$ means stronger shrinkage; and
- the Bayesian and optimization viewpoints are the same mathematics in different language.

## 4. Lasso

Lasso solves

$$
\widehat{w}_{\mathrm{lasso}}
\in
\arg\min_w
\frac{1}{n}\|y - Xw\|_2^2 + \lambda \|w\|_1,
$$

Unlike ridge, this objective is not differentiable at coordinates where $w_j = 0$.
So we cannot derive a linear closed form by setting the gradient equal to zero.

### Subgradient condition

For the absolute value,

$$
\partial |w_j|
=
\begin{cases}
\{1\}, & w_j > 0, \\
[-1,1], & w_j = 0, \\
\{-1\}, & w_j < 0.
\end{cases}
$$

Hence the first-order optimality condition for lasso is

$$
0 \in \frac{2}{n}X^\top(Xw - y) + \lambda \partial \|w\|_1,
$$

meaning coordinatewise

$$
0 \in \frac{2}{n}x_j^\top(Xw - y) + \lambda \partial |w_j|.
$$

If $w_j \neq 0$, then the subgradient is $\pm 1$.
If $w_j = 0$, then the subgradient may be any value in $[-1,1]$.

This interval at zero is exactly what allows an optimum to land with $w_j = 0$ on a whole region of the parameter space.

## 5. Geometric explanation of sparsity

In two dimensions, the lasso-constrained problem is

$$
\min_w \frac{1}{n}\|y - Xw\|_2^2
\quad
\text{subject to}
\quad
\|w\|_1 \leq c.
$$

The feasible set is a diamond with corners on the coordinate axes.
The quadratic loss contours are ellipses.
The first ellipse that touches the feasible region frequently touches at a corner.

At a corner, one coordinate is zero.
In higher dimensions, the same phenomenon occurs on faces and corners of the $\ell_1$ polytope.
That is the geometric source of sparsity.

By contrast, an $\ell_2$ ball has no axis-aligned corners.
So ridge typically gives small coefficients, not exact zeros.

## 6. Lasso as MAP estimation

If the coordinates of $w$ are independent with Laplace prior

$$
p(w_j) = \frac{\alpha}{2}\exp(-\alpha |w_j|),
$$

then

$$
p(w) \propto \exp(-\alpha \|w\|_1).
$$

Under the same Gaussian observation model,

$$
-\log p(w \mid X, y, \sigma^2)
=
\frac{1}{2\sigma^2}\|y - Xw\|_2^2 + \alpha \|w\|_1 + \text{constant}.
$$

So lasso is MAP estimation under a Laplace prior.

This prior is sharply peaked at zero compared with a Gaussian prior.
That sharp peak is the probabilistic analogue of the geometric corners in the $\ell_1$ ball.

## 7. Scope notes

Lasso is often presented as feature selection.
That is useful but incomplete.

- When predictors are highly correlated, lasso may select one variable somewhat arbitrarily.
- Ridge tends to distribute weight across correlated predictors more smoothly.
- Elastic net combines both penalties to balance sparsity and stability.

So the right comparison is not "ridge versus lasso in the abstract," but rather "which inductive bias matches the structure of the problem?"

## ML relevance

Ridge and lasso are central because they make regularization concrete.
They show that model complexity can be controlled:

- analytically through modified normal equations;
- geometrically through constraint sets;
- probabilistically through priors and MAP estimation; and
- empirically through validation on held-out data.

These themes reappear throughout machine learning, including neural network weight decay, sparse coding, Bayesian regression, and compressed sensing.
