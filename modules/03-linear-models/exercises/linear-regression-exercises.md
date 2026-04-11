---
title: "Linear Regression Exercises"
module: "03-linear-models"
lesson: "linear-regression-exercises"
doc_type: "exercise"
topic: "linear-regression"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
  - "00-math-toolkit/probability"
  - "01-optimization/convexity-and-optimization"
  - "02-statistical-learning/statistical-learning-foundations"
  - "03-linear-models/linear-regression"
updated: "2026-04-11"
owner: "curriculum-team"
tags:
  - "linear-models"
  - "linear-regression"
  - "ridge-regression"
  - "lasso"
  - "maximum-likelihood"
---

## Purpose

These exercises reinforce the linear regression module from algebraic, geometric, probabilistic, and computational viewpoints.

## Exercise 1. Scalar to matrix notation

Suppose

$$
\hat{y}_i = w^\top x_i
$$

for $i=1,\dots,n$.

1. Write the least-squares objective in summation form.
2. Define the design matrix $X$ and target vector $y$.
3. Rewrite the objective as a squared Euclidean norm in matrix form.
4. State the dimensions of $X$, $w$, $y$, and $Xw$.

## Exercise 2. Deriving the normal equations

Let

$$
L(w) = \frac{1}{n}\|y - Xw\|_2^2.
$$

1. Expand $L(w)$ into quadratic form.
2. Differentiate with respect to $w$.
3. Set the gradient to zero and derive the normal equations.
4. State the condition under which the solution can be written as

$$
\widehat{w} = (X^\top X)^{-1}X^\top y.
$$

## Exercise 3. Projection interpretation

Let $\widehat{w}$ be a least-squares solution and let

$$
\hat{y} = X\widehat{w},
\qquad
r = y - \hat{y}.
$$

1. Show that $X^\top r = 0$.
2. Explain why this means $r$ is orthogonal to the column space of $X$.
3. Explain why $\hat{y}$ is the orthogonal projection of $y$ onto $\operatorname{col}(X)$.
4. State one reason the projection viewpoint is useful when $X^\top X$ is singular.

## Exercise 4. Pseudoinverse and non-uniqueness

Suppose two columns of $X$ are identical.

1. Explain why $X^\top X$ is singular.
2. Does least squares still have a minimizer?
3. Are the fitted values $X\widehat{w}$ unique?
4. Are the coefficients $\widehat{w}$ unique?
5. What special role does the Moore-Penrose pseudoinverse play?

## Exercise 5. Gaussian-noise model

Assume

$$
Y \mid X, w, \sigma^2 \sim \mathcal{N}(Xw, \sigma^2 I_n).
$$

1. Write the likelihood $p(y \mid X, w, \sigma^2)$.
2. Compute the log-likelihood up to additive constants.
3. Show that maximizing likelihood in $w$ is equivalent to minimizing squared error.
4. Explain in words why least squares and MLE coincide in this model.

## Exercise 6. Interpreting coefficients

Consider a multiple linear regression model with features:

- age in years;
- annual income in dollars; and
- years of education.

The target is yearly spending on a product in dollars.

1. Interpret the coefficient of income.
2. Explain why coefficient interpretation is conditional on the other features being held fixed.
3. Describe one situation in which the coefficient interpretation becomes unstable.

## Exercise 7. Ridge regression derivation

Consider

$$
J(w) = \frac{1}{n}\|y - Xw\|_2^2 + \lambda \|w\|_2^2.
$$

1. Differentiate $J(w)$.
2. Derive the ridge normal equations.
3. Explain why $X^\top X + n\lambda I$ is invertible for $\lambda > 0$.
4. State one statistical benefit and one numerical benefit of ridge regression.

## Exercise 8. Ridge as MAP

Assume

$$
y \mid X, w, \sigma^2 \sim \mathcal{N}(Xw, \sigma^2 I_n)
$$

and

$$
w \sim \mathcal{N}(0, \tau^2 I_d).
$$

1. Write the negative log-posterior up to additive constants.
2. Show that the MAP estimator solves a ridge-regularized optimization problem.
3. Explain how $\tau^2$ controls the strength of shrinkage.

## Exercise 9. Lasso geometry

Answer the following conceptual questions.

1. What is the $\ell_1$ norm of a vector?
2. Describe the shape of the set $\{w \in \mathbb{R}^2 : \|w\|_1 \leq c\}$.
3. Why do sharp corners in the feasible set promote exact zeros in the solution?
4. Why does ridge regression not usually produce the same degree of sparsity?

## Exercise 10. Subgradient reasoning for lasso

For lasso,

$$
\min_w \frac{1}{n}\|y - Xw\|_2^2 + \lambda \|w\|_1,
$$

1. Write the subgradient of $|w_j|$.
2. State the first-order optimality condition coordinatewise.
3. Explain why the interval subgradient at $w_j = 0$ matters.

## Exercise 11. Polynomial regression and overfitting

Suppose a one-dimensional regression problem is modeled using polynomial features up to degree $p$.

1. Explain why the model is still linear in the parameters.
2. Describe what underfitting looks like when $p$ is too small.
3. Describe what overfitting looks like when $p$ is too large.
4. Explain why cross-validation is a better guide than training error for choosing $p$.

## Exercise 12. Worked computation

Let

$$
X =
\begin{bmatrix}
1 & 0 \\
1 & 1 \\
1 & 2
\end{bmatrix},
\qquad
y =
\begin{bmatrix}
1 \\
2 \\
2
\end{bmatrix}.
$$

1. Compute $X^\top X$ and $X^\top y$.
2. Solve the normal equations for $\widehat{w}$.
3. Compute the fitted values $\hat{y}$.
4. Compute the residual vector $r$.
5. Verify directly that $X^\top r = 0$.

## Exercise 13. Notebook analysis

Run [LM-01-linear-regression.ipynb](../notebooks/LM-01-linear-regression.ipynb).

1. Which polynomial degree visibly underfits the synthetic data?
2. Which degree provides a reasonable middle-ground fit?
3. Which high-degree model overfits the noisy sample most strongly?
4. How do the ridge and lasso coefficient paths change as regularization strength increases?
5. What changes when you increase the sample size or reduce the noise level?

## Exercise 14. Reflection

Write a short response to the following prompt:

Why is linear regression best understood as a meeting point of linear algebra, optimization, and probability rather than as merely a formula for fitting a line?
