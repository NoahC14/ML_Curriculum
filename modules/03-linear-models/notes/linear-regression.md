---
title: "Linear Regression"
module: "03-linear-models"
lesson: "linear-regression"
doc_type: "notes"
topic: "least-squares-regression"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
  - "00-math-toolkit/probability"
  - "01-optimization/convexity-and-optimization"
  - "02-statistical-learning/statistical-learning-foundations"
updated: "2026-04-11"
owner: "curriculum-team"
tags:
  - "linear-models"
  - "linear-regression"
  - "least-squares"
  - "ridge-regression"
  - "lasso"
  - "maximum-likelihood"
  - "map-estimation"
---

## Purpose

These notes introduce linear regression as the first fully worked supervised learning model in the curriculum.
The aim is to connect four viewpoints that students should learn to move between fluently:

- linear regression as empirical risk minimization under squared loss;
- linear regression as orthogonal projection in a vector space;
- linear regression as maximum likelihood under Gaussian noise; and
- regularized linear regression as complexity control and, in the ridge case, as MAP estimation.

## Learning objectives

After working through this note, you should be able to:

- write linear regression in scalar, vector, and matrix notation;
- derive the least-squares objective and interpret its geometry;
- explain the normal equations and the role of the Gram matrix $X^\top X$;
- distinguish existence from uniqueness of least-squares solutions;
- explain how multicollinearity and conditioning affect estimation;
- derive the Gaussian-noise likelihood and recover least squares as MLE;
- explain ridge regression as coefficient shrinkage and Gaussian-prior MAP estimation;
- explain why lasso can set coefficients exactly to zero; and
- identify underfitting, good fit, and overfitting in polynomial regression experiments.

## 1. The supervised regression problem

We observe training data

$$
S = \{(x_i, y_i)\}_{i=1}^n,
$$

where each feature vector satisfies $x_i \in \mathbb{R}^d$ and each target satisfies $y_i \in \mathbb{R}$.
In a linear model, we predict

$$
\hat{y}_i = f_w(x_i) = w^\top x_i + b,
$$

where $w \in \mathbb{R}^d$ is the coefficient vector and $b \in \mathbb{R}$ is the intercept.

It is often convenient to absorb the intercept into the parameter vector.
If we augment each input with a leading $1$, then we can write

$$
\tilde{x}_i =
\begin{bmatrix}
1 \\
x_i
\end{bmatrix}
\in \mathbb{R}^{d+1},
\qquad
\tilde{w} =
\begin{bmatrix}
b \\
w
\end{bmatrix}
\in \mathbb{R}^{d+1},
$$

so that

$$
f_{\tilde{w}}(x_i) = \tilde{w}^\top \tilde{x}_i.
$$

To simplify notation, these notes mostly suppress the tilde and assume the intercept has already been handled.

## 2. Least squares as empirical risk minimization

For regression, the canonical loss is squared loss:

$$
\ell(\hat{y}, y) = (\hat{y} - y)^2.
$$

The empirical risk for a linear predictor is

$$
\widehat{R}_n(w) = \frac{1}{n}\sum_{i=1}^n (y_i - w^\top x_i)^2.
$$

Minimizing this objective gives the **least-squares estimator**:

$$
\widehat{w}
\in
\arg\min_{w \in \mathbb{R}^d}
\frac{1}{n}\sum_{i=1}^n (y_i - w^\top x_i)^2.
$$

Introduce the design matrix

$$
X =
\begin{bmatrix}
- & x_1^\top & - \\
- & x_2^\top & - \\
& \vdots & \\
- & x_n^\top & -
\end{bmatrix}
\in \mathbb{R}^{n \times d},
\qquad
y =
\begin{bmatrix}
y_1 \\
\vdots \\
y_n
\end{bmatrix}
\in \mathbb{R}^n.
$$

Then the objective becomes

$$
L(w) = \frac{1}{n}\|y - Xw\|_2^2.
$$

The residual vector is

$$
r(w) = y - Xw.
$$

Least squares chooses $w$ so that the Euclidean length of the residual vector is as small as possible.

### Why squared loss is mathematically convenient

The squared-loss objective is:

- differentiable;
- convex in $w$;
- quadratic in $w$; and
- solvable either analytically or by generic optimization methods.

Expanding the objective gives

$$
L(w) = \frac{1}{n}(y^\top y - 2y^\top Xw + w^\top X^\top X w).
$$

This makes the quadratic structure explicit.

## 3. The normal equations

Taking the gradient of $L(w)$ yields

$$
\nabla_w L(w) = -\frac{2}{n}X^\top(y - Xw)
= \frac{2}{n}(X^\top X w - X^\top y).
$$

Setting the gradient to zero gives the **normal equations**:

$$
X^\top X \widehat{w} = X^\top y.
$$

If $X^\top X$ is invertible, then

$$
\widehat{w} = (X^\top X)^{-1}X^\top y.
$$

This is the familiar closed form.
The derivation is developed carefully in [normal-equations.md](../derivations/normal-equations.md).

### Existence and uniqueness

The least-squares objective always has at least one minimizer because it is continuous, convex, and grows quadratically in the directions seen by $X$.
Uniqueness is more delicate.

- If the columns of $X$ are linearly independent, then $X^\top X$ is positive definite and the minimizer is unique.
- If the columns of $X$ are linearly dependent, then $X^\top X$ is singular and there may be infinitely many minimizers.

In the rank-deficient case, all minimizers produce the same fitted values $X\widehat{w}$, even though the parameter vectors may differ.
One canonical choice is the minimum-norm solution

$$
\widehat{w} = X^+ y,
$$

where $X^+$ is the Moore-Penrose pseudoinverse.

## 4. Geometry: projection onto the column space

The geometric meaning of least squares is essential.
The vector of fitted values

$$
\hat{y} = X\widehat{w}
$$

must lie in the column space of $X$:

$$
\hat{y} \in \operatorname{col}(X) \subseteq \mathbb{R}^n.
$$

Least squares chooses the point in $\operatorname{col}(X)$ closest to $y$ in Euclidean distance.
Equivalently, $\hat{y}$ is the orthogonal projection of $y$ onto $\operatorname{col}(X)$.

The residual vector

$$
r = y - \hat{y}
$$

is therefore orthogonal to every column of $X$:

$$
X^\top r = 0.
$$

This is exactly the normal-equation condition

$$
X^\top(y - X\widehat{w}) = 0.
$$

So the optimization statement and the geometric statement are the same result viewed from two angles.

### Hat matrix

When $X^\top X$ is invertible, the fitted values can be written as

$$
\hat{y} = X(X^\top X)^{-1}X^\top y = Hy,
$$

where

$$
H = X(X^\top X)^{-1}X^\top
$$

is the **hat matrix**.
It maps the observed target vector $y$ to its projection onto $\operatorname{col}(X)$.

Important properties:

- $H$ is symmetric;
- $H$ is idempotent, meaning $H^2 = H$; and
- $\operatorname{rank}(H) = \operatorname{rank}(X)$.

These facts encode the geometry of projection.

## 5. Statistical model and Gaussian-noise viewpoint

To move from optimization to probability, specify a stochastic data model.
Assume

$$
Y_i = w^\top x_i + \varepsilon_i,
$$

where the noise variables satisfy

$$
\varepsilon_i \overset{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2).
$$

Conditioned on the design matrix $X$, this implies

$$
Y \mid X, w, \sigma^2 \sim \mathcal{N}(Xw, \sigma^2 I_n).
$$

The conditional likelihood is

$$
p(y \mid X, w, \sigma^2)
=
(2\pi \sigma^2)^{-n/2}
\exp\left(
-\frac{1}{2\sigma^2}\|y - Xw\|_2^2
\right).
$$

Taking the negative log-likelihood and dropping constants that do not depend on $w$ gives

$$
-\log p(y \mid X, w, \sigma^2)
=
\frac{1}{2\sigma^2}\|y - Xw\|_2^2 + \text{constant}.
$$

Therefore maximizing likelihood in $w$ is equivalent to minimizing squared error.
Under Gaussian noise, least squares is the **maximum likelihood estimator**.

The detailed argument is also included in [normal-equations.md](../derivations/normal-equations.md).

## 6. Interpreting the coefficients

In a linear model, each coefficient describes the change in the prediction produced by a one-unit change in a feature, holding the remaining features fixed.
If the model includes an intercept and feature $j$ is not standardized, then

$$
w_j
$$

has units of

$$
\frac{\text{units of target}}{\text{units of feature } j}.
$$

This interpretability is one reason linear regression remains important even when more flexible models are available.

However, coefficient interpretation becomes unstable when features are strongly correlated.
Then multiple coefficient vectors may produce similar predictions.

## 7. Conditioning and multicollinearity

The matrix

$$
X^\top X
$$

is the Gram matrix of the feature columns.
When features are nearly linearly dependent, $X^\top X$ becomes ill-conditioned.

Consequences include:

- unstable coefficient estimates;
- high variance of the estimator;
- numerical sensitivity to small perturbations; and
- difficulty interpreting individual coefficients.

This matters even when prediction remains acceptable.
Linear regression therefore forces us to distinguish:

- the problem of fitting the conditional mean; from
- the problem of estimating a stable parameter vector.

Regularization addresses this tension.

## 8. Ridge regression

Ridge regression augments least squares with an $\ell_2$ penalty:

$$
\widehat{w}_{\mathrm{ridge}}
\in
\arg\min_w
\frac{1}{n}\|y - Xw\|_2^2 + \lambda \|w\|_2^2,
$$

where $\lambda \geq 0$ controls the strength of shrinkage.

Effects of the penalty:

- large coefficients are discouraged;
- collinearity becomes less damaging;
- the problem becomes strictly convex for $\lambda > 0$; and
- variance is reduced, usually at the cost of adding some bias.

The ridge normal equations are

$$
(X^\top X + n\lambda I)\widehat{w}_{\mathrm{ridge}} = X^\top y.
$$

If the intercept is represented explicitly, it is common not to penalize it.

### Geometric view of ridge

Ridge can be interpreted either as:

- minimizing squared loss subject to an $\ell_2$ ball constraint $\|w\|_2^2 \leq c$; or
- minimizing penalized squared loss.

The constraint set is round and smooth.
Because the loss contours are ellipsoids, the solution is typically shrunk continuously toward the origin rather than driven to exact zeros.

### Probabilistic view of ridge

Assume the same Gaussian noise model as before and add a Gaussian prior

$$
w \sim \mathcal{N}(0, \tau^2 I).
$$

Then

$$
-\log p(w) = \frac{1}{2\tau^2}\|w\|_2^2 + \text{constant}.
$$

Maximizing the posterior is equivalent to minimizing

$$
\frac{1}{2\sigma^2}\|y - Xw\|_2^2 + \frac{1}{2\tau^2}\|w\|_2^2.
$$

So ridge regression is a **MAP estimator** under a Gaussian prior.
This is derived in [ridge-lasso.md](../derivations/ridge-lasso.md).

## 9. Lasso regression

Lasso replaces the $\ell_2$ penalty with an $\ell_1$ penalty:

$$
\widehat{w}_{\mathrm{lasso}}
\in
\arg\min_w
\frac{1}{n}\|y - Xw\|_2^2 + \lambda \|w\|_1,
$$

where

$$
\|w\|_1 = \sum_{j=1}^d |w_j|.
$$

Lasso remains convex, but it is no longer differentiable at coordinates where $w_j = 0$.
That non-smooth geometry is not a nuisance detail.
It is the reason exact sparsity can appear.

### Why lasso yields sparse solutions

The $\ell_1$ constraint region is a diamond in two dimensions, and more generally a polytope with sharp corners aligned to the coordinate axes.
Quadratic loss contours often touch that region at a corner.
At a corner, one or more coordinates are exactly zero.

So the geometry of the feasible set promotes sparse solutions.
Ridge shrinks.
Lasso both shrinks and performs variable selection.

### Probabilistic view of lasso

If

$$
p(w_j) \propto \exp(-\alpha |w_j|),
$$

independently across coordinates, then the prior is Laplace and the negative log-prior is proportional to $\|w\|_1$.
Therefore lasso can also be interpreted as MAP estimation, but unlike ridge it usually has no closed-form solution and is solved numerically.

## 10. Choosing between unregularized, ridge, and lasso

A useful first-pass comparison is:

- ordinary least squares: unbiased under the classical correctly specified model, but high variance when features are noisy or collinear;
- ridge: good when many features carry signal and stability matters more than exact sparsity;
- lasso: good when a sparse representation is plausible and feature selection is useful.

In practice, hyperparameters such as $\lambda$ are chosen by validation or cross-validation, not by training error alone.

## 11. Underfitting, good fit, and overfitting

Even linear regression can overfit when the feature representation is flexible.
For example, if we use polynomial features

$$
\phi(x) = (1, x, x^2, \ldots, x^p),
$$

then the model is still linear in its parameters:

$$
f_w(x) = w^\top \phi(x),
$$

but its function class becomes more expressive as $p$ increases.

This gives a clean demonstration of:

- **underfitting** at low degree, when the model cannot represent the underlying pattern;
- **good fit** at intermediate degree, when the signal is captured without chasing noise; and
- **overfitting** at high degree, when the curve bends to match sample-specific fluctuations.

The notebook [LM-01-linear-regression.ipynb](../notebooks/LM-01-linear-regression.ipynb) visualizes this progression directly.

## 12. Category theory insertion point

The canonical mathematics of linear regression does not require category theory.
Still, there is a disciplined structural reading available.

A feature map

$$
\phi : \mathcal{X} \to \mathbb{R}^d
$$

followed by a linear functional

$$
\ell_w : \mathbb{R}^d \to \mathbb{R}
$$

is a composition

$$
\mathcal{X} \xrightarrow{\phi} \mathbb{R}^d \xrightarrow{\ell_w} \mathbb{R}.
$$

This viewpoint clarifies that changing the feature representation changes the object on which the linear map acts.
For this module, that perspective is explanatory rather than foundational.

## 13. Unity Theory insertion point

Any Unity Theory material here should remain companion-only.
One disciplined interpretation is that regression tries to stabilize a coherent scalar summary of structured variation in the input.
That language may be useful philosophically, but the primary content remains least squares, projection, Gaussian likelihood, and regularization.

## Summary

Linear regression is simple enough to solve exactly and rich enough to expose the central themes of machine learning.

- Optimization gives the least-squares objective.
- Geometry shows that least squares is orthogonal projection.
- Probability shows that least squares is MLE under Gaussian noise.
- Regularization shows how bias can be traded for variance reduction.
- Ridge and lasso reveal two distinct notions of simplicity: small norm and sparsity.

Because these viewpoints agree on one model, linear regression is a foundational bridge between linear algebra, probability, optimization, and statistical learning.

## References

- Trevor Hastie, Robert Tibshirani, and Jerome Friedman, *The Elements of Statistical Learning*, 2nd ed.
- Christopher Bishop, *Pattern Recognition and Machine Learning*.
- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*.
