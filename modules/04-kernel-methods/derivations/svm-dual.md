---
title: "Dual Derivation for the Soft-Margin SVM"
module: "04-kernel-methods"
lesson: "svm-dual"
doc_type: "derivation"
topic: "lagrangian-duality"
status: "draft"
prerequisites:
  - "01-optimization/convexity-and-optimization"
  - "01-optimization/kkt-conditions"
  - "04-kernel-methods/kernel-methods"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "svm"
  - "duality"
  - "lagrangian"
  - "kernel-methods"
  - "support-vectors"
---

## Purpose

This derivation shows how the soft-margin SVM becomes a dual quadratic program whose data dependence appears only through inner products.
That is the step that makes kernelization possible.

## 1. Primal problem

Let training data be

$$
\{(x_i,y_i)\}_{i=1}^n,
\qquad
x_i \in \mathbb{R}^d,
\qquad
y_i \in \{-1,+1\}.
$$

The soft-margin primal SVM is

$$
\min_{w,b,\xi}
\frac{1}{2}\|w\|_2^2 + C\sum_{i=1}^n \xi_i
$$

subject to

$$
y_i(w^\top x_i + b) \geq 1 - \xi_i,
\qquad
\xi_i \geq 0,
\qquad
i=1,\dots,n.
$$

Here:

- $w \in \mathbb{R}^d$ is the normal vector of the separating hyperplane;
- $b \in \mathbb{R}$ is the offset;
- $\xi_i \in \mathbb{R}$ are slack variables; and
- $C > 0$ controls the tradeoff between margin size and margin violations.

## 2. Put constraints in standard form

Rewrite the constraints as

$$
1 - \xi_i - y_i(w^\top x_i + b) \leq 0
$$

and

$$
-\xi_i \leq 0.
$$

Introduce Lagrange multipliers

$$
\alpha_i \geq 0
\quad \text{for } 1 - \xi_i - y_i(w^\top x_i + b) \leq 0
$$

and

$$
\mu_i \geq 0
\quad \text{for } -\xi_i \leq 0.
$$

## 3. Form the Lagrangian

The Lagrangian is

$$
\mathcal{L}(w,b,\xi,\alpha,\mu)
= \frac{1}{2}\|w\|_2^2 + C\sum_{i=1}^n \xi_i
+ \sum_{i=1}^n \alpha_i \bigl(1 - \xi_i - y_i(w^\top x_i + b)\bigr)
- \sum_{i=1}^n \mu_i \xi_i.
$$

Expand and group like terms:

$$
\mathcal{L}
= \frac{1}{2}\|w\|_2^2
+ \sum_{i=1}^n \alpha_i
+ \sum_{i=1}^n \xi_i(C - \alpha_i - \mu_i)
- \sum_{i=1}^n \alpha_i y_i w^\top x_i
- b\sum_{i=1}^n \alpha_i y_i.
$$

To obtain the dual function, minimize $\mathcal{L}$ over the primal variables $w$, $b$, and $\xi$.

## 4. Stationarity with respect to \(w\)

Differentiate with respect to $w$:

$$
\nabla_w \mathcal{L}
= w - \sum_{i=1}^n \alpha_i y_i x_i.
$$

Set the gradient to zero:

$$
w = \sum_{i=1}^n \alpha_i y_i x_i.
$$

This already shows that the separating direction lies in the span of the training inputs.

## 5. Stationarity with respect to \(b\)

Differentiate with respect to $b$:

$$
\frac{\partial \mathcal{L}}{\partial b}
= -\sum_{i=1}^n \alpha_i y_i.
$$

Set this to zero:

$$
\sum_{i=1}^n \alpha_i y_i = 0.
$$

This is the equality constraint in the dual problem.

## 6. Stationarity with respect to \(\xi_i\)

For each $i$,

$$
\frac{\partial \mathcal{L}}{\partial \xi_i}
= C - \alpha_i - \mu_i.
$$

Set this to zero:

$$
\mu_i = C - \alpha_i.
$$

Since $\mu_i \geq 0$, we obtain

$$
\alpha_i \leq C.
$$

Together with $\alpha_i \geq 0$, this yields the box constraint

$$
0 \leq \alpha_i \leq C.
$$

## 7. Substitute back into the Lagrangian

Use

$$
w = \sum_{i=1}^n \alpha_i y_i x_i.
$$

Then

$$
\frac{1}{2}\|w\|_2^2
= \frac{1}{2}
\left\langle
\sum_{i=1}^n \alpha_i y_i x_i,
\sum_{j=1}^n \alpha_j y_j x_j
\right\rangle
= \frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n \alpha_i\alpha_j y_i y_j x_i^\top x_j.
$$

Also,

$$
\sum_{i=1}^n \alpha_i y_i w^\top x_i
= \sum_{i=1}^n \alpha_i y_i
\left(
\sum_{j=1}^n \alpha_j y_j x_j^\top x_i
\right)
= \sum_{i=1}^n \sum_{j=1}^n \alpha_i\alpha_j y_i y_j x_i^\top x_j.
$$

The $\xi_i$ terms vanish because stationarity enforces

$$
C - \alpha_i - \mu_i = 0.
$$

The $b$ term vanishes because

$$
\sum_{i=1}^n \alpha_i y_i = 0.
$$

Therefore the dual function is

$$
g(\alpha)
= \sum_{i=1}^n \alpha_i
- \frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n \alpha_i\alpha_j y_i y_j x_i^\top x_j.
$$

The dual optimization problem is thus

$$
\max_{\alpha \in \mathbb{R}^n}
\sum_{i=1}^n \alpha_i
- \frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n \alpha_i\alpha_j y_i y_j x_i^\top x_j
$$

subject to

$$
0 \leq \alpha_i \leq C,
\qquad
\sum_{i=1}^n \alpha_i y_i = 0.
$$

This is a concave quadratic maximization with linear constraints.

## 8. Recovering the predictor

After solving for $\alpha^\star$, the weight vector is

$$
w^\star = \sum_{i=1}^n \alpha_i^\star y_i x_i.
$$

Prediction uses

$$
f(x) = (w^\star)^\top x + b^\star
= \sum_{i=1}^n \alpha_i^\star y_i x_i^\top x + b^\star.
$$

Only indices with $\alpha_i^\star > 0$ contribute.
These are the support vectors.

For any support vector with $0 < \alpha_i^\star < C$, complementary slackness gives

$$
y_i\bigl((w^\star)^\top x_i + b^\star\bigr) = 1,
$$

so one can recover $b^\star$ from

$$
b^\star = y_i - (w^\star)^\top x_i.
$$

In practice, averaging over margin support vectors improves numerical stability.

## 9. Kernelization

The dual objective and prediction rule depend on data only through dot products $x_i^\top x_j$ and $x_i^\top x$.
Therefore, if a kernel satisfies

$$
K(x,z) = \langle \phi(x), \phi(z) \rangle,
$$

we replace

$$
x_i^\top x_j \rightsquigarrow K(x_i,x_j),
\qquad
x_i^\top x \rightsquigarrow K(x_i,x).
$$

The dual becomes

$$
\max_{\alpha}
\sum_{i=1}^n \alpha_i
- \frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n
\alpha_i\alpha_j y_i y_j K(x_i,x_j)
$$

subject to the same constraints

$$
0 \leq \alpha_i \leq C,
\qquad
\sum_{i=1}^n \alpha_i y_i = 0.
$$

Prediction becomes

$$
f(x)
= \sum_{i=1}^n \alpha_i^\star y_i K(x_i,x) + b^\star.
$$

This is the kernel SVM.

## 10. ML interpretation

The dual derivation matters for three reasons:

1. it exposes the role of pairwise similarities;
2. it shows why only support vectors determine the classifier; and
3. it turns a nonlinear classifier in input space into a convex optimization problem over coefficients.

The SVM is therefore a clean example of how optimization, geometry, and representation interact.
