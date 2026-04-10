---
title: "KKT Conditions"
module: "01-optimization"
lesson: "kkt-conditions"
doc_type: "derivation"
topic: "constrained-optimization"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
  - "00-math-toolkit/multivariable-calculus"
  - "01-optimization/convexity-and-optimization"
updated: "2026-04-09"
owner: "curriculum-team"
tags:
  - "optimization"
  - "kkt"
  - "lagrangian"
  - "duality"
---

## Purpose

This derivation explains where the Karush-Kuhn-Tucker conditions come from and how to interpret each term geometrically.
The goal here is not full constraint qualification theory, but a clean derivation that is sufficient for later ML applications such as SVMs and regularized convex programs.

## Problem setup

Consider

$$
\min_{\mathbf{x} \in \mathbb{R}^d} f(\mathbf{x})
$$

subject to inequality constraints

$$
g_i(\mathbf{x}) \leq 0, \qquad i=1,\dots,m,
$$

and equality constraints

$$
h_j(\mathbf{x}) = 0, \qquad j=1,\dots,p.
$$

Assume all functions are differentiable.

Define the Lagrangian

$$
\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})
= f(\mathbf{x})
+ \sum_{i=1}^m \lambda_i g_i(\mathbf{x})
+ \sum_{j=1}^p \nu_j h_j(\mathbf{x}),
$$

where the inequality multipliers satisfy $\lambda_i \geq 0$.

## 1. Equality-constrained warm-up

If only the equality constraints are present, a feasible perturbation $\mathbf{d}$ at $\mathbf{x}^\star$ must satisfy the linearized conditions

$$
\nabla h_j(\mathbf{x}^\star)^\top \mathbf{d} = 0
\qquad \text{for all } j.
$$

So feasible directions lie in the tangent space

$$
T_{\mathbf{x}^\star}
= \left\{
\mathbf{d} :
\nabla h_j(\mathbf{x}^\star)^\top \mathbf{d} = 0
\text{ for all } j
\right\}.
$$

At a local constrained minimum, the directional derivative of $f$ along every feasible direction must vanish:

$$
\nabla f(\mathbf{x}^\star)^\top \mathbf{d} = 0
\qquad \text{for all } \mathbf{d} \in T_{\mathbf{x}^\star}.
$$

Therefore $\nabla f(\mathbf{x}^\star)$ belongs to the span of the normal vectors $\nabla h_j(\mathbf{x}^\star)$.
So there exist multipliers $\nu_1^\star, \dots, \nu_p^\star$ such that

$$
\nabla f(\mathbf{x}^\star)
+ \sum_{j=1}^p \nu_j^\star \nabla h_j(\mathbf{x}^\star)
= 0.
$$

This is the stationarity equation for equality-constrained optimization.

## 2. Why inequality multipliers must be nonnegative

Now include a single inequality constraint $g(\mathbf{x}) \leq 0$.
Suppose $g(\mathbf{x}^\star) < 0$.
Then the constraint is locally inactive and should not influence first-order stationarity.
This suggests its multiplier should be zero.

Suppose instead that $g(\mathbf{x}^\star)=0$, so the constraint is active.
Locally feasible directions must satisfy

$$
\nabla g(\mathbf{x}^\star)^\top \mathbf{d} \leq 0,
$$

meaning movement is permitted only toward the feasible side of the boundary.
The constraint gradient points outward from the feasible region.
To balance the objective gradient against outward normals, the corresponding multiplier must enter with a nonnegative coefficient.

That is why the Lagrangian uses $\lambda_i \geq 0$ for minimization problems with $g_i(\mathbf{x}) \leq 0$.

## 3. Stationarity with active constraints

At a regular local minimizer $\mathbf{x}^\star$, the objective gradient can be decomposed into active constraint normals:

$$
\nabla f(\mathbf{x}^\star)
+ \sum_{i=1}^m \lambda_i^\star \nabla g_i(\mathbf{x}^\star)
+ \sum_{j=1}^p \nu_j^\star \nabla h_j(\mathbf{x}^\star)
= 0.
$$

Only active inequalities should matter.
If a constraint is inactive, then $g_i(\mathbf{x}^\star) < 0$ and its multiplier should vanish.
This is encoded by complementary slackness.

## 4. Complementary slackness

For each inequality constraint,

$$
\lambda_i^\star g_i(\mathbf{x}^\star)=0.
$$

This compact equation means exactly one of the following occurs:

- $g_i(\mathbf{x}^\star) < 0$ and then $\lambda_i^\star = 0$;
- $g_i(\mathbf{x}^\star) = 0$ and then $\lambda_i^\star$ may be positive.

An inactive inequality contributes no force term.
An active inequality contributes a normal vector scaled by its multiplier.

## 5. The KKT system

Collecting the preceding conditions gives the KKT system:

### Stationarity

$$
\nabla f(\mathbf{x}^\star)
+ \sum_{i=1}^m \lambda_i^\star \nabla g_i(\mathbf{x}^\star)
+ \sum_{j=1}^p \nu_j^\star \nabla h_j(\mathbf{x}^\star)
= 0.
$$

### Primal feasibility

$$
g_i(\mathbf{x}^\star) \leq 0,
\qquad
h_j(\mathbf{x}^\star)=0.
$$

### Dual feasibility

$$
\lambda_i^\star \geq 0.
$$

### Complementary slackness

$$
\lambda_i^\star g_i(\mathbf{x}^\star)=0.
$$

Under suitable constraint qualifications, these are necessary at a local optimum.
For convex problems they often become sufficient as well.

## 6. Sufficiency in the convex case

Assume:

- $f$ is convex;
- each inequality function $g_i$ is convex;
- each equality function $h_j$ is affine.

Suppose $(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ satisfies the KKT conditions.
Then $\mathbf{x}^\star$ is globally optimal.

### Proof sketch

By convexity of $f$ and each $g_i$, and affinity of each $h_j$,

$$
f(\mathbf{x}) \geq f(\mathbf{x}^\star) + \nabla f(\mathbf{x}^\star)^\top(\mathbf{x}-\mathbf{x}^\star),
$$

$$
g_i(\mathbf{x}) \geq g_i(\mathbf{x}^\star) + \nabla g_i(\mathbf{x}^\star)^\top(\mathbf{x}-\mathbf{x}^\star),
$$

$$
h_j(\mathbf{x}) = h_j(\mathbf{x}^\star) + \nabla h_j(\mathbf{x}^\star)^\top(\mathbf{x}-\mathbf{x}^\star).
$$

Multiply the inequality constraints by $\lambda_i^\star \geq 0$, sum everything, and substitute the stationarity equation.
For any feasible $\mathbf{x}$, the linear terms cancel and the remaining KKT terms reduce to

$$
f(\mathbf{x}) \geq f(\mathbf{x}^\star)
- \sum_{i=1}^m \lambda_i^\star g_i(\mathbf{x}^\star).
$$

By complementary slackness, the final sum is zero.
Hence

$$
f(\mathbf{x}) \geq f(\mathbf{x}^\star)
$$

for every feasible $\mathbf{x}$.
So $\mathbf{x}^\star$ is globally optimal.

## 7. Duality connection

The dual function is

$$
q(\boldsymbol{\lambda}, \boldsymbol{\nu})
= \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}).
$$

When $\lambda_i \geq 0$, weak duality gives

$$
q(\boldsymbol{\lambda}, \boldsymbol{\nu}) \leq p^\star,
$$

where $p^\star$ is the primal optimal value.

At a KKT point for a convex problem with strong duality, the primal and dual optima meet:

$$
q(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star) = f(\mathbf{x}^\star) = p^\star.
$$

So KKT conditions connect:

- geometry of feasible directions;
- algebra of the Lagrangian;
- optimality certificates; and
- the equality of primal and dual values in well-behaved convex problems.

## 8. ML-facing example: hard-margin SVM

Write the constraints as

$$
g_i(\mathbf{w}, b) = 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b) \leq 0.
$$

The primal problem is

$$
\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|_2^2
\quad \text{subject to} \quad
g_i(\mathbf{w}, b) \leq 0.
$$

The Lagrangian is

$$
\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha})
= \frac{1}{2}\|\mathbf{w}\|_2^2
+ \sum_{i=1}^n \alpha_i \bigl(1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b)\bigr),
$$

with $\alpha_i \geq 0$.

KKT says:

- stationarity with respect to $\mathbf{w}$ gives
  $$
  \mathbf{w}^\star = \sum_{i=1}^n \alpha_i^\star y_i \mathbf{x}_i;
  $$
- stationarity with respect to $b$ gives
  $$
  \sum_{i=1}^n \alpha_i^\star y_i = 0;
  $$
- complementary slackness gives
  $$
  \alpha_i^\star \bigl(1 - y_i(\mathbf{w}^{\star\top}\mathbf{x}_i + b^\star)\bigr)=0.
  $$

So only points on the margin can carry positive multipliers.
Those are the support vectors.

## References

- Stephen Boyd and Lieven Vandenberghe, *Convex Optimization*, Chapter 5.
- Jorge Nocedal and Stephen Wright, *Numerical Optimization*, sections on constrained optimality conditions.
