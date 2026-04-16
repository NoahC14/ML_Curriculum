---
title: "Optimization Theory Exercises"
module: "01-optimization"
lesson: "optimization-theory-exercises"
doc_type: "exercise"
topic: "optimization-theory"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
  - "00-math-toolkit/multivariable-calculus"
  - "01-optimization/convexity-and-optimization"
updated: "2026-04-09"
owner: "curriculum-team"
tags:
  - "optimization"
  - "convexity"
  - "kkt"
  - "duality"
  - "machine-learning"
---

## Purpose

These exercises move from core convexity definitions to constrained optimality and duality intuition.

- Tier 1: geometric fluency and derivative checks.
- Tier 2: proofs and derivations of optimality conditions.
- Tier 3: ML-facing synthesis around PCA, SVMs, and non-convex objectives.

## Exercise 1: Convex combinations and feasible sets

**Taxonomy**

- `difficulty`: `foundational`
- `type`: `proof`
- `tags`: `convexity`, `feasible-set`, `definition-check`

> **Problem.** Let
> $$
> \mathcal{C} = \{\mathbf{x} \in \mathbb{R}^2 : x_1 \geq 0,\ x_2 \geq 0,\ x_1 + x_2 \leq 1\}.
> $$
> Show directly from the definition that $\mathcal{C}$ is convex.

**Deliverables**

- A short proof using an arbitrary convex combination of two feasible points.

## Exercise 2: Non-convex feasible region

> **Problem.** Consider
> $$
> \mathcal{A} = \{\mathbf{x} \in \mathbb{R}^2 : 1 \leq \|\mathbf{x}\|_2 \leq 2\}.
> $$
> Show that $\mathcal{A}$ is not convex.

**Hints**

- Pick two points on the inner boundary whose midpoint falls into the hole.

**Deliverables**

- Two feasible points.
- A one-sentence convexity failure argument.

## Exercise 3: Convexity of a quadratic

> **Problem.** Let
> $$
> f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top Q \mathbf{x} + \mathbf{b}^\top \mathbf{x} + c,
> $$
> where $Q = Q^\top$.
> Show that $f$ is convex if $Q \succeq 0$ and strongly convex if $Q \succeq \mu I$ for some $\mu > 0$.

**Hints**

- Compute the Hessian.
- Use the second-order characterization of convexity.

**Deliverables**

- A short derivation.

## Exercise 4: First-order condition for convexity

> **Problem.** Assume $f : \mathbb{R}^d \to \mathbb{R}$ is differentiable and convex.
> Prove that
> $$
> f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x})
> $$
> for all $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$.

**Hints**

- Apply convexity to $\mathbf{x} + t(\mathbf{y} - \mathbf{x})$.
- Differentiate with respect to $t$ at $t=0$.

**Deliverables**

- A proof.

## Exercise 5: Stationary points and classification

> **Problem.** Let
> $$
> f(x_1, x_2) = x_1^2 + 4x_2^2 - 2x_1x_2 - 2x_1.
> $$
> Find all stationary points and classify them using the Hessian.

**Deliverables**

- The gradient.
- The stationary point.
- The Hessian.
- The classification with a one-sentence justification.

## Exercise 6: Global optimality in the convex case

> **Problem.** Let $f$ be differentiable and convex.
> Prove that if $\nabla f(\mathbf{x}^\star)=0$, then $\mathbf{x}^\star$ is a global minimizer.

**Deliverables**

- A proof using the first-order convexity inequality.

## Exercise 7: Lagrange multipliers for a sphere constraint

> **Problem.** Minimize
> $$
> f(\mathbf{x}) = x_1 + 2x_2 + 3x_3
> $$
> subject to
> $$
> x_1^2 + x_2^2 + x_3^2 = 1.
> $$
> Derive the Lagrange equations and solve for the minimizer.

**Hints**

- The gradient of the objective must be parallel to the gradient of the constraint.

**Deliverables**

- The Lagrangian.
- The stationarity equations.
- The minimizing point.

## Exercise 8: PCA as constrained optimization

> **Problem.** Let $S \in \mathbb{R}^{d \times d}$ be symmetric.
> Show that maximizing $\mathbf{v}^\top S\mathbf{v}$ subject to $\|\mathbf{v}\|_2 = 1$ leads to the eigenvalue equation
> $$
> S\mathbf{v} = \lambda \mathbf{v}.
> $$
> Explain which eigenvector solves the maximization problem.

**Deliverables**

- A derivation.
- A one-sentence ML interpretation.

## Exercise 9: KKT system for a one-dimensional inequality problem

> **Problem.** Solve
> $$
> \min_x \ (x-2)^2
> \quad \text{subject to} \quad
> x \geq 0.
> $$
> Rewrite the constraint in the form $g(x) \leq 0$, write the KKT conditions, and solve them.

**Deliverables**

- The constraint function $g$.
- The full KKT system.
- The optimizer and multiplier.

## Exercise 10: Active and inactive constraints

> **Problem.** Consider
> $$
> \min_{x_1,x_2} \ x_1^2 + x_2^2
> \quad \text{subject to} \quad
> x_1 + x_2 \geq 1,\ x_1 \geq 0,\ x_2 \geq 0.
> $$
> Determine which constraints are active at the optimum and solve the problem using KKT.

**Hints**

- Write all inequalities in the form $g_i(\mathbf{x}) \leq 0$.
- Use symmetry if you see it, but still verify through KKT.

**Deliverables**

- The optimizer.
- The active set.
- The nonzero multipliers.

## Exercise 11: Weak duality check

> **Problem.** Consider
> $$
> \min_x \ x^2
> \quad \text{subject to} \quad
> x \geq 1.
> $$
> Form the Lagrangian and the dual function, then verify directly that the dual value is always a lower bound on the primal optimum.

**Deliverables**

- The Lagrangian.
- The dual function.
- A short weak-duality verification.

## Exercise 12: Hard-margin SVM KKT interpretation

> **Problem.** For hard-margin SVM,
> $$
> \min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|_2^2
> \quad \text{subject to} \quad
> y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 \quad \text{for } i=1,\dots,n,
> $$
> write the Lagrangian and derive the stationarity conditions with respect to $\mathbf{w}$ and $b$.
> Then explain the meaning of complementary slackness in this setting.

**Deliverables**

- The Lagrangian.
- The stationarity equations.
- A short explanation of why only support vectors have positive multipliers.

## Exercise 13: A non-convex landscape

> **Problem.** Consider
> $$
> f(x) = x^4 - 3x^2.
> $$
> Find all stationary points and classify them.
> Then explain why this example shows that stationary points alone do not solve non-convex optimization.

**Deliverables**

- The first derivative.
- The second derivative.
- All stationary points with classifications.
- A short conceptual explanation.

## Exercise 14: Logistic loss Hessian

> **Problem.** Let
> $$
> \ell(z) = \log(1 + e^{-z}).
> $$
> Compute $\ell'(z)$ and $\ell''(z)$ and conclude that $\ell$ is convex.
> Briefly explain why this matters for linear classification objectives built from logistic loss.

**Deliverables**

- The first and second derivatives.
- A one-sentence convexity conclusion.
- A one-sentence ML relevance statement.
