---
title: "Convexity and First-Order Lower Bounds"
module: "01-optimization"
lesson: "convexity-basics"
doc_type: "notes"
topic: "convexity"
status: "draft"
prerequisites:
  - "00-math-toolkit/multivariable-calculus"
  - "00-math-toolkit/linear-algebra"
updated: "2026-04-09"
owner: "curriculum-team"
tags:
  - "optimization"
  - "convexity"
---

# Convexity and First-Order Lower Bounds

## Motivation
Convexity gives a clean bridge from geometry to optimization guarantees. In machine learning, it is one of the main reasons linear models and least-squares objectives are analytically tractable.

## Assumptions and Notation
Let $f : \mathbb{R}^d \to \mathbb{R}$ be differentiable. Let $x, y \in \mathbb{R}^d$.

> **Definition.** The function $f$ is convex if for every $t \in [0,1]$,
> $f(tx + (1-t)y) \leq t f(x) + (1-t) f(y)$.

## Main Result
> **Proposition.** If $f$ is differentiable and convex, then
> $f(y) \geq f(x) + \nabla f(x)^\top (y - x)$.

## Computational Interpretation
The gradient defines a supporting hyperplane. Gradient-based optimization exploits this local linear information to choose a descent direction even when the full objective is hard to minimize in closed form.
