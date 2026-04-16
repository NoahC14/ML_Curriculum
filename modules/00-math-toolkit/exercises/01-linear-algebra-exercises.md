---
title: "Linear Algebra Exercises"
module: "00-math-toolkit"
lesson: "linear-algebra-exercises"
doc_type: "exercise"
topic: "linear-algebra"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
updated: "2026-04-09"
owner: "curriculum-team"
tags:
  - "linear-algebra"
  - "exercises"
  - "pca"
  - "svd"
---

## Purpose

These exercises are organized into three tiers:

- Tier 1: core fluency with definitions, computations, and standard identities;
- Tier 2: multi-step derivations and interpretation;
- Tier 3: ML-facing synthesis.

## Exercise 1: Span and linear dependence

**Taxonomy**

- `difficulty`: `foundational`
- `type`: `proof`
- `tags`: `span`, `linear-dependence`, `vector-space`

> **Problem.** Let
> $$
> \mathbf{v}_1 =
> \begin{bmatrix}
> 1 \\ 0 \\ 1
> \end{bmatrix},
> \quad
> \mathbf{v}_2 =
> \begin{bmatrix}
> 0 \\ 1 \\ 1
> \end{bmatrix},
> \quad
> \mathbf{v}_3 =
> \begin{bmatrix}
> 1 \\ 1 \\ 2
> \end{bmatrix}.
> $$
> Determine whether the set $\{\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3\}$ is linearly independent. If not, express one vector as a linear combination of the others.

**Hints**

- Set $\alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \alpha_3 \mathbf{v}_3 = \mathbf{0}$.
- Compare coordinates.

**Deliverables**

- A short algebraic argument.

## Exercise 2: Orthogonal projection

> **Problem.** Let
> $$
> \mathbf{U} =
> \begin{bmatrix}
> 1/\sqrt{2} \\
> 1/\sqrt{2}
> \end{bmatrix}
> \in \mathbb{R}^{2 \times 1}.
> $$
> Compute the projector $\mathbf{P} = \mathbf{U}\mathbf{U}^\top$, and use it to project
> $$
> \mathbf{x} =
> \begin{bmatrix}
> 3 \\ 1
> \end{bmatrix}
> $$
> onto the line spanned by $\mathbf{U}$.

**Hints**

- First verify that the column of $\mathbf{U}$ has unit norm.
- Then compute $\mathbf{P}\mathbf{x}$.

**Deliverables**

- The matrix $\mathbf{P}$.
- The projected vector.

## Exercise 3: Positive semidefinite covariance

> **Problem.** Let $\mathbf{x} \in \mathbb{R}^d$ be a random vector with covariance matrix $\boldsymbol{\Sigma}$. Prove that $\boldsymbol{\Sigma}$ is positive semidefinite.

**Hints**

- Start from $\mathbf{v}^\top \boldsymbol{\Sigma}\mathbf{v}$ for an arbitrary $\mathbf{v} \in \mathbb{R}^d$.
- Rewrite the result as an expectation of a square.

**Deliverables**

- A proof.

## Exercise 4: Symmetric eigenvector orthogonality

> **Problem.** Let $\mathbf{A} \in \mathbb{R}^{d \times d}$ be symmetric. Prove that eigenvectors corresponding to distinct eigenvalues are orthogonal.

**Hints**

- Compare $\mathbf{u}^\top \mathbf{A}\mathbf{v}$ and $(\mathbf{A}\mathbf{u})^\top \mathbf{v}$.

**Deliverables**

- A proof.

## Exercise 5: Spectral decomposition in a concrete case

> **Problem.** Consider
> $$
> \mathbf{A} =
> \begin{bmatrix}
> 2 & 1 \\
> 1 & 2
> \end{bmatrix}.
> $$
> Find its eigenvalues and an orthonormal eigenbasis. Then write the spectral decomposition $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$.

**Hints**

- Solve $\det(\mathbf{A} - \lambda \mathbf{I}) = 0$.
- Normalize the eigenvectors.

**Deliverables**

- Eigenvalues.
- Orthonormal eigenvectors.
- The decomposition.

## Exercise 6: First principal component from constrained optimization

> **Problem.** Let $\mathbf{S} \in \mathbb{R}^{d \times d}$ be a symmetric sample covariance matrix. Show that maximizing $\mathbf{v}^\top \mathbf{S}\mathbf{v}$ subject to $\|\mathbf{v}\|_2 = 1$ leads to the eigenvalue equation $\mathbf{S}\mathbf{v} = \lambda \mathbf{v}$.

**Hints**

- Use a Lagrange multiplier for the unit-norm constraint.

**Deliverables**

- A derivation.
- One sentence explaining the ML meaning of the solution.

## Exercise 7: Compute an SVD

> **Problem.** Let
> $$
> \mathbf{A} =
> \begin{bmatrix}
> 3 & 0 \\
> 0 & 1
> \end{bmatrix}.
> $$
> Compute an SVD of $\mathbf{A}$.

**Hints**

- Start from $\mathbf{A}^\top \mathbf{A}$.
- This matrix is already diagonal.

**Deliverables**

- The matrices $\mathbf{U}$, $\boldsymbol{\Sigma}$, and $\mathbf{V}$.
- A sentence describing the geometric action of $\mathbf{A}$.

## Exercise 8: Rank-one approximation

> **Problem.** Suppose
> $$
> \mathbf{A} = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^\top + \sigma_2 \mathbf{u}_2 \mathbf{v}_2^\top,
> \quad \sigma_1 \geq \sigma_2 > 0.
> $$
> Propose the best rank-one approximation suggested by the SVD and explain why it preserves the dominant action of the map.

**Hints**

- Keep the leading singular component.

**Deliverables**

- The proposed rank-one approximation.
- A short geometric interpretation.

## Exercise 9: Whitening transform

> **Problem.** Let
> $$
> \boldsymbol{\Sigma} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top
> $$
> be a positive definite covariance matrix. Show that the transform
> $$
> \mathbf{W} = \boldsymbol{\Lambda}^{-1/2}\mathbf{Q}^\top
> $$
> whitens the data, meaning that if $\mathbf{z} = \mathbf{W}\mathbf{x}$ for centered $\mathbf{x}$ with covariance $\boldsymbol{\Sigma}$, then $\mathrm{Cov}(\mathbf{z}) = \mathbf{I}$.

**Hints**

- Use $\mathrm{Cov}(\mathbf{W}\mathbf{x}) = \mathbf{W}\mathrm{Cov}(\mathbf{x})\mathbf{W}^\top$.

**Deliverables**

- A derivation.
- One sentence explaining one practical risk of whitening.

## Exercise 10: Conditioning and regression

> **Problem.** In least squares, the normal equations are
> $$
> \mathbf{X}^\top \mathbf{X}\mathbf{w} = \mathbf{X}^\top \mathbf{y}.
> $$
> Explain why highly collinear columns of $\mathbf{X}$ can make solving this system numerically unstable. State the role of small singular values in your explanation.

**Hints**

- Connect collinearity to near-linear dependence.
- Think about what this implies for the smallest singular value.

**Deliverables**

- A short written explanation.

## Exercise 11: Proof-style comparison of eigendecomposition and SVD

> **Problem.** Explain why eigendecomposition is not the right universal tool for arbitrary rectangular data matrices, but SVD is. Your answer should include a formal point about domain restrictions and a geometric point about what SVD adds.

**Hints**

- Ask which matrices have eigenvalues and eigenvectors in the same ambient space.
- Ask what happens when input and output dimensions differ.

**Deliverables**

- A short proof-style explanation in prose.

## Exercise 12: ML application mini-analysis

> **Problem.** You are given a centered data matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ with $d$ very large and strong feature correlations. Write a brief analysis answering all of the following:
>
> 1. Why might PCA be useful before fitting a downstream model?
> 2. How do eigenvalues or singular values help choose a reduced dimension $k$?
> 3. What information might be lost by truncating the spectrum?

**Hints**

- Connect PCA to variance, redundancy, and orthogonal coordinates.

**Deliverables**

- A three-part written response.
