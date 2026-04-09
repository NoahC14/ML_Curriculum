---
title: "Derivation of the Singular Value Decomposition"
module: "00-math-toolkit"
lesson: "svd-derivation"
doc_type: "derivation"
topic: "svd"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
updated: "2026-04-09"
owner: "curriculum-team"
tags:
  - "linear-algebra"
  - "svd"
  - "matrix-factorization"
---

## Goal

Derive the singular value decomposition for a real matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ and connect the factorization to PCA, low-rank approximation, and conditioning.

## Assumptions and Notation

Let $\mathbf{A} \in \mathbb{R}^{m \times n}$. We denote by $r = \mathrm{rank}(\mathbf{A})$ its rank. The matrices $\mathbf{A}^\top \mathbf{A} \in \mathbb{R}^{n \times n}$ and $\mathbf{A}\mathbf{A}^\top \in \mathbb{R}^{m \times m}$ are symmetric and positive semidefinite.

We will derive

$$
\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top,
$$

where:

- $\mathbf{U} \in \mathbb{R}^{m \times m}$ is orthogonal;
- $\mathbf{V} \in \mathbb{R}^{n \times n}$ is orthogonal;
- $\boldsymbol{\Sigma} \in \mathbb{R}^{m \times n}$ is diagonal-like with nonnegative diagonal entries.

## Step 1: Spectral decomposition of $\mathbf{A}^\top \mathbf{A}$

Because $\mathbf{A}^\top \mathbf{A}$ is real symmetric, the spectral theorem gives an orthogonal matrix $\mathbf{V}$ and a diagonal matrix $\boldsymbol{\Lambda}$ such that

$$
\mathbf{A}^\top \mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top.
$$

Write the eigenvalues as

$$
\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n \geq 0.
$$

The nonnegativity follows from

$$
\mathbf{x}^\top \mathbf{A}^\top \mathbf{A}\mathbf{x}
= (\mathbf{A}\mathbf{x})^\top(\mathbf{A}\mathbf{x})
= \|\mathbf{A}\mathbf{x}\|_2^2
\geq 0.
$$

Define the singular values by

$$
\sigma_i = \sqrt{\lambda_i}, \quad i = 1, \ldots, n.
$$

Exactly $r$ of these singular values are positive.

## Step 2: Build the right singular vectors

Let $\mathbf{v}_1, \ldots, \mathbf{v}_n$ be the orthonormal eigenvectors of $\mathbf{A}^\top \mathbf{A}$, so

$$
\mathbf{A}^\top \mathbf{A}\mathbf{v}_i = \lambda_i \mathbf{v}_i = \sigma_i^2 \mathbf{v}_i.
$$

These are the right singular vectors. They identify input directions whose image norms under $\mathbf{A}$ are especially simple:

$$
\|\mathbf{A}\mathbf{v}_i\|_2^2
= \mathbf{v}_i^\top \mathbf{A}^\top \mathbf{A}\mathbf{v}_i
= \mathbf{v}_i^\top (\sigma_i^2 \mathbf{v}_i)
= \sigma_i^2.
$$

Therefore

$$
\|\mathbf{A}\mathbf{v}_i\|_2 = \sigma_i.
$$

## Step 3: Build the left singular vectors

For each $i$ with $\sigma_i > 0$, define

$$
\mathbf{u}_i = \frac{1}{\sigma_i}\mathbf{A}\mathbf{v}_i.
$$

These vectors are well-defined because $\sigma_i \neq 0$. We now prove that they are orthonormal.

Let $i, j \leq r$. Then

$$
\mathbf{u}_i^\top \mathbf{u}_j
= \left(\frac{1}{\sigma_i}\mathbf{A}\mathbf{v}_i\right)^\top
\left(\frac{1}{\sigma_j}\mathbf{A}\mathbf{v}_j\right)
= \frac{1}{\sigma_i \sigma_j}\mathbf{v}_i^\top \mathbf{A}^\top \mathbf{A}\mathbf{v}_j.
$$

Since $\mathbf{v}_j$ is an eigenvector of $\mathbf{A}^\top \mathbf{A}$,

$$
\mathbf{u}_i^\top \mathbf{u}_j
= \frac{1}{\sigma_i \sigma_j}\mathbf{v}_i^\top (\sigma_j^2 \mathbf{v}_j)
= \frac{\sigma_j}{\sigma_i}\mathbf{v}_i^\top \mathbf{v}_j.
$$

Because the $\mathbf{v}_i$ are orthonormal, $\mathbf{v}_i^\top \mathbf{v}_j = 0$ for $i \neq j$ and $1$ for $i=j$. Hence

$$
\mathbf{u}_i^\top \mathbf{u}_j = \delta_{ij}.
$$

So $\mathbf{u}_1, \ldots, \mathbf{u}_r$ are orthonormal.

## Step 4: Relate $\mathbf{A}$ to these vectors

By construction,

$$
\mathbf{A}\mathbf{v}_i = \sigma_i \mathbf{u}_i
\quad \text{for } i = 1, \ldots, r.
$$

For $i > r$, we have $\sigma_i = 0$, so $\lambda_i = 0$. Then

$$
\|\mathbf{A}\mathbf{v}_i\|_2^2
= \mathbf{v}_i^\top \mathbf{A}^\top \mathbf{A}\mathbf{v}_i
= \mathbf{v}_i^\top \mathbf{0}
= 0,
$$

which implies

$$
\mathbf{A}\mathbf{v}_i = \mathbf{0}.
$$

Thus the action of $\mathbf{A}$ on the orthonormal basis $\{\mathbf{v}_i\}_{i=1}^n$ is completely known:

$$
\mathbf{A}\mathbf{v}_i =
\begin{cases}
\sigma_i \mathbf{u}_i, & i \leq r, \\
\mathbf{0}, & i > r.
\end{cases}
$$

## Step 5: Extend to orthogonal bases

The vectors $\mathbf{u}_1, \ldots, \mathbf{u}_r$ are orthonormal in $\mathbb{R}^m$. Extend them to an orthonormal basis $\mathbf{u}_1, \ldots, \mathbf{u}_m$ of $\mathbb{R}^m$.

Form the orthogonal matrices

$$
\mathbf{U} = [\mathbf{u}_1 \ \cdots \ \mathbf{u}_m],
\quad
\mathbf{V} = [\mathbf{v}_1 \ \cdots \ \mathbf{v}_n].
$$

Define $\boldsymbol{\Sigma} \in \mathbb{R}^{m \times n}$ by

$$
\Sigma_{ii} = \sigma_i \quad \text{for } i = 1, \ldots, r,
$$

and all remaining entries equal to zero.

Then for each basis vector $\mathbf{e}_i \in \mathbb{R}^n$,

$$
\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top \mathbf{v}_i
= \mathbf{U}\boldsymbol{\Sigma}\mathbf{e}_i
=
\begin{cases}
\sigma_i \mathbf{u}_i, & i \leq r, \\
\mathbf{0}, & i > r.
\end{cases}
$$

This matches $\mathbf{A}\mathbf{v}_i$ for every $i$. Since the $\mathbf{v}_i$ form a basis of $\mathbb{R}^n$, the two linear maps are equal:

$$
\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top.
$$

This is the singular value decomposition.

## Equivalent outer-product form

From the matrix product,

$$
\mathbf{A} = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^\top.
$$

Each term $\sigma_i \mathbf{u}_i \mathbf{v}_i^\top$ is a rank-one matrix. SVD therefore decomposes $\mathbf{A}$ into orthogonal rank-one components ordered by strength.

## Relation to $\mathbf{A}\mathbf{A}^\top$

For each $i \leq r$,

$$
\mathbf{A}\mathbf{A}^\top \mathbf{u}_i
= \mathbf{A}\mathbf{A}^\top \left(\frac{1}{\sigma_i}\mathbf{A}\mathbf{v}_i\right)
= \frac{1}{\sigma_i}\mathbf{A}(\mathbf{A}^\top \mathbf{A}\mathbf{v}_i)
= \frac{1}{\sigma_i}\mathbf{A}(\sigma_i^2 \mathbf{v}_i)
= \sigma_i^2 \mathbf{u}_i.
$$

So the left singular vectors are eigenvectors of $\mathbf{A}\mathbf{A}^\top$ with the same nonzero eigenvalues $\sigma_i^2$.

This is why PCA can be computed either from $\mathbf{X}^\top \mathbf{X}$ or directly from the SVD of $\mathbf{X}$.

## Result

> **Result.** Every real matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ admits an SVD
>
> $$
> \mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top,
> $$
>
> where $\mathbf{U}$ and $\mathbf{V}$ are orthogonal and $\boldsymbol{\Sigma}$ has nonnegative diagonal entries $\sigma_1 \geq \cdots \geq \sigma_r > 0$ followed by zeros.

The singular values are unique. The singular vectors are unique up to sign in simple-eigenvalue cases and up to orthogonal changes within degenerate singular subspaces.

## ML Relevance

### PCA

If centered data are stored in $\mathbf{X} \in \mathbb{R}^{n \times d}$ and

$$
\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top,
$$

then

$$
\mathbf{X}^\top \mathbf{X} = \mathbf{V}\boldsymbol{\Sigma}^\top \boldsymbol{\Sigma}\mathbf{V}^\top.
$$

So the right singular vectors are principal directions, and the covariance eigenvalues are proportional to the squared singular values.

### Low-rank compression

The truncated SVD

$$
\mathbf{A}_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top
$$

keeps the dominant components of the map. In ML this appears in dimensionality reduction, latent semantic analysis, matrix completion, recommender systems, and parameter compression.

### Conditioning

If $\mathbf{A}$ is square and invertible, then

$$
\|\mathbf{A}\|_2 = \sigma_1,
\quad
\|\mathbf{A}^{-1}\|_2 = \frac{1}{\sigma_r},
\quad
\kappa_2(\mathbf{A}) = \frac{\sigma_1}{\sigma_r}.
$$

Thus small singular values indicate near-collapse of some directions and possible numerical instability.

### Whitening

If a covariance matrix has eigendecomposition $\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$, then whitening rescales the principal axes by $\Lambda^{-1/2}$. This is the same spectral logic as SVD, specialized to a symmetric positive definite matrix.

## Limitations

- SVD explains linear structure, not nonlinear manifolds by itself.
- Computing a full SVD can be expensive for large matrices, which motivates randomized and iterative methods.
- Singular directions are sensitive to perturbations when singular values are close together.

## References

- Strang, G. (2016). *Introduction to Linear Algebra*. Wellesley-Cambridge Press.
- Axler, S. (2015). *Linear Algebra Done Right*. Springer.
- Trefethen, L. N., and Bau, D. (1997). *Numerical Linear Algebra*. SIAM.
