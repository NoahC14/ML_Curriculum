---
title: "Linear Algebra Solutions"
module: "00-math-toolkit"
lesson: "linear-algebra-solutions"
doc_type: "solution"
topic: "linear-algebra"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
updated: "2026-04-09"
owner: "curriculum-team"
tags:
  - "linear-algebra"
  - "solutions"
  - "pca"
  - "svd"
---

## Solution 1: Span and linear dependence

Set

$$
\alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \alpha_3 \mathbf{v}_3 = \mathbf{0}.
$$

Coordinate-wise,

$$
\alpha_1 + \alpha_3 = 0,
\quad
\alpha_2 + \alpha_3 = 0,
\quad
\alpha_1 + \alpha_2 + 2\alpha_3 = 0.
$$

The third equation is implied by the first two. Taking $\alpha_3 = 1$ gives $\alpha_1 = -1$ and $\alpha_2 = -1$. Therefore the vectors are linearly dependent, and

$$
\mathbf{v}_3 = \mathbf{v}_1 + \mathbf{v}_2.
$$

## Solution 2: Orthogonal projection

First,

$$
\mathbf{P} = \mathbf{U}\mathbf{U}^\top
=
\begin{bmatrix}
1/\sqrt{2} \\
1/\sqrt{2}
\end{bmatrix}
\begin{bmatrix}
1/\sqrt{2} & 1/\sqrt{2}
\end{bmatrix}
=
\begin{bmatrix}
1/2 & 1/2 \\
1/2 & 1/2
\end{bmatrix}.
$$

Then

$$
\mathbf{P}\mathbf{x}
=
\begin{bmatrix}
1/2 & 1/2 \\
1/2 & 1/2
\end{bmatrix}
\begin{bmatrix}
3 \\ 1
\end{bmatrix}
=
\begin{bmatrix}
2 \\ 2
\end{bmatrix}.
$$

So the projection of $\mathbf{x}$ onto the line $x_1 = x_2$ is $(2,2)^\top$.

## Solution 3: Positive semidefinite covariance

Let $\boldsymbol{\mu} = \mathbb{E}[\mathbf{x}]$. For any $\mathbf{v} \in \mathbb{R}^d$,

$$
\mathbf{v}^\top \boldsymbol{\Sigma}\mathbf{v}
= \mathbf{v}^\top \mathbb{E}\left[(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^\top\right]\mathbf{v}
= \mathbb{E}\left[\mathbf{v}^\top (\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{v}\right].
$$

Rewrite the scalar inside the expectation:

$$
\mathbf{v}^\top (\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{v}
= \left((\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{v}\right)^2 \geq 0.
$$

Therefore

$$
\mathbf{v}^\top \boldsymbol{\Sigma}\mathbf{v}
= \mathbb{E}\left[\left((\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{v}\right)^2\right]
\geq 0.
$$

So $\boldsymbol{\Sigma}$ is positive semidefinite.

## Solution 4: Symmetric eigenvector orthogonality

Let $\mathbf{A}$ be symmetric and suppose

$$
\mathbf{A}\mathbf{u} = \lambda_u \mathbf{u},
\quad
\mathbf{A}\mathbf{v} = \lambda_v \mathbf{v},
\quad
\lambda_u \neq \lambda_v.
$$

Then

$$
\mathbf{u}^\top \mathbf{A}\mathbf{v} = \lambda_v \mathbf{u}^\top \mathbf{v}.
$$

Because $\mathbf{A}$ is symmetric,

$$
\mathbf{u}^\top \mathbf{A}\mathbf{v}
= (\mathbf{A}\mathbf{u})^\top \mathbf{v}
= (\lambda_u \mathbf{u})^\top \mathbf{v}
= \lambda_u \mathbf{u}^\top \mathbf{v}.
$$

Thus

$$
(\lambda_u - \lambda_v)\mathbf{u}^\top \mathbf{v} = 0.
$$

Since $\lambda_u \neq \lambda_v$, it follows that $\mathbf{u}^\top \mathbf{v} = 0$.

## Solution 5: Spectral decomposition in a concrete case

Compute the characteristic polynomial:

$$
\det(\mathbf{A} - \lambda \mathbf{I})
=
\det
\begin{bmatrix}
2-\lambda & 1 \\
1 & 2-\lambda
\end{bmatrix}
= (2-\lambda)^2 - 1.
$$

So

$$
(2-\lambda)^2 - 1 = 0
\quad \Longrightarrow \quad
\lambda \in \{3, 1\}.
$$

For $\lambda = 3$, an eigenvector is $(1,1)^\top$, which normalizes to

$$
\mathbf{q}_1 = \frac{1}{\sqrt{2}}
\begin{bmatrix}
1 \\ 1
\end{bmatrix}.
$$

For $\lambda = 1$, an eigenvector is $(1,-1)^\top$, which normalizes to

$$
\mathbf{q}_2 = \frac{1}{\sqrt{2}}
\begin{bmatrix}
1 \\ -1
\end{bmatrix}.
$$

Thus

$$
\mathbf{Q} =
\begin{bmatrix}
1/\sqrt{2} & 1/\sqrt{2} \\
1/\sqrt{2} & -1/\sqrt{2}
\end{bmatrix},
\quad
\boldsymbol{\Lambda} =
\begin{bmatrix}
3 & 0 \\
0 & 1
\end{bmatrix},
$$

and

$$
\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top.
$$

## Solution 6: First principal component from constrained optimization

Consider

$$
\max_{\|\mathbf{v}\|_2 = 1} \mathbf{v}^\top \mathbf{S}\mathbf{v}.
$$

Introduce the Lagrangian

$$
\mathcal{L}(\mathbf{v}, \lambda) = \mathbf{v}^\top \mathbf{S}\mathbf{v} - \lambda(\mathbf{v}^\top \mathbf{v} - 1).
$$

Differentiate with respect to $\mathbf{v}$:

$$
\nabla_{\mathbf{v}}\mathcal{L} = 2\mathbf{S}\mathbf{v} - 2\lambda \mathbf{v}.
$$

Setting the gradient equal to zero gives

$$
\mathbf{S}\mathbf{v} = \lambda \mathbf{v}.
$$

So an optimizer must be an eigenvector of $\mathbf{S}$. The maximizing direction is the eigenvector associated with the largest eigenvalue, which means PCA chooses the direction of greatest sample variance.

## Solution 7: Compute an SVD

Here

$$
\mathbf{A}^\top \mathbf{A} =
\begin{bmatrix}
9 & 0 \\
0 & 1
\end{bmatrix}.
$$

Its eigenvalues are $9$ and $1$, so the singular values are $3$ and $1$. The standard basis vectors are already orthonormal eigenvectors, so we may take

$$
\mathbf{V} = \mathbf{I},
\quad
\boldsymbol{\Sigma} =
\begin{bmatrix}
3 & 0 \\
0 & 1
\end{bmatrix}.
$$

Then

$$
\mathbf{U} = \mathbf{A}\mathbf{V}\boldsymbol{\Sigma}^{-1} = \mathbf{I}.
$$

One valid SVD is therefore

$$
\mathbf{A} = \mathbf{I}
\begin{bmatrix}
3 & 0 \\
0 & 1
\end{bmatrix}
\mathbf{I}^\top.
$$

Geometrically, $\mathbf{A}$ stretches the first coordinate by a factor of $3$ and the second by a factor of $1$.

## Solution 8: Rank-one approximation

The SVD-suggested rank-one approximation is

$$
\mathbf{A}_1 = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^\top.
$$

This keeps the dominant singular direction and discards the weaker component $\sigma_2 \mathbf{u}_2 \mathbf{v}_2^\top$. Geometrically, it preserves the strongest stretching action of the map while collapsing the smaller orthogonal mode.

## Solution 9: Whitening transform

Let $\mathbf{z} = \mathbf{W}\mathbf{x}$ with centered $\mathbf{x}$ and

$$
\mathbf{W} = \boldsymbol{\Lambda}^{-1/2}\mathbf{Q}^\top.
$$

Then

$$
\mathrm{Cov}(\mathbf{z})
= \mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^\top
= \boldsymbol{\Lambda}^{-1/2}\mathbf{Q}^\top
\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top
\mathbf{Q}\boldsymbol{\Lambda}^{-1/2}.
$$

Using $\mathbf{Q}^\top \mathbf{Q} = \mathbf{I}$,

$$
\mathrm{Cov}(\mathbf{z})
= \boldsymbol{\Lambda}^{-1/2}\boldsymbol{\Lambda}\boldsymbol{\Lambda}^{-1/2}
= \mathbf{I}.
$$

So the transformed coordinates have identity covariance. One practical risk is that very small eigenvalues produce very large inverse square roots, which can amplify noise.

## Solution 10: Conditioning and regression

If the columns of $\mathbf{X}$ are highly collinear, then some column is nearly a linear combination of the others. That means $\mathbf{X}$ nearly collapses at least one direction in parameter space, so its smallest singular value is close to zero. Consequently $\mathbf{X}^\top \mathbf{X}$ has a very small eigenvalue, making the normal-equation system ill-conditioned. Small perturbations in data or floating-point roundoff can then cause large changes in the computed coefficient vector.

## Solution 11: Proof-style comparison of eigendecomposition and SVD

Eigendecomposition requires a square matrix $\mathbf{A} \in \mathbb{R}^{d \times d}$ because the equation $\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$ compares vectors in the same ambient space. For a rectangular matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ with $m \neq n$, the input $\mathbf{v}$ lies in $\mathbb{R}^n$ while $\mathbf{A}\mathbf{v}$ lies in $\mathbb{R}^m$, so the eigenvector equation is not even type-consistent. SVD resolves this by introducing right singular vectors in the input space and left singular vectors in the output space, with singular values describing how strongly each input direction is mapped into an output direction. Geometrically, SVD therefore handles maps between different-dimensional spaces and separates change of basis from anisotropic scaling.

## Solution 12: ML application mini-analysis

1. PCA can be useful because strong feature correlations imply redundant directions. PCA rotates the data into orthogonal directions ordered by variance, which can reduce noise, simplify downstream optimization, and produce a lower-dimensional representation.
2. Eigenvalues of the covariance matrix, or equivalently squared singular values of the centered data matrix, measure how much variance each principal direction explains. A common heuristic is to choose $k$ so that the retained cumulative variance exceeds a target threshold.
3. Truncating the spectrum discards low-variance directions. This may remove mostly noise, but it can also erase rare but predictive structure, class-separating information, or interpretable features that happen not to dominate total variance.
