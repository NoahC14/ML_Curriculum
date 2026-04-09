---
title: "Linear Algebra Primer for Machine Learning"
module: "00-math-toolkit"
lesson: "linear-algebra-primer"
doc_type: "notes"
topic: "linear-algebra"
status: "draft"
prerequisites:
  - "proof-comfort"
  - "basic-calculus"
updated: "2026-04-09"
owner: "curriculum-team"
tags:
  - "linear-algebra"
  - "ml-foundations"
  - "spectral-methods"
  - "svd"
---

## Motivation

Linear algebra is the native language of machine learning. Data tables are matrices, features are vectors, covariance is a symmetric matrix, principal component analysis is a spectral problem, and least squares is an orthogonal projection problem. The goal of this note is not to cover all of linear algebra. It is to build the part of the subject that an ML practitioner repeatedly uses when deriving algorithms and interpreting models.

Two viewpoints will appear throughout:

- the **algebraic viewpoint**, where we manipulate coordinates, matrices, and identities; and
- the **geometric viewpoint**, where we reason about directions, subspaces, angles, projections, and distortions.

## Assumptions and Notation

We write vectors in lowercase bold, such as $\mathbf{x} \in \mathbb{R}^d$, and matrices in uppercase bold, such as $\mathbf{A} \in \mathbb{R}^{m \times n}$. The transpose of $\mathbf{A}$ is $\mathbf{A}^\top$. The identity matrix of size $d$ is $\mathbf{I}_d$.

If $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$, then their standard inner product is

$$
\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^\top \mathbf{y} = \sum_{i=1}^d x_i y_i.
$$

The Euclidean norm is $\|\mathbf{x}\|_2 = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}$. We will also use $\|\cdot\|_1$ and $\|\cdot\|_\infty$ for vector norms, and the Frobenius norm $\|\mathbf{A}\|_F$ for matrices:

$$
\|\mathbf{A}\|_F^2 = \sum_{i=1}^m \sum_{j=1}^n A_{ij}^2.
$$

The column space of $\mathbf{A}$ is the span of its columns. The null space of $\mathbf{A}$ is

$$
\mathcal{N}(\mathbf{A}) = \{\mathbf{x} \in \mathbb{R}^n : \mathbf{A}\mathbf{x} = \mathbf{0}\}.
$$

## Vectors, Linear Combinations, and Subspaces

Vectors in $\mathbb{R}^d$ can be interpreted as points, displacements, or feature collections. In ML, a data example with $d$ features is usually represented by a vector $\mathbf{x} \in \mathbb{R}^d$.

> **Definition.** A set $V \subseteq \mathbb{R}^d$ is a subspace if it is closed under vector addition and scalar multiplication.

Subspaces matter because many ML procedures search over a constrained family of directions. For example, a low-rank approximation restricts attention to a lower-dimensional subspace, and linear regression predicts inside the span of the columns of the design matrix.

If $\mathbf{v}_1, \ldots, \mathbf{v}_k \in \mathbb{R}^d$, their span is

$$
\mathrm{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}
= \left\{\sum_{i=1}^k \alpha_i \mathbf{v}_i : \alpha_i \in \mathbb{R}\right\}.
$$

Geometrically, the span is the collection of directions you can reach by mixing the given vectors. Algebraically, it is the image of a matrix whose columns are those vectors.

> **Example.** If $\mathbf{X} \in \mathbb{R}^{n \times d}$ is a design matrix, then every prediction vector of a linear model $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$ lies in the column space of $\mathbf{X}$. Ordinary least squares therefore finds the point in $\mathrm{col}(\mathbf{X})$ closest to $\mathbf{y}$.

## Matrices as Linear Maps

A matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ defines a linear map from $\mathbb{R}^n$ to $\mathbb{R}^m$ by $\mathbf{x} \mapsto \mathbf{A}\mathbf{x}$. This point of view is more useful than thinking of a matrix as a table of numbers.

Every linear map satisfies

$$
\mathbf{A}(\alpha \mathbf{x} + \beta \mathbf{y}) = \alpha \mathbf{A}\mathbf{x} + \beta \mathbf{A}\mathbf{y}.
$$

Geometrically, $\mathbf{A}$ can rotate, reflect, stretch, compress, or collapse directions. SVD will later make this description exact.

Three subspaces organize the behavior of $\mathbf{A}$:

- $\mathrm{col}(\mathbf{A}) \subseteq \mathbb{R}^m$;
- $\mathrm{row}(\mathbf{A}) = \mathrm{col}(\mathbf{A}^\top) \subseteq \mathbb{R}^n$;
- $\mathcal{N}(\mathbf{A}) \subseteq \mathbb{R}^n$.

In data analysis, rank measures how many independent directions remain after redundant features are removed.

> **Definition.** The rank of $\mathbf{A}$ is the dimension of $\mathrm{col}(\mathbf{A})$, equivalently the number of linearly independent columns of $\mathbf{A}$.

Low rank is central in recommendation systems, PCA, latent factor models, and compressed representations.

## Inner Products, Norms, and Orthogonality

The inner product turns geometry into algebra. Two vectors are orthogonal when

$$
\langle \mathbf{x}, \mathbf{y} \rangle = 0.
$$

Orthogonality is important because orthogonal directions do not interfere under projection, variance decomposition, or least-squares optimization.

> **Proposition.** If $\mathbf{u}_1, \ldots, \mathbf{u}_k$ are orthonormal vectors in $\mathbb{R}^d$, then for any coefficients $c_1, \ldots, c_k$,
>
> $$
> \left\|\sum_{i=1}^k c_i \mathbf{u}_i\right\|_2^2 = \sum_{i=1}^k c_i^2.
> $$

> **Proof Sketch.** Expand the squared norm using the inner product. The cross terms vanish because $\langle \mathbf{u}_i, \mathbf{u}_j \rangle = 0$ for $i \neq j$, and $\|\mathbf{u}_i\|_2^2 = 1$.

This is the finite-dimensional form of the Pythagorean theorem. In ML, orthonormal bases make coordinate systems stable, which is one reason PCA directions are chosen to be orthonormal.

### Projection Onto a Subspace

If $\mathbf{U} \in \mathbb{R}^{d \times k}$ has orthonormal columns, then the orthogonal projector onto $\mathrm{col}(\mathbf{U})$ is

$$
\mathbf{P} = \mathbf{U}\mathbf{U}^\top.
$$

To see this, decompose any vector $\mathbf{x}$ into a component inside the subspace and a residual orthogonal to it:

$$
\mathbf{x} = \mathbf{U}\mathbf{U}^\top \mathbf{x} + (\mathbf{I} - \mathbf{U}\mathbf{U}^\top)\mathbf{x}.
$$

The second term is orthogonal to every column of $\mathbf{U}$ because

$$
\mathbf{U}^\top(\mathbf{I} - \mathbf{U}\mathbf{U}^\top)\mathbf{x}
= \mathbf{U}^\top \mathbf{x} - \mathbf{U}^\top \mathbf{U}\mathbf{U}^\top \mathbf{x}
= \mathbf{U}^\top \mathbf{x} - \mathbf{U}^\top \mathbf{x}
= \mathbf{0}.
$$

> **ML Interpretation.** Least squares, PCA reconstruction, and linear autoencoder analyses all depend on orthogonal projection. The residual vector represents information not captured by the chosen subspace.

## Matrix Norms and Conditioning

Norms quantify size; conditioning quantifies sensitivity.

The operator norm induced by the Euclidean norm is

$$
\|\mathbf{A}\|_2 = \max_{\|\mathbf{x}\|_2 = 1} \|\mathbf{A}\mathbf{x}\|_2.
$$

This is the largest factor by which $\mathbf{A}$ can stretch a unit vector. Later we will see that it equals the largest singular value of $\mathbf{A}$.

If $\mathbf{A}$ is invertible, its condition number in the Euclidean norm is

$$
\kappa_2(\mathbf{A}) = \|\mathbf{A}\|_2 \|\mathbf{A}^{-1}\|_2.
$$

Large condition number means small perturbations in data may cause large perturbations in the solution. In regression, the conditioning of $\mathbf{X}^\top \mathbf{X}$ strongly affects numerical stability.

> **Warning.** A conceptually valid algorithm may still behave poorly in floating-point arithmetic if the relevant matrix is ill-conditioned.

## Symmetric Matrices, Quadratic Forms, and Positive Definiteness

Many important ML matrices are symmetric: covariance matrices, Hessians of twice-differentiable scalar functions, graph Laplacians, and kernel Gram matrices.

For a symmetric matrix $\mathbf{A} \in \mathbb{R}^{d \times d}$, the quadratic form is

$$
q(\mathbf{x}) = \mathbf{x}^\top \mathbf{A}\mathbf{x}.
$$

> **Definition.** A symmetric matrix $\mathbf{A}$ is positive semidefinite if $\mathbf{x}^\top \mathbf{A}\mathbf{x} \geq 0$ for all $\mathbf{x} \in \mathbb{R}^d$. It is positive definite if the inequality is strict for all nonzero $\mathbf{x}$.

### Why covariance matrices are positive semidefinite

Let $\mathbf{x} \in \mathbb{R}^d$ be a random vector with mean $\boldsymbol{\mu} = \mathbb{E}[\mathbf{x}]$. Its covariance matrix is

$$
\boldsymbol{\Sigma} = \mathbb{E}\left[(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^\top\right].
$$

For any $\mathbf{v} \in \mathbb{R}^d$,

$$
\mathbf{v}^\top \boldsymbol{\Sigma} \mathbf{v}
= \mathbb{E}\left[\mathbf{v}^\top(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{v}\right]
= \mathbb{E}\left[\left((\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{v}\right)^2\right]
\geq 0.
$$

So $\boldsymbol{\Sigma}$ is positive semidefinite. The scalar above is exactly the variance of the projection of the data onto direction $\mathbf{v}$.

> **ML Interpretation.** PCA chooses the direction $\mathbf{v}$ maximizing $\mathbf{v}^\top \boldsymbol{\Sigma} \mathbf{v}$ subject to $\|\mathbf{v}\|_2 = 1$. This turns dimensionality reduction into a constrained quadratic optimization problem.

## Eigenvalues and Eigenvectors

> **Definition.** Let $\mathbf{A} \in \mathbb{R}^{d \times d}$. A nonzero vector $\mathbf{v}$ is an eigenvector of $\mathbf{A}$ with eigenvalue $\lambda$ if
>
> $$
> \mathbf{A}\mathbf{v} = \lambda \mathbf{v}.
> $$

Eigenvectors identify directions that a linear map leaves on the same line. The map may scale the direction, reverse it, or leave it unchanged.

The characteristic equation

$$
\det(\mathbf{A} - \lambda \mathbf{I}) = 0
$$

determines the eigenvalues.

### Symmetric matrices have orthogonal eigenvectors

> **Proposition.** If $\mathbf{A}$ is symmetric and $\mathbf{u}, \mathbf{v}$ are eigenvectors with distinct eigenvalues $\lambda_u \neq \lambda_v$, then $\mathbf{u}$ and $\mathbf{v}$ are orthogonal.

> **Proof.** Since $\mathbf{A}\mathbf{u} = \lambda_u \mathbf{u}$ and $\mathbf{A}\mathbf{v} = \lambda_v \mathbf{v}$,
>
> $$
> \mathbf{u}^\top \mathbf{A}\mathbf{v} = \lambda_v \mathbf{u}^\top \mathbf{v}.
> $$
>
> Because $\mathbf{A}$ is symmetric,
>
> $$
> \mathbf{u}^\top \mathbf{A}\mathbf{v}
> = (\mathbf{A}\mathbf{u})^\top \mathbf{v}
> = (\lambda_u \mathbf{u})^\top \mathbf{v}
> = \lambda_u \mathbf{u}^\top \mathbf{v}.
> $$
>
> Therefore
>
> $$
> \lambda_u \mathbf{u}^\top \mathbf{v} = \lambda_v \mathbf{u}^\top \mathbf{v}.
> $$
>
> Rearranging gives
>
> $$
> (\lambda_u - \lambda_v)\mathbf{u}^\top \mathbf{v} = 0.
> $$
>
> Since $\lambda_u \neq \lambda_v$, we conclude $\mathbf{u}^\top \mathbf{v} = 0$.

This orthogonality is the reason symmetric matrices admit especially clean decompositions.

## Spectral Decomposition

The most useful eigendecomposition in ML is for real symmetric matrices.

> **Theorem.** If $\mathbf{A} \in \mathbb{R}^{d \times d}$ is symmetric, then there exists an orthogonal matrix $\mathbf{Q} \in \mathbb{R}^{d \times d}$ and a diagonal matrix $\boldsymbol{\Lambda} \in \mathbb{R}^{d \times d}$ such that
>
> $$
> \mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top.
> $$
>
> The diagonal entries of $\boldsymbol{\Lambda}$ are the eigenvalues of $\mathbf{A}$, and the columns of $\mathbf{Q}$ are orthonormal eigenvectors.

### Derivation of the symmetric eigendecomposition

We will derive the structure and use the standard fact that a real symmetric matrix has at least one real eigenvalue with an associated eigenvector.

1. Let $\mathbf{q}_1$ be a unit eigenvector of $\mathbf{A}$ with eigenvalue $\lambda_1$.
2. Extend $\mathbf{q}_1$ to an orthonormal basis of $\mathbb{R}^d$, and let $\mathbf{Q}_1$ be the orthogonal matrix whose first column is $\mathbf{q}_1$.
3. Compute

$$
\mathbf{Q}_1^\top \mathbf{A}\mathbf{Q}_1 =
\begin{bmatrix}
\lambda_1 & \mathbf{b}^\top \\
\mathbf{b} & \mathbf{B}
\end{bmatrix}.
$$

We now show $\mathbf{b} = \mathbf{0}$. Since $\mathbf{A}\mathbf{q}_1 = \lambda_1 \mathbf{q}_1$, we have

$$
\mathbf{Q}_1^\top \mathbf{A}\mathbf{q}_1
= \mathbf{Q}_1^\top (\lambda_1 \mathbf{q}_1)
= \lambda_1 \mathbf{Q}_1^\top \mathbf{q}_1
= \lambda_1 \mathbf{e}_1.
$$

But $\mathbf{Q}_1^\top \mathbf{A}\mathbf{q}_1$ is exactly the first column of $\mathbf{Q}_1^\top \mathbf{A}\mathbf{Q}_1$, so that first column must be

$$
\begin{bmatrix}
\lambda_1 \\
\mathbf{0}
\end{bmatrix}.
$$

Because $\mathbf{Q}_1^\top \mathbf{A}\mathbf{Q}_1$ is symmetric, the first row is also $(\lambda_1, \mathbf{0}^\top)$. Therefore

$$
\mathbf{Q}_1^\top \mathbf{A}\mathbf{Q}_1 =
\begin{bmatrix}
\lambda_1 & \mathbf{0}^\top \\
\mathbf{0} & \mathbf{B}
\end{bmatrix},
$$

where $\mathbf{B} \in \mathbb{R}^{(d-1)\times(d-1)}$ is symmetric.

4. Apply the same argument recursively to $\mathbf{B}$.

After $d$ steps we obtain an orthogonal matrix $\mathbf{Q}$ whose columns are orthonormal eigenvectors and a diagonal matrix $\boldsymbol{\Lambda}$ of eigenvalues such that

$$
\mathbf{Q}^\top \mathbf{A}\mathbf{Q} = \boldsymbol{\Lambda}.
$$

Multiplying on the left by $\mathbf{Q}$ and on the right by $\mathbf{Q}^\top$ yields

$$
\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top.
$$

### Consequences

If $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$, then for any vector $\mathbf{x}$,

$$
\mathbf{x}^\top \mathbf{A}\mathbf{x}
= \mathbf{x}^\top \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top \mathbf{x}
= \mathbf{z}^\top \boldsymbol{\Lambda}\mathbf{z}
= \sum_{i=1}^d \lambda_i z_i^2,
$$

where $\mathbf{z} = \mathbf{Q}^\top \mathbf{x}$. Therefore:

- $\mathbf{A}$ is positive semidefinite if and only if all $\lambda_i \geq 0$;
- $\mathbf{A}$ is positive definite if and only if all $\lambda_i > 0$.

This gives a clean spectral test for positive definiteness.

### Worked Example: PCA as spectral decomposition

Let centered data points be rows of $\mathbf{X} \in \mathbb{R}^{n \times d}$, so the sample covariance is

$$
\mathbf{S} = \frac{1}{n}\mathbf{X}^\top \mathbf{X}.
$$

The first principal component solves

$$
\max_{\|\mathbf{v}\|_2 = 1} \mathbf{v}^\top \mathbf{S}\mathbf{v}.
$$

Introduce a Lagrange multiplier $\lambda$:

$$
\mathcal{L}(\mathbf{v}, \lambda) = \mathbf{v}^\top \mathbf{S}\mathbf{v} - \lambda(\mathbf{v}^\top \mathbf{v} - 1).
$$

Setting the gradient with respect to $\mathbf{v}$ equal to zero gives

$$
2\mathbf{S}\mathbf{v} - 2\lambda \mathbf{v} = \mathbf{0},
$$

so

$$
\mathbf{S}\mathbf{v} = \lambda \mathbf{v}.
$$

Therefore principal components are eigenvectors of the covariance matrix. The explained variance along $\mathbf{v}$ is the corresponding eigenvalue.

> **ML Interpretation.** PCA is not just "rotate the data." It finds an orthonormal basis ordered by variance captured, which is why it underlies dimensionality reduction, denoising, whitening, and latent representation analysis.

## Singular Value Decomposition

Eigendecomposition applies to square matrices and is especially clean for symmetric ones. SVD applies to every real matrix.

> **Theorem.** For any $\mathbf{A} \in \mathbb{R}^{m \times n}$ of rank $r$, there exist orthogonal matrices $\mathbf{U} \in \mathbb{R}^{m \times m}$ and $\mathbf{V} \in \mathbb{R}^{n \times n}$ and a diagonal-like matrix $\boldsymbol{\Sigma} \in \mathbb{R}^{m \times n}$ such that
>
> $$
> \mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top,
> $$
>
> where the nonzero diagonal entries $\sigma_1 \geq \cdots \geq \sigma_r > 0$ are the singular values of $\mathbf{A}$.

The full derivation appears in [SVD derivation](../derivations/svd-derivation.md). Here is the structural idea.

1. The matrix $\mathbf{A}^\top \mathbf{A}$ is symmetric positive semidefinite.
2. Therefore it has an orthonormal eigenbasis:

$$
\mathbf{A}^\top \mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top,
$$

with $\lambda_i \geq 0$.

3. Define singular values by $\sigma_i = \sqrt{\lambda_i}$.
4. For each $\sigma_i > 0$, define

$$
\mathbf{u}_i = \frac{1}{\sigma_i}\mathbf{A}\mathbf{v}_i.
$$

Then the vectors $\mathbf{u}_i$ are orthonormal, and

$$
\mathbf{A}\mathbf{v}_i = \sigma_i \mathbf{u}_i.
$$

Thus $\mathbf{A}$ maps the right singular direction $\mathbf{v}_i$ to the left singular direction $\mathbf{u}_i$ scaled by $\sigma_i$.

### Geometric interpretation

SVD says that any linear map can be decomposed into three stages:

1. rotate or reflect the input space by $\mathbf{V}^\top$;
2. stretch or compress orthogonal directions by $\boldsymbol{\Sigma}$;
3. rotate or reflect the output space by $\mathbf{U}$.

This is one of the most useful geometric statements in all of linear algebra.

### Low-rank approximation

If

$$
\mathbf{A} = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^\top,
$$

then the rank-$k$ truncation is

$$
\mathbf{A}_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top.
$$

Keeping only the largest singular values preserves the most energetic directions of the map. In practice this drives PCA, latent semantic analysis, collaborative filtering, and matrix compression.

> **ML Interpretation.** If $\mathbf{X}$ is centered data, then PCA can be computed from the SVD $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$. The right singular vectors are principal directions, and the squared singular values scaled by $1/n$ are covariance eigenvalues.

## Whitening and Decorrelated Coordinates

Suppose $\boldsymbol{\Sigma} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ is the covariance of centered data and all eigenvalues are positive. Define the whitening transform

$$
\mathbf{W} = \boldsymbol{\Lambda}^{-1/2}\mathbf{Q}^\top.
$$

If $\mathbf{z} = \mathbf{W}\mathbf{x}$, then the covariance of $\mathbf{z}$ is

$$
\mathrm{Cov}(\mathbf{z})
= \mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^\top
= \boldsymbol{\Lambda}^{-1/2}\mathbf{Q}^\top
\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top
\mathbf{Q}\boldsymbol{\Lambda}^{-1/2}
= \mathbf{I}.
$$

So whitening rotates into the eigenbasis and rescales by inverse standard deviation. This is useful for optimization, preprocessing, and understanding isotropic latent coordinates.

> **Warning.** Whitening can amplify noise in directions with very small eigenvalues, which is why regularized whitening is common in practice.

## Linear Algebra in Matrix Calculus Preview

Later modules will differentiate scalar objectives with respect to vectors and matrices. Linear algebra controls the shape of those derivatives.

For example, if $\mathbf{X} \in \mathbb{R}^{n \times d}$, $\mathbf{w} \in \mathbb{R}^d$, and $\mathbf{y} \in \mathbb{R}^n$, then the least-squares objective

$$
L(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2
$$

can be expanded as

$$
\begin{aligned}
L(\mathbf{w})
&= (\mathbf{X}\mathbf{w} - \mathbf{y})^\top(\mathbf{X}\mathbf{w} - \mathbf{y}) \\
&= \mathbf{w}^\top \mathbf{X}^\top \mathbf{X}\mathbf{w} - 2\mathbf{y}^\top \mathbf{X}\mathbf{w} + \mathbf{y}^\top \mathbf{y}.
\end{aligned}
$$

The matrix $\mathbf{X}^\top \mathbf{X}$ is symmetric positive semidefinite, which is why the curvature of least squares is well behaved. Setting the gradient to zero later yields the normal equations

$$
\mathbf{X}^\top \mathbf{X}\mathbf{w} = \mathbf{X}^\top \mathbf{y}.
$$

> **ML Interpretation.** When gradients and Hessians appear in optimization or backpropagation, they are not separate from linear algebra. They are linear maps and quadratic forms acting on parameter perturbations.

## What to Retain for Later Modules

The linear algebra facts that recur most often in this curriculum are:

- linear models live in column spaces;
- least squares is orthogonal projection;
- covariance matrices are symmetric positive semidefinite;
- PCA is an eigenvalue problem;
- SVD describes every matrix as orthogonal changes of basis plus anisotropic scaling;
- conditioning controls numerical sensitivity;
- positive definiteness governs curvature, invertibility, and optimization stability.

These ideas will reappear in optimization, kernel methods, probabilistic modeling, neural networks, graph learning, and transformer mechanics.

## References

- Axler, S. (2015). *Linear Algebra Done Right*. Springer.
- Strang, G. (2016). *Introduction to Linear Algebra*. Wellesley-Cambridge Press.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
