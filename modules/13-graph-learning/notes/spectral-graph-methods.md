---
title: "Spectral Graph Methods"
module: "13-graph-learning"
lesson: "spectral-graph-methods"
doc_type: "notes"
topic: "laplacian-graph-fourier-transform-graph-convolution"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
  - "13-graph-learning/graph-learning"
updated: "2026-04-13"
owner: "curriculum-team"
tags:
  - "graph-learning"
  - "spectral-methods"
  - "laplacian"
  - "gcn"
---

## Purpose

These notes develop the spectral view of graphs.
The key idea is that the graph Laplacian plays the role that the ordinary Fourier basis plays on Euclidean domains:
it gives an orthogonal basis of graph modes, ordered from smooth to oscillatory.

This viewpoint lets us define graph convolution and then derive the GCN layer as a tractable approximation.
For the message-passing-first presentation that this note complements, see [graph-learning.md](./graph-learning.md).

## Learning objectives

After working through this note, you should be able to:

- define combinatorial and normalized graph Laplacians;
- interpret the Laplacian quadratic form as a smoothness penalty on graph signals;
- define the graph Fourier transform using Laplacian eigenvectors;
- express graph convolution as spectral filtering; and
- derive the GCN propagation rule as a low-order spectral approximation.

## 1. Graph signals and the Laplacian

Let $G = (V,E)$ be an undirected graph with $n$ nodes.
A graph signal is a function

$$
f : V \to \mathbb{R},
$$

which we represent as a vector $f \in \mathbb{R}^n$.

The combinatorial Laplacian is

$$
L = D - A.
$$

For weighted undirected graphs, $L$ is symmetric and positive semidefinite.

### 1.1 Quadratic form and smoothness

For any $f \in \mathbb{R}^n$,

$$
f^\top L f
=
f^\top D f - f^\top A f
=
\frac{1}{2}\sum_{i,j} A_{ij}(f_i - f_j)^2.
$$

So $f^\top L f$ is small when adjacent nodes have similar values.
This is why low-frequency graph signals are smooth over edges.

### 1.2 Normalized Laplacian

Two common normalized forms are

$$
L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2}
$$

and

$$
L_{\mathrm{rw}} = I - D^{-1}A.
$$

The symmetric normalized Laplacian is especially convenient because it remains symmetric for undirected graphs.

## 2. Eigenvectors as graph Fourier modes

Because $L_{\mathrm{sym}}$ is real symmetric, it has an orthonormal eigendecomposition

$$
L_{\mathrm{sym}} = U \Lambda U^\top,
$$

where:

- $U = [u_1,\dots,u_n]$ contains orthonormal eigenvectors;
- $\Lambda = \operatorname{diag}(\lambda_1,\dots,\lambda_n)$ with $0 = \lambda_1 \le \cdots \le \lambda_n$.

The eigenvectors are graph analogues of Fourier basis functions.
Small eigenvalues correspond to smooth modes.
Large eigenvalues correspond to more oscillatory variation across edges.

### 2.1 Graph Fourier transform

The graph Fourier transform of a signal $f$ is

$$
\hat{f} = U^\top f,
$$

and the inverse transform is

$$
f = U \hat{f}.
$$

So a graph signal can be analyzed frequency by frequency in the Laplacian eigenbasis.

## 3. Graph convolution as spectral filtering

In Euclidean signal processing, convolution corresponds to multiplication in the Fourier domain.
The graph analogue is to filter Fourier coefficients by a spectral transfer function $g_\theta$:

$$
g_\theta \star f = U g_\theta(\Lambda) U^\top f.
$$

Here $g_\theta(\Lambda)$ is a diagonal matrix that scales each graph frequency.

This definition is mathematically clean but computationally expensive because a full eigendecomposition is costly on large graphs.
It is also tied to one fixed graph basis, which complicates transfer across graphs.

## 4. Polynomial filters and localization

If we restrict $g_\theta(\Lambda)$ to a polynomial in $\Lambda$, then

$$
g_\theta(L)f
=
\sum_{k=0}^K \theta_k L^k f.
$$

That matters because powers of the Laplacian or adjacency operator remain localized:
$L^k f$ depends on information within roughly $k$ hops.
So polynomial spectral filters become local graph operators.

This is the main bridge between spectral and spatial graph learning.

### 4.1 Chebyshev approximation

A standard approach is to approximate the spectral filter using Chebyshev polynomials:

$$
g_\theta(L)f \approx \sum_{k=0}^K \theta_k T_k(\tilde{L})f,
$$

where

$$
\tilde{L} = \frac{2}{\lambda_{\max}}L - I.
$$

This avoids explicit eigendecomposition and yields a $K$-hop localized filter.

## 5. Deriving GCN as a first-order spectral approximation

The GCN layer can be derived by simplifying the Chebyshev filter.

### 5.1 First-order truncation

Take $K=1$:

$$
g_\theta(L)f \approx \theta_0 f + \theta_1 \tilde{L}f.
$$

If we further approximate $\lambda_{\max} \approx 2$, then

$$
\tilde{L} \approx L - I.
$$

For the symmetric normalized Laplacian,

$$
L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2},
$$

so

$$
L_{\mathrm{sym}} - I = -D^{-1/2} A D^{-1/2}.
$$

Thus

$$
g_\theta(L_{\mathrm{sym}})f
\approx
\theta_0 f - \theta_1 D^{-1/2} A D^{-1/2}f.
$$

If we tie coefficients by setting $\theta = \theta_0 = -\theta_1$, then

$$
g_\theta(L_{\mathrm{sym}})f
\approx
\theta
\left(I + D^{-1/2} A D^{-1/2}\right)f.
$$

This is already a graph smoothing operator plus a learnable scalar weight.

### 5.2 Renormalization trick

Directly using

$$
I + D^{-1/2} A D^{-1/2}
$$

can be numerically awkward because the spectrum may not stay well scaled.
GCN therefore introduces self-loops first:

$$
\hat{A} = A + I,
\qquad
\hat{D}_{ii} = \sum_j \hat{A}_{ij},
$$

and uses the renormalized propagation operator

$$
\hat{S} = \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}.
$$

For node-feature matrices $X$, one GCN layer becomes

$$
H^{(1)} = \sigma(\hat{S} X W).
$$

Stacking layers gives

$$
H^{(\ell+1)} = \sigma(\hat{S} H^{(\ell)} W^{(\ell)}).
$$

This is the familiar spatial-looking formula, but now we can see where it came from:
it is a first-order approximation to spectral graph convolution.

## 6. Interpretation of the GCN operator

The operator $\hat{S}$ does two things at once:

- it mixes each node with its neighbors;
- it rescales by degrees so high-degree nodes do not dominate.

Repeated applications of $\hat{S}$ suppress high-frequency components and amplify smooth components.
That explains both the usefulness and the danger of GCNs:

- useful because homophilous labels often correlate with smooth signals on the graph;
- dangerous because too much smoothing erases discriminative differences.

## 7. Spectral intuition for graph transformers

Graph transformers are usually presented in spatial or attention-based language, but the spectral perspective still helps.
Attention can be understood as learning a richer, data-dependent propagation operator rather than relying on a fixed Laplacian-derived one.

In that sense:

- GCN uses a fixed low-pass filter derived from the graph;
- attention-based graph models learn more adaptive filters;
- positional or Laplacian eigenvector encodings inject spectral structure back into transformer-style models.

## 8. Limitations of pure spectral methods

- The full eigendecomposition is expensive on large graphs.
- Filters expressed in one graph eigenbasis do not transfer cleanly to different graphs.
- Pure low-pass filtering is not always the right bias, especially under heterophily.
- Spectral constructions are mathematically elegant, but spatial implementations are often easier to scale.

That is why modern graph learning usually combines spectral intuition with spatial architectures.

## 9. Summary

The Laplacian measures how rapidly a signal changes across edges.
Its eigenvectors define graph frequencies, and graph convolution becomes spectral filtering in that basis.
Polynomial approximations make those filters local.
GCN emerges by taking a first-order approximation and renormalizing the resulting operator.

This derivation matters because it explains why GCN behaves like learned graph smoothing rather than as an arbitrary neural architecture.

## References

- Chung, *Spectral Graph Theory*.
- Hammond, Vandergheynst, and Gribonval, *Wavelets on Graphs via Spectral Graph Theory*.
- Defferrard, Bresson, and Vandergheynst, *Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering*.
- Kipf and Welling, *Semi-Supervised Classification with Graph Convolutional Networks*.
