---
title: "Derivation of Self-Attention"
module: "10-transformers-llms"
lesson: "self-attention"
doc_type: "derivation"
topic: "scaled-dot-product-attention"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
  - "09-sequence-models/sequence-modeling"
  - "10-transformers-llms/transformer-foundations"
updated: "2026-04-13"
owner: "curriculum-team"
tags:
  - "transformers"
  - "self-attention"
  - "scaled-dot-product"
  - "matrix-derivation"
---

## Purpose

This derivation makes the self-attention mechanism explicit from token-wise weighted averaging to the standard matrix formula

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

The emphasis is on dimensions, algebra, and the meaning of each matrix product.

## 1. Assumptions and notation

Let the input sequence representation be

$$
X \in \mathbb{R}^{T \times d_{\mathrm{model}}},
$$

where:

- $T$ is the sequence length;
- $d_{\mathrm{model}}$ is the token representation width.

Define learned projection matrices

$$
W_Q \in \mathbb{R}^{d_{\mathrm{model}} \times d_k},
\qquad
W_K \in \mathbb{R}^{d_{\mathrm{model}} \times d_k},
\qquad
W_V \in \mathbb{R}^{d_{\mathrm{model}} \times d_v}.
$$

Then

$$
Q = XW_Q \in \mathbb{R}^{T \times d_k},
\qquad
K = XW_K \in \mathbb{R}^{T \times d_k},
\qquad
V = XW_V \in \mathbb{R}^{T \times d_v}.
$$

Write the $i$th rows as

$$
q_i^\top \in \mathbb{R}^{1 \times d_k},
\qquad
k_i^\top \in \mathbb{R}^{1 \times d_k},
\qquad
v_i^\top \in \mathbb{R}^{1 \times d_v}.
$$

Equivalently, as column vectors,

$$
q_i, k_i \in \mathbb{R}^{d_k},
\qquad
v_i \in \mathbb{R}^{d_v}.
$$

## 2. Token-wise derivation

Fix a query position $i$.
We want token $i$ to aggregate information from all tokens $j = 1,\dots,T$.

### 2.1 Compatibility scores

Define the raw score between query $i$ and key $j$ by

$$
e_{ij} = q_i^\top k_j.
$$

This is a scalar because

$$
q_i^\top \in \mathbb{R}^{1 \times d_k},
\qquad
k_j \in \mathbb{R}^{d_k \times 1}.
$$

So the product has shape

$$
(1 \times d_k)(d_k \times 1) = 1 \times 1.
$$

### 2.2 Normalize into attention weights

The scores $e_{ij}$ can be any real numbers, but we want weights that are nonnegative and sum to one.
Apply the softmax over $j$:

$$
\alpha_{ij}
=
\frac{\exp(e_{ij} / \sqrt{d_k})}
{\sum_{\ell=1}^T \exp(e_{i\ell} / \sqrt{d_k})}.
$$

For fixed $i$,

$$
\alpha_{ij} \geq 0,
\qquad
\sum_{j=1}^T \alpha_{ij} = 1.
$$

### 2.3 Weighted aggregation

The output for token $i$ is

$$
c_i = \sum_{j=1}^T \alpha_{ij} v_j
\in \mathbb{R}^{d_v}.
$$

Since the coefficients are nonnegative and sum to one, $c_i$ is a convex combination of the value vectors.
So attention does not invent information at this stage; it reweights and mixes existing value vectors.

## 3. Recovering the matrix formula

### 3.1 Score matrix

The $(i,j)$ entry of $QK^\top$ is

$$
(QK^\top)_{ij} = q_i^\top k_j.
$$

Check the dimensions:

$$
Q \in \mathbb{R}^{T \times d_k},
\qquad
K^\top \in \mathbb{R}^{d_k \times T},
$$

so

$$
QK^\top \in \mathbb{R}^{T \times T}.
$$

This matrix stores all pairwise query-key scores at once.

### 3.2 Row-wise softmax

Define

$$
A = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)
\in \mathbb{R}^{T \times T},
$$

where softmax is applied independently to each row.
Then the $(i,j)$ entry of $A$ is precisely $\alpha_{ij}$.

### 3.3 Multiply by values

Now compute

$$
C = AV.
$$

Since

$$
A \in \mathbb{R}^{T \times T},
\qquad
V \in \mathbb{R}^{T \times d_v},
$$

we obtain

$$
C \in \mathbb{R}^{T \times d_v}.
$$

Its $i$th row is

$$
c_i^\top
=
\sum_{j=1}^T A_{ij} v_j^\top
=
\sum_{j=1}^T \alpha_{ij} v_j^\top,
$$

which matches the token-wise derivation.

Therefore

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

## 4. Why divide by $\sqrt{d_k}$?

Assume for intuition that the coordinates of $q_i$ and $k_j$ are independent with

$$
\mathbb{E}[q_{ir}] = \mathbb{E}[k_{jr}] = 0,
\qquad
\operatorname{Var}(q_{ir}) = \operatorname{Var}(k_{jr}) = 1.
$$

Then

$$
q_i^\top k_j = \sum_{r=1}^{d_k} q_{ir} k_{jr}.
$$

Because the summands are approximately mean-zero with variance near one,

$$
\operatorname{Var}(q_i^\top k_j) \approx d_k.
$$

So the standard deviation grows like $\sqrt{d_k}$.
Without scaling, larger key dimension means larger logits, which drives the softmax toward near-one-hot outputs and poorer gradient conditioning.

Dividing by $\sqrt{d_k}$ keeps the score scale more stable across dimensions:

$$
\operatorname{Var}\!\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right)
\approx 1.
$$

> **Proof Sketch.** Use $\operatorname{Var}(aZ) = a^2 \operatorname{Var}(Z)$ with $a = 1/\sqrt{d_k}$.

## 5. Causal masking

In autoregressive decoding, token $i$ must not attend to future positions $j > i$.
Introduce a mask matrix

$$
M \in \mathbb{R}^{T \times T},
\qquad
M_{ij}
=
\begin{cases}
0, & j \leq i, \\
-\infty, & j > i.
\end{cases}
$$

Then the masked attention formula is

$$
\operatorname{MaskedAttention}(Q,K,V)
=
\operatorname{softmax}\!\left(
\frac{QK^\top}{\sqrt{d_k}} + M
\right)V.
$$

If $j > i$, then the masked logit is $-\infty$, so after softmax

$$
\alpha_{ij} = 0.
$$

This enforces the causal factorization used in language modeling.

## 6. Multi-head extension

For heads $h=1,\dots,H$, define

$$
Q^{(h)} = XW_Q^{(h)},
\qquad
K^{(h)} = XW_K^{(h)},
\qquad
V^{(h)} = XW_V^{(h)}.
$$

Each head computes

$$
C^{(h)}
=
\operatorname{softmax}\!\left(
\frac{Q^{(h)} K^{(h)\top}}{\sqrt{d_k}}
\right)V^{(h)}
\in \mathbb{R}^{T \times d_v}.
$$

Concatenate along the feature dimension:

$$
Z = \operatorname{Concat}(C^{(1)}, \dots, C^{(H)})
\in \mathbb{R}^{T \times (H d_v)}.
$$

Then project back to model width:

$$
Y = ZW_O,
\qquad
W_O \in \mathbb{R}^{H d_v \times d_{\mathrm{model}}}.
$$

## Result

Self-attention is a row-wise normalized matrix of pairwise query-key similarities used to average value vectors:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

The formula is compact, but each factor has a concrete role:

- $QK^\top$ builds pairwise compatibility scores;
- softmax converts scores into stochastic weights;
- multiplication by $V$ aggregates content.

## ML relevance

This derivation is the algebraic core of transformers.
It explains:

- why attention parallelizes across all positions;
- how causal masks yield autoregressive decoding;
- why dimensional scaling matters for optimization; and
- how multi-head attention generalizes the basic mechanism.

## Limitations

- The variance argument for $\sqrt{d_k}$ is heuristic and initialization-based.
- The derivation does not address memory or compute complexity.
- Interpretability claims about attention cannot be read directly from the algebra alone.
