---
title: "Transformer Exercises"
module: "10-transformers-llms"
lesson: "transformer-exercises"
doc_type: "exercise"
topic: "attention-positional-encoding-pretraining-scaling"
status: "draft"
prerequisites:
  - "09-sequence-models/sequence-modeling"
  - "10-transformers-llms/transformer-foundations"
  - "10-transformers-llms/pretraining-and-scaling"
updated: "2026-04-13"
owner: "curriculum-team"
tags:
  - "transformers"
  - "self-attention"
  - "positional-encoding"
  - "pretraining"
  - "scaling-laws"
---

## Purpose

These exercises reinforce the derivation of self-attention, the role of positional encoding, and the training pipeline behind modern transformer language models.
They mix symbolic derivation, architectural comparison, and objective-level interpretation.

## Exercise 1: annotate every dimension

Let

$$
X \in \mathbb{R}^{T \times d_{\mathrm{model}}},
\qquad
W_Q, W_K \in \mathbb{R}^{d_{\mathrm{model}} \times d_k},
\qquad
W_V \in \mathbb{R}^{d_{\mathrm{model}} \times d_v}.
$$

1. State the dimensions of $Q$, $K$, and $V$.
2. State the dimensions of $QK^\top$.
3. State the dimensions of $\operatorname{softmax}(QK^\top / \sqrt{d_k})$.
4. State the dimensions of the final attention output.
5. Explain why the output has one row per input token.

## Exercise 2: derive the scaling factor

Assume the coordinates of $q_i$ and $k_j$ are independent, mean-zero, and unit-variance.

1. Write $q_i^\top k_j$ as a sum over coordinates.
2. Compute the variance of the sum under the independence approximation.
3. Show that dividing by $\sqrt{d_k}$ makes the variance approximately constant in $d_k$.
4. Explain why large unscaled logits make the softmax harder to optimize.
5. State one limitation of this derivation.

## Exercise 3: weighted averaging interpretation

For a fixed query position $i$, let

$$
\alpha_{ij}
=
\frac{\exp(e_{ij})}{\sum_{\ell=1}^T \exp(e_{i\ell})},
\qquad
c_i = \sum_{j=1}^T \alpha_{ij} v_j.
$$

1. Prove that $\alpha_{ij} \geq 0$.
2. Prove that $\sum_j \alpha_{ij} = 1$.
3. Conclude that $c_i$ is a convex combination of the value vectors.
4. Explain what happens when one score $e_{ij}$ is much larger than all others.

## Exercise 4: permutation invariance without position

Let $P$ be a permutation matrix acting on the rows of $X$.

1. Show that if $Q = XW_Q$, then $Q' = PXW_Q = PQ$.
2. Show similarly that $K' = PK$ and $V' = PV$.
3. Derive the transformed score matrix $Q'K'^\top$.
4. Explain why plain self-attention is permutation-equivariant.
5. Give a concrete language example showing why this is unacceptable without positional information.

## Exercise 5: sinusoidal versus learned positional encoding

1. Write the sinusoidal encoding formulas for even and odd coordinates.
2. Explain why multiple frequencies are used.
3. State one advantage of sinusoidal encodings.
4. State one advantage of learned positional embeddings.
5. Give one setting where relative position information may matter more than absolute position information.

## Exercise 6: construct a causal mask

Let $T = 4$.

1. Write the $4 \times 4$ causal mask matrix $M$ with entries $0$ or $-\infty$.
2. Which entries prevent token 2 from attending to future tokens?
3. After adding $M$ before the softmax, what must the attention weights from position 2 to positions 3 and 4 become?
4. Explain why this mask is consistent with the factorization

$$
p(x_{1:T}) = \prod_{t=1}^T p(x_t \mid x_{<t}).
$$

## Exercise 7: compare transformer architectures

For each architecture below, state the dominant pretraining objective and a natural task family:

1. encoder-only;
2. decoder-only;
3. encoder-decoder.

Then answer:

4. Which family is most natural for bidirectional token representation learning?
5. Which family is most natural for free-form text generation?
6. Which family is most natural for translation?

## Exercise 8: MLM and CLM with concrete examples

Consider the sentence

$$
\text{``the cat sat on the mat''}.
$$

1. Write one MLM training example using a masked token.
2. Write three CLM conditional prediction targets from the same sentence.
3. Explain why MLM can use right context but CLM cannot.
4. Explain why CLM is better aligned with autoregressive sampling.

## Exercise 9: scaling laws as empirical findings

1. What does it mean to say a scaling law is empirical rather than a theorem?
2. State the basic qualitative lesson of Kaplan et al.
3. State the compute-allocation lesson of Hoffmann et al.
4. Why can a larger model still be inefficient if the training token budget is too small?
5. Give one reason a scaling trend measured on one setup may not transfer perfectly to another.

## Exercise 10: instruction tuning and RLHF

1. Explain the difference between pretraining and instruction tuning.
2. What kind of data is used in supervised instruction tuning?
3. What kind of data is used to train a reward model in RLHF?
4. Why is a preference model not the same thing as a ground-truth notion of correctness?
5. State one failure mode that can arise during preference-based post-training.
