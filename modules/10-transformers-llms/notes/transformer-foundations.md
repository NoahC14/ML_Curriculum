---
title: "Transformer Foundations"
module: "10-transformers-llms"
lesson: "transformer-foundations"
doc_type: "notes"
topic: "self-attention-positional-encoding-encoder-decoder"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
  - "00-math-toolkit/information-theory"
  - "06-neural-networks/neural-networks-first-principles"
  - "07-deep-learning-systems/training-deep-networks"
  - "09-sequence-models/sequence-modeling"
updated: "2026-04-13"
owner: "curriculum-team"
tags:
  - "transformers"
  - "self-attention"
  - "multi-head-attention"
  - "positional-encoding"
  - "encoder-decoder"
---

## Purpose

These notes derive the transformer architecture from the limitations of recurrent sequence models and from the algebra of matrix-based attention.
The central goal is to understand why self-attention is expressive, parallelizable, and now dominant in modern language modeling, while keeping the derivation explicit enough to support later implementation work.

## Learning objectives

After working through this note, you should be able to:

- derive scaled dot-product attention in both token-wise and matrix form;
- annotate the dimensions of the query, key, and value projections;
- explain why the factor $1 / \sqrt{d_k}$ stabilizes attention logits;
- state why plain self-attention is permutation-equivariant and therefore needs positional information;
- compare sinusoidal and learned positional encodings;
- describe multi-head attention as parallel subspace-wise attention;
- distinguish encoder self-attention, decoder masked self-attention, and encoder-decoder cross-attention; and
- connect transformer blocks to modern large language model architectures.

## 1. From recurrent bottlenecks to attention

In an RNN or LSTM, the hidden state $h_t$ must summarize everything relevant from the prefix $x_{1:t}$.
That design creates two pressure points:

- long-range dependencies must be compressed into a fixed-width state; and
- training requires sequential recurrence, so parallelism across time is limited.

Attention relaxes both constraints.
Instead of forcing all past information through a single recurrent state, a token at position $i$ can directly score and aggregate representations from all positions $j$ in the sequence.
This turns sequence modeling into a learned content-based retrieval problem.

## 2. Setup and notation

Let

$$
X \in \mathbb{R}^{T \times d_{\mathrm{model}}}
$$

be the matrix whose $i$th row $x_i^\top$ is the representation of token $i$.
Here:

- $T$ is sequence length;
- $d_{\mathrm{model}}$ is the embedding or residual width.

Single-head attention introduces three learned linear maps:

$$
W_Q \in \mathbb{R}^{d_{\mathrm{model}} \times d_k},
\qquad
W_K \in \mathbb{R}^{d_{\mathrm{model}} \times d_k},
\qquad
W_V \in \mathbb{R}^{d_{\mathrm{model}} \times d_v}.
$$

Define

$$
Q = XW_Q \in \mathbb{R}^{T \times d_k},
\qquad
K = XW_K \in \mathbb{R}^{T \times d_k},
\qquad
V = XW_V \in \mathbb{R}^{T \times d_v}.
$$

The $i$th rows are denoted

$$
q_i^\top, \quad k_i^\top, \quad v_i^\top.
$$

Intuitively:

- the query asks what information a token is looking for;
- the key describes what information each token offers;
- the value is the content that will actually be aggregated.

## 3. Deriving scaled dot-product attention

For a fixed query position $i$, define the unnormalized compatibility score with position $j$ by

$$
e_{ij} = q_i^\top k_j.
$$

This is a learned similarity score in key-query space.
Convert these scores into nonnegative weights with a row-wise softmax:

$$
\alpha_{ij}
=
\frac{\exp(e_{ij})}{\sum_{\ell=1}^T \exp(e_{i\ell})}.
$$

The context vector for token $i$ is then

$$
c_i = \sum_{j=1}^T \alpha_{ij} v_j.
$$

Because the weights sum to one, $c_i$ is a weighted average of the value vectors.

### 3.1 Matrix form

Stack the scores for all query-key pairs:

$$
S = QK^\top \in \mathbb{R}^{T \times T}.
$$

Apply the softmax row-wise:

$$
A = \operatorname{softmax}(S),
$$

where each row of $A$ sums to one.
Then the attention output is

$$
C = AV \in \mathbb{R}^{T \times d_v}.
$$

The $(i,j)$ entry of $A$ is exactly $\alpha_{ij}$, so the $i$th row of $C$ is $c_i^\top$.

### 3.2 Why the scaling factor appears

If the coordinates of $q_i$ and $k_j$ are approximately independent with mean zero and variance one, then

$$
q_i^\top k_j = \sum_{r=1}^{d_k} q_{ir} k_{jr}.
$$

Under that crude but useful approximation,

$$
\mathbb{E}[q_{ir}k_{jr}] = 0,
\qquad
\operatorname{Var}(q_{ir}k_{jr}) \approx 1,
$$

so

$$
\operatorname{Var}(q_i^\top k_j) \approx d_k.
$$

Hence the typical logit magnitude grows like $\sqrt{d_k}$.
Large logits push the softmax into saturation, making one position dominate and making gradients small or unstable.

To keep the score scale roughly dimension-independent, transformers use

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

The factor $1 / \sqrt{d_k}$ is therefore a variance-normalization heuristic grounded in the dot-product scale.

> **Remark.** This is not a proof that the scaled logits have ideal variance in every trained network.
> It is a design argument showing why unscaled dot products become numerically harder to optimize as $d_k$ grows.

## 4. Why attention needs positional information

Plain self-attention has no inherent notion of token order.
If we permute the rows of $X$ by a permutation matrix $P \in \mathbb{R}^{T \times T}$, then

$$
Q' = PXW_Q = PQ,
\qquad
K' = PK,
\qquad
V' = PV.
$$

The score matrix becomes

$$
Q'K'^\top = (PQ)(PK)^\top = PQK^\top P^\top.
$$

Row-wise softmax commutes with the same simultaneous row-column permutation, so the output is permuted in the same way:

$$
\operatorname{Attention}(PQ,PK,PV) = P \operatorname{Attention}(Q,K,V).
$$

This is permutation-equivariance.
Without extra structure, the model cannot distinguish

- "dog bites man" from
- "man bites dog"

if the multiset of token embeddings is the same.
Positional information breaks that symmetry.

## 5. Positional encoding

There are two broad strategies.

### 5.1 Sinusoidal positional encoding

Vaswani et al. proposed a deterministic encoding

$$
\operatorname{PE}(\mathrm{pos}, 2r)
=
\sin\!\left(\frac{\mathrm{pos}}{10000^{2r / d_{\mathrm{model}}}}\right),
$$

$$
\operatorname{PE}(\mathrm{pos}, 2r+1)
=
\cos\!\left(\frac{\mathrm{pos}}{10000^{2r / d_{\mathrm{model}}}}\right).
$$

The position vector $\operatorname{PE}(\mathrm{pos}) \in \mathbb{R}^{d_{\mathrm{model}}}$ is added to the token embedding.
Different frequencies allow the model to represent both local and global relative variation.

Advantages:

- no learned position parameters are needed;
- the encoding extends naturally to longer lengths than seen in training, at least heuristically;
- relative offsets can be approximately reconstructed from phase relationships.

### 5.2 Learned positional embeddings

An alternative is to learn

$$
p_1, \dots, p_T \in \mathbb{R}^{d_{\mathrm{model}}}
$$

and use $x_i + p_i$ at input time.
This is simpler and often works well in practice, but extrapolation beyond the training length is less principled.

### 5.3 Relative and modern positional schemes

Many modern LLMs use relative or rotary schemes rather than plain absolute embeddings.
The important foundational point is not the specific variant but the reason one is needed:
attention alone does not encode order.

## 6. Multi-head attention

Single-head attention produces one set of weights and one aggregation pattern.
Multi-head attention runs several attention mechanisms in parallel:

$$
Q^{(h)} = X W_Q^{(h)},
\qquad
K^{(h)} = X W_K^{(h)},
\qquad
V^{(h)} = X W_V^{(h)},
\qquad
h = 1,\dots,H.
$$

Each head computes

$$
\mathrm{head}^{(h)}
=
\operatorname{softmax}\!\left(
\frac{Q^{(h)} K^{(h)\top}}{\sqrt{d_k}}
\right)
V^{(h)}.
$$

Concatenate the head outputs:

$$
Z = \operatorname{Concat}\bigl(\mathrm{head}^{(1)}, \dots, \mathrm{head}^{(H)}\bigr)
\in \mathbb{R}^{T \times (H d_v)}.
$$

Then apply an output projection

$$
Y = ZW_O,
\qquad
W_O \in \mathbb{R}^{H d_v \times d_{\mathrm{model}}}.
$$

Different heads can specialize to different relations:

- local syntactic dependencies;
- long-range subject-verb agreement;
- delimiter matching;
- retrieval of factual cues;
- copy-like behavior.

Multi-head attention is therefore best understood as multiple learned relation spaces rather than one monolithic attention map.

## 7. Transformer blocks

A standard transformer layer combines attention with a position-wise feedforward network.
In modern notation, one block usually contains:

1. multi-head attention;
2. residual connection;
3. layer normalization;
4. feedforward network applied independently at each position;
5. another residual connection and normalization.

The feedforward sublayer typically has the form

$$
\operatorname{FFN}(x)
=
W_2 \sigma(W_1 x + b_1) + b_2,
$$

applied to each token position separately.

Attention mixes information across positions.
The FFN mixes information across channels within a position.

## 8. Encoder, decoder, and cross-attention

### 8.1 Encoder

An encoder stack applies bidirectional self-attention.
Each token may attend to every other token in the input.
This is natural for representation learning tasks such as classification, retrieval, or masked-token prediction.

### 8.2 Decoder

A decoder stack uses masked self-attention so that token $i$ can attend only to positions $j \leq i$.
This preserves the autoregressive factorization

$$
p(x_{1:T}) = \prod_{t=1}^T p(x_t \mid x_{<t}).
$$

Masked attention is therefore essential for causal language modeling.

### 8.3 Encoder-decoder architecture

Sequence-to-sequence transformers, such as the original translation model in Vaswani et al., combine:

- an encoder that builds contextual source representations; and
- a decoder that performs masked self-attention over the target prefix and cross-attention to the encoder outputs.

In cross-attention, the decoder hidden states provide queries while the encoder outputs provide keys and values:

$$
Q = H_{\mathrm{dec}} W_Q,
\qquad
K = H_{\mathrm{enc}} W_K,
\qquad
V = H_{\mathrm{enc}} W_V.
$$

This lets the decoder decide which source positions matter while generating each target token.

## 9. Masked attention in detail

Let $M \in \mathbb{R}^{T \times T}$ be a mask matrix with

$$
M_{ij}
=
\begin{cases}
0, & j \leq i, \\
-\infty, & j > i.
\end{cases}
$$

Then causal attention uses

$$
\operatorname{softmax}\!\left(
\frac{QK^\top}{\sqrt{d_k}} + M
\right)V.
$$

The $-\infty$ entries become zero after softmax, so future positions receive zero probability mass.

## 10. Architectural lineages of modern LLMs

Three families matter most:

- encoder-only models, such as BERT, emphasize bidirectional representations and masked-token objectives;
- encoder-decoder models, such as T5, emphasize conditional generation;
- decoder-only models, such as GPT-style systems, emphasize causal next-token prediction and generation.

Current frontier LLMs are predominantly decoder-only because the causal objective scales cleanly and aligns naturally with text generation.
But the underlying transformer machinery still comes from the general encoder-decoder framework.

## 11. Computational interpretation

Self-attention can be read as a differentiable memory lookup:

- keys index content;
- queries choose which content matters;
- values are the retrieved payloads.

In matrix form, the entire sequence performs this lookup in parallel.
That parallelism is one reason transformers displaced RNNs at scale, despite their quadratic attention cost in sequence length.

## 12. Category-theoretic insertion point

At this stage the category-theory role should stay modest.
One useful structural reading is that a transformer block composes:

- a relation-building map from tokens to pairwise compatibilities;
- a normalization map from compatibilities to stochastic weights; and
- a weighted aggregation map from values to updated representations.

This does not replace the linear algebra.
It simply highlights that attention is a compositional pipeline rather than an opaque primitive.

## 13. Limitations and scope notes

- Standard self-attention has $O(T^2)$ score interactions, which becomes expensive for long contexts.
- Attention weights are not identical to explanations; high weight does not by itself prove causal importance.
- The $\sqrt{d_k}$ argument is a stabilizing heuristic, not a universal theorem of good optimization.
- Modern architectures add many refinements, including rotary position embeddings, grouped-query attention, mixture-of-experts layers, and improved normalization choices.

## References

- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Bahdanau, D., Cho, K., and Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- Raffel, C. et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. JMLR. [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)
