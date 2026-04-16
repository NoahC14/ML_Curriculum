---
title: "Sequence Modeling Exercises"
module: "09-sequence-models"
lesson: "sequence-exercises"
doc_type: "exercise"
topic: "rnn-bptt-lstm-gru-seq2seq-decoding"
status: "draft"
prerequisites:
  - "06-neural-networks/backpropagation"
  - "07-deep-learning-systems/training-deep-networks"
  - "09-sequence-models/sequence-modeling"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "sequence-modeling"
  - "rnn"
  - "bptt"
  - "lstm"
  - "gru"
  - "teacher-forcing"
  - "beam-search"
---

## Purpose

These exercises reinforce the structure, derivations, and design choices behind recurrent sequence models.
They mix symbolic derivation, interpretation, and short design analysis.

## Exercise 1: first-order versus finite-window dependence

**Taxonomy**

- `difficulty`: `foundational`
- `type`: `analysis`
- `tags`: `markov-assumption`, `context-window`, `sequence-dependence`

Suppose a language process is modeled by

$$
p(w_t \mid w_{1:t-1}).
$$

1. State the first-order Markov assumption.
2. State the $k$th-order Markov assumption.
3. Explain why an $n$-gram model is a finite-order Markov model.
4. Give one linguistic dependency that a small fixed window is likely to miss.
5. Explain how a hidden state $h_t$ attempts to overcome the fixed-window limitation.

## Exercise 2: forward pass of a simple RNN

Consider

$$
a_t = W_{xh}x_t + W_{hh}h_{t-1} + b_h,
\qquad
h_t = \tanh(a_t),
\qquad
o_t = W_{hy}h_t + b_y.
$$

Assume

$$
x_t \in \mathbb{R}^d,
\qquad
h_t \in \mathbb{R}^m,
\qquad
o_t \in \mathbb{R}^p.
$$

1. State the dimensions of $W_{xh}$, $W_{hh}$, $W_{hy}$, $b_h$, and $b_y$.
2. Explain what it means for the parameters to be shared across time.
3. Write the computation graph for times $t=1,2,3$ in words or diagram form.
4. Explain why the unrolled graph can be regarded as a depth-$T$ network.

## Exercise 3: BPTT recursion

Let

$$
\mathcal{L} = \sum_{t=1}^T \ell_t
$$

for the simple RNN above, and define

$$
\delta_t = \frac{\partial \mathcal{L}}{\partial a_t}.
$$

1. Show that

$$
\frac{\partial \mathcal{L}}{\partial h_t}
=
g_t^{\mathrm{out}} + W_{hh}^\top \delta_{t+1},
$$

where $g_t^{\mathrm{out}} = \frac{\partial \ell_t}{\partial h_t}$.

2. Deduce that

$$
\delta_t
=
\left(g_t^{\mathrm{out}} + W_{hh}^\top \delta_{t+1}\right) \odot \phi'(a_t).
$$

3. Derive

$$
\frac{\partial \mathcal{L}}{\partial W_{hh}}
=
\sum_{t=1}^T \delta_t h_{t-1}^\top.
$$

4. Explain in words why the sum over time appears.
5. Explain why BPTT is a special case of ordinary backpropagation on an unrolled graph.

## Exercise 4: gradient pathologies

Suppose

$$
J_j = W_{hh}^\top \operatorname{Diag}(\phi'(a_j)).
$$

1. Write the Jacobian product that transports gradient information from time $t+k$ back to time $t$.
2. If $\|J_j\|_2 \leq \alpha < 1$ for all $j$, prove a norm bound showing vanishing behavior.
3. If $\|J_j\|_2 \geq \beta > 1$ often enough, explain why exploding gradients may occur.
4. Why do saturating nonlinearities such as sigmoid and $\tanh$ worsen vanishing gradients?
5. Give two practical mitigation strategies and state whether they solve the problem fundamentally or only partially.

## Exercise 5: truncated BPTT

You are training on a sequence of length $T=500$ but backpropagating only through windows of length $\tau=40$.

1. Describe how truncated BPTT works operationally.
2. Which hidden states must still be carried forward in the forward pass?
3. Which dependencies cannot be learned exactly under this approximation?
4. Why is truncated BPTT still useful in practice?

## Exercise 6: LSTM gate roles

Consider the LSTM equations

$$
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f),
$$

$$
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i),
$$

$$
\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c),
$$

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t,
$$

$$
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o),
\qquad
h_t = o_t \odot \tanh(c_t).
$$

1. State the role of each of the four named gates or states: $f_t$, $i_t$, $\tilde{c}_t$, $o_t$.
2. Show that

$$
\frac{\partial c_t}{\partial c_{t-1}} = \operatorname{Diag}(f_t).
$$

3. Explain why this derivative is more stable than repeatedly multiplying by a dense recurrent matrix alone.
4. What failure mode can still occur if the forget gate is consistently near $0$?
5. Why is the output gate useful even though the memory cell already stores information?

## Exercise 7: GRU simplification

For the GRU equations

$$
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z),
\qquad
r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r),
$$

$$
\tilde{h}_t = \tanh(W_h x_t + U_h(r_t \odot h_{t-1}) + b_h),
$$

$$
h_t = (1-z_t)\odot h_{t-1} + z_t \odot \tilde{h}_t,
$$

1. Explain how the update gate $z_t$ blends old and new information.
2. Explain the role of the reset gate $r_t$.
3. Compare the GRU state update with the LSTM cell update.
4. Give one reason a practitioner might prefer a GRU over an LSTM.
5. Give one reason the reverse choice might still be reasonable.

## Exercise 8: teacher forcing

A decoder models

$$
p(y_u \mid y_{1:u-1}, x_{1:T}).
$$

1. Define teacher forcing precisely.
2. Why does teacher forcing typically make optimization easier?
3. Define exposure bias.
4. Give one example of how an early decoding mistake can cascade.
5. Name one strategy, other than plain teacher forcing, that attempts to reduce the train-test mismatch.

## Exercise 9: beam search

Suppose the beam width is $B=2$.
At decoding step $u=1$, the model assigns:

- token A: probability $0.6$;
- token B: probability $0.3$;
- token C: probability $0.1$.

At step $u=2$, the conditional probabilities are:

- from A: next-token probabilities $(0.4, 0.35, 0.25)$ for tokens D, E, F;
- from B: next-token probabilities $(0.9, 0.05, 0.05)$ for tokens D, E, F.

1. Using cumulative log-probability, list the two beam hypotheses after step $1$.
2. Expand those hypotheses and compute their cumulative scores after step $2$.
3. Which two hypotheses remain in the beam?
4. Why might greedy decoding choose a worse final sequence here?
5. Why is beam search still not globally optimal in general?

## Exercise 10: seq2seq bottleneck and attention precursor

Consider a classical encoder-decoder RNN that uses only the final encoder state $h_T^{\mathrm{enc}}$ to initialize the decoder.

1. Explain why this creates a fixed-size information bottleneck.
2. Why does the bottleneck become more severe as the source sequence grows longer or more information-dense?
3. Write a one-sentence description of attention as a remedy.
4. Explain how attention shortens the path between a source token and a decoder decision.
5. Why does this motivation matter even if the final deployed model is a transformer rather than an RNN?
