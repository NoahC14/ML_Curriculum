---
title: "Sequence Modeling"
module: "09-sequence-models"
lesson: "sequence-modeling"
doc_type: "notes"
topic: "rnn-lstm-gru-seq2seq-attention-transition"
status: "draft"
prerequisites:
  - "02-statistical-learning/statistical-learning-foundations"
  - "05-probabilistic-modeling/graphical-models"
  - "06-neural-networks/neural-networks-first-principles"
  - "06-neural-networks/backpropagation"
  - "07-deep-learning-systems/training-deep-networks"
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
  - "seq2seq"
  - "attention"
---

## Purpose

These notes study sequence models in the form that historically led from Markov assumptions to recurrent neural networks and then to gated architectures.
The technical goal is to understand how sequential dependence is modeled, why recurrent training is difficult, and why those difficulties motivate attention-based models later in the course.

RNNs are no longer the dominant architecture for large-scale language modeling, but they remain essential for two reasons:

- they make temporal credit assignment explicit;
- they show concretely which optimization and memory problems transformers are designed to avoid.

## Learning objectives

After working through this note, you should be able to:

- distinguish finite-order Markov assumptions from hidden-state sequence models;
- write an RNN forward pass in both recursive and unrolled form;
- derive backpropagation through time as ordinary backpropagation on an unrolled graph with shared parameters;
- explain the vanishing and exploding gradient problem in terms of repeated Jacobian products;
- derive the LSTM cell and motivate the input, forget, and output gates;
- explain how GRUs simplify LSTMs;
- describe teacher forcing, exposure bias, greedy decoding, and beam search;
- work through a basic sequence-to-sequence example; and
- state the core memory bottleneck that motivates attention.

## 1. Why sequences are different

In ordinary supervised learning we often assume examples are independent:

$$
(x^{(1)}, y^{(1)}), \dots, (x^{(n)}, y^{(n)}).
$$

A sequence task instead receives ordered data

$$
x_{1:T} = (x_1, \dots, x_T),
$$

possibly with targets

$$
y_{1:T} = (y_1, \dots, y_T)
\quad \text{or} \quad
y.
$$

Order matters.
The interpretation of token $x_t$ usually depends on what came before it.
Examples include:

- language modeling: predict $x_{t+1}$ from $x_{1:t}$;
- sequence labeling: map $x_{1:T}$ to $y_{1:T}$;
- sequence classification: map $x_{1:T}$ to one label $y$;
- sequence-to-sequence translation: map source sequence $x_{1:T}$ to target sequence $y_{1:U}$.

The central modeling question is how to summarize past information compactly enough to make prediction at the next step.

## 2. Markov assumptions and their limits

### 2.1 Finite-order Markov structure

For a stochastic process $(X_t)$, a first-order Markov assumption states

$$
p(x_t \mid x_{1:t-1}) = p(x_t \mid x_{t-1}).
$$

More generally, a $k$th-order Markov model assumes

$$
p(x_t \mid x_{1:t-1}) = p(x_t \mid x_{t-k:t-1}).
$$

This reduces long-history prediction to a fixed-size context window.
Classical $n$-gram language models are a standard example.

### 2.2 Why Markov models are often too weak

Finite-order Markov assumptions are attractive because they make likelihoods and dynamic-programming algorithms tractable.
But they have two structural weaknesses:

- long-range dependencies require very large state spaces;
- the model must treat each context pattern separately unless we impose parameter sharing.

For example, in language the subject of a sentence may appear many tokens before the verb.
A fixed-window Markov model may miss that dependence unless the window is widened drastically.

This motivates hidden-state models.
Instead of conditioning directly on the raw past, we build a learned state summary:

$$
h_t = \text{summary of } x_{1:t}.
$$

That summary should be low-dimensional enough to learn and rich enough to support prediction.

## 3. Recurrent neural networks

### 3.1 State update

An RNN defines a hidden state $h_t \in \mathbb{R}^m$ recursively:

$$
h_t = \phi(W_{xh} x_t + W_{hh} h_{t-1} + b_h),
$$

where

$$
x_t \in \mathbb{R}^d,
\qquad
W_{xh} \in \mathbb{R}^{m \times d},
\qquad
W_{hh} \in \mathbb{R}^{m \times m},
\qquad
b_h \in \mathbb{R}^m.
$$

The hidden state is then mapped to an output,

$$
o_t = W_{hy} h_t + b_y,
\qquad
\hat{y}_t = \psi(o_t),
$$

with

$$
W_{hy} \in \mathbb{R}^{p \times m},
\qquad
b_y \in \mathbb{R}^p.
$$

The same parameters are reused at every time step.
This parameter tying is the defining inductive bias of an RNN.

### 3.2 Unrolling through time

Although the recurrence is written compactly, the computational graph over a finite sequence is

$$
h_0 \to h_1 \to h_2 \to \cdots \to h_T.
$$

Unrolling reveals that an RNN is a deep network of depth $T$ whose layers share weights.
For a sequence loss

$$
\mathcal{L} = \sum_{t=1}^T \ell_t(\hat{y}_t, y_t),
$$

the graph is just a repeated composition.
That is why backpropagation through time is not a new calculus rule.
It is ordinary backpropagation applied to the unrolled graph.

### 3.3 A concrete language-model form

In a next-token language model, the input is often an embedding $e_t = E[w_t]$ of token $w_t$.
Then

$$
h_t = \tanh(W_{xh} e_t + W_{hh} h_{t-1} + b_h),
$$

and

$$
p(w_{t+1} = j \mid w_{1:t})
=
\frac{\exp((W_{hy} h_t + b_y)_j)}
{\sum_{k=1}^{V} \exp((W_{hy} h_t + b_y)_k)},
$$

where $V$ is the vocabulary size.

The model can in principle use all prior tokens through $h_t$.
In practice, the hidden state may fail to preserve useful information over long horizons.

## 4. Backpropagation through time

## 4.1 Setup

Consider a simple RNN with hidden preactivation

$$
a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h,
\qquad
h_t = \phi(a_t),
$$

and output

$$
o_t = W_{hy} h_t + b_y,
\qquad
\hat{y}_t = \psi(o_t).
$$

Let the total loss be

$$
\mathcal{L} = \sum_{t=1}^T \ell_t(\hat{y}_t, y_t).
$$

Define the hidden-state error signal

$$
\delta_t := \frac{\partial \mathcal{L}}{\partial a_t} \in \mathbb{R}^m.
$$

### 4.2 Why later losses affect earlier states

The hidden state $h_t$ affects the loss at time $t$, but also all later losses through $h_{t+1}, h_{t+2}, \dots$.
So

$$
\frac{\partial \mathcal{L}}{\partial h_t}
=
\frac{\partial \ell_t}{\partial h_t}
+ \frac{\partial \mathcal{L}_{t+1:T}}{\partial h_t},
$$

where $\mathcal{L}_{t+1:T} = \sum_{s=t+1}^T \ell_s$.

If we define

$$
g_t^{\text{out}} := \frac{\partial \ell_t}{\partial h_t},
$$

then by the chain rule

$$
\frac{\partial \mathcal{L}}{\partial h_t}
=
g_t^{\text{out}}
+
\left(\frac{\partial a_{t+1}}{\partial h_t}\right)^\top
 \frac{\partial \mathcal{L}}{\partial a_{t+1}}.
$$

Because

$$
\frac{\partial a_{t+1}}{\partial h_t} = W_{hh},
$$

we get

$$
\frac{\partial \mathcal{L}}{\partial h_t}
=
g_t^{\text{out}} + W_{hh}^\top \delta_{t+1}.
$$

Applying the activation derivative gives

$$
\delta_t
=
\left(g_t^{\text{out}} + W_{hh}^\top \delta_{t+1}\right) \odot \phi'(a_t).
$$

This is the core BPTT recursion.

### 4.3 Parameter gradients

Because parameters are shared across time, their total gradient is the sum of time-step contributions:

$$
\frac{\partial \mathcal{L}}{\partial W_{xh}}
=
\sum_{t=1}^T \delta_t x_t^\top,
$$

$$
\frac{\partial \mathcal{L}}{\partial W_{hh}}
=
\sum_{t=1}^T \delta_t h_{t-1}^\top,
$$

$$
\frac{\partial \mathcal{L}}{\partial b_h}
=
\sum_{t=1}^T \delta_t.
$$

Similarly,

$$
\frac{\partial \mathcal{L}}{\partial W_{hy}}
=
\sum_{t=1}^T \frac{\partial \ell_t}{\partial o_t} h_t^\top,
\qquad
\frac{\partial \mathcal{L}}{\partial b_y}
=
\sum_{t=1}^T \frac{\partial \ell_t}{\partial o_t}.
$$

This is exactly what we should expect from shared weights:
every time step uses the same parameters, so every time step contributes to the same gradient accumulator.

### 4.4 BPTT as a special case of backpropagation

There is no separate principle called "through time" in the calculus.
If we create copies

$$
W_{hh}^{(1)}, \dots, W_{hh}^{(T)}
$$

for the unrolled network, ordinary backpropagation gives one gradient for each copy.
Parameter tying then imposes

$$
W_{hh}^{(1)} = \cdots = W_{hh}^{(T)} = W_{hh},
$$

so the true gradient is

$$
\frac{\partial \mathcal{L}}{\partial W_{hh}}
=
\sum_{t=1}^T
\frac{\partial \mathcal{L}}{\partial W_{hh}^{(t)}}.
$$

That is BPTT.
It is backpropagation on an unrolled graph followed by summation over tied-parameter copies.

### 4.5 Truncated BPTT

For long sequences, storing every hidden state and backpropagating through the full history is expensive.
A common approximation is truncated BPTT:

- run the RNN forward for a chunk of length $\tau$;
- backpropagate only through that chunk;
- carry the hidden state forward but stop the gradient at chunk boundaries.

This lowers cost but weakens long-range credit assignment.

## 5. Vanishing and exploding gradients

### 5.1 Repeated Jacobian products

The recurrence for $\delta_t$ contains repeated multiplication by $W_{hh}^\top$ and by activation derivatives.
If we expand backward from time $t+k$ to time $t$, then ignoring output-specific details we obtain a factor of the form

$$
\prod_{j=t+1}^{t+k}
\left(
W_{hh}^\top \operatorname{Diag}(\phi'(a_j))
\right).
$$

The norm of this product controls how much later losses influence earlier states.

If

$$
\left\|
W_{hh}^\top \operatorname{Diag}(\phi'(a_j))
\right\| < 1
$$

typically, gradients shrink exponentially with distance.
If the norm is often greater than $1$, they can grow exponentially.

### 5.2 A simple bound

Suppose

$$
\|W_{hh}\|_2 \leq \rho
\qquad \text{and} \qquad
\|\operatorname{Diag}(\phi'(a_j))\|_2 \leq \gamma
$$

for all relevant $j$.
Then

$$
\left\|
\prod_{j=t+1}^{t+k}
\left(
W_{hh}^\top \operatorname{Diag}(\phi'(a_j))
\right)
\right\|_2
\leq
(\rho \gamma)^k.
$$

Therefore:

- if $\rho \gamma < 1$, gradient signals decay geometrically;
- if $\rho \gamma > 1$, they may blow up geometrically.

For $\tanh$ and sigmoid activations, $\gamma \leq 1$ and is often much smaller in saturated regions.
That makes vanishing especially common.

### 5.3 Empirical demonstration

The notebook [SEQ-01-rnn-from-scratch](../notebooks/SEQ-01-rnn-from-scratch.ipynb) measures hidden-state gradient norms as sequence length increases.
Two phenomena appear:

- with a recurrent matrix scaled below the stability threshold, the earliest-step gradients collapse toward zero;
- with a recurrent matrix scaled above it, norms spike and optimization becomes erratic.

This is one of the central historical reasons that gated RNNs and later attention-based models were developed.

### 5.4 Practical mitigations

Before LSTMs and GRUs became standard, practitioners used several partial fixes:

- orthogonal or carefully scaled initialization;
- $\tanh$ instead of sigmoid hidden states;
- gradient clipping;
- truncated BPTT;
- better optimizers and normalization choices.

These help, but they do not solve the memory bottleneck fundamentally.

## 6. Long short-term memory networks

### 6.1 Motivation

A vanilla RNN stores history only in the hidden state $h_t$.
Every update overwrites the same vector through one nonlinear recurrence.
That makes it hard to preserve information over long spans while still adapting rapidly to new inputs.

LSTMs introduce an explicit memory cell $c_t$ and gates that control what to erase, write, and expose.
The design goal is to create a pathway along which gradients can propagate more stably.

### 6.2 Equations

Given input $x_t \in \mathbb{R}^d$, previous hidden state $h_{t-1} \in \mathbb{R}^m$, and previous cell state $c_{t-1} \in \mathbb{R}^m$, define

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
$$

$$
h_t = o_t \odot \tanh(c_t).
$$

All gate vectors lie in $(0,1)^m$ because of the sigmoid nonlinearity.

### 6.3 Why each gate exists

- Forget gate $f_t$:
  controls how much of the old memory $c_{t-1}$ is retained.
  If $f_t \approx 1$, memory persists; if $f_t \approx 0$, memory is erased.

- Input gate $i_t$:
  controls how much new candidate information enters memory.

- Candidate state $\tilde{c}_t$:
  proposes content that might be written into memory.

- Output gate $o_t$:
  controls how much of the memory is exposed to the hidden state used for prediction.

### 6.4 Why LSTMs help gradients

The cell update

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

contains an additive path.
Differentiating with respect to the previous cell state gives

$$
\frac{\partial c_t}{\partial c_{t-1}} = \operatorname{Diag}(f_t).
$$

If the forget gate stays near $1$ on relevant coordinates, then gradients can travel along the cell state without repeated multiplication by a full dense recurrent matrix.
This is not perfect memory, but it is much less fragile than the vanilla RNN recurrence.

### 6.5 Limitations

LSTMs still process tokens sequentially.
That implies:

- limited parallelism over time;
- long dependency paths from distant tokens to current decisions;
- compression of the source sequence into a finite-dimensional state in encoder-decoder settings.

Those limitations will matter when we reach transformers.

## 7. Gated recurrent units

GRUs simplify the LSTM by combining some gates and removing the separate cell state.
A common formulation is

$$
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z),
$$

$$
r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r),
$$

$$
\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h),
$$

$$
h_t = (1-z_t)\odot h_{t-1} + z_t \odot \tilde{h}_t.
$$

Interpretation:

- update gate $z_t$ plays a role similar to a combined write/keep decision;
- reset gate $r_t$ controls how much of the old state is used in building the candidate state.

GRUs often perform comparably to LSTMs with fewer parameters and a simpler state structure.
The main conceptual point is the same:
gating introduces adaptive additive paths that improve temporal credit assignment.

## 8. Sequence-to-sequence modeling

### 8.1 Encoder-decoder idea

In machine translation, the source and target lengths may differ.
A classic seq2seq RNN uses:

- an encoder that reads the source sequence $x_{1:T}$ and produces a final state $h_T^{\text{enc}}$;
- a decoder that generates target tokens one step at a time conditioned on that state.

Formally,

$$
h_t^{\text{enc}} = f_{\text{enc}}(x_t, h_{t-1}^{\text{enc}}),
$$

and decoder states follow

$$
s_u = f_{\text{dec}}(y_{u-1}, s_{u-1}, h_T^{\text{enc}}),
$$

with token probabilities

$$
p(y_u \mid y_{1:u-1}, x_{1:T}) = \psi(W s_u + b).
$$

### 8.2 Concrete example

Suppose the source is

$$
\texttt{[je, suis, ici]}
$$

and the target is

$$
\texttt{[i, am, here]}.
$$

The encoder compresses the French sequence into a final representation.
The decoder then predicts:

- first token conditioned on a start symbol and encoder summary;
- second token conditioned on the first target token and updated decoder state;
- and so on until an end symbol is produced.

This framework works, but it creates a bottleneck:
the entire source must pass through a fixed-size vector if we use only the final encoder state.

## 9. Teacher forcing and exposure bias

During training of an autoregressive decoder, we often feed the true previous token rather than the model's own previous prediction.
This is teacher forcing.

If the target sequence is $y_{1:U}$, decoder training uses

$$
p(y_u \mid y_{1:u-1}^{\text{gold}}, x_{1:T})
$$

for each $u$.

Teacher forcing stabilizes optimization because the model sees correct histories.
It also lets us compute all stepwise losses against known prefixes in parallel across training examples.

However, inference differs:
the model must condition on its own sampled or decoded outputs,

$$
p(y_u \mid \hat{y}_{1:u-1}, x_{1:T}).
$$

This train-test mismatch is called exposure bias.
An early decoding mistake changes the future context and can compound.

The notebook [SEQ-02-lstm-language-model](../notebooks/SEQ-02-lstm-language-model.ipynb) shows this explicitly in a small character-level setting.

## 10. Greedy decoding and beam search

### 10.1 Greedy decoding

Greedy decoding selects

$$
\hat{y}_u = \arg\max_j p(y_u = j \mid \hat{y}_{1:u-1}, x_{1:T})
$$

at each step.

This is cheap but myopic.
A locally best token can block a better later continuation.

### 10.2 Beam search

Beam search keeps the top $B$ partial hypotheses at each step according to cumulative log-probability.
For a candidate sequence $y_{1:u}$, score

$$
\operatorname{score}(y_{1:u})
=
\sum_{t=1}^u \log p(y_t \mid y_{1:t-1}, x_{1:T}).
$$

Algorithmically:

1. start with the begin-of-sequence token;
2. expand each live hypothesis by one token;
3. keep the top $B$ scored continuations;
4. stop when enough end-of-sequence hypotheses appear or a maximum length is reached.

Beam search is still approximate.
It improves over greedy decoding by exploring several plausible futures, but it does not guarantee the globally optimal sequence.

Length normalization is often added because raw log-probability favors shorter outputs.

## 11. Why attention is the next step

Classical encoder-decoder RNNs ask the decoder to recover all relevant source information from one compressed state.
For long or information-rich sequences, this is a severe bottleneck.

Attention loosens that bottleneck.
Instead of forcing all information into one vector, the decoder can query encoder states directly:

$$
\text{context at step } u
\approx
\sum_{t=1}^{T} \alpha_{u,t} h_t^{\text{enc}},
\qquad
\sum_{t=1}^{T} \alpha_{u,t} = 1.
$$

The conceptual change is decisive:

- RNNs transport information through repeated state updates;
- attention creates direct, content-dependent access paths to earlier positions.

That shortens dependency paths and weakens the memory bottleneck that plagues recurrent models.

## 12. Structural insertion points

### 12.1 Category theory insertion point

A sequence model can be viewed concretely as repeated composition of a state-transition morphism

$$
f : X \times H \to H
$$

and an output morphism

$$
g : H \to Y.
$$

The categorical value here is modest but real:
it highlights that recurrence is composition with shared structure, while seq2seq models compose an encoder and decoder through an intermediate state object.
This viewpoint clarifies architecture composition, but it does not replace the linear algebra or optimization analysis.

### 12.2 Unity Theory insertion point

No Unity Theory vocabulary is needed to understand this module.
If a companion note is later added, the appropriate role would be interpretive only:
sequence models as systems that compress temporal experience into state under resource constraints.

## 13. Summary

- Markov models use a fixed explicit context; RNNs replace that with a learned hidden state.
- BPTT is just backpropagation on an unrolled, tied-parameter computation graph.
- Repeated Jacobian products cause vanishing and exploding gradients.
- LSTMs and GRUs introduce gates and additive memory paths to stabilize temporal credit assignment.
- Teacher forcing improves training but creates exposure bias at inference time.
- Beam search partially corrects greedy decoding's myopia.
- The fixed-state bottleneck of recurrent seq2seq models motivates attention and then transformers.
