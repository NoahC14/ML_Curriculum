---
title: "Pretraining and Scaling"
module: "10-transformers-llms"
lesson: "pretraining-and-scaling"
doc_type: "notes"
topic: "mlm-clm-scaling-laws-fine-tuning-instruction-tuning-rlhf"
status: "draft"
prerequisites:
  - "00-math-toolkit/information-theory"
  - "07-deep-learning-systems/training-deep-networks"
  - "09-sequence-models/sequence-modeling"
  - "10-transformers-llms/transformer-foundations"
updated: "2026-04-13"
owner: "curriculum-team"
tags:
  - "pretraining"
  - "masked-language-modeling"
  - "causal-language-modeling"
  - "scaling-laws"
  - "instruction-tuning"
  - "rlhf"
---

## Purpose

These notes explain how transformer architectures are trained into practical language models.
The focus is on the two dominant pretraining objectives, the empirical logic of scaling laws, and the post-training pipeline that turns a base model into an instruction-following assistant.

## Learning objectives

After working through this note, you should be able to:

- distinguish masked language modeling from causal language modeling;
- write the optimization target for each objective and explain it with concrete token examples;
- interpret scaling laws as empirical findings rather than first-principles theorems;
- describe the role of supervised fine-tuning and instruction tuning;
- explain the standard RLHF pipeline at a systems level; and
- state why modern LLM practice is concentrated around decoder-only causal pretraining plus post-training.

## 1. Why pretraining works

Large-scale language pretraining uses raw text as a supervision source.
The model is asked to predict missing or next tokens, so every document yields many training targets.

This makes pretraining attractive because it is:

- self-supervised, so labels are not required;
- statistically rich, since natural language contains syntax, semantics, world knowledge, and discourse structure;
- transferable, because the same representation can be adapted to many downstream tasks.

From an information-theoretic viewpoint, better token prediction means lower surprisal on the data distribution.
That is closely tied to lower negative log-likelihood and lower cross-entropy.

## 2. Masked language modeling

Masked language modeling, or MLM, corrupts an observed sequence and trains the model to reconstruct selected tokens from the surrounding context.
Let $x_{1:T}$ be the original token sequence and let $M \subseteq \{1,\dots,T\}$ be the set of masked positions.
Then the MLM objective is

$$
\mathcal{L}_{\mathrm{MLM}}(\theta)
=
- \sum_{t \in M} \log p_\theta(x_t \mid x_{\backslash M}),
$$

where $x_{\backslash M}$ denotes the partially observed sequence with masks or replacements inserted.

### 2.1 Concrete example

Take the sentence

$$
\text{``the cat sat on the mat''}.
$$

If we mask the third token, the model sees

$$
\text{``the cat [MASK] on the mat''}
$$

and should assign high probability to

$$
\text{``sat''}.
$$

Because the model can use both left and right context, MLM is naturally bidirectional.
That is why it fits encoder-style architectures well.

### 2.2 BERT-style corruption

BERT popularized a practical corruption scheme in which only some positions are selected, and selected tokens are not always replaced by the explicit mask token.
The exact corruption details matter in implementation, but the conceptual objective is still contextual reconstruction.

### 2.3 Strengths and limitations

Strengths:

- strong bidirectional representations;
- natural for classification and retrieval tasks;
- efficient reuse of every sentence as self-supervision.

Limitations:

- the model is not trained directly for left-to-right generation;
- the special masking pattern creates a train-test mismatch for generation tasks;
- decoder-style sampling is less natural than under a causal objective.

## 3. Causal language modeling

Causal language modeling, or CLM, uses the autoregressive factorization

$$
p_\theta(x_{1:T}) = \prod_{t=1}^T p_\theta(x_t \mid x_{<t}),
$$

so the negative log-likelihood is

$$
\mathcal{L}_{\mathrm{CLM}}(\theta)
=
- \sum_{t=1}^T \log p_\theta(x_t \mid x_{<t}).
$$

During training, the model receives the true prefix and predicts the next token.
This is teacher forcing in autoregressive form.

### 3.1 Concrete example

For the tokenized phrase

$$
(\text{``the''}, \text{``cat''}, \text{``sat''}, \text{``on''}, \text{``the''}, \text{``mat''}),
$$

the model learns distributions such as

$$
p_\theta(\text{``cat''} \mid \text{``the''}),
\qquad
p_\theta(\text{``sat''} \mid \text{``the cat''}),
\qquad
p_\theta(\text{``mat''} \mid \text{``the cat sat on the''}).
$$

At inference time, we can sample or decode one token at a time using exactly the same factorization.

### 3.2 Why CLM dominates modern LLMs

CLM aligns naturally with open-ended generation.
It also avoids a dedicated mask token and scales cleanly to very large corpora and decoder-only architectures.
That is why GPT-style systems use causal training.

## 4. Other denoising objectives

MLM and CLM are the two foundational objectives in this module, but they are not the only ones.
Modern encoder-decoder systems often use span corruption or text-to-text denoising.
The important curriculum point is that all of these are structured ways of turning raw text into predictive supervision.

## 5. Scaling laws

Scaling laws describe empirical regularities observed when model size, dataset size, and compute budget are increased systematically.
They are not derivable from standard statistical learning theory in any simple closed form.
They are measurements.

### 5.1 Basic intuition

When holding most other choices fixed, test loss often decreases approximately as a power law in:

- parameter count;
- number of training tokens; and
- optimization compute.

This does **not** mean unlimited scaling always helps.
It means that within broad regimes, the returns are smooth and predictable enough to guide engineering decisions.

### 5.2 Kaplan-style observation

Kaplan et al. reported that language-model loss follows approximate power-law trends over several orders of magnitude.
The main takeaway for curriculum purposes is:

- bigger models help;
- more data helps;
- more compute helps; and
- these tradeoffs can be studied quantitatively rather than by guesswork.

### 5.3 Chinchilla-style correction

Later work by Hoffmann et al. showed that many large models had been undertrained relative to their parameter count.
The updated lesson is that compute-optimal performance depends on balancing model size and dataset size rather than maximizing only one of them.

> **Warning.** Scaling laws are empirical summaries of observed training runs.
> They are useful for planning, but they are not universal physical laws and can shift when architectures, optimizers, data mixtures, or context lengths change.

## 6. Fine-tuning

After pretraining, a base model can be adapted to a downstream task by optimizing on task-specific examples.
If $(x,y)$ denotes an input-output pair, fine-tuning usually minimizes

$$
- \log p_\theta(y \mid x)
$$

or a closely related sequence loss.

Fine-tuning can be:

- full-parameter, where all parameters are updated; or
- parameter-efficient, where only a small subset or adapter mechanism is trained.

The curriculum-level point is that pretraining builds a broad prior over language, while fine-tuning specializes that prior.

## 7. Instruction tuning

Instruction tuning is supervised fine-tuning on prompt-response pairs designed to teach the model how to follow task descriptions.
Examples include:

- summarization prompts;
- question-answering prompts;
- classification in natural-language form;
- multi-turn assistant-style conversations.

A base CLM model predicts the next token well, but that alone does not guarantee it will answer in the format a user wants.
Instruction tuning reshapes behavior toward helpful task completion.

## 8. RLHF overview

RLHF stands for reinforcement learning from human feedback.
The classic pipeline has three stages:

1. pretrain a base language model;
2. supervised fine-tune it on demonstrations;
3. collect preference comparisons and use them to optimize behavior.

### 8.1 Reward modeling

Human annotators compare candidate outputs for the same prompt.
A reward model is trained to assign higher scores to preferred responses.

### 8.2 Policy optimization

The language model is then optimized to produce outputs that score well under the reward model while staying close to the supervised model.
Historically, PPO has been a common choice for this stage.

### 8.3 Why RLHF is only an overview here

The exact post-training stack changes quickly.
Many production systems now mix or replace PPO-style RLHF with direct preference optimization, rejection sampling, constitutional methods, or other preference-learning pipelines.
So the stable foundation is the framework:

- demonstrations shape initial behavior;
- preferences provide a richer signal than next-token likelihood alone;
- optimization must manage reward hacking, distribution shift, and alignment tradeoffs.

## 9. Modern LLM recipe at a glance

A simplified modern pipeline is:

1. tokenize a very large text corpus;
2. pretrain a transformer, usually decoder-only, with CLM;
3. optionally continue training on curated domain data;
4. instruction tune on prompt-response demonstrations;
5. apply preference-based post-training such as RLHF or a close variant;
6. evaluate for capability, robustness, and safety.

Encoder-only and encoder-decoder transformers remain important, but the dominant foundation for chat-oriented LLMs is large-scale causal pretraining plus post-training.

## 10. Computational and statistical cautions

- Better pretraining loss does not guarantee safe or reliable downstream behavior.
- Scaling improves average performance but can also amplify memorization, bias, and misuse risk.
- RLHF improves preference alignment only relative to the annotator distribution and reward design.
- Evaluation must therefore include capability benchmarks, calibration checks, and safety-oriented audits.

## References

- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Kaplan, J. et al. (2020). *Scaling Laws for Neural Language Models*. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- Brown, T. B. et al. (2020). *Language Models are Few-Shot Learners*. NeurIPS. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- Hoffmann, J. et al. (2022). *Training Compute-Optimal Large Language Models*. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback*. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
