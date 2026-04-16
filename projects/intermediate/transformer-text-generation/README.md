# Transformer Text Generation

## Project overview

This project asks students to train a small autoregressive transformer and analyze how model design and training choices affect text generation quality, optimization behavior, and computational cost.

The project should remain pedagogical rather than industrial. The goal is to understand transformer mechanics in a reproducible small-scale setting, not to imitate large-scale pretraining.

Students must make and justify decisions about:

- tokenization granularity or vocabulary setup;
- context length;
- transformer depth and width;
- learning-rate schedule and warmup; and
- sampling strategy at generation time.

## Recommended dataset

Use one compact corpus such as:

- Tiny Shakespeare or another public-domain dramatic text;
- a small domain-specific text collection with consistent style; or
- a cleaned subset of a larger corpus with explicit documentation of the sampling procedure.

The dataset should support repeated training runs within the compute budget.

## Prerequisites

Students should be comfortable with:

- optimization and diagnostics ideas from Module 07;
- sequence modeling from Module 09;
- self-attention, positional encoding, and pretraining objectives from Module 10; and
- cross-entropy evaluation and basic sampling heuristics.

## Learning objectives

By the end of the project, students should be able to:

1. implement or adapt a small transformer text-generation pipeline;
2. justify transformer hyperparameters in terms of context, capacity, and training stability;
3. use diagnostics to interpret loss trajectories and schedule behavior;
4. evaluate generations with both quantitative and qualitative criteria; and
5. explain tradeoffs between model quality, compute cost, and controllability.

## Estimated completion time

`14-20 hours`

## Compute guidance

- Standard path: single GPU preferred.
- Reduced-scale path: use character-level tokenization, `2-4` layers, smaller embedding size, shorter contexts, fewer training steps, and a smaller corpus.

Reduced-scale work is acceptable if the student states the reduced configuration and avoids overclaiming from the resulting generations.

## Required deliverables

Submit:

- a reproducible notebook or script pipeline;
- sample generations under at least two decoding settings;
- training and schedule diagnostics;
- a short report embedded in the notebook or as markdown; and
- a concise compute-and-limitations note.

Instructors can grade the submission with [`rubric.md`](./rubric.md).

## Required tasks

### 1. Define the language-modeling setup

State and justify:

- corpus choice;
- tokenization scheme;
- context window;
- train/validation/test split; and
- model size target.

### 2. Train a baseline transformer

The baseline must specify:

- number of layers and attention heads;
- embedding dimension;
- optimizer and weight decay;
- learning-rate schedule, including warmup if used; and
- stopping criterion.

### 3. Compare one meaningful design variation

Choose one focused comparison such as:

- shorter versus longer context;
- no warmup versus warmup;
- smaller versus larger model;
- different tokenization choices; or
- greedy decoding versus temperature/top-k sampling.

### 4. Run diagnostics

Use at least two diagnostic views, such as:

- training and validation loss curves;
- learning-rate schedule plots;
- gradient-norm monitoring; or
- token-level error concentration by position.

### 5. Evaluate generation quality

Include:

- held-out loss or perplexity;
- qualitative generation samples;
- a brief analysis of repetition, coherence, or degeneration; and
- a discussion of how the chosen decoding strategy changes outputs.

### 6. Write the analysis

Address the following prompts:

1. Which design decision most affected model quality?
2. Which design decision most affected training stability or efficiency?
3. What failure patterns appear in the generated text?
4. How well do the diagnostics explain those failures?
5. What would be the next scaling step if compute increased modestly?

## Suggested workflow

1. Verify the pipeline on a very small corpus slice before longer runs.
2. Keep the main comparison narrow enough that compute does not dominate the schedule.
3. Save small checkpoints or logs so the analysis can cite actual evidence.
4. Treat generated text as evidence to interpret, not as a marketing sample.

## Expected submission quality

A strong submission:

- demonstrates clear understanding of the small-transformer training loop;
- justifies architecture and optimization choices with concrete tradeoffs;
- uses both metrics and generations to evaluate model quality; and
- writes carefully about what small-scale text generation can and cannot show.
