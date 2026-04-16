# Sequence Prediction with Diagnostics Analysis

## Project overview

This project asks students to build a sequence prediction model and use diagnostics to explain how training dynamics relate to model quality. The main emphasis is on understanding sequential optimization behavior, not merely obtaining the lowest perplexity or loss.

Students should compare at least two sequence-model configurations such as:

- vanilla RNN versus LSTM;
- LSTM without clipping versus LSTM with clipping;
- short-context versus longer-context training; or
- recurrent model versus a lightweight attention-augmented variant.

The analysis should connect observed behavior to the pathologies discussed in Modules 07 and 09.

## Recommended dataset

Use one lightweight sequence dataset such as:

- character-level text from a public-domain corpus;
- a small word-level language-modeling corpus;
- synthetic sequence prediction data with controlled memory length; or
- a univariate or low-dimensional time-series forecasting task with clear sequential structure.

Choose a dataset that allows repeated experiments within the time budget.

## Prerequisites

Students should be comfortable with:

- training stability concepts from Module 07;
- recurrent modeling, teacher forcing, and sequence pathologies from Module 09; and
- shared diagnostics for loss, gradients, and confusion-style analyses where applicable.

## Learning objectives

By the end of the project, students should be able to:

1. implement and train a sequence model under a controlled experimental setup;
2. diagnose vanishing or exploding behavior using gradient-focused evidence;
3. explain how architectural and optimization choices affect memory and stability;
4. evaluate sequence predictions with task-appropriate metrics; and
5. write a technical argument connecting theory, diagnostics, and observed errors.

## Estimated completion time

`10-14 hours`

## Compute guidance

- Standard path: CPU or single GPU.
- Reduced-scale path: shorten sequence length, use smaller hidden states, train on a smaller corpus, and limit the number of comparison runs.

Reduced-scale submissions remain acceptable if the report explains how the reduced setting affects external validity.

## Required deliverables

Submit:

- a reproducible notebook or script pipeline;
- a short report embedded in the notebook or as markdown;
- gradient or training diagnostics plots;
- a table of model settings and results; and
- a short failure-analysis section.

Instructors can grade the submission with [`rubric.md`](./rubric.md).

## Required tasks

### 1. Define the prediction task

State:

- what is being predicted;
- what the sequence length and target horizon are; and
- which failure modes you expect to matter.

### 2. Train at least two model variants

Compare at least two variants that differ in architecture or stabilization strategy. At minimum, justify:

- hidden size or model dimension;
- sequence length or context window;
- optimizer and learning rate;
- clipping or regularization choices; and
- training budget.

### 3. Run diagnostics during training

Use the shared diagnostics toolkit or equivalent custom analysis to inspect:

- loss curves;
- gradient norms over time; and
- at least one additional signal such as activation statistics, error concentration by position, or prediction entropy.

### 4. Evaluate prediction quality

Use task-appropriate metrics such as:

- perplexity or cross-entropy for language modeling;
- sequence accuracy or token accuracy;
- mean squared error for forecasting; or
- qualitative examples of successes and failures.

### 5. Write the analysis

Address the following prompts:

1. Which model or stabilization strategy gave the best tradeoff between performance and training stability?
2. Did the diagnostics reveal vanishing, exploding, or other optimization pathologies?
3. What kinds of sequence dependencies were hardest for the model?
4. Which design choice would you change first if given more compute?
5. How do your observations connect back to the theory of recurrent or sequential composition?

## Suggested workflow

1. Start on a tiny subset and confirm the training loop behaves sensibly.
2. Track gradients early rather than only after failure.
3. Keep the comparison narrow enough that design conclusions remain interpretable.
4. Use a few qualitative examples to complement aggregate metrics.

## Expected submission quality

A strong submission:

- isolates a genuine sequence-model design question;
- uses diagnostics to explain training behavior;
- distinguishes optimization failure from modeling insufficiency; and
- writes clearly about what the model can and cannot remember.
