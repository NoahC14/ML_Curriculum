# CNN Architecture Comparison

## Project overview

This project asks students to compare two convolutional image-classification pipelines on the same vision task and explain how architectural and training choices affect accuracy, optimization, and error patterns.

The core requirement is not just to train two models. Students must make and justify design decisions about:

- baseline architecture choice;
- normalization and regularization;
- optimization schedule;
- data augmentation; and
- evaluation criteria beyond top-1 accuracy.

The finished artifact should read like a compact model-comparison report grounded in Modules 07-08.

## Recommended dataset

Use one small-to-medium vision dataset such as:

- CIFAR-10;
- Fashion-MNIST with a deliberately stronger CNN comparison setup; or
- a curated subset of `torchvision` image folders with at least `5` classes.

The dataset should be large enough to make regularization and architecture choices matter, but small enough to finish within the compute budget.

## Prerequisites

Students should be comfortable with:

- backpropagation and optimization from Module 06;
- normalization, regularization, residual thinking, and diagnostics from Module 07;
- convolution, pooling, transfer learning, and vision evaluation from Module 08; and
- confusion matrices and metric reporting from the shared evaluation toolkit.

## Learning objectives

By the end of the project, students should be able to:

1. implement and train two CNN-based classifiers under a controlled comparison;
2. justify architectural differences in terms of receptive fields, depth, normalization, and parameter count;
3. use training diagnostics to analyze convergence and failure modes;
4. compare models with both scalar metrics and class-level error analysis; and
5. write a short evidence-based argument about which design choices mattered most.

## Estimated completion time

`12-16 hours`

## Compute guidance

- Standard path: single GPU, roughly `4-8 GB` VRAM, moderate batch sizes.
- Reduced-scale path: use a smaller data subset, downsample images if justified, cap training at fewer epochs, and compare shallower models.

Reduced-scale submissions are acceptable if the report explicitly states what was reduced and how that limits the conclusions.

## Required deliverables

Submit:

- a reproducible notebook or script pipeline;
- a short report embedded in the notebook or as markdown;
- at least `3` figures or tables, including one diagnostic plot; and
- brief appendix notes describing hyperparameter choices.

Instructors can grade the submission with [`rubric.md`](./rubric.md).

## Required tasks

### 1. Frame the comparison

Choose and justify a pair such as:

- plain CNN versus residual CNN;
- shallow CNN versus deeper CNN with normalization;
- scratch-trained CNN versus transfer-learning classifier head.

State what you expect the comparison to reveal before training.

### 2. Build a reproducible training pipeline

Your pipeline must:

- define train, validation, and test splits;
- report dataset size, class balance, and image shape;
- keep preprocessing and augmentation explicit;
- fix random seeds where practical; and
- record model parameter counts.

### 3. Train and tune both models

For each model, document and justify:

- optimizer choice;
- learning-rate schedule;
- normalization strategy;
- regularization strategy; and
- stopping criterion or epoch budget.

### 4. Run diagnostics

Use at least two diagnostics from `shared/src/training_diagnostics.py` or equivalent analysis:

- loss curves;
- gradient-norm traces;
- activation-distribution summaries; or
- confusion diagnostics.

Explain what the diagnostics suggest about optimization quality, not just final performance.

### 5. Compare performance and behavior

Report:

- validation and test metrics;
- confusion matrix or per-class breakdown;
- runtime or training-efficiency comparison; and
- at least one small ablation or controlled modification.

### 6. Write the analysis

Address the following prompts:

1. Which architectural change produced the largest practical benefit?
2. Which optimization or regularization choice mattered most?
3. Did the diagnostics support your explanation of model behavior?
4. Which classes or image types remained difficult, and why?
5. Under what constraints would you choose the weaker model anyway?

## Suggested workflow

1. Start with a minimal baseline that trains end to end.
2. Freeze the dataset split and evaluation code early.
3. Change one major design variable at a time.
4. Use diagnostics before adding more tuning.
5. Keep the final writeup focused on evidence, not benchmark theater.

## Expected submission quality

A strong submission:

- compares two genuinely distinct CNN design choices;
- uses diagnostics to support claims about optimization;
- keeps the evaluation protocol controlled and fair; and
- ties architectural behavior back to the theory of convolution, normalization, and depth.
