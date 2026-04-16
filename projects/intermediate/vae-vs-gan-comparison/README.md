# Generative Model Comparison: VAE vs GAN

## Project overview

This project asks students to compare a variational autoencoder and a generative adversarial network on the same image-generation task. The goal is to analyze the tradeoff between likelihood-oriented latent modeling and adversarial sample realism in a setting small enough to run responsibly.

Students must make and justify decisions about:

- dataset and image resolution;
- latent dimension;
- encoder, decoder, generator, and discriminator capacity;
- optimization and stabilization choices; and
- evaluation criteria for both reconstruction and generation quality.

The final artifact should explain not only which model looked better, but which objective each model was actually optimizing and how that shaped behavior.

## Recommended dataset

Use one lightweight image dataset such as:

- MNIST or Fashion-MNIST;
- Kuzushiji-MNIST;
- a grayscale or low-resolution subset of CIFAR-10; or
- another small image dataset with a justified preprocessing pipeline.

Avoid datasets that require large-resolution image synthesis or multi-GPU training.

## Prerequisites

Students should be comfortable with:

- deep-learning optimization and diagnostics from Module 07;
- latent-variable reasoning and ELBO ideas from Module 11;
- adversarial training basics and instability issues from Module 11; and
- standard image-evaluation practices for small-scale experiments.

## Learning objectives

By the end of the project, students should be able to:

1. implement or adapt a VAE and a GAN for a shared image domain;
2. explain the difference between reconstruction-oriented and adversarial objectives;
3. compare latent structure, sample quality, and training stability;
4. justify stabilization choices such as regularization, normalization, or training-ratio changes; and
5. write a nuanced analysis of what the comparison does and does not show.

## Estimated completion time

`14-18 hours`

## Compute guidance

- Standard path: single GPU preferred.
- Reduced-scale path: grayscale images, lower resolution, smaller latent dimension, fewer epochs, and simpler convolutional blocks.

Reduced-scale work is acceptable if the report explains the consequences for sample fidelity and generality.

## Required deliverables

Submit:

- a reproducible notebook or script pipeline;
- generated-sample figures for both models;
- reconstruction examples for the VAE;
- training diagnostics or stability notes for the GAN;
- a short report embedded in the notebook or as markdown.

Instructors can grade the submission with [`rubric.md`](./rubric.md).

## Required tasks

### 1. Define the comparison setup

State and justify:

- dataset choice;
- image preprocessing;
- latent dimension;
- architecture family; and
- evaluation strategy.

### 2. Train a VAE

Document:

- encoder and decoder design;
- reconstruction loss choice;
- KL weighting if modified; and
- evidence about latent-space organization or reconstruction behavior.

### 3. Train a GAN

Document:

- generator and discriminator design;
- adversarial objective used;
- stabilization choices such as label smoothing, normalization, or update-ratio control; and
- any signs of instability or mode collapse.

### 4. Compare outputs and objectives

At minimum, compare:

- reconstruction quality;
- sample diversity;
- sample realism at the chosen resolution;
- training stability; and
- which claims can be supported by the selected metrics and visual evidence.

### 5. Write the analysis

Address the following prompts:

1. Which model gave the most useful latent representation, and why?
2. Which model produced the most convincing samples, and under what standard?
3. What instabilities appeared during GAN training?
4. What limitations make this comparison incomplete or scale-dependent?
5. In what application would you prefer the VAE over the GAN, or vice versa?

## Suggested workflow

1. Start with the VAE to anchor the dataset and architecture pipeline.
2. Use simple GAN baselines before attempting extra stabilizers.
3. Keep visual comparisons on the same sample grid and scale.
4. Avoid strong claims about generative quality without discussing the metric limitations.

## Expected submission quality

A strong submission:

- reflects a correct understanding of ELBO-based versus adversarial training;
- treats generative evaluation carefully and modestly;
- documents instability rather than hiding it; and
- gives a balanced written comparison of objectives, outputs, and limitations.
