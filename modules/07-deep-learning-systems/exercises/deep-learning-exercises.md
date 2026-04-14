---
title: "Deep Learning Systems Exercises"
module: "07-deep-learning-systems"
lesson: "deep-learning-exercises"
doc_type: "exercise"
topic: "normalization-regularization-residual-training"
status: "draft"
prerequisites:
  - "01-optimization/convexity-and-optimization"
  - "01-optimization/OPT-02-stochastic-and-momentum"
  - "06-neural-networks/neural-networks-first-principles"
  - "06-neural-networks/initialization-and-normalization"
  - "07-deep-learning-systems/training-deep-networks"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "deep-learning"
  - "normalization"
  - "dropout"
  - "weight-decay"
  - "residual-connections"
  - "gradient-clipping"
---

## Purpose

These exercises reinforce the mathematical and practical ideas behind stable deep-network training.
They combine derivation, interpretation, and design judgment.

## Exercise 1: batch normalization moments

Let $x_1,\dots,x_m$ be a batch of scalar activations, and define

$$
\hat{x}_i = \frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\varepsilon}},
\qquad
\mu_B = \frac{1}{m}\sum_{i=1}^m x_i,
\qquad
\sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i-\mu_B)^2.
$$

1. Show that $\frac{1}{m}\sum_{i=1}^m \hat{x}_i = 0$.
2. Show that $\frac{1}{m}\sum_{i=1}^m \hat{x}_i^2 = \frac{\sigma_B^2}{\sigma_B^2+\varepsilon}$.
3. Explain why the variance is only approximately $1$ when $\varepsilon > 0$.
4. State the role of the learnable parameters $\gamma$ and $\beta$.

## Exercise 2: layer norm versus batch norm

Let $H \in \mathbb{R}^{m \times d}$ be a hidden-activation matrix.

1. Write the mean and variance formulas used by batch normalization.
2. Write the mean and variance formulas used by layer normalization.
3. For each method, identify which entries of $H$ influence the normalization of one coordinate $H_{ij}$.
4. Explain why layer normalization is insensitive to batch size.
5. Explain why batch normalization behaves differently during training and inference.

## Exercise 3: backward-pass structure for batch norm

Suppose

$$
y_i = \gamma \hat{x}_i + \beta
$$

with upstream gradients $g_i = \frac{\partial \mathcal{L}}{\partial y_i}$.

1. Derive $\frac{\partial \mathcal{L}}{\partial \beta}$.
2. Derive $\frac{\partial \mathcal{L}}{\partial \gamma}$.
3. Explain in words why $\frac{\partial \mathcal{L}}{\partial x_i}$ depends on all examples in the batch rather than only on $x_i$.
4. Why does this coupling disappear in layer normalization across different training examples?

## Exercise 4: weight decay as shrinkage

Consider the regularized objective

$$
\mathcal{J}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2}\|\theta\|_2^2.
$$

1. Compute $\nabla_\theta \mathcal{J}(\theta)$.
2. Derive the gradient-descent update with step size $\eta$.
3. Show explicitly where multiplicative shrinkage appears.
4. Explain why this motivates the phrase "weight decay."
5. Briefly explain why decoupled weight decay differs from naive $L_2$ penalty inside Adam.

## Exercise 5: dropout expectation

Let $m_j \sim \mathrm{Bernoulli}(q)$ independently and define inverted dropout by

$$
\tilde{h}_j = \frac{m_j}{q} h_j.
$$

1. Show that $\mathbb{E}[\tilde{h}_j] = h_j$.
2. Compute $\operatorname{Var}(\tilde{h}_j)$ in terms of $h_j$ and $q$.
3. Explain why smaller $q$ increases stochasticity.
4. Give one argument for dropout as regularization.
5. Give one argument for dropout as approximate ensemble averaging.

## Exercise 6: residual Jacobian

Consider a residual block

$$
h_{\ell+1} = h_\ell + F_\ell(h_\ell).
$$

1. Compute $\frac{\partial h_{\ell+1}}{\partial h_\ell}$.
2. Compare this with the Jacobian of a plain block $h_{\ell+1} = F_\ell(h_\ell)$.
3. Explain why the identity term helps gradient transport.
4. Under what conditions could a residual network still suffer optimization instability?

## Exercise 7: schedule design

You are training a deep network and observe the following:

- loss drops rapidly for the first few epochs;
- validation accuracy improves early;
- later, training loss oscillates and validation accuracy stagnates.

1. Give two plausible learning-rate explanations.
2. Describe a schedule change that could address the issue.
3. Explain when warmup would be useful.
4. Explain why a constant tiny learning rate is not a satisfactory universal fix.

## Exercise 8: gradient clipping

Suppose a gradient vector $g$ has Euclidean norm $\|g\|_2 = 30$ and the clipping threshold is $\tau = 5$.

1. Write the clipped gradient explicitly in terms of $g$.
2. What is the norm of the clipped gradient?
3. Does gradient clipping change direction, magnitude, or both?
4. Give one training regime where clipping is especially useful.
5. Explain why frequent clipping is a warning sign rather than a complete solution.

## Exercise 9: training pathology diagnosis

A 20-layer plain MLP without normalization is trained on a synthetic classification task.
Training accuracy barely improves, and gradient norms in early layers are near zero.

1. Identify the likely pathology.
2. Name two interventions that directly target it.
3. Explain why simply training longer may not help.
4. State one diagnostic plot or statistic you would inspect in PyTorch.

## Exercise 10: system design comparison

You must choose one configuration for each case below.
In each answer, justify the choice in 3 to 5 sentences.

1. A CNN trained with batch size 256 on image data.
2. A transformer trained with variable-length sequences and small effective batch size.
3. A very deep feedforward network that shows gradient decay with depth.
4. A model that overfits despite stable training loss.

For each case, discuss at least two of:
normalization, residual structure, dropout, weight decay, learning-rate schedule, or clipping.
