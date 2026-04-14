---
title: "Training Deep Networks"
module: "07-deep-learning-systems"
lesson: "training-deep-networks"
doc_type: "notes"
topic: "normalization-regularization-residuals"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
  - "00-math-toolkit/multivariable-calculus"
  - "01-optimization/convexity-and-optimization"
  - "01-optimization/OPT-02-stochastic-and-momentum"
  - "06-neural-networks/neural-networks-first-principles"
  - "06-neural-networks/initialization-and-normalization"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "deep-learning"
  - "batch-normalization"
  - "layer-normalization"
  - "dropout"
  - "weight-decay"
  - "residual-connections"
  - "gradient-clipping"
  - "learning-rate-schedules"
---

## Purpose

These notes study the engineering choices that make deep networks train reliably in practice.
The central theme is that depth magnifies small numerical and statistical problems:
activation scales drift, gradients become poorly conditioned, optimization becomes brittle, and generalization can degrade even when training loss falls.

The goal is not to memorize isolated tricks.
The goal is to see how normalization, regularization, initialization, residual design, and schedule choices jointly shape gradient flow and optimization geometry.

## Learning objectives

After working through this note, you should be able to:

- derive the forward and backward structure of batch normalization and layer normalization;
- explain when batch normalization and layer normalization differ in behavior and why;
- interpret dropout both as stochastic regularization and as approximate model averaging;
- distinguish explicit weight decay from optimizer-side implementation details;
- motivate residual connections through gradient transport and effective conditioning;
- describe when learning-rate schedules and gradient clipping stabilize training; and
- state brief hardware and scaling considerations without confusing them for the core mathematical ideas.

## 1. Why deep networks are hard to train

For a depth-$L$ feedforward network,

$$
h^{(\ell)} = \phi\!\left(W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}\right),
\qquad \ell = 1,\dots,L.
$$

The backward pass contains repeated products of Jacobians:

$$
\frac{\partial \mathcal{L}}{\partial h^{(0)}} =
\frac{\partial \mathcal{L}}{\partial h^{(L)}}
\prod_{\ell=L}^{1}
\frac{\partial h^{(\ell)}}{\partial h^{(\ell-1)}}.
$$

If the operator norms of these Jacobians are typically below $1$, gradients tend to vanish.
If they are typically above $1$, gradients tend to explode.
Even when norms are acceptable on average, correlations, saturation, and batch-level noise can make optimization erratic.

In practice, we therefore manage several coupled objects:

- the scale of activations;
- the scale of gradients;
- the noise level of stochastic updates;
- the conditioning of the optimization problem; and
- the degree of implicit or explicit regularization.

## 2. Batch normalization

### 2.1 Forward definition

For one scalar feature over a mini-batch $\{x_1,\dots,x_m\}$, define

$$
\mu_B = \frac{1}{m}\sum_{i=1}^m x_i,
\qquad
\sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i-\mu_B)^2.
$$

Batch normalization standardizes and then reparameterizes:

$$
\hat{x}_i = \frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\varepsilon}},
\qquad
y_i = \gamma \hat{x}_i + \beta.
$$

For vector activations, this is applied featurewise across the mini-batch.
In convolutional networks, the average is usually taken across batch and spatial locations per channel.

### 2.2 Why it helps

Batch normalization does not merely center and scale activations.
It changes the parameterization seen by the optimizer.
When intermediate coordinates stay in a predictable range, gradients are often less sensitive to scale mismatches and deeper models become easier to optimize.

Common practical effects are:

- faster early training;
- reduced sensitivity to initialization;
- milder gradient pathologies;
- some regularization through batch-statistic noise.

The phrase "internal covariate shift" is historically associated with batch normalization, but modern interpretations emphasize smoother optimization and better conditioning more than a literal distribution-shift explanation.

### 2.3 Backward structure

Let $g_i = \frac{\partial \mathcal{L}}{\partial y_i}$.
Then

$$
\frac{\partial \mathcal{L}}{\partial \beta} = \sum_{i=1}^m g_i,
\qquad
\frac{\partial \mathcal{L}}{\partial \gamma} = \sum_{i=1}^m g_i \hat{x}_i.
$$

The input gradient couples all examples in the batch:

$$
\frac{\partial \mathcal{L}}{\partial x_i}
=
\frac{\gamma}{m\sqrt{\sigma_B^2+\varepsilon}}
\left[
m g_i
- \sum_{j=1}^m g_j
- \hat{x}_i \sum_{j=1}^m g_j \hat{x}_j
\right].
$$

This coupling matters.
One sample's gradient depends on the others through the batch statistics.
That is one reason batch normalization behaves differently for very small batches and why inference requires a separate treatment.

### 2.4 Training versus inference

During training, batch statistics are computed from the current mini-batch.
During inference, we typically use running estimates of the mean and variance:

$$
\mu_{\mathrm{run}} \leftarrow \rho \mu_{\mathrm{run}} + (1-\rho)\mu_B,
\qquad
\sigma^2_{\mathrm{run}} \leftarrow \rho \sigma^2_{\mathrm{run}} + (1-\rho)\sigma_B^2.
$$

This train-test mismatch is usually acceptable for moderate batch sizes, but it becomes fragile when batches are tiny or highly nonstationary.

## 3. Layer normalization

### 3.1 Forward definition

Batch normalization normalizes across examples.
Layer normalization normalizes within a single example.

For one hidden vector $x \in \mathbb{R}^d$,

$$
\mu(x) = \frac{1}{d}\sum_{j=1}^d x_j,
\qquad
\sigma^2(x) = \frac{1}{d}\sum_{j=1}^d (x_j-\mu(x))^2,
$$

and

$$
\hat{x}_j = \frac{x_j-\mu(x)}{\sqrt{\sigma^2(x)+\varepsilon}},
\qquad
y_j = \gamma_j \hat{x}_j + \beta_j.
$$

The learnable parameters are per feature, but the statistics are computed per sample.

### 3.2 Consequences

Because layer normalization does not depend on other examples in the batch, it:

- behaves the same at training and inference time;
- works well with batch size $1$;
- is natural for sequence models and transformers, where tokenwise or samplewise normalization is convenient.

However, it does not use batch-level noise as implicit regularization and does not stabilize convolutional training in quite the same way batch normalization often does.

### 3.3 Comparison with batch normalization

The critical distinction is the axis of normalization.

| Method | Statistics computed over | Training/inference mismatch | Works well with tiny batches | Common use |
| --- | --- | --- | --- | --- |
| Batch norm | examples in a batch, featurewise | yes | often no | CNNs, medium-to-large batch MLPs |
| Layer norm | features in one example | no | yes | RNNs, transformers, small-batch settings |

Empirically, if you shrink the batch size enough, batch normalization becomes noisy and sometimes unstable, while layer normalization changes very little.

## 4. Regularization

### 4.1 Weight decay

For parameters $\theta$, $L_2$-regularized training minimizes

$$
\mathcal{J}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2}\|\theta\|_2^2.
$$

Then

$$
\nabla_\theta \mathcal{J}(\theta) = \nabla_\theta \mathcal{L}(\theta) + \lambda \theta.
$$

Under vanilla gradient descent with step size $\eta$,

$$
\theta_{t+1}
=
\theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t) - \eta \lambda \theta_t
=
(1-\eta\lambda)\theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t).
$$

This multiplicative shrinkage explains the name weight decay.
For adaptive optimizers, "adding $\lambda \theta$ to the gradient" and "decoupled weight decay" are not exactly the same update, so one should distinguish optimizer implementation from the underlying regularization idea.

Weight decay penalizes large weights, usually improving generalization and sometimes helping optimization by discouraging unnecessarily sharp parameter growth.

### 4.2 Dropout

Let $h \in \mathbb{R}^d$ be a hidden activation vector.
Dropout samples a mask $m_j \sim \mathrm{Bernoulli}(q)$ and forms

$$
\tilde{h}_j = \frac{m_j}{q} h_j.
$$

This is inverted dropout.
The factor $1/q$ keeps the activation expectation fixed:

$$
\mathbb{E}[\tilde{h}_j] = h_j.
$$

At training time, units are randomly removed.
At test time, the full network is used with no mask because the scaling was already applied during training.

Two complementary viewpoints are useful:

- **Regularization viewpoint.** The network cannot rely on any one hidden unit always being present, so it learns more distributed and robust representations.
- **Ensemble viewpoint.** Each dropout mask corresponds to a thinned subnetwork. Training shares parameters across many such subnetworks, and inference approximates averaging over that ensemble.

Dropout is often strongest in fully connected layers and less essential in architectures that already have strong regularizers such as heavy data augmentation, batch normalization, or residual design.

## 5. Residual connections

### 5.1 From direct mapping to residual mapping

A residual block writes

$$
h_{\ell+1} = h_\ell + F_\ell(h_\ell; \theta_\ell),
$$

instead of

$$
h_{\ell+1} = F_\ell(h_\ell; \theta_\ell).
$$

If the desired transformation is close to the identity, the residual form only needs to learn the correction.
That already reduces representational burden.

### 5.2 Gradient-flow motivation

Differentiate the residual block:

$$
\frac{\partial h_{\ell+1}}{\partial h_\ell}
=
I + \frac{\partial F_\ell}{\partial h_\ell}.
$$

For a deep residual stack,

$$
\frac{\partial \mathcal{L}}{\partial h_\ell}
=
\frac{\partial \mathcal{L}}{\partial h_L}
\prod_{k=\ell}^{L-1}
\left(I + \frac{\partial F_k}{\partial h_k}\right).
$$

The identity term gives gradients a direct transport path through depth.
This does not magically eliminate all conditioning problems, but it reduces the need for every layerwise Jacobian to preserve signal perfectly on its own.

This is the main optimization reason residual networks scale so much better than plain deep stacks.

### 5.3 Interaction with normalization

Residual blocks are often paired with normalization.
In pre-norm transformers, for example, normalization is applied before the residual branch:

$$
h_{\ell+1} = h_\ell + F_\ell(\mathrm{LN}(h_\ell)).
$$

This improves gradient flow through very deep stacks because the skip path itself remains clean.

## 6. Learning-rate schedules

The learning rate controls how far the optimizer moves in parameter space per step.
If it is too large, optimization oscillates or diverges.
If it is too small, training stalls long before a useful solution is reached.

Modern training usually benefits from a schedule rather than a constant step size.
Common patterns are:

- step decay;
- cosine decay;
- linear warmup followed by decay;
- one-cycle schedules.

The main idea is simple:
large steps help move rapidly early in training, while smaller steps later help refine a solution without bouncing around narrow directions.

Warmup is especially useful when normalization, adaptive optimization, or large-batch training make the first few updates unusually sensitive.

## 7. Gradient clipping

Gradient clipping is a safeguard against rare but destructive updates.
The most common form clips the global gradient norm:

$$
g \leftarrow
\begin{cases}
g & \|g\|_2 \le \tau, \\
\tau \dfrac{g}{\|g\|_2} & \|g\|_2 > \tau.
\end{cases}
$$

This leaves direction unchanged while limiting magnitude.
Clipping is particularly useful in recurrent models, unstable deep stacks, or any training regime with occasional gradient spikes.

Clipping is not a substitute for correct modeling choices.
If clipping activates constantly, the underlying problem is often poor architecture, bad initialization, or an overly aggressive learning rate.

## 8. Putting the pieces together

A practical deep-network training recipe often looks like this:

1. choose an architecture with stable signal paths, often including residual structure;
2. initialize weights with a variance rule matched to the activation;
3. use normalization appropriate to the architecture and batch regime;
4. choose an optimizer and learning-rate schedule that match the noise scale of training;
5. add regularization such as weight decay, dropout, and data augmentation as needed;
6. monitor losses, gradient norms, activation statistics, and train-validation gaps.

These techniques are not independent.
For example, heavy dropout may be unnecessary when strong augmentation and weight decay are already present, and batch normalization can change the effective scale seen by the optimizer enough to justify larger learning rates.

## 9. Brief scaling and hardware considerations

This module is not a systems-engineering course, but a few practical constraints matter:

- larger batches improve hardware throughput but can reduce gradient noise and change optimization behavior;
- mixed precision improves speed and memory use but introduces numerical considerations such as loss scaling;
- activation checkpointing trades compute for memory;
- distributed training changes synchronization costs and can make batch-normalization statistics awkward across devices.

The mathematical lesson is that scaling decisions affect optimization through batch statistics, noise scale, and numeric precision.
They are implementation details, but not irrelevant ones.

## 10. Worked comparison: batch norm versus layer norm

Consider hidden activations $H \in \mathbb{R}^{m \times d}$.
Batch normalization computes statistics down each column:

$$
\mu_j^{\mathrm{BN}} = \frac{1}{m}\sum_{i=1}^m H_{ij},
\qquad
(\sigma_j^2)^{\mathrm{BN}} = \frac{1}{m}\sum_{i=1}^m (H_{ij} - \mu_j^{\mathrm{BN}})^2.
$$

Layer normalization computes statistics across each row:

$$
\mu_i^{\mathrm{LN}} = \frac{1}{d}\sum_{j=1}^d H_{ij},
\qquad
(\sigma_i^2)^{\mathrm{LN}} = \frac{1}{d}\sum_{j=1}^d (H_{ij} - \mu_i^{\mathrm{LN}})^2.
$$

So BN makes each feature comparable across examples, while LN makes each example internally standardized across features.
That axis choice predicts the empirical behavior:

- BN often excels when mini-batches are informative and stable;
- LN is more robust when sequence length varies, batch size is tiny, or inference behavior must match training exactly.

## 11. Common failure patterns

When training a deep model, diagnose these patterns early:

- **vanishing gradients:** early layers learn slowly, gradient norms decay with depth, training plateaus;
- **exploding gradients:** loss spikes, parameter norms jump, updates become unstable;
- **normalization mismatch:** batch-norm performance changes sharply with batch size or between training and evaluation;
- **overfitting:** training loss falls while validation loss rises;
- **schedule mismatch:** loss decreases initially but later oscillates because the learning rate stays too large.

The notebooks in this module make these pathologies concrete with small PyTorch experiments.

## Category theory insertion point

The structural lesson is compositional:
deep models are long morphism chains, and training stability depends on whether composition preserves informative signal.
Normalization and residual maps can be interpreted as local reparameterizations that improve the behavior of that composition without changing the global task.

## Unity Theory insertion point

Any Unity-oriented discussion here should remain companion material.
If one wants to treat normalization or residual structure as preserving coherent information flow across scales, that claim should be presented as interpretation, not as a replacement for the standard optimization account.

## Summary

Training deep networks is fundamentally about keeping optimization in a regime where signal survives depth and stochastic updates remain informative.
Batch normalization and layer normalization stabilize internal representations in different ways.
Dropout and weight decay regularize different aspects of model capacity.
Residual connections provide direct signal paths that substantially ease deep optimization.
Learning-rate schedules and gradient clipping then manage the dynamics of the optimizer itself.
