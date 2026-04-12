---
title: "Diffusion Models"
module: "11-generative-models"
lesson: "diffusion-models"
doc_type: "notes"
topic: "diffusion-models"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability"
  - "00-math-toolkit/information-theory"
  - "05-probabilistic-modeling/bayesian-inference"
  - "06-neural-networks/README"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "generative-models"
  - "diffusion"
  - "ddpm"
  - "denoising"
---

## Purpose

These notes introduce diffusion models through the denoising diffusion probabilistic model (DDPM) viewpoint.
The emphasis is on the mathematics of the forward noising process, the learned reverse process, and why denoising becomes a practical training objective.

## Learning objectives

After working through this note, you should be able to:

- define the DDPM forward process and closed-form marginal at time $t$;
- explain the reverse denoising process mathematically;
- state the common noise-prediction training objective;
- compare diffusion models with GANs and VAEs in terms of stability and sampling cost; and
- interpret diffusion generation as iterative denoising from noise to data.

## 1. Forward diffusion process

Let $x_0 \sim p_{\mathrm{data}}$ be a data sample.
Choose a variance schedule $\beta_1,\dots,\beta_T$ with $0 < \beta_t < 1$ and define $\alpha_t = 1-\beta_t$.
The forward process is a Markov chain

$$
q(x_t \mid x_{t-1})
=
\mathcal{N}\!\left(x_t; \sqrt{\alpha_t}\,x_{t-1}, (1-\alpha_t)I\right).
$$

Each step slightly shrinks the signal and injects Gaussian noise.
After many steps, the distribution approaches an isotropic Gaussian.

## 2. Closed-form noising at arbitrary time

Define

$$
\bar{\alpha}_t = \prod_{s=1}^t \alpha_s.
$$

Then one can sample $x_t$ directly from $x_0$:

$$
q(x_t \mid x_0)
=
\mathcal{N}\!\left(x_t; \sqrt{\bar{\alpha}_t}\,x_0, (1-\bar{\alpha}_t)I\right).
$$

Equivalently,

$$
x_t
=
\sqrt{\bar{\alpha}_t}\,x_0
+
\sqrt{1-\bar{\alpha}_t}\,\varepsilon,
\qquad
\varepsilon \sim \mathcal{N}(0,I).
$$

This identity is crucial because it lets us generate noisy training pairs $(x_0,x_t)$ without simulating the whole chain step by step.

## 3. Reverse generative process

Generation runs the chain backward.
Start from

$$
x_T \sim \mathcal{N}(0,I)
$$

and learn reverse transitions

$$
p_\theta(x_{t-1}\mid x_t).
$$

If the forward increments are small, the reverse transitions are also approximately Gaussian.
A common parameterization is

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal{N}\!\left(x_{t-1}; \mu_\theta(x_t,t), \Sigma_\theta(x_t,t)\right).
$$

In the basic DDPM setup, the variance is fixed or simplified, and a neural network learns the mean indirectly by predicting the noise component.

## 4. Noise-prediction objective

Using

$$
x_t
=
\sqrt{\bar{\alpha}_t}\,x_0
+
\sqrt{1-\bar{\alpha}_t}\,\varepsilon,
$$

train a neural network $\varepsilon_\theta(x_t,t)$ to predict $\varepsilon$.
The practical loss is

$$
\mathcal{L}_{\mathrm{simple}}(\theta)
=
\mathbb{E}_{x_0,\varepsilon,t}
\left[
\left\|
\varepsilon - \varepsilon_\theta(x_t,t)
\right\|_2^2
\right].
$$

This looks like ordinary regression, which is one reason diffusion training is stable in practice.

## 5. Why denoising works

At time $t$, the model receives a partially corrupted sample $x_t$ and a time index.
To predict the noise correctly, it must learn which directions in input space correspond to signal and which correspond to random perturbation.
Repeated across noise scales, this teaches the model how to move from high-entropy noise back toward the data manifold.

From a more formal viewpoint, the denoising objective is linked to a variational bound on log-likelihood and to score matching.
For this module, the DDPM denoising view is the primary exposition because it is the most operational for implementation.

## 6. Reverse sampling algorithm

Starting from $x_T \sim \mathcal{N}(0,I)$, for $t=T,T-1,\dots,1$:

1. predict noise $\hat{\varepsilon} = \varepsilon_\theta(x_t,t)$;
2. estimate a mean for $x_{t-1}$ using $\hat{\varepsilon}$ and the schedule;
3. add Gaussian noise when $t>1$ according to the reverse variance;
4. continue until $x_0$ is obtained.

This iterative reverse process is the main computational cost of diffusion models.

## 7. Worked intuition in low dimension

Suppose the data lie on two clusters in $\mathbb{R}^2$.
The forward process gradually turns those clusters into something close to a single Gaussian cloud.
The reverse model learns how to "unmix" that cloud:
at high noise levels it learns broad global corrections, and at low noise levels it learns fine local adjustments.

This explains why diffusion can model complicated multimodal distributions without adversarial training.

## 8. Comparison with other paradigms

Compared with GANs:

- diffusion training is usually more stable;
- diffusion sampling is slower;
- both can achieve high sample quality.

Compared with VAEs:

- diffusion often yields better perceptual samples;
- VAEs offer a more direct latent representation story;
- diffusion derivations rely on stochastic chains rather than one global latent code.

Compared with autoregressive models:

- diffusion does not require a natural token order;
- autoregressive models keep exact tractable conditionals;
- diffusion relies on iterative denoising instead of sequential factorization.

## 9. Caveats

### 9.1 Sampling cost

Basic DDPMs require many reverse steps, which can make generation slow.

### 9.2 Schedule dependence

Performance depends on the variance schedule, parameterization, and sampler.

### 9.3 Theoretical richness

Diffusion models connect to variational inference, denoising, stochastic differential equations, and score-based models.
That richness is useful, but it can obscure the core implementation idea if presented all at once.

## 10. Summary

Diffusion models learn to reverse a gradual noising process.
The forward process is analytically specified, the reverse process is learned, and the common practical objective is noise prediction.
Their main advantage is a strong combination of sample quality and training stability.
Their main cost is iterative sampling and greater mathematical overhead than simpler generative models.
