---
title: "Generative Models Overview"
module: "11-generative-models"
lesson: "generative-models-overview"
doc_type: "notes"
topic: "generative-modeling"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability"
  - "00-math-toolkit/information-theory"
  - "01-optimization/convexity-and-optimization"
  - "05-probabilistic-modeling/bayesian-inference"
  - "05-probabilistic-modeling/em-algorithm"
  - "06-neural-networks/README"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "generative-models"
  - "autoregressive-models"
  - "vae"
  - "gan"
  - "diffusion"
---

## Purpose

These notes compare the four dominant paradigms of modern generative modeling:
autoregressive models, variational autoencoders, generative adversarial networks, and diffusion models.
The main goal is to understand what each family optimizes, what distributions it represents naturally, and what tradeoffs it makes among likelihood, sample quality, latent structure, and computational cost.

## Learning objectives

After working through this note, you should be able to:

- distinguish explicit density models from implicit generative models;
- explain the factorization behind autoregressive likelihood models;
- state the optimization target for VAEs, GANs, and diffusion models;
- compare the paradigms on tractable likelihood, sample quality, latent structure, and training stability; and
- choose an appropriate generative paradigm for a concrete ML task.

## 1. What is a generative model?

A generative model specifies how data $x \in \mathcal{X}$ arise, either by:

- modeling a probability density or mass function $p_\theta(x)$ directly;
- introducing latent variables $z$ and defining a joint model $p_\theta(x,z)$; or
- defining a stochastic procedure whose output distribution approximates the data distribution $p_{\mathrm{data}}$.

The central task is to learn parameters $\theta$ so that the model distribution is close to the data distribution.
Different model families operationalize "close" in different ways:

- maximum likelihood or an approximation to it;
- adversarial distinguishability;
- denoising or score-matching objectives; or
- reconstruction plus regularization of latent structure.

## 2. Four paradigms at a glance

### 2.1 Autoregressive models

Autoregressive models use the chain rule:

$$
p_\theta(x_1,\dots,x_T)
= \prod_{t=1}^T p_\theta(x_t \mid x_{<t}),
$$

where $x_{<t} = (x_1,\dots,x_{t-1})$.
This makes the likelihood exact and tractable as long as each conditional factor is tractable.

Examples include:

- $n$-gram language models;
- recurrent neural language models;
- PixelRNN and PixelCNN;
- transformer language models.

Strengths:

- exact log-likelihood evaluation;
- conceptually clean probabilistic semantics;
- straightforward training by teacher-forced maximum likelihood.

Weaknesses:

- sequential sampling can be slow;
- global latent structure is not built in;
- long-range coherence can be hard without large-capacity architectures.

### 2.2 Variational autoencoders

VAEs introduce a latent variable $z \in \mathbb{R}^d$ and a joint model

$$
p_\theta(x,z) = p(z)p_\theta(x \mid z).
$$

Because $\log p_\theta(x)$ usually requires integrating out $z$,
VAEs optimize a lower bound using an encoder distribution $q_\phi(z \mid x)$.

Strengths:

- principled latent-variable model;
- amortized inference through the encoder;
- continuous latent spaces that support interpolation and representation learning.

Weaknesses:

- the ELBO is only a lower bound on log-likelihood;
- simple decoders may produce blurry samples;
- the latent channel can collapse if the decoder is too expressive.

### 2.3 Generative adversarial networks

GANs define a generator $x = G_\theta(\varepsilon)$ with noise input $\varepsilon \sim p(\varepsilon)$.
A discriminator $D_\psi(x)$ is trained to distinguish real samples from generated samples.
The generator is trained adversarially to fool the discriminator.

Strengths:

- often excellent perceptual sample quality;
- no need to specify an explicit likelihood;
- flexible generators for complex data manifolds.

Weaknesses:

- unstable minimax training;
- no tractable normalized likelihood;
- latent representation quality depends on architecture and regularization rather than a probabilistic posterior.

### 2.4 Diffusion models

Diffusion models define a forward noising process that gradually corrupts data and a learned reverse process that denoises.
In a DDPM-style formulation, the forward chain is

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(\sqrt{1-\beta_t}\,x_{t-1}, \beta_t I\right),
\qquad t=1,\dots,T.
$$

The model learns a reverse transition

$$
p_\theta(x_{t-1}\mid x_t).
$$

Strengths:

- highly stable training relative to GANs;
- strong sample quality;
- clear connection to likelihood bounds, denoising, and score estimation.

Weaknesses:

- sampling is usually much slower than one-shot generators;
- derivations are more involved than for VAEs or autoregressive models;
- latent structure is less direct than in explicit latent-variable models.

## 3. A structural comparison

### 3.1 Likelihood

Autoregressive models provide exact tractable likelihood:

$$
\log p_\theta(x)
= \sum_{t=1}^T \log p_\theta(x_t \mid x_{<t}).
$$

VAEs provide a lower bound:

$$
\log p_\theta(x)
\geq
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
- D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p(z)\right).
$$

Standard GANs do not provide a tractable normalized likelihood.
Diffusion models admit likelihood-related training objectives, but the most common practical objective is a denoising loss rather than exact maximum likelihood.

### 3.2 Sample quality

In modern practice:

- GANs historically emphasized perceptual sharpness;
- diffusion models now often dominate on sample fidelity and diversity;
- autoregressive models can be excellent when the sequential factorization matches the modality;
- VAEs often trade some sharpness for stable latent-variable learning.

Perceptual quality is not identical to likelihood.
A model can assign high likelihood yet produce visually mediocre samples, or produce sharp samples without calibrated probabilities.

### 3.3 Latent structure

VAEs offer the clearest built-in latent representation because the model explicitly introduces $z$ and learns an encoder.
GANs can have useful latent spaces, but this is not enforced by the original objective.
Diffusion models can be paired with latent spaces, especially in latent diffusion systems, but the core DDPM presentation is not primarily a latent-variable representation-learning story.
Autoregressive models typically treat previous tokens or pixels as context rather than compressing the data into one global latent code.

### 3.4 Optimization and stability

Autoregressive models reduce to supervised conditional prediction.
VAEs use stochastic gradient ascent on an ELBO and are usually stable.
GANs optimize a saddle-point problem, so gradient dynamics can cycle or collapse.
Diffusion models typically optimize a denoising regression loss, which is empirically stable but computationally expensive.

### 3.5 Sampling cost

Autoregressive sampling is sequential in the number of tokens or pixels.
VAEs and GANs can generate in one forward pass after drawing a latent noise vector.
Diffusion models require many denoising steps unless accelerated samplers are used.

## 4. Choosing among paradigms

Choose an autoregressive model when:

- exact likelihood matters;
- the data have a natural ordering;
- sequential prediction is already the main task.

Choose a VAE when:

- you want a meaningful continuous latent space;
- posterior inference or structured representation learning matters;
- stable training is more important than the sharpest possible samples.

Choose a GAN when:

- sample realism is the main goal;
- tractable likelihood is not required;
- you can afford careful architecture and optimization tuning.

Choose a diffusion model when:

- high sample quality and stable optimization matter;
- slower sampling is acceptable or can be mitigated;
- the modality supports denoising-based generation.

## 5. Worked comparison table

| Paradigm | Primary objective | Tractable likelihood? | Latent structure | Typical training stability | Typical sampling speed |
| --- | --- | --- | --- | --- | --- |
| Autoregressive | Maximize $\log p_\theta(x)$ via chain rule | Yes, exact | Weak global latent structure by default | High | Slow, sequential |
| VAE | Maximize ELBO | Approximate lower bound | Strong, explicit | High | Fast |
| GAN | Adversarial minimax game | No | Implicit unless regularized | Low to moderate | Fast |
| Diffusion | Denoising / variational objective | Approximate, often indirect | Moderate, usually implicit in basic DDPMs | High | Slow |

## 6. Common misconceptions

### Misconception 1: Better likelihood implies better samples

Not necessarily.
Likelihood rewards global probability mass allocation, not human perceptual sharpness alone.

### Misconception 2: GANs are "more probabilistic" because they are generative

GANs define a sample distribution through a generator, but the standard formulation does not produce a tractable density $p_\theta(x)$.

### Misconception 3: Diffusion models are unrelated to latent-variable thinking

The training story is different from a VAE, but diffusion models still involve hidden random variables along the noising path and can be interpreted through variational bounds.

## 7. Category-theoretic insertion point

At a light structural level, each paradigm defines a map from a source of randomness to data space:

- autoregressive models compose conditional morphisms over a sequence;
- VAEs compose a prior, decoder, and approximate inference map;
- GANs compose latent noise with a generator and an adversarial critic;
- diffusion models compose many small stochastic kernels along forward and reverse chains.

The category-theoretic value here is organizational rather than foundational:
composition clarifies pipeline structure, but it does not replace the probability theory.

## 8. Summary

The four major generative paradigms differ mainly in what they optimize and what they make easy:

- autoregressive models make likelihood easy;
- VAEs make latent-variable learning easy;
- GANs make one-shot high-quality sampling possible but hard to optimize;
- diffusion models make high-quality generation stable, at the cost of iterative sampling.

No single paradigm dominates every criterion.
The practical choice depends on whether the task prioritizes likelihood, representation learning, perceptual quality, or sampling efficiency.
