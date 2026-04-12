---
title: "Generative Adversarial Networks"
module: "11-generative-models"
lesson: "gans"
doc_type: "notes"
topic: "generative-adversarial-networks"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability"
  - "01-optimization/convexity-and-optimization"
  - "06-neural-networks/README"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "generative-models"
  - "gan"
  - "minimax"
  - "implicit-models"
---

## Purpose

These notes introduce GANs as implicit generative models trained through a two-player minimax game.
The emphasis is on what the discriminator estimates, why the generator can improve without explicit likelihoods, and why the optimization is both powerful and fragile.

## Learning objectives

After working through this note, you should be able to:

- write the original GAN minimax objective;
- explain the roles of the generator and discriminator;
- interpret the discriminator as a density-ratio classifier at optimum;
- describe the Nash-equilibrium intuition behind GAN training; and
- identify common failure modes such as mode collapse and unstable gradients.

## 1. Setup

Let $p_{\mathrm{data}}$ denote the data distribution over $x \in \mathcal{X}$.
Let $\varepsilon \sim p(\varepsilon)$ be a simple noise variable, often Gaussian.
The generator defines

$$
x = G_\theta(\varepsilon),
$$

which induces a model distribution $p_\theta$ over data space.

The discriminator is a function

$$
D_\psi(x) \in (0,1),
$$

interpreted as the estimated probability that $x$ came from the real data rather than the generator.

## 2. The original minimax objective

The classical GAN objective is

$$
\min_\theta \max_\psi V(\psi,\theta),
$$

where

$$
V(\psi,\theta)
=
\mathbb{E}_{x\sim p_{\mathrm{data}}}[\log D_\psi(x)]
+
\mathbb{E}_{x\sim p_\theta}[\log (1-D_\psi(x))].
$$

The discriminator tries to maximize this objective by assigning high scores to real data and low scores to generated samples.
The generator tries to minimize it by making generated samples look real.

## 3. Why this works without explicit likelihood

The generator never needs to evaluate a normalized density $p_\theta(x)$.
It only needs to sample from its own distribution by drawing $\varepsilon$ and computing $G_\theta(\varepsilon)$.

This makes GANs implicit models:
they define a distribution through sampling, not through a closed-form likelihood function.

## 4. Optimal discriminator and density ratios

For fixed generator distribution $p_\theta$, the discriminator optimization decouples pointwise in $x$.
The optimal discriminator is

$$
D^*(x)
=
\frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x)+p_\theta(x)}.
$$

So the discriminator estimates a density-ratio-derived quantity.
If the two distributions match exactly, then

$$
D^*(x) = \frac{1}{2}
$$

for all $x$ in their support.

This is the equilibrium signal:
the discriminator cannot do better than random guessing when the generator has matched the data distribution.

## 5. Nash-equilibrium intuition

GAN training is not simple minimization.
It is a game between two players with different objectives:

- discriminator: classify real versus fake;
- generator: produce fakes that the discriminator cannot distinguish from real data.

At an ideal equilibrium:

- $p_\theta = p_{\mathrm{data}}$;
- the discriminator outputs $1/2$ everywhere on the support;
- neither player can improve unilaterally.

This is the Nash-equilibrium intuition behind the original GAN theory.

## 6. Practical generator loss

In practice, many implementations replace the generator's minimization of

$$
\mathbb{E}_{x\sim p_\theta}[\log (1-D_\psi(x))]
$$

with the non-saturating loss

$$
\min_\theta
-\mathbb{E}_{\varepsilon\sim p(\varepsilon)}[\log D_\psi(G_\theta(\varepsilon))].
$$

This has the same qualitative goal but yields stronger gradients early in training, when the discriminator can easily reject fake samples.

## 7. Failure modes

### 7.1 Mode collapse

The generator may map many latent codes to a small set of outputs.
Then sample quality may look locally good while diversity is poor.

### 7.2 Training instability

Because GAN training is a saddle-point problem, gradient descent dynamics can oscillate.
Improving one player changes the loss landscape seen by the other.

### 7.3 Vanishing gradients

If the discriminator becomes too strong too quickly, the generator may receive weak learning signals.

## 8. Standard stabilizing ideas

Common practical modifications include:

- non-saturating generator loss;
- Wasserstein objectives;
- gradient penalties;
- spectral normalization;
- careful architecture choices and balanced update schedules.

These improve optimization, but the core tradeoff remains:
GANs can produce excellent samples, yet their training is usually less robust than VAE or diffusion training.

## 9. ML interpretation

GANs are useful when:

- perceptual realism matters more than tractable likelihood;
- one wants direct sample synthesis;
- evaluation is based on sample quality and diversity rather than calibrated probabilities.

They are less natural when:

- exact uncertainty quantification is required;
- explicit inference over latent variables is central;
- the application needs stable optimization with minimal tuning.

## 10. Category-theoretic insertion point

Structurally, GANs compose:

- a noise source $\varepsilon \mapsto G_\theta(\varepsilon)$;
- a critic $x \mapsto D_\psi(x)$.

The distinctive feature is not the maps alone but the adversarial coupling between their objectives.
Composition explains the pipeline; game-theoretic optimization explains the difficulty.

## 11. Summary

GANs replace explicit likelihood fitting with adversarial learning.
Their main advantage is strong sample realism and fast generation.
Their main costs are optimization instability, lack of tractable likelihood, and weaker built-in latent semantics than in VAEs.
