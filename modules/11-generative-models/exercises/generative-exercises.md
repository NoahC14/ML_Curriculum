---
title: "Generative Modeling Exercises"
module: "11-generative-models"
lesson: "generative-exercises"
doc_type: "exercise"
topic: "generative-modeling"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability"
  - "00-math-toolkit/information-theory"
  - "05-probabilistic-modeling/em-algorithm"
  - "11-generative-models/notes/generative-models-overview"
  - "11-generative-models/notes/vaes"
  - "11-generative-models/notes/gans"
  - "11-generative-models/notes/diffusion-models"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "generative-models"
  - "vae"
  - "gan"
  - "diffusion"
  - "autoregressive"
---

## Purpose

These exercises reinforce the mathematical foundations and tradeoffs of the four main generative-modeling paradigms studied in this module.

## Exercise 1. Autoregressive factorization

Let $x=(x_1,x_2,x_3)$ be a discrete random vector.

1. Write the chain-rule factorization of $p(x_1,x_2,x_3)$.
2. Explain why this gives an exact likelihood model.
3. State one computational reason training is easy and one reason sampling is slow.
4. Give one modality where autoregressive factorization is natural and one where it is less natural.

## Exercise 2. ELBO from Jensen's inequality

Let $q(z\mid x)$ be any density.

1. Starting from $\log p_\theta(x)=\log \int p_\theta(x,z)\,dz$, insert $q(z\mid x)$ into the integral.
2. Apply Jensen's inequality to derive a lower bound.
3. Show that the lower bound can be written as
   $$
   \mathbb{E}_{q(z\mid x)}[\log p_\theta(x\mid z)] - D_{\mathrm{KL}}(q(z\mid x)\|p(z)).
   $$
4. State the condition under which the bound is tight.

## Exercise 3. Closed-form Gaussian KL

Let

$$
q(z\mid x)=\mathcal{N}\!\bigl(\mu,\operatorname{diag}(\sigma^2)\bigr),
\qquad
p(z)=\mathcal{N}(0,I).
$$

1. Derive the KL divergence $D_{\mathrm{KL}}(q(z\mid x)\|p(z))$.
2. State how this term behaves when $\mu=0$ and $\sigma_j^2=1$ for all $j$.
3. Explain why this KL term regularizes the latent space.

## Exercise 4. Reparameterization trick

Suppose $z \sim \mathcal{N}(\mu_\phi(x), \operatorname{diag}(\sigma_\phi^2(x)))$.

1. Write $z$ as a deterministic function of $\mu_\phi(x)$, $\sigma_\phi(x)$, and a parameter-free noise variable.
2. Explain why this helps gradient-based optimization.
3. State one latent-variable family where the reparameterization trick is less direct.

## Exercise 5. Optimal discriminator in a GAN

For fixed generator distribution $p_\theta$, maximize

$$
\mathbb{E}_{x\sim p_{\mathrm{data}}}[\log D(x)]
+
\mathbb{E}_{x\sim p_\theta}[\log(1-D(x))]
$$

pointwise in $x$.

1. Differentiate the objective with respect to $D(x)$.
2. Solve for the optimal $D^*(x)$.
3. State what $D^*(x)$ becomes when $p_\theta = p_{\mathrm{data}}$.
4. Explain why this supports the Nash-equilibrium interpretation.

## Exercise 6. GAN tradeoffs

Answer each in 2-4 sentences.

1. Why can GANs produce high-quality samples without maximizing likelihood?
2. Why is mode collapse harmful even when individual samples look realistic?
3. Why can a too-strong discriminator harm generator training?

## Exercise 7. Diffusion forward process

Let

$$
q(x_t\mid x_{t-1})
=
\mathcal{N}\!\left(\sqrt{\alpha_t}\,x_{t-1}, (1-\alpha_t)I\right).
$$

1. Define $\bar{\alpha}_t$.
2. Show that $x_t$ can be written as
   $$
   x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon.
   $$
3. Explain in words what happens to the data distribution as $t$ increases.

## Exercise 8. Diffusion reverse process

1. Why is the reverse process $p_\theta(x_{t-1}\mid x_t)$ generative?
2. What object does the model commonly predict in a DDPM implementation?
3. Why does this turn training into a regression problem?
4. State one reason diffusion training is often more stable than GAN training.

## Exercise 9. Model selection by application

Choose the most appropriate paradigm and justify it briefly.

1. A language-modeling task where exact token likelihood matters.
2. A representation-learning task where interpolation in latent space is important.
3. A high-fidelity image synthesis task where slow sampling is acceptable.
4. A low-latency image generator where one-shot sampling matters more than tractable likelihood.

## Exercise 10. Comparative reflection

Compare autoregressive models, VAEs, GANs, and diffusion models along these axes:

1. tractable likelihood;
2. latent structure;
3. training stability;
4. sampling speed;
5. sample quality.

For each axis, identify one model family that is usually strongest and one that is usually weakest, then explain the tradeoff.
