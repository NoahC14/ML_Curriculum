---
title: "ELBO Derivation for Variational Autoencoders"
module: "11-generative-models"
lesson: "elbo-derivation"
doc_type: "derivation"
topic: "elbo"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability"
  - "00-math-toolkit/information-theory"
  - "05-probabilistic-modeling/em-algorithm"
  - "11-generative-models/notes/vaes"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "generative-models"
  - "vae"
  - "elbo"
  - "kl-divergence"
---

## Purpose

This derivation develops the evidence lower bound for a latent-variable model from scratch.
The point is not only to arrive at the final formula, but to see exactly where Jensen's inequality, KL divergence, and approximate inference enter the argument.

## 1. Setup

Let $x \in \mathcal{X}$ be one observed example and let $z \in \mathbb{R}^d$ be a latent variable.
Assume a generative model

$$
p_\theta(x,z) = p(z)p_\theta(x\mid z).
$$

The marginal likelihood of the observation is

$$
p_\theta(x) = \int p_\theta(x,z)\,dz.
$$

Taking logarithms gives

$$
\log p_\theta(x)
=
\log \int p_\theta(x,z)\,dz.
$$

This quantity is difficult to optimize directly because the integral is often intractable.

## 2. Introduce an auxiliary density

Let $q_\phi(z\mid x)$ be any density whose support covers the relevant posterior mass.
Insert it inside the integral:

$$
\log p_\theta(x)
=
\log \int q_\phi(z\mid x)\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\,dz.
$$

Rewrite the integral as an expectation under $q_\phi(z\mid x)$:

$$
\log p_\theta(x)
=
\log
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right].
$$

## 3. Apply Jensen's inequality

Because $\log$ is concave,

$$
\log \mathbb{E}[Y] \geq \mathbb{E}[\log Y]
$$

for any positive random variable $Y$.
Therefore

$$
\log p_\theta(x)
\geq
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log \frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right].
$$

Define the evidence lower bound

$$
\mathcal{L}(x;\theta,\phi)
=
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log \frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right].
$$

This proves

$$
\mathcal{L}(x;\theta,\phi) \leq \log p_\theta(x).
$$

## 4. Expand the ELBO

Using $p_\theta(x,z)=p(z)p_\theta(x\mid z)$,

$$
\mathcal{L}(x;\theta,\phi)
=
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log p_\theta(x\mid z)
+
\log p(z)
-
\log q_\phi(z\mid x)
\right].
$$

Split the expectation term by term:

$$
\mathcal{L}(x;\theta,\phi)
=
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
+
\mathbb{E}_{q_\phi(z\mid x)}[\log p(z) - \log q_\phi(z\mid x)].
$$

Recognize the KL divergence

$$
D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p(z)\right)
=
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log \frac{q_\phi(z\mid x)}{p(z)}
\right].
$$

Therefore

$$
\mathcal{L}(x;\theta,\phi)
=
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
- D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p(z)\right).
$$

This is the familiar VAE objective.

## 5. Derive the exact identity with the posterior

Start from the KL divergence to the true posterior:

$$
D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\right)
=
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log \frac{q_\phi(z\mid x)}{p_\theta(z\mid x)}
\right].
$$

Use Bayes' rule:

$$
p_\theta(z\mid x)
=
\frac{p_\theta(x,z)}{p_\theta(x)}.
$$

Substitute into the KL:

$$
D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\right)
=
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log q_\phi(z\mid x)
-
\log p_\theta(x,z)
+
\log p_\theta(x)
\right].
$$

Since $\log p_\theta(x)$ does not depend on $z$, pull it out of the expectation:

$$
D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\right)
=
\log p_\theta(x)
-
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log p_\theta(x,z)
-
\log q_\phi(z\mid x)
\right].
$$

Recognize the ELBO inside the expectation:

$$
D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\right)
=
\log p_\theta(x) - \mathcal{L}(x;\theta,\phi).
$$

Rearranging,

$$
\log p_\theta(x)
=
\mathcal{L}(x;\theta,\phi)
+
D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\right).
$$

This identity shows exactly why the ELBO becomes tight when the approximate posterior equals the true posterior.

## 6. Dataset version

For iid data $x_{1:n}$, the objective is

$$
\sum_{i=1}^n \mathcal{L}(x_i;\theta,\phi)
=
\sum_{i=1}^n
\left[
\mathbb{E}_{q_\phi(z_i\mid x_i)}[\log p_\theta(x_i\mid z_i)]
- D_{\mathrm{KL}}\!\left(q_\phi(z_i\mid x_i)\,\|\,p(z_i)\right)
\right].
$$

In stochastic optimization, we estimate this sum with mini-batches and Monte Carlo samples from the encoder.

## 7. Closed-form KL for a diagonal Gaussian encoder

Assume

$$
q_\phi(z\mid x)=\mathcal{N}\!\bigl(\mu,\operatorname{diag}(\sigma^2)\bigr),
\qquad
p(z)=\mathcal{N}(0,I).
$$

Then

$$
D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p(z)\right)
=
\frac{1}{2}\sum_{j=1}^d
\left(
\mu_j^2 + \sigma_j^2 - 1 - \log \sigma_j^2
\right).
$$

This formula is especially important in implementation because it avoids Monte Carlo noise in the KL term.

## 8. ML interpretation

The ELBO contains two pressures:

- fit the data well through the decoder;
- keep the approximate posterior near a simple prior so latent samples remain generative.

Too much emphasis on reconstruction can overfit and fragment the latent space.
Too much emphasis on the KL term can force the latent code to carry too little information.

## 9. Summary

The ELBO emerges from a simple move:
introduce an auxiliary posterior approximation and apply Jensen's inequality.
Its gap from the true log-likelihood is exactly the KL divergence between the approximate and true posterior.
This makes VAEs both a generative-modeling method and an approximate-inference method.
