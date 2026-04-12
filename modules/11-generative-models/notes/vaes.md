---
title: "Variational Autoencoders"
module: "11-generative-models"
lesson: "vaes"
doc_type: "notes"
topic: "variational-autoencoders"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability"
  - "00-math-toolkit/information-theory"
  - "05-probabilistic-modeling/em-algorithm"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "generative-models"
  - "vae"
  - "elbo"
  - "latent-variables"
  - "reparameterization"
---

## Purpose

These notes develop variational autoencoders as latent-variable generative models trained by optimizing an evidence lower bound.
The emphasis is on the mathematical reason the ELBO appears, the role of amortized inference, and the practical tradeoff between reconstruction fidelity and latent regularization.

## Learning objectives

After working through this note, you should be able to:

- write the latent-variable model underlying a VAE;
- explain why the posterior $p_\theta(z\mid x)$ is usually intractable;
- derive and interpret the ELBO at a high level;
- explain the reparameterization trick;
- identify posterior collapse and its causes; and
- connect VAE training to representation learning and sample generation.

## 1. Latent-variable setup

Let $x \in \mathcal{X}$ denote observed data and let $z \in \mathbb{R}^d$ denote a latent variable.
A VAE specifies

$$
p_\theta(x,z) = p(z)p_\theta(x\mid z).
$$

where $p(z)$ is usually a simple prior such as $\mathcal{N}(0,I)$.
The marginal likelihood is

$$
p_\theta(x) = \int p(z)p_\theta(x\mid z)\,dz.
$$

The challenge is that the posterior

$$
p_\theta(z\mid x) = \frac{p(z)p_\theta(x\mid z)}{p_\theta(x)}
$$

is generally intractable because the denominator requires the same integral.

## 2. Approximate inference with an encoder

Introduce an inference model

$$
q_\phi(z\mid x),
$$

often called the encoder.
For continuous latent variables, a common choice is a diagonal Gaussian:

$$
q_\phi(z\mid x) = \mathcal{N}\!\bigl(z;\mu_\phi(x), \operatorname{diag}(\sigma_\phi^2(x))\bigr).
$$

The VAE jointly learns:

- decoder parameters $\theta$ in $p_\theta(x\mid z)$;
- encoder parameters $\phi$ in $q_\phi(z\mid x)$.

This is amortized inference: instead of solving a separate optimization problem for each example $x$, one neural network predicts an approximate posterior for all examples.

## 3. The ELBO

For any distribution $q_\phi(z\mid x)$,

$$
\log p_\theta(x)
=
\mathcal{L}(x;\theta,\phi)
+
D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\right),
$$

where

$$
\mathcal{L}(x;\theta,\phi)
=
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
- D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p(z)\right).
$$

Because KL divergence is nonnegative,

$$
\mathcal{L}(x;\theta,\phi) \leq \log p_\theta(x).
$$

The first term is a reconstruction term.
The second term regularizes the approximate posterior toward the prior.

## 4. Interpreting the two ELBO terms

### 4.1 Reconstruction term

The quantity

$$
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
$$

encourages the decoder to assign high probability to the observed example when $z$ is drawn from the encoder.
For Bernoulli decoders this becomes a binary cross-entropy term.
For Gaussian decoders it becomes a squared-error-like term up to constants.

### 4.2 KL regularizer

The term

$$
D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p(z)\right)
$$

discourages the encoder from placing arbitrary isolated codes for each data point.
It encourages the latent codes to stay in a region where prior samples are plausible, which is why drawing $z \sim p(z)$ can produce new samples.

## 5. Reparameterization trick

To optimize the ELBO with gradient methods, we need gradients through samples from $q_\phi(z\mid x)$.
For Gaussian latent variables, write

$$
z = \mu_\phi(x) + \sigma_\phi(x)\odot \varepsilon,
\qquad
\varepsilon \sim \mathcal{N}(0,I).
$$

The randomness is now isolated in $\varepsilon$, which does not depend on $\phi$.
This lets us differentiate the Monte Carlo estimate of the reconstruction term with respect to encoder parameters.

## 6. Worked example: Gaussian prior and Bernoulli decoder

Suppose:

- $p(z) = \mathcal{N}(0,I)$;
- $q_\phi(z\mid x)$ is diagonal Gaussian;
- $p_\theta(x\mid z)$ is Bernoulli with mean vector $f_\theta(z)$.

Then for one example $x \in \{0,1\}^m$ the ELBO estimate is

$$
\widehat{\mathcal{L}}(x;\theta,\phi)
=
\log p_\theta(x\mid z)
- D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,\mathcal{N}(0,I)\right),
\qquad
z = \mu_\phi(x) + \sigma_\phi(x)\odot \varepsilon.
$$

For a diagonal Gaussian encoder with log-variance vector $\log \sigma^2(x)$, the KL term has the closed form

$$
D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,\mathcal{N}(0,I)\right)
=
\frac{1}{2}\sum_{j=1}^d
\left(
\mu_j(x)^2 + \sigma_j(x)^2 - 1 - \log \sigma_j(x)^2
\right).
$$

## 7. Relation to EM and probabilistic modeling

VAEs and EM both optimize lower bounds on log-likelihood.
The difference is that EM often uses an exact posterior in the E-step when the model permits it, while a VAE learns a restricted parametric approximation $q_\phi(z\mid x)$.

So the VAE can be viewed as:

- a latent-variable model;
- an approximate-inference method;
- a neural parameterization of both the generative and inference maps.

## 8. Failure modes and caveats

### 8.1 Posterior collapse

If the decoder is very strong, it may model $x$ well without using $z$.
Then $q_\phi(z\mid x)$ can collapse toward the prior, making the latent code uninformative.

### 8.2 Blurry samples

Likelihood-based objectives often average over multiple plausible outputs.
In image domains this can lead to samples that look smooth or blurry compared with GAN or diffusion outputs.

### 8.3 ELBO is not exact likelihood

A higher ELBO implies a better lower bound, not necessarily a better exact likelihood, unless the approximate posterior is also tight.

## 9. ML interpretation

VAEs are especially useful when the task needs both generation and structure:

- representation learning;
- anomaly detection via reconstruction and likelihood surrogates;
- semi-supervised learning with latent factors;
- conditional generation with interpretable controls.

## 10. Category-theoretic insertion point

The core VAE pipeline can be viewed as two composable stochastic maps:

- encoder: $x \mapsto q_\phi(z\mid x)$;
- decoder: $z \mapsto p_\theta(x\mid z)$.

This viewpoint highlights composition and approximation.
The encoder is not the inverse of the decoder in a strict sense; it is an approximate inference morphism that makes the optimization tractable.

## 11. Summary

A VAE is a latent-variable generative model trained by maximizing an ELBO.
Its main advantage is a structured latent space with stable optimization.
Its main cost is that the optimization target is a bound rather than the exact log-likelihood, and high-fidelity samples can be harder to obtain than in adversarial or diffusion models.
