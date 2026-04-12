---
title: "Expectation-Maximization"
module: "05-probabilistic-modeling"
lesson: "em-algorithm"
doc_type: "notes"
topic: "expectation-maximization"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability"
  - "00-math-toolkit/information-theory"
  - "01-optimization/convexity-and-optimization"
  - "05-probabilistic-modeling/bayesian-inference"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "probabilistic-modeling"
  - "em"
  - "elbo"
  - "latent-variables"
  - "gmm"
---

## Purpose

These notes introduce the expectation-maximization algorithm as a method for optimizing likelihoods with latent variables.
The main emphasis is conceptual: EM alternates between constructing a tractable lower bound and maximizing that bound.
This is the right bridge from classical latent-variable models to later variational methods.

## Learning objectives

After working through this note, you should be able to:

- explain why latent variables make maximum-likelihood optimization difficult;
- define the EM auxiliary function and connect it to the ELBO;
- describe the E-step and M-step in words and equations;
- explain the monotonic-improvement property of EM;
- derive the Gaussian-mixture E-step responsibilities; and
- distinguish EM from gradient descent and from full Bayesian inference.

## 1. The latent-variable likelihood problem

Suppose observed data are $x_{1:n}$ and each observation has an associated latent variable $z_i$.
The complete-data model is

$$
p(x_i,z_i \mid \theta),
$$

while the observed-data likelihood is

$$
p(x_i \mid \theta) = \sum_{z_i} p(x_i,z_i \mid \theta)
$$

for discrete latent variables, or

$$
p(x_i \mid \theta) = \int p(x_i,z_i \mid \theta)\,dz_i
$$

for continuous ones.

The log-likelihood becomes

$$
\ell(\theta) = \sum_{i=1}^n \log p(x_i \mid \theta).
$$

The difficulty is that the logarithm sits outside the sum or integral over latent structure.
Even when the complete-data model is simple, the marginal likelihood may not be.

## 2. Lower bounds from auxiliary distributions

Introduce any distribution $q_i(z_i)$ over the latent variable for observation $i$.
Then

$$
\log p(x_i \mid \theta)
= \log \sum_{z_i} q_i(z_i)\frac{p(x_i,z_i \mid \theta)}{q_i(z_i)}.
$$

By Jensen's inequality,

$$
\log p(x_i \mid \theta)
\geq
\sum_{z_i} q_i(z_i)\log \frac{p(x_i,z_i \mid \theta)}{q_i(z_i)}.
$$

Summing over $i$ yields an evidence lower bound

$$
\mathcal{L}(q,\theta)
= \sum_{i=1}^n \mathbb{E}_{q_i}\!\left[\log p(x_i,z_i \mid \theta)\right]
  + \sum_{i=1}^n H(q_i),
$$

where $H(q_i) = -\mathbb{E}_{q_i}[\log q_i(z_i)]$ is the entropy of $q_i$.

This is the same structural object that later appears in variational inference.

## 3. The ELBO identity

The lower bound becomes especially informative when we compare it directly to the log-likelihood:

$$
\log p(x_i \mid \theta)
= \mathcal{L}_i(q_i,\theta)
 + D_{\mathrm{KL}}\!\left(q_i(z_i)\,\|\,p(z_i \mid x_i,\theta)\right).
$$

Therefore:

- the ELBO is tight exactly when $q_i(z_i)=p(z_i\mid x_i,\theta)$;
- maximizing the ELBO can be done by alternating over $q$ and $\theta$; and
- EM is the special case where the variational family is unrestricted enough to set $q_i$ equal to the exact posterior under the current parameters.

## 4. E-step and M-step

At iteration $t$, EM alternates:

### E-step

Set

$$
q_i^{(t+1)}(z_i) = p(z_i \mid x_i,\theta^{(t)}).
$$

This choice makes the ELBO tight at $\theta=\theta^{(t)}$.

### M-step

Choose new parameters by maximizing the ELBO with $q$ fixed:

$$
\theta^{(t+1)}
= \arg\max_\theta \sum_{i=1}^n \mathbb{E}_{q_i^{(t+1)}}[\log p(x_i,z_i \mid \theta)].
$$

Because the entropy term does not depend on $\theta$, the M-step maximizes the expected complete-data log-likelihood.

This motivates the standard auxiliary function

$$
Q(\theta \mid \theta^{(t)})
= \sum_{i=1}^n
\mathbb{E}_{p(z_i \mid x_i,\theta^{(t)})}
\left[\log p(x_i,z_i \mid \theta)\right].
$$

## 5. Why EM improves the likelihood

At the E-step, the ELBO is tight at the current iterate:

$$
\ell(\theta^{(t)})
= \mathcal{L}(q^{(t+1)},\theta^{(t)}).
$$

At the M-step, we choose $\theta^{(t+1)}$ so that

$$
\mathcal{L}(q^{(t+1)},\theta^{(t+1)})
\geq
\mathcal{L}(q^{(t+1)},\theta^{(t)}).
$$

Since $\ell(\theta) \geq \mathcal{L}(q,\theta)$ for every $q$,

$$
\ell(\theta^{(t+1)})
\geq
\mathcal{L}(q^{(t+1)},\theta^{(t+1)})
\geq
\ell(\theta^{(t)}).
$$

So EM produces a non-decreasing sequence of observed-data likelihood values.

> **Caution.** Non-decreasing does not mean globally optimal.
> EM can converge to local optima or saddle-like stationary points, and initialization matters.

## 6. Gaussian mixture models as the canonical example

For a $K$-component Gaussian mixture model,

$$
p(x_i,z_i=k \mid \theta)
= \pi_k \,\mathcal{N}(x_i \mid \mu_k,\Sigma_k),
$$

where $\theta = \{\pi_k,\mu_k,\Sigma_k\}_{k=1}^K$ and $\sum_k \pi_k = 1$.

The observed-data likelihood is

$$
p(x_i \mid \theta)
= \sum_{k=1}^K \pi_k \,\mathcal{N}(x_i \mid \mu_k,\Sigma_k).
$$

### E-step responsibilities

The posterior probability that point $x_i$ belongs to component $k$ is

$$
\gamma_{ik}
= p(z_i=k \mid x_i,\theta)
= \frac{\pi_k \,\mathcal{N}(x_i \mid \mu_k,\Sigma_k)}
{\sum_{j=1}^K \pi_j \,\mathcal{N}(x_i \mid \mu_j,\Sigma_j)}.
$$

These are called responsibilities.

### M-step updates

Let

$$
N_k = \sum_{i=1}^n \gamma_{ik}.
$$

Then the maximizers are

$$
\pi_k^{\text{new}} = \frac{N_k}{n},
$$

$$
\mu_k^{\text{new}} = \frac{1}{N_k}\sum_{i=1}^n \gamma_{ik} x_i,
$$

$$
\Sigma_k^{\text{new}}
= \frac{1}{N_k}\sum_{i=1}^n \gamma_{ik}
(x_i-\mu_k^{\text{new}})(x_i-\mu_k^{\text{new}})^\top.
$$

The formulas look like weighted empirical estimates because the latent assignments are replaced by their posterior expectations.

## 7. Hard assignments versus soft assignments

K-means and GMM-EM are closely related, but they are not the same algorithm.

- K-means uses hard assignments and minimizes squared distortion.
- GMM-EM uses soft assignments and maximizes a probabilistic likelihood.

In a limiting regime with spherical covariances and vanishing variance, GMM responsibilities become nearly hard and the link to K-means becomes visible.

## 8. Exact versus approximate inference around EM

EM mixes two inferential regimes.

- In the E-step, the latent posterior under the current parameters is computed exactly for the chosen model family.
- In the overall learning problem, we still optimize a non-convex likelihood and do not integrate over parameter uncertainty.

So EM should be understood as:

- exact conditional inference over latent variables inside each iteration; but
- approximate global optimization for the full learning problem.

This distinction matters because later variational inference loosens the E-step as well, using a restricted family $q_\phi(z)$ when exact posteriors are no longer tractable.

## 9. Practical failure modes

EM is elegant, but not automatic.
Common pathologies include:

- singular covariance estimates in GMMs;
- dead components with tiny effective sample size $N_k$;
- strong dependence on initialization; and
- slow convergence near a stationary point.

Common mitigations include covariance regularization, multiple random restarts, and early stopping based on lower-bound or likelihood change.

## Summary

EM is a coordinate-ascent procedure on an ELBO for latent-variable models.
The E-step sets the auxiliary distribution to the exact latent posterior under current parameters, and the M-step maximizes the expected complete-data log-likelihood.
This yields monotonic likelihood improvement and makes GMMs and other latent-variable models computationally accessible.
The same ELBO structure later reappears in variational inference and variational autoencoders.
