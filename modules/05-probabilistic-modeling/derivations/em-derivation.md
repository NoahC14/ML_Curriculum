---
title: "EM Derivation from the ELBO"
module: "05-probabilistic-modeling"
lesson: "em-derivation"
doc_type: "derivation"
topic: "expectation-maximization"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability"
  - "00-math-toolkit/information-theory"
  - "01-optimization/convexity-and-optimization"
  - "05-probabilistic-modeling/em-algorithm"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "probabilistic-modeling"
  - "em"
  - "elbo"
  - "jensen"
  - "latent-variables"
---

## Purpose

This derivation shows carefully how EM arises from a lower bound on the observed-data log-likelihood.
The key point is not just the mechanics of the algorithm.
The key point is that EM is an ELBO method with an exact E-step.

## 1. Setup

Let $x_{1:n}$ be observed data and let $z_{1:n}$ be latent variables.
Let $\theta$ denote the model parameters.
The complete-data joint distribution factorizes over examples as

$$
p(x_{1:n},z_{1:n}\mid\theta)
= \prod_{i=1}^n p(x_i,z_i \mid \theta).
$$

The observed-data log-likelihood is

$$
\ell(\theta)
= \log p(x_{1:n}\mid\theta)
= \sum_{i=1}^n \log p(x_i\mid\theta),
$$

where

$$
p(x_i\mid\theta)
= \sum_{z_i} p(x_i,z_i\mid\theta)
$$

for discrete latent variables, or the analogous integral for continuous ones.

## 2. Insert an auxiliary distribution

For each $i$, let $q_i(z_i)$ be any distribution whose support contains the support of the posterior.
Then

$$
\log p(x_i\mid\theta)
= \log \sum_{z_i} q_i(z_i)\frac{p(x_i,z_i\mid\theta)}{q_i(z_i)}.
$$

Applying Jensen's inequality to the concave logarithm gives

$$
\log p(x_i\mid\theta)
\geq
\sum_{z_i} q_i(z_i)\log \frac{p(x_i,z_i\mid\theta)}{q_i(z_i)}.
$$

Define

$$
\mathcal{L}_i(q_i,\theta)
= \sum_{z_i} q_i(z_i)\log \frac{p(x_i,z_i\mid\theta)}{q_i(z_i)}.
$$

Summing over examples,

$$
\mathcal{L}(q,\theta)
= \sum_{i=1}^n \mathcal{L}_i(q_i,\theta)
$$

is an evidence lower bound on $\ell(\theta)$.

## 3. Separate expected log joint and entropy

Expand the bound:

$$
\mathcal{L}(q,\theta)
= \sum_{i=1}^n \sum_{z_i} q_i(z_i)\log p(x_i,z_i\mid\theta)
 - \sum_{i=1}^n \sum_{z_i} q_i(z_i)\log q_i(z_i).
$$

Therefore,

$$
\mathcal{L}(q,\theta)
= \sum_{i=1}^n \mathbb{E}_{q_i}[\log p(x_i,z_i\mid\theta)]
  + \sum_{i=1}^n H(q_i),
$$

where

$$
H(q_i) = -\mathbb{E}_{q_i}[\log q_i(z_i)].
$$

The second term does not depend on $\theta$.
So maximizing the ELBO with respect to $\theta$ only requires maximizing the expected complete-data log-likelihood.

## 4. Derive the KL decomposition

Start from the posterior

$$
p(z_i\mid x_i,\theta)
= \frac{p(x_i,z_i\mid\theta)}{p(x_i\mid\theta)}.
$$

Take logarithms:

$$
\log p(x_i\mid\theta)
= \log p(x_i,z_i\mid\theta) - \log p(z_i\mid x_i,\theta).
$$

Multiply both sides by $q_i(z_i)$ and sum over $z_i$:

$$
\log p(x_i\mid\theta)
= \sum_{z_i} q_i(z_i)\log p(x_i,z_i\mid\theta)
 - \sum_{z_i} q_i(z_i)\log p(z_i\mid x_i,\theta).
$$

Add and subtract $\sum_{z_i} q_i(z_i)\log q_i(z_i)$:

$$
\log p(x_i\mid\theta)
= \left[
\sum_{z_i} q_i(z_i)\log p(x_i,z_i\mid\theta)
 - \sum_{z_i} q_i(z_i)\log q_i(z_i)
\right]
$$

$$
\qquad
+ \left[
\sum_{z_i} q_i(z_i)\log q_i(z_i)
 - \sum_{z_i} q_i(z_i)\log p(z_i\mid x_i,\theta)
\right].
$$

Recognize the first bracket as $\mathcal{L}_i(q_i,\theta)$ and the second as a KL divergence:

$$
\log p(x_i\mid\theta)
= \mathcal{L}_i(q_i,\theta)
 + D_{\mathrm{KL}}\!\left(q_i(z_i)\,\|\,p(z_i\mid x_i,\theta)\right).
$$

Summing over $i$ yields

$$
\ell(\theta)
= \mathcal{L}(q,\theta)
 + \sum_{i=1}^n D_{\mathrm{KL}}\!\left(q_i(z_i)\,\|\,p(z_i\mid x_i,\theta)\right).
$$

Because KL divergence is nonnegative, $\mathcal{L}(q,\theta)$ is indeed a lower bound.

## 5. Exact maximization over q: the E-step

For fixed $\theta^{(t)}$, the ELBO is maximized over $q$ by minimizing the KL term.
The minimum KL value is zero, achieved when

$$
q_i^{(t+1)}(z_i) = p(z_i\mid x_i,\theta^{(t)}).
$$

This is the E-step.

At this choice,

$$
\ell(\theta^{(t)})
= \mathcal{L}(q^{(t+1)},\theta^{(t)}).
$$

So the bound is tight at the current parameter value.

## 6. Maximization over theta: the M-step

With $q^{(t+1)}$ fixed, maximize the ELBO over $\theta$:

$$
\theta^{(t+1)}
= \arg\max_\theta \mathcal{L}(q^{(t+1)},\theta).
$$

Since the entropy term does not depend on $\theta$,

$$
\theta^{(t+1)}
= \arg\max_\theta \sum_{i=1}^n
\mathbb{E}_{q_i^{(t+1)}}[\log p(x_i,z_i\mid\theta)].
$$

Using the E-step identity, this becomes

$$
\theta^{(t+1)}
= \arg\max_\theta Q(\theta \mid \theta^{(t)}),
$$

where

$$
Q(\theta \mid \theta^{(t)})
= \sum_{i=1}^n
\mathbb{E}_{p(z_i\mid x_i,\theta^{(t)})}
[\log p(x_i,z_i\mid\theta)].
$$

This is the usual M-step form.

## 7. Monotonic improvement proof

We now show that the observed-data likelihood does not decrease.

First, by the M-step,

$$
\mathcal{L}(q^{(t+1)},\theta^{(t+1)})
\geq
\mathcal{L}(q^{(t+1)},\theta^{(t)}).
$$

Second, by E-step tightness,

$$
\mathcal{L}(q^{(t+1)},\theta^{(t)})
= \ell(\theta^{(t)}).
$$

Third, because $\ell(\theta)$ upper-bounds the ELBO for any fixed $q$,

$$
\ell(\theta^{(t+1)})
\geq
\mathcal{L}(q^{(t+1)},\theta^{(t+1)}).
$$

Combining the three inequalities,

$$
\ell(\theta^{(t+1)})
\geq
\mathcal{L}(q^{(t+1)},\theta^{(t+1)})
\geq
\mathcal{L}(q^{(t+1)},\theta^{(t)})
= \ell(\theta^{(t)}).
$$

Hence EM monotonically increases the observed-data likelihood.

## 8. Interpretation

The derivation shows exactly where EM sits in the broader inference landscape:

- it is an ELBO method;
- its E-step is exact because the auxiliary family is unrestricted enough to match the true latent posterior;
- its M-step optimizes expected complete-data log-likelihood rather than the intractable marginal log-likelihood directly; and
- it performs coordinate ascent on $(q,\theta)$.

Variational inference generalizes this picture by restricting $q$ to a tractable family when the true posterior is itself intractable.

## Summary

EM follows from a Jensen lower bound on the observed-data log-likelihood.
The E-step makes the bound tight by setting the auxiliary distribution equal to the current latent posterior, and the M-step maximizes the resulting expected complete-data objective.
That is why EM is best understood as a special ELBO algorithm, not merely as an ad hoc alternating procedure.
