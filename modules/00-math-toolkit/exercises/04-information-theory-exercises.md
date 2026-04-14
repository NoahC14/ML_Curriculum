---
title: "Information Theory Exercises"
module: "00-math-toolkit"
lesson: "information-theory-exercises"
doc_type: "exercise"
topic: "information-theory"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability-statistics"
  - "00-math-toolkit/information-theory"
updated: "2026-04-09"
owner: "curriculum-team"
tags:
  - "information-theory"
  - "entropy"
  - "kl-divergence"
  - "cross-entropy"
  - "mutual-information"
---

## Purpose

These exercises emphasize the information-theoretic quantities that appear repeatedly in machine learning objectives. The progression is:

- Tier 1: basic computation and identity checking;
- Tier 2: derivation and interpretation; and
- Tier 3: explicit ML connections.

## Exercise 1: Entropy of a Bernoulli variable

> **Problem.** Let $X \sim \mathrm{Bernoulli}(p)$.
> Derive
>
> $$
> H(X) = -p\log p - (1-p)\log(1-p).
> $$
>
> Then show that $H(X)$ is maximized at $p = 1/2$.

**Hints**

- Write the entropy sum over the two possible outcomes.
- Differentiate with respect to $p$.

**Deliverables**

- The closed-form entropy expression.
- A short argument for the maximizing value of $p$.

## Exercise 2: Cross-entropy decomposition

> **Problem.** Let $P$ and $Q$ be discrete distributions on the same support. Starting from the definitions, prove that
>
> $$
> H(P,Q) = H(P) + D_{\mathrm{KL}}(P \| Q).
> $$

**Hints**

- Expand the KL divergence.
- Group the terms containing $\log p(x)$ and $\log q(x)$.

**Deliverables**

- A derivation.
- One sentence explaining why minimizing cross-entropy with respect to $Q$ is equivalent to minimizing KL divergence from $P$ to $Q$.

## Exercise 3: Classification loss as empirical cross-entropy

> **Problem.** Consider a multiclass classifier that outputs predicted probabilities $\hat{\mathbf{p}}^{(i)} \in \mathbb{R}^K$ for examples $i=1,\dots,n$. Let $\mathbf{y}^{(i)}$ be the one-hot encoded labels. Show that the average negative log-likelihood
>
> $$
> -\frac{1}{n}\sum_{i=1}^n \sum_{k=1}^K y_k^{(i)} \log \hat{p}^{(i)}_k
> $$
>
> is the empirical cross-entropy loss.

**Hints**

- Use the fact that exactly one entry of each one-hot label vector is equal to $1$.
- Interpret the average over the sample as an empirical expectation.

**Deliverables**

- A short derivation.
- A brief explanation of why confident wrong predictions are penalized strongly.

## Exercise 4: Nonnegativity of KL divergence

> **Problem.** Prove that $D_{\mathrm{KL}}(P\|Q) \geq 0$ for discrete distributions $P$ and $Q$, with equality if and only if $P=Q$ on the support of $P$.

**Hints**

- Use Jensen's inequality for the concave logarithm, or cite Gibbs' inequality and explain how it applies.

**Deliverables**

- A proof or proof sketch.

## Exercise 5: KL divergence between Bernoulli distributions

> **Problem.** Let $P=\mathrm{Bernoulli}(p)$ and $Q=\mathrm{Bernoulli}(q)$.
> Derive
>
> $$
> D_{\mathrm{KL}}(P\|Q)
> = p\log \frac{p}{q} + (1-p)\log \frac{1-p}{1-q}.
> $$
>
> Evaluate the divergence when $(p,q)=(0.8,0.5)$ and when $(p,q)=(0.8,0.8)$.

**Hints**

- Write the sum over the outcomes $0$ and $1$.
- The second case should expose the equality condition.

**Deliverables**

- The formula.
- The two numerical values.

## Exercise 6: Mutual information as KL divergence

> **Problem.** Prove that
>
> $$
> I(X;Y) = D_{\mathrm{KL}}(p(x,y)\|p(x)p(y)).
> $$
>
> Explain why this implies that mutual information is nonnegative and vanishes exactly when $X$ and $Y$ are independent.

**Hints**

- Compare the definition of KL divergence with the definition of mutual information.

**Deliverables**

- A derivation.
- A one-paragraph interpretation.

## Exercise 7: A binary symmetric channel

> **Problem.** Let $X \sim \mathrm{Bernoulli}(1/2)$ and let $Y$ be obtained by flipping $X$ independently with probability $\varepsilon \in [0,1/2]$.
>
> 1. Compute $H(X)$.
> 2. Compute $H(X \mid Y)$.
> 3. Deduce $I(X;Y)$.
> 4. Describe what happens as $\varepsilon \to 0$ and as $\varepsilon \to 1/2$.

**Hints**

- Symmetry implies that the posterior uncertainty after observing $Y$ matches the flip entropy.
- Use $I(X;Y)=H(X)-H(X \mid Y)$.

**Deliverables**

- The closed-form expression for $I(X;Y)$.
- A short interpretation of the noise parameter.

## Exercise 8: Data processing intuition in a feature pipeline

> **Problem.** Suppose a raw input $X$ is mapped to hand-designed features $Z=g(X)$ and then to a classifier output $\hat{Y}=h(Z)$. Give a clear argument for why the data processing inequality suggests
>
> $$
> I(X;\hat{Y}) \leq I(X;Z).
> $$
>
> Then describe one practical risk this creates when feature engineering is too aggressive.

**Hints**

- Treat the pipeline as a Markov chain $X \to Z \to \hat{Y}$.
- Focus on what information can be lost at the feature step.

**Deliverables**

- A short conceptual argument.
- One concrete ML example.

## Exercise 9: KL divergence in a VAE objective

> **Problem.** Consider the VAE evidence lower bound
>
> $$
> \mathcal{L}_{\mathrm{ELBO}}(x)
> = \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]
> - D_{\mathrm{KL}}(q_\phi(z \mid x)\|p(z)).
> $$
>
> Explain what would likely happen if the KL term were removed entirely. Then explain what can go wrong if the KL term dominates too strongly.

**Hints**

- Think about the geometry of the latent space in both extremes.
- Relate the KL term to regularization toward the prior.

**Deliverables**

- A short written analysis.

## Exercise 10: Differential entropy of a Gaussian

> **Problem.** Let $X \sim \mathcal{N}(\mu,\sigma^2)$.
> Show that
>
> $$
> h(X) = \frac{1}{2}\log(2\pi e \sigma^2).
> $$
>
> Explain why the result depends on $\sigma^2$ but not on $\mu$.

**Hints**

- Substitute the Gaussian density into the definition of differential entropy.
- Use $\mathbb{E}[(X-\mu)^2]=\sigma^2$.

**Deliverables**

- A derivation.
- A short interpretation.
