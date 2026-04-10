---
title: "Information Theory Primer for Machine Learning"
module: "00-math-toolkit"
lesson: "information-theory-primer"
doc_type: "notes"
topic: "information-theory"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability"
updated: "2026-04-09"
owner: "curriculum-team"
tags:
  - "information-theory"
  - "entropy"
  - "kl-divergence"
  - "mutual-information"
  - "cross-entropy"
---

## Motivation

Information-theoretic quantities appear throughout machine learning even when the model is not presented as an information theory model. Classification losses are cross-entropies, variational inference introduces KL divergence, representation learning objectives often control mutual information, and generative modeling repeatedly compares distributions rather than single point predictions.

This note keeps the scope practical. The goal is not to build a full communications-theory course. The goal is to develop the small collection of quantities that repeatedly appear in ML objectives:

- entropy as uncertainty or coding cost under an optimal code;
- cross-entropy as expected log loss under a candidate predictive distribution;
- KL divergence as a directed discrepancy between distributions;
- mutual information as the amount of dependence between variables; and
- the data processing inequality as a structural warning that post-processing cannot create information about an upstream source.

## Assumptions and Notation

Let $X$ be a discrete random variable with support $\mathcal{X}$ and probability mass function $p(x) = \mathbb{P}(X = x)$. Let $Y$ be another random variable, with joint distribution $p(x, y)$ and marginals $p(x)$ and $p(y)$.

For continuous random variables with density $p(x)$ on $\mathbb{R}^d$, integrals replace sums. We use $\log$ for the natural logarithm, so information is measured in nats. If base-2 logarithms are used instead, the units are bits.

When we compare two distributions $P$ and $Q$ on the same space, we write $p$ and $q$ for their densities or mass functions. We assume $q(x) > 0$ whenever $p(x) > 0$ if a KL divergence is to be finite.

## Shannon Entropy

> **Definition.** The Shannon entropy of a discrete random variable $X$ with mass function $p$ is
>
> $$
> H(X) = -\sum_{x \in \mathcal{X}} p(x)\log p(x).
> $$

Entropy is the expected surprisal $-\log p(X)$. Rare events carry larger surprisal, and entropy averages that quantity under the true distribution.

Two interpretations are worth keeping in mind:

- **uncertainty interpretation:** larger entropy means the outcome is harder to predict in advance;
- **coding interpretation:** entropy is the lower bound on the average code length achievable by an optimal code, up to standard integer-length caveats.

> **Example.** If $X \sim \mathrm{Bernoulli}(p)$, then
>
> $$
> H(X) = -p\log p - (1-p)\log(1-p).
> $$
>
> This quantity is maximized at $p = 1/2$, where the two outcomes are equally likely.

> **Remark.** Entropy depends on the distribution, not on the labels of the outcomes. A fair coin and a fair two-class label have the same entropy.

### Why Entropy Matters in ML

High-entropy predictive targets are intrinsically less predictable than low-entropy ones. In classification, label noise raises irreducible uncertainty. In generative modeling, entropy helps distinguish sharp distributions from diffuse ones. In reinforcement learning, entropy regularization encourages policies that remain exploratory rather than collapsing too early.

## Differential Entropy

For a continuous random variable $X$ with density $p$ on $\mathbb{R}^d$, the analogous quantity is

> **Definition.** The differential entropy of $X$ is
>
> $$
> h(X) = -\int_{\mathbb{R}^d} p(x)\log p(x)\,dx,
> $$
>
> provided the integral exists.

Differential entropy behaves differently from discrete entropy.

> **Warning.** Differential entropy is not the entropy of a discretized variable in the naive sense. It can be negative, and it is not invariant under arbitrary smooth reparameterizations.

For a Gaussian random variable $X \sim \mathcal{N}(\mu, \sigma^2)$,

$$
h(X) = \frac{1}{2}\log(2\pi e \sigma^2).
$$

So increasing variance increases differential entropy. This matches the intuition that a wider Gaussian is more spread out, but the caveat above matters: unlike KL divergence or mutual information, differential entropy itself is not always the safest quantity to interpret geometrically.

## Cross-Entropy

Suppose the true distribution is $P$ with mass function $p$, but a model predicts $Q$ with mass function $q$.

> **Definition.** The cross-entropy from $P$ to $Q$ is
>
> $$
> H(P, Q) = -\sum_{x \in \mathcal{X}} p(x)\log q(x).
> $$

Cross-entropy is the expected log loss incurred when samples come from $P$ but we score them using $Q$.

It decomposes as

$$
H(P, Q) = H(P) + D_{\mathrm{KL}}(P \| Q),
$$

so minimizing cross-entropy with respect to $Q$ is equivalent to minimizing KL divergence from the true distribution to the model distribution.

### Cross-Entropy and Classification Loss

In multiclass classification, a model outputs probabilities $\hat{\mathbf{p}} = (\hat{p}_1, \ldots, \hat{p}_K)$ over $K$ classes. If the true label is encoded as a one-hot vector $\mathbf{y}$, the per-example cross-entropy loss is

$$
\ell(\mathbf{y}, \hat{\mathbf{p}})
= -\sum_{k=1}^K y_k \log \hat{p}_k.
$$

Because a one-hot label has exactly one coordinate equal to $1$, this reduces to

$$
\ell(\mathbf{y}, \hat{\mathbf{p}}) = -\log \hat{p}_{k^\star},
$$

where $k^\star$ is the correct class.

> **Example.** If the correct class has predicted probability $0.9$, the loss is $-\log 0.9$. If it has predicted probability $0.01$, the loss is much larger. Cross-entropy therefore punishes confident mistakes sharply.

This is why softmax classification is usually trained by negative log-likelihood: it is exactly empirical cross-entropy minimization.

## KL Divergence

> **Definition.** The Kullback-Leibler divergence from $P$ to $Q$ is
>
> $$
> D_{\mathrm{KL}}(P \| Q)
> = \sum_{x \in \mathcal{X}} p(x)\log \frac{p(x)}{q(x)}.
> $$

For continuous densities,

$$
D_{\mathrm{KL}}(P \| Q)
= \int p(x)\log \frac{p(x)}{q(x)}\,dx.
$$

KL divergence is not a metric:

- it is not symmetric;
- it does not satisfy the triangle inequality; and
- it can be infinite when $Q$ assigns zero probability to an event that $P$ allows.

Still, it is fundamental because it quantifies how much extra log loss is paid when using $Q$ instead of the true distribution $P$.

> **Proposition.** $D_{\mathrm{KL}}(P \| Q) \geq 0$, with equality if and only if $P = Q$ almost everywhere.

> **Proof Sketch.** Apply Jensen's inequality to the concave logarithm, or equivalently use Gibbs' inequality.

### Worked Example: Bernoulli KL Divergence

If $P = \mathrm{Bernoulli}(p)$ and $Q = \mathrm{Bernoulli}(q)$, then

$$
D_{\mathrm{KL}}(P \| Q)
= p\log \frac{p}{q} + (1-p)\log \frac{1-p}{1-q}.
$$

This quantity is zero at $p=q$ and grows as the two Bernoulli parameters separate.

### KL Divergence in Variational Autoencoders and Regularization

Variational autoencoders optimize an evidence lower bound of the form

$$
\mathcal{L}_{\mathrm{ELBO}}(x)
= \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]
 - D_{\mathrm{KL}}\!\left(q_\phi(z \mid x)\,\|\,p(z)\right).
$$

The first term encourages good reconstruction or likelihood under the decoder. The KL term encourages the approximate posterior $q_\phi(z \mid x)$ to stay close to the prior $p(z)$.

This KL term acts as a regularizer with probabilistic meaning:

- it discourages latent codes from drifting too far from the prior;
- it promotes a latent space that can be sampled from globally; and
- it balances fidelity against compression.

Related KL penalties appear in many settings beyond VAEs, including trust-region policy updates, distillation, posterior regularization, and distribution matching.

## Mutual Information

> **Definition.** The mutual information between random variables $X$ and $Y$ is
>
> $$
> I(X;Y)
> = \sum_{x,y} p(x,y)\log \frac{p(x,y)}{p(x)p(y)}.
> $$

Equivalent forms are

$$
I(X;Y) = D_{\mathrm{KL}}\!\left(p(x,y)\,\|\,p(x)p(y)\right),
$$

$$
I(X;Y) = H(X) - H(X \mid Y),
$$

and symmetrically

$$
I(X;Y) = H(Y) - H(Y \mid X).
$$

So mutual information measures how much knowing one variable reduces uncertainty about the other.

> **Remark.** Since it is a KL divergence between the joint distribution and the product of marginals, mutual information is always nonnegative and equals zero exactly when $X$ and $Y$ are independent.

### Worked Example: Deterministic Copy

If $Y=X$ almost surely, then $H(X \mid Y)=0$, so

$$
I(X;Y) = H(X).
$$

A perfect copy carries all the information in the original variable.

### Mutual Information in Representation Learning

Representation learning often asks for a representation $Z = f_\theta(X)$ that preserves task-relevant information while discarding nuisance variation. Mutual information gives one way to express this goal:

- maximize $I(Z;Y)$ when $Y$ is a target or supervisory signal;
- sometimes constrain or reduce $I(Z;X)$ to encourage compression;
- compare different views of the same data by increasing agreement-related lower bounds on mutual information.

Contrastive learning objectives, the information bottleneck viewpoint, and several self-supervised methods can all be read through this lens, even when the implemented objective is a tractable surrogate rather than an exact mutual information computation.

## Data Processing Inequality

Suppose random variables form a Markov chain

$$
X \to Z \to Y,
$$

meaning that $Y$ depends on $X$ only through $Z$. Then:

> **Proposition.** If $X \to Z \to Y$ is a Markov chain, then
>
> $$
> I(X;Y) \leq I(X;Z).
> $$

This is the data processing inequality.

### Intuition

Once information has been discarded by passing from $X$ to $Z$, later processing cannot reconstruct what was lost unless extra side information is introduced. A deterministic feature map, a noisy channel, or a compression step may preserve some information about $X$, but no downstream transformation can create new information about the original source out of nothing.

For ML, this gives a useful design intuition:

- every representation trades off preservation against compression;
- lossy preprocessing can permanently remove task-relevant signal; and
- deeper pipelines do not guarantee richer information about the input, only different transformations of what has already been retained.

## Summary of Core Identities

The small set of identities below covers a large fraction of ML usage:

$$
H(X) = -\sum_x p(x)\log p(x),
$$

$$
H(P,Q) = H(P) + D_{\mathrm{KL}}(P \| Q),
$$

$$
D_{\mathrm{KL}}(P \| Q) = \mathbb{E}_{X \sim P}\left[\log \frac{p(X)}{q(X)}\right],
$$

$$
I(X;Y) = D_{\mathrm{KL}}(p(x,y)\|p(x)p(y)) = H(X) - H(X \mid Y).
$$

These quantities are unified by one theme: they compare uncertainty under a true distribution, a model distribution, or a joint dependence structure. That is exactly the setting of modern machine learning.

## Scope Notes and Common Misconceptions

- Entropy is not a synonym for disorder in a vague everyday sense. In this course it is a precisely defined functional of a distribution.
- Differential entropy should be used carefully. It is not as robust an invariant as KL divergence or mutual information.
- KL divergence is directed. Minimizing $D_{\mathrm{KL}}(P\|Q)$ and minimizing $D_{\mathrm{KL}}(Q\|P)$ usually produce different behavior.
- Mutual information is conceptually useful but often hard to estimate in high dimensions, so practical objectives use approximations, lower bounds, or related surrogates.

## ML Takeaways

- Cross-entropy is the standard loss for probabilistic classification because it is negative log-likelihood under the model.
- KL divergence appears whenever the objective asks one distribution to approximate or regularize another, especially in variational inference and latent-variable models.
- Mutual information provides a language for what a representation preserves about labels, inputs, or paired views.
- The data processing inequality explains why preprocessing and latent bottlenecks must be chosen carefully: discarded information is usually gone for good.
