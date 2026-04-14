---
title: "Bayesian Inference"
module: "05-probabilistic-modeling"
lesson: "bayesian-inference"
doc_type: "notes"
topic: "bayesian-inference"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability"
  - "00-math-toolkit/information-theory"
  - "02-statistical-learning/statistical-learning-foundations"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "probabilistic-modeling"
  - "bayesian-inference"
  - "posterior"
  - "map"
  - "latent-variables"
---

## Purpose

These notes introduce the Bayesian viewpoint for machine learning.
The goal is to make four ideas operational:

- inference updates uncertainty rather than producing only point estimates;
- priors, likelihoods, and posteriors play different roles;
- latent-variable models extend this viewpoint by treating some quantities as unobserved; and
- exact Bayesian inference is often intractable, which motivates approximation methods used later in variational inference and deep generative modeling.

## Learning objectives

After working through this note, you should be able to:

- state Bayes' rule and explain the roles of prior, likelihood, evidence, and posterior;
- distinguish MLE, MAP estimation, and posterior inference;
- explain posterior predictive distributions and why they support uncertainty quantification;
- describe latent-variable models and give common ML examples;
- explain conjugacy at a high level; and
- identify when exact inference is feasible and when approximate inference is needed.

## 1. Bayesian modeling as uncertainty updating

In frequentist estimation, parameters are usually treated as fixed unknown values.
In Bayesian modeling, unknown parameters are modeled as random variables with a prior distribution.
Observed data update that prior into a posterior.

Let $\theta$ denote parameters and let $x_{1:n} = (x_1,\dots,x_n)$ denote observed data.
Bayes' rule gives

$$
p(\theta \mid x_{1:n})
= \frac{p(x_{1:n} \mid \theta)\,p(\theta)}{p(x_{1:n})},
$$

where

$$
p(x_{1:n}) = \int p(x_{1:n} \mid \theta)\,p(\theta)\,d\theta
$$

is the marginal likelihood or evidence.

The pieces have different meanings:

- $p(\theta)$ is the prior, encoding uncertainty before seeing the current dataset;
- $p(x_{1:n}\mid\theta)$ is the likelihood, describing how parameters generate data;
- $p(\theta\mid x_{1:n})$ is the posterior, the updated uncertainty; and
- $p(x_{1:n})$ normalizes the posterior and measures how well the model explains the data overall.

## 2. Likelihood, posterior mode, and full posterior

Three inferential objects are often confused.

### Maximum likelihood estimation

MLE chooses

$$
\hat{\theta}_{\mathrm{MLE}}
= \arg\max_\theta p(x_{1:n}\mid \theta).
$$

It uses the data model but ignores any prior uncertainty.

### Maximum a posteriori estimation

MAP chooses

$$
\hat{\theta}_{\mathrm{MAP}}
= \arg\max_\theta p(\theta \mid x_{1:n})
= \arg\max_\theta p(x_{1:n}\mid\theta)p(\theta).
$$

MAP is a point estimate, not full Bayesian inference.
It adds prior regularization but still collapses uncertainty to one value.

### Full posterior inference

Full Bayesian inference keeps the whole posterior distribution $p(\theta\mid x_{1:n})$.
This allows interval estimates, posterior predictive distributions, and propagation of parameter uncertainty into downstream decisions.

> **Remark.** Many regularized estimators can be interpreted as MAP estimators.
> Ridge regression corresponds to a Gaussian prior on coefficients, while lasso corresponds to a Laplace prior.

## 3. Posterior predictive distributions

For a new observation $x_\star$, Bayesian prediction averages over parameter uncertainty:

$$
p(x_\star \mid x_{1:n})
= \int p(x_\star \mid \theta)\,p(\theta \mid x_{1:n})\,d\theta.
$$

This is the posterior predictive distribution.
It differs from plug-in prediction, which would replace $\theta$ by a single estimate such as $\hat{\theta}_{\mathrm{MLE}}$ or $\hat{\theta}_{\mathrm{MAP}}$.

The posterior predictive distribution is often preferable when uncertainty matters:

- small-data settings;
- safety-critical decisions;
- active learning and experiment design; and
- any setting where confidence calibration matters.

## 4. A worked Bernoulli-Beta example

Suppose $Y_i \in \{0,1\}$ are iid Bernoulli with unknown success probability $\theta$:

$$
p(y_{1:n}\mid \theta)
= \prod_{i=1}^n \theta^{y_i}(1-\theta)^{1-y_i}
= \theta^{s}(1-\theta)^{n-s},
$$

where $s=\sum_{i=1}^n y_i$.

Choose a Beta prior

$$
\theta \sim \mathrm{Beta}(\alpha,\beta),
\qquad
p(\theta) \propto \theta^{\alpha-1}(1-\theta)^{\beta-1}.
$$

Then the posterior is

$$
p(\theta\mid y_{1:n})
\propto \theta^{s+\alpha-1}(1-\theta)^{n-s+\beta-1},
$$

so

$$
\theta \mid y_{1:n} \sim \mathrm{Beta}(\alpha+s,\beta+n-s).
$$

This is a conjugate update: prior and posterior stay in the same family.
The posterior mean is

$$
\mathbb{E}[\theta \mid y_{1:n}]
= \frac{\alpha+s}{\alpha+\beta+n}.
$$

The formula shows how prior pseudo-counts and observed counts combine.

## 5. Conjugacy and why it matters

A prior is conjugate to a likelihood family when the posterior belongs to the same distributional family as the prior.
Conjugacy matters because it gives exact updates in closed form.

Classic examples include:

- Beta-Bernoulli or Beta-Binomial;
- Dirichlet-Categorical or Dirichlet-Multinomial; and
- Gaussian-Gaussian models with known variance.

Conjugacy is pedagogically important, but modern ML often operates outside conjugate settings.
Neural latent-variable models, large graphical models, and hierarchical Bayesian models usually require numerical approximation.

## 6. Latent variables

A latent variable is an unobserved random variable introduced to explain observable structure.
If $x$ is observed and $z$ is latent, a common model factorization is

$$
p(x,z) = p(x\mid z)\,p(z).
$$

The latent variable may represent:

- cluster identity in a Gaussian mixture model;
- topic proportions or topic assignments in a topic model;
- hidden state in a hidden Markov model; or
- low-dimensional representation in a variational autoencoder.

Latent variables make models more expressive, but they also make inference harder because the posterior

$$
p(z,\theta \mid x)
$$

usually requires marginalization over unobserved quantities.

## 7. Hierarchical models and partial pooling

Bayesian models can place priors on parameters and hyperpriors on prior parameters.
This leads to hierarchical models such as

$$
\theta_j \sim p(\theta_j \mid \eta),
\qquad
\eta \sim p(\eta).
$$

Hierarchical structure is useful when multiple related groups share statistical strength.
In ML terms, it supports partial pooling: each group gets its own parameters, but those parameters are coupled through shared higher-level uncertainty.

This idea is closely related to multi-task learning and structured regularization.

## 8. Exact versus approximate inference

The central computational task in Bayesian modeling is usually one of these:

- compute the posterior $p(\theta \mid x)$;
- compute a marginal likelihood $p(x)$;
- compute a posterior predictive distribution $p(x_\star \mid x)$; or
- compute posterior marginals such as $p(z_i \mid x)$.

### Exact inference

Exact inference is available when these quantities can be computed analytically or by finite exact elimination.
Common cases:

- conjugate models;
- small discrete graphical models; and
- Gaussian models with tractable linear algebra.

### Approximate inference

Approximate inference is needed when posterior normalization or marginalization is computationally prohibitive.
Common approaches:

- Laplace approximation;
- expectation-maximization for maximum-likelihood or MAP estimation with latent variables;
- variational inference; and
- Markov chain Monte Carlo.

> **Scope note.** EM is not full Bayesian posterior inference.
> It is an optimization procedure for latent-variable likelihoods or MAP objectives.
> It becomes relevant here because it uses the same conditional-expectation structure that later appears in variational inference and the ELBO.

## 9. ML examples that motivate the module

### Naive Bayes

Naive Bayes is a probabilistic classifier that models

$$
p(y)\prod_{j=1}^d p(x_j \mid y).
$$

It uses strong conditional-independence assumptions, but it is simple, fast, and often surprisingly competitive on text-style sparse features.

### Gaussian mixture models

GMMs introduce a latent component variable $Z$ and model

$$
p(x) = \sum_{k=1}^K \pi_k\,\mathcal{N}(x \mid \mu_k,\Sigma_k).
$$

The latent assignments are not observed, so learning naturally leads to EM.

### Variational autoencoders

VAEs use continuous latent variables and learn both a generative model $p_\theta(x\mid z)$ and an approximate posterior $q_\phi(z\mid x)$.
Their training objective is an ELBO, so they should be understood as a scalable descendant of the latent-variable viewpoint introduced here.

## 10. Limits and modeling discipline

Probabilistic models are not automatically better than deterministic ones.
They can be misspecified, badly calibrated, or computationally expensive.
Three questions should be asked explicitly:

1. What random variables are being modeled?
2. What conditional independence or distributional assumptions are being imposed?
3. Which uncertainty is represented: data noise, parameter uncertainty, latent uncertainty, or all three?

Clear answers to those questions make later probabilistic deep-learning material much easier to interpret.

## Summary

Bayesian inference treats learning as uncertainty updating.
The posterior combines prior information and likelihood information, and the posterior predictive distribution propagates that uncertainty into future predictions.
Latent-variable models enrich expressivity by introducing hidden explanatory structure, but they turn inference into a marginalization problem.
That computational difficulty is the bridge to EM, graphical models, and variational methods.
