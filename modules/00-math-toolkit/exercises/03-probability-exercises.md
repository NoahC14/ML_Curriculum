---
title: "Probability and Statistics Exercises"
module: "00-math-toolkit"
lesson: "probability-statistics-exercises"
doc_type: "exercise"
topic: "probability-statistics"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability-statistics"
updated: "2026-04-09"
owner: "curriculum-team"
tags:
  - "probability"
  - "statistics"
  - "bayes"
  - "mle"
  - "map"
---

## Purpose

These exercises mix proof-style reasoning, calculation, and ML interpretation.

- Tier 1: definitions, core manipulations, and standard computations;
- Tier 2: derivations and probabilistic modeling arguments;
- Tier 3: ML-facing synthesis and interpretation.

## Exercise 1: PMF normalization

> **Problem.** Let $X$ take values in $\{0,1,2\}$ with
> $$
> p_X(0) = c, \qquad p_X(1) = 2c, \qquad p_X(2) = 3c.
> $$
> Find $c$, then compute $\mathbb{E}[X]$ and $\mathrm{Var}(X)$.

**Hints**

- A PMF must sum to $1$.
- Use $\mathrm{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$.

**Deliverables**

- The value of $c$.
- The expectation and variance.

## Exercise 2: Variance identity proof

> **Problem.** Prove that
> $$
> \mathrm{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2.
> $$

**Hints**

- Expand $(X-\mathbb{E}[X])^2$.
- Use linearity of expectation.

**Deliverables**

- A proof.

## Exercise 3: Covariance matrix is positive semidefinite

> **Problem.** Let $\mathbf{X} \in \mathbb{R}^d$ have covariance matrix $\boldsymbol{\Sigma}$. Prove that $\boldsymbol{\Sigma}$ is positive semidefinite.

**Hints**

- Start from $\mathbf{v}^\top \boldsymbol{\Sigma}\mathbf{v}$.
- Rewrite the expression as an expectation of a square.

**Deliverables**

- A proof.
- One sentence explaining why this matters for PCA.

## Exercise 4: Correlation and feature redundancy

> **Problem.** Suppose two centered features $X_1$ and $X_2$ have correlation $0.99$. Explain why this can cause trouble for linear regression, and describe one reason PCA might help.

**Hints**

- Connect large correlation to near-collinearity.
- Think about the conditioning of the design matrix.

**Deliverables**

- A short written explanation.

## Exercise 5: Joint, marginal, and conditional probabilities

> **Problem.** A binary label $Y \in \{0,1\}$ and binary feature $X \in \{0,1\}$ satisfy
> $$
> P(X=1,Y=1)=0.30,\quad
> P(X=1,Y=0)=0.10,\quad
> P(X=0,Y=1)=0.20,\quad
> P(X=0,Y=0)=0.40.
> $$
> Compute:
>
> 1. $P(Y=1)$,
> 2. $P(X=1)$,
> 3. $P(Y=1 \mid X=1)$,
> 4. whether $X$ and $Y$ are independent.

**Hints**

- Marginals are sums over the other variable.
- Independence requires $P(X,Y)=P(X)P(Y)$.

**Deliverables**

- All four quantities or conclusions, with brief justification.

## Exercise 6: Bayes' theorem for spam filtering

> **Problem.** In an email population, $20\%$ of messages are spam. The word "free" appears in $60\%$ of spam emails and $5\%$ of non-spam emails. Compute the posterior probability that an email is spam given that it contains the word "free".

**Hints**

- Use Bayes' theorem.
- Compute the evidence term with the law of total probability.

**Deliverables**

- The posterior probability.
- Two sentences interpreting the result in ML language.

## Exercise 7: Bernoulli MLE derivation

> **Problem.** Let $y_1,\ldots,y_n \in \{0,1\}$ be i.i.d. Bernoulli$(\theta)$ observations. Derive the MLE of $\theta$ from the log-likelihood.

**Hints**

- Write the likelihood as a product.
- Differentiate the log-likelihood and set the derivative to zero.
- Check concavity or otherwise justify that the critical point is a maximum.

**Deliverables**

- A full derivation.

## Exercise 8: Gaussian mean MLE with known variance

> **Problem.** Let $x_1,\ldots,x_n$ be i.i.d. $\mathcal{N}(\mu,\sigma^2)$ with $\sigma^2$ known. Derive the MLE of $\mu$.

**Hints**

- Ignore constants that do not depend on $\mu$.
- Differentiate with respect to $\mu$.

**Deliverables**

- A derivation.
- One sentence explaining why the result is statistically intuitive.

## Exercise 9: Gaussian variance MLE

> **Problem.** Let $x_1,\ldots,x_n$ be i.i.d. $\mathcal{N}(\mu,\sigma^2)$ with both parameters unknown. Assuming $\hat{\mu}_{\mathrm{MLE}} = \bar{x}$, derive the MLE of $\sigma^2$.

**Hints**

- Differentiate the Gaussian log-likelihood with respect to $\sigma^2$.
- Substitute $\bar{x}$ after solving for the optimizer.

**Deliverables**

- A derivation.
- A sentence comparing the denominator $n$ here to the $n-1$ denominator used in unbiased variance estimation.

## Exercise 10: MAP as regularized estimation

> **Problem.** Suppose $w \in \mathbb{R}^d$ is a scalar model parameter for a one-dimensional regression problem with Gaussian likelihood and Gaussian prior
> $$
> w \sim \mathcal{N}(0,\tau^2).
> $$
> Show that MAP estimation is equivalent to minimizing a squared-error objective plus an $\ell_2$ penalty term.

**Hints**

- Write the log-posterior as log-likelihood plus log-prior.
- Drop constants independent of $w$.

**Deliverables**

- A derivation.
- A sentence interpreting the prior as shrinkage.

## Exercise 11: Beta-Bernoulli conjugacy

> **Problem.** Let $\theta \sim \mathrm{Beta}(\alpha,\beta)$ and let $y_1,\ldots,y_n \mid \theta$ be i.i.d. Bernoulli$(\theta)$. Derive the posterior distribution of $\theta$.

**Hints**

- Multiply the likelihood by the Beta prior density.
- Collect the powers of $\theta$ and $(1-\theta)$.

**Deliverables**

- The posterior family and updated parameters.
- A short explanation of what the prior contributes when $n$ is small.

## Exercise 12: Categorical counts and Dirichlet intuition

> **Problem.** A three-class classifier sees counts $(15, 3, 2)$ in a minibatch. If the prior over class probabilities is Dirichlet$(1,1,1)$, what are the posterior parameters? Explain what this means as a smoothing rule.

**Hints**

- Add observed counts to prior pseudocounts.

**Deliverables**

- The posterior parameters.
- A short explanation.

## Exercise 13: Poisson modeling decision

> **Problem.** Give one ML-relevant example where a Poisson model is more appropriate than a Gaussian model. State the random variable, why count structure matters, and one limitation of the Poisson assumption.

**Hints**

- Think about nonnegative integer-valued targets.
- Recall that a Poisson model ties the mean and variance together.

**Deliverables**

- A short written response.

## Exercise 14: Generative versus discriminative framing

> **Problem.** In a binary classification setting, explain the difference between modeling $p(y \mid \mathbf{x})$ directly and modeling $p(\mathbf{x} \mid y)$ together with $p(y)$. State one advantage and one disadvantage of the generative approach.

**Hints**

- Connect the second approach to Bayes' theorem.
- Think about what extra structure a generative model must learn.

**Deliverables**

- A short prose comparison.

## Exercise 15: Sample covariance in matrix form

> **Problem.** Let centered data points be rows of a matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$. Show that the sample covariance matrix can be written as
> $$
> \mathbf{S} = \frac{1}{n}\mathbf{X}^\top \mathbf{X}.
> $$
> Then explain why the eigenvectors of $\mathbf{S}$ are relevant for PCA.

**Hints**

- Expand the matrix product coordinatewise.
- Connect covariance directions to explained variance.

**Deliverables**

- A derivation.
- A brief PCA interpretation.
