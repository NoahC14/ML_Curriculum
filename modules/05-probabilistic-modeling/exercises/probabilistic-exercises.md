---
title: "Probabilistic Modeling Exercises"
module: "05-probabilistic-modeling"
lesson: "probabilistic-exercises"
doc_type: "exercise"
topic: "probabilistic-modeling"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability"
  - "00-math-toolkit/information-theory"
  - "05-probabilistic-modeling/bayesian-inference"
  - "05-probabilistic-modeling/em-algorithm"
  - "05-probabilistic-modeling/graphical-models"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "probabilistic-modeling"
  - "bayesian-inference"
  - "em"
  - "graphical-models"
  - "latent-variables"
---

## Purpose

These exercises reinforce the core ideas of probabilistic modeling: Bayesian updating, latent variables, EM, and graphical-model inference.

## Exercise 1. Likelihood, MAP, and posterior prediction

**Taxonomy**

- `difficulty`: `intermediate`
- `type`: `derivation`
- `tags`: `likelihood`, `map-estimation`, `posterior-predictive`

Suppose $Y_1,\dots,Y_n$ are iid Bernoulli with unknown parameter $\theta$.

1. Write the likelihood $p(y_{1:n}\mid\theta)$.
2. Assume a $\mathrm{Beta}(\alpha,\beta)$ prior. Write the posterior distribution.
3. Write the MAP estimator when $\alpha>1$ and $\beta>1$.
4. Write the posterior predictive probability that $Y_{n+1}=1$.
5. Explain in words how the posterior predictive differs from plugging the posterior mean into a Bernoulli model.

## Exercise 2. Conjugacy and pseudo-count interpretation

For the Beta-Bernoulli model:

1. Show that observing one success increments the first Beta parameter by one.
2. Show that observing one failure increments the second Beta parameter by one.
3. Interpret $\alpha-1$ and $\beta-1$ as prior pseudo-counts.
4. Give one situation where a strong prior would be reasonable and one where it would be dangerous.

## Exercise 3. Latent-variable factorization

Consider a Gaussian mixture model with latent component variable $Z \in \{1,\dots,K\}$ and observation $X \in \mathbb{R}^d$.

1. Write the complete-data joint distribution $p(x,z)$.
2. Marginalize out $Z$ to obtain $p(x)$.
3. Explain why maximizing $\log p(x)$ directly is harder than maximizing $\log p(x,z)$ with known $z$.
4. State what the latent variable represents in practical clustering terms.

## Exercise 4. ELBO identity for EM

Let $q(z)$ be any distribution over the latent variable $z$ for one observation $x$.

1. Show that
   $$
   \log p(x\mid\theta)
   =
   \mathcal{L}(q,\theta)
   + D_{\mathrm{KL}}(q(z)\,\|\,p(z\mid x,\theta)).
   $$
2. State why this implies $\mathcal{L}(q,\theta)\leq \log p(x\mid\theta)$.
3. State which choice of $q$ makes the lower bound tight.
4. Explain why EM can be interpreted as coordinate ascent.

## Exercise 5. Responsibilities in a two-component mixture

Suppose a one-dimensional mixture has two components with weights $\pi_1,\pi_2$, means $\mu_1,\mu_2$, and common known variance $\sigma^2$.

1. Write the responsibility $\gamma_{i1}=p(z_i=1\mid x_i,\theta)$.
2. Express $\gamma_{i2}$ in terms of $\gamma_{i1}$.
3. Describe what happens to the responsibilities when $x_i$ is much closer to $\mu_1$ than to $\mu_2$.
4. Describe what happens when the two component densities overlap strongly.

## Exercise 6. EM versus K-means

Answer each prompt in two or three sentences.

1. What does K-means optimize?
2. What does GMM-EM optimize?
3. What is the difference between hard assignments and soft assignments?
4. In what limiting sense can K-means be viewed as related to a Gaussian mixture model?

## Exercise 7. Directed graphical models

Consider the naive Bayes factorization

$$
p(y,x_1,\dots,x_d) = p(y)\prod_{j=1}^d p(x_j\mid y).
$$

1. Draw the directed graph in words or diagram form.
2. State the conditional independence assumption encoded by the graph.
3. Explain why this factorization is computationally attractive for classification.
4. Give one domain where the assumption is unrealistic but still useful.

## Exercise 8. Hidden Markov model inference

For an HMM with latent states $Z_{1:T}$ and observations $X_{1:T}$:

1. Write the joint distribution factorization.
2. Name one exact inference problem solved by the forward-backward algorithm.
3. Name one exact inference problem solved by the Viterbi algorithm.
4. Explain why dynamic programming is possible in this model.

## Exercise 9. Undirected graphical models and partition functions

For an undirected graphical model with clique potentials $\psi_c(x_c)$:

1. Write the normalized joint distribution.
2. Define the partition function $Z$.
3. Explain why computing $Z$ can be difficult.
4. State one modeling situation where an undirected model is more natural than a directed one.

## Exercise 10. Exact versus approximate inference

For each scenario below, state whether exact inference is typically feasible, sometimes feasible, or usually intractable, and explain briefly why.

1. Naive Bayes classification with discrete features.
2. Marginal inference in a small tree-structured Bayesian network.
3. Marginal inference in a large loopy Markov random field.
4. Posterior inference in a variational autoencoder.

## Exercise 11. Short proof exercise

Give a concise proof sketch that each EM iteration does not decrease the observed-data likelihood.
Your answer should explicitly reference:

1. E-step tightness of the ELBO;
2. M-step maximization of the ELBO; and
3. nonnegativity of KL divergence.

## Exercise 12. Modeling reflection

Choose one of the following models: naive Bayes, GMM, HMM, or a VAE.
Write a short paragraph answering:

1. What are the observed variables?
2. What are the latent variables or uncertain parameters?
3. Which conditional independence assumptions are essential?
4. Which inference problem is central?
5. What is one likely failure mode if the modeling assumptions are wrong?
