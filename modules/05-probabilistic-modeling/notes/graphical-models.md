---
title: "Graphical Models"
module: "05-probabilistic-modeling"
lesson: "graphical-models"
doc_type: "notes"
topic: "graphical-models"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability"
  - "05-probabilistic-modeling/bayesian-inference"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "probabilistic-modeling"
  - "graphical-models"
  - "bayesian-networks"
  - "markov-random-fields"
  - "inference"
---

## Purpose

These notes introduce graphical models as compact languages for factorizing joint distributions and reasoning about conditional independence.
The scope is intentionally limited to what later modules need: directed and undirected models, examples of each, and a clear distinction between exact and approximate inference.

## Learning objectives

After working through this note, you should be able to:

- explain how a graph encodes factorization assumptions;
- distinguish directed graphical models from undirected graphical models;
- read local conditional-independence claims from a model specification;
- describe at least two standard examples of graphical models used in ML; and
- explain why exact inference is tractable in some graphs and intractable in others.

## 1. Why graphical models matter

A full joint distribution over many variables is hard to specify directly.
Graphical models reduce complexity by exposing structure:

- which variables depend directly on which others;
- which conditional independences are assumed; and
- how the joint distribution factorizes into local pieces.

They are useful because they separate modeling questions from inference questions.
One can specify a principled joint model first, then ask how to compute marginals, conditionals, or MAP states.

## 2. Directed graphical models

A directed graphical model, or Bayesian network, is a directed acyclic graph whose nodes are random variables.
If $\mathrm{Pa}(X_i)$ denotes the parents of $X_i$, then the joint distribution factorizes as

$$
p(x_1,\dots,x_n)
= \prod_{i=1}^n p(x_i \mid \mathrm{pa}(x_i)).
$$

The graph states that each variable is conditionally independent of its non-descendants given its parents.

### Example 1: Naive Bayes

Naive Bayes has one class node $Y$ and observed feature nodes $X_1,\dots,X_d$, each with parent $Y$.
The factorization is

$$
p(y,x_1,\dots,x_d)
= p(y)\prod_{j=1}^d p(x_j \mid y).
$$

The key assumption is conditional independence of features given the class label.
This assumption is usually false exactly, but it often gives a good bias-variance tradeoff in high-dimensional classification tasks such as text categorization.

### Example 2: Hidden Markov models

An HMM uses latent states $Z_1,\dots,Z_T$ and observations $X_1,\dots,X_T$ with factorization

$$
p(z_1)\prod_{t=2}^T p(z_t \mid z_{t-1}) \prod_{t=1}^T p(x_t \mid z_t).
$$

This is a directed time-series model.
It supports exact dynamic-programming algorithms such as forward-backward for marginals and Viterbi for MAP state sequences.

## 3. Undirected graphical models

An undirected graphical model, often called a Markov random field, uses an undirected graph to represent symmetric local interactions.
The joint distribution factorizes over cliques:

$$
p(x)
= \frac{1}{Z}\prod_{c \in \mathcal{C}} \psi_c(x_c),
$$

where $\psi_c$ are nonnegative potential functions and

$$
Z = \sum_x \prod_{c \in \mathcal{C}} \psi_c(x_c)
$$

or the corresponding integral in the continuous case.

The normalizing constant $Z$ is the partition function.
It is often the main computational bottleneck.

### Example 3: Ising model or pairwise binary MRF

For binary variables $x_i \in \{-1,+1\}$ on a graph,

$$
p(x)
\propto
\exp\left(
\sum_i b_i x_i + \sum_{(i,j)\in E} w_{ij} x_i x_j
\right).
$$

This model captures local coupling and appears in statistical physics, image denoising, and energy-based modeling.

### Example 4: Conditional random fields

A conditional random field models $p(y \mid x)$ directly using undirected structure over output variables.
Linear-chain CRFs are a discriminative alternative to HMMs for structured prediction tasks such as sequence labeling.

## 4. Reading conditional independence

Graphical models matter because independence assumptions simplify both data requirements and computation.

### Directed case

In directed graphs, d-separation determines whether a set of nodes is conditionally independent of another set given evidence.
For this module, the operational idea is enough:

- chains and forks are blocked by conditioning on the middle node;
- colliders are opened, not blocked, by conditioning on the collider or its descendants.

### Undirected case

In undirected graphs, a set of nodes is conditionally independent of another set given a separator set when every path between them passes through the separator.

These rules are not only logical bookkeeping.
They tell us which quantities can be computed locally and which messages need to pass through a graph.

## 5. Exact inference tasks

Once a graphical model is specified, common inference tasks include:

- marginal inference: compute $p(x_i)$ or $p(z_i \mid x_{\mathrm{obs}})$;
- MAP inference: find the most likely latent assignment;
- conditional prediction: compute $p(y \mid x)$; and
- likelihood evaluation: compute $p(x)$ or $p(x_{\mathrm{obs}})$.

Exact inference is feasible when graph structure is simple enough.
Examples:

- naive Bayes classification;
- forward-backward in HMMs;
- variable elimination on small trees; and
- belief propagation on trees.

Tree-structured or low-treewidth graphs are the main tractable regime.

## 6. Approximate inference

Approximate inference is needed when exact marginalization or exact partition-function computation becomes too expensive.
This happens quickly in loopy graphs, high-dimensional latent-variable models, and richly coupled posterior distributions.

Main approximate families:

- Monte Carlo methods such as Gibbs sampling and Metropolis-Hastings;
- variational methods that optimize a tractable surrogate family;
- loopy belief propagation; and
- Laplace or local Gaussian approximations.

> **Key distinction.** Exact inference computes the desired quantity without approximation error other than numerical precision.
> Approximate inference trades exactness for tractability because the exact quantity is computationally prohibitive.

That distinction must remain explicit later when moving from HMMs and small Bayesian networks to VAEs and large probabilistic deep models.

## 7. Choosing between directed and undirected models

Directed models are often preferred when:

- there is a natural generative story;
- causally ordered or time-ordered structure matters; or
- local conditional distributions are easy to specify.

Undirected models are often preferred when:

- interactions are symmetric;
- compatibility functions are easier to define than normalized conditionals; or
- the task is discriminative structured prediction, as in CRFs.

Both are languages for factorization, not competing philosophies.
The right choice is driven by the dependency structure and inference burden of the problem.

## 8. Relevance for later modules

Graphical models provide the conceptual scaffolding for several later topics:

- HMMs and sequence models;
- topic models and other latent-variable models;
- variational autoencoders, where approximate posterior inference is central; and
- energy-based or structured-output models.

Thinking in terms of nodes, factors, and inference queries helps organize models that would otherwise look like unrelated algorithms.

## Summary

Directed graphical models factorize joint distributions into local conditional distributions, while undirected graphical models factorize them into clique potentials and a partition function.
Naive Bayes and HMMs are standard directed examples; pairwise MRFs and CRFs are standard undirected ones.
The main computational question is inference: when the graph is simple, exact inference is possible; when the graph is large or loopy, approximate inference becomes necessary.
