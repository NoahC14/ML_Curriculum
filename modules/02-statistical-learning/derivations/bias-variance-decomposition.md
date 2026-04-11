---
title: "Bias-Variance Decomposition"
module: "02-statistical-learning"
lesson: "bias-variance-decomposition"
doc_type: "derivation"
topic: "squared-loss"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability"
  - "01-optimization/convexity-and-optimization"
  - "02-statistical-learning/statistical-learning-foundations"
updated: "2026-04-11"
owner: "curriculum-team"
tags:
  - "statistical-learning"
  - "bias-variance"
  - "squared-loss"
  - "regression"
---

## Purpose

This derivation proves the bias-variance decomposition for squared loss.
The aim is to make explicit which randomness is being averaged over and where the irreducible noise term enters.

## Setup and notation

Let $(X,Y)$ be drawn from an unknown joint distribution.
Assume the conditional mean function is

$$
f^\star(x) = \mathbb{E}[Y \mid X=x].
$$

Define the conditional noise variance

$$
\sigma^2(x) = \operatorname{Var}(Y \mid X=x).
$$

Let $S$ denote a random training sample and let $\widehat{f}_S$ be the predictor produced by a learning algorithm from $S$.
For a fixed input $x$, the quantity $\widehat{f}_S(x)$ is random because it depends on the random sample $S$.

We will derive the conditional expected squared prediction error

$$
\mathbb{E}_{S,Y \mid X=x}\bigl[(Y - \widehat{f}_S(x))^2\bigr].
$$

The expectation is over:

- the random sample $S$, which changes the fitted predictor; and
- the random response $Y$ at the same input value $x$.

## Step 1. Add and subtract the conditional mean

Start from

$$
Y - \widehat{f}_S(x)
= \bigl(Y - f^\star(x)\bigr) + \bigl(f^\star(x) - \widehat{f}_S(x)\bigr).
$$

Squaring gives

$$
(Y - \widehat{f}_S(x))^2
= (Y - f^\star(x))^2
+ \bigl(f^\star(x) - \widehat{f}_S(x)\bigr)^2
+ 2\bigl(Y - f^\star(x)\bigr)\bigl(f^\star(x) - \widehat{f}_S(x)\bigr).
$$

Take conditional expectation given $X=x$:

$$
\mathbb{E}\bigl[(Y - \widehat{f}_S(x))^2 \mid X=x\bigr]
= A + B + C,
$$

where

$$
A = \mathbb{E}\bigl[(Y - f^\star(x))^2 \mid X=x\bigr],
$$

$$
B = \mathbb{E}\bigl[\bigl(f^\star(x) - \widehat{f}_S(x)\bigr)^2 \mid X=x\bigr],
$$

and

$$
C = 2\mathbb{E}\bigl[\bigl(Y - f^\star(x)\bigr)\bigl(f^\star(x) - \widehat{f}_S(x)\bigr) \mid X=x\bigr].
$$

## Step 2. Identify the irreducible noise term

Because $f^\star(x) = \mathbb{E}[Y \mid X=x]$,

$$
A = \operatorname{Var}(Y \mid X=x) = \sigma^2(x).
$$

This is the irreducible noise.
No learning algorithm can remove it because it is inherent in the conditional distribution of $Y$ at input $x$.

## Step 3. Show that the cross term vanishes

Assume the fresh response $Y$ at input $x$ is independent of the training sample $S$ given $X=x$.
Then $Y - f^\star(x)$ is conditionally independent of $\widehat{f}_S(x)$ given $X=x$.

Therefore

$$
\mathbb{E}\bigl[\bigl(Y - f^\star(x)\bigr)\bigl(f^\star(x) - \widehat{f}_S(x)\bigr) \mid X=x\bigr]
$$

factorizes into

$$
\mathbb{E}[Y - f^\star(x) \mid X=x]
\cdot
\mathbb{E}[f^\star(x) - \widehat{f}_S(x) \mid X=x].
$$

But

$$
\mathbb{E}[Y - f^\star(x) \mid X=x]
= \mathbb{E}[Y \mid X=x] - f^\star(x)
= 0.
$$

Hence

$$
C = 0.
$$

So far we have proved

$$
\mathbb{E}_{S,Y \mid X=x}\bigl[(Y - \widehat{f}_S(x))^2\bigr]
= \sigma^2(x)
+ \mathbb{E}_S\bigl[\bigl(f^\star(x) - \widehat{f}_S(x)\bigr)^2\bigr].
$$

## Step 4. Decompose the algorithmic term into bias and variance

Define the mean predictor at input $x$:

$$
\bar{f}(x) = \mathbb{E}_S[\widehat{f}_S(x)].
$$

Add and subtract $\bar{f}(x)$:

$$
f^\star(x) - \widehat{f}_S(x)
= \bigl(f^\star(x) - \bar{f}(x)\bigr) + \bigl(\bar{f}(x) - \widehat{f}_S(x)\bigr).
$$

Square:

$$
\bigl(f^\star(x) - \widehat{f}_S(x)\bigr)^2
= \bigl(f^\star(x) - \bar{f}(x)\bigr)^2
+ \bigl(\bar{f}(x) - \widehat{f}_S(x)\bigr)^2
+ 2\bigl(f^\star(x) - \bar{f}(x)\bigr)\bigl(\bar{f}(x) - \widehat{f}_S(x)\bigr).
$$

Now take expectation over $S$.
The first term is constant with respect to $S$, so it stays as is.
For the cross term:

$$
\mathbb{E}_S\bigl[\bar{f}(x) - \widehat{f}_S(x)\bigr]
= \bar{f}(x) - \mathbb{E}_S[\widehat{f}_S(x)]
= 0.
$$

Therefore the cross term again disappears, giving

$$
\mathbb{E}_S\bigl[\bigl(f^\star(x) - \widehat{f}_S(x)\bigr)^2\bigr]
= \bigl(f^\star(x) - \bar{f}(x)\bigr)^2
+ \mathbb{E}_S\bigl[\bigl(\widehat{f}_S(x) - \bar{f}(x)\bigr)^2\bigr].
$$

Define:

$$
\operatorname{Bias}(\widehat{f}_S(x)) = \bar{f}(x) - f^\star(x),
$$

$$
\operatorname{Var}(\widehat{f}_S(x)) = \mathbb{E}_S\bigl[\bigl(\widehat{f}_S(x) - \bar{f}(x)\bigr)^2\bigr].
$$

Then

$$
\mathbb{E}_{S,Y \mid X=x}\bigl[(Y - \widehat{f}_S(x))^2\bigr]
= \sigma^2(x)
+ \operatorname{Bias}(\widehat{f}_S(x))^2
+ \operatorname{Var}(\widehat{f}_S(x)).
$$

This is the pointwise bias-variance decomposition for squared loss.

## Step 5. Integrated version over the input distribution

If we also average over $X$, then

$$
\mathbb{E}_{X,S,Y}\bigl[(Y - \widehat{f}_S(X))^2\bigr]
= \mathbb{E}_X[\sigma^2(X)]
+ \mathbb{E}_X[\operatorname{Bias}(\widehat{f}_S(X))^2]
+ \mathbb{E}_X[\operatorname{Var}(\widehat{f}_S(X))].
$$

This version is often the one used when discussing average test mean squared error.

## Interpretation

- The noise term $\sigma^2(x)$ is determined by the data-generating process.
- The bias term measures systematic mismatch between the average learned predictor and the regression function.
- The variance term measures instability with respect to the sample.

Typical patterns:

- low-capacity models tend to have larger bias and smaller variance;
- high-capacity models tend to have smaller bias and larger variance; and
- regularization often increases bias slightly while decreasing variance substantially.

## Scope notes

This decomposition is exact for squared loss.
There are analogous decompositions for other losses, but they are usually less clean and often require different notions of bias and variance.

Also note that the decomposition concerns expected prediction error across repeated samples.
It does not say that every flexible model is bad or that every simple model is good.
Its role is explanatory: it tells us which terms are moving when model complexity changes.

## ML relevance

The decomposition explains why:

- polynomial models of increasing degree first improve and then degrade on test error;
- ensemble methods can reduce variance by averaging unstable predictors;
- ridge regression can improve prediction by sacrificing a little bias to gain a larger variance reduction; and
- cross-validation is useful because it estimates predictive error rather than training error.

The accompanying notebook [SL-01-bias-variance-demo.ipynb](../notebooks/SL-01-bias-variance-demo.ipynb) shows this behavior empirically.
