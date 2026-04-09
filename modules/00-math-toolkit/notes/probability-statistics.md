---
title: "Probability and Statistics Primer for Machine Learning"
module: "00-math-toolkit"
lesson: "probability-statistics-primer"
doc_type: "notes"
topic: "probability-statistics"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
  - "00-math-toolkit/multivariable-calculus"
updated: "2026-04-09"
owner: "curriculum-team"
tags:
  - "probability"
  - "statistics"
  - "bayesian-inference"
  - "mle"
  - "map"
---

## Motivation

Machine learning models are built to reason under uncertainty. Labels are noisy, features vary across populations, and future data are never identical to past data. Probability gives the language for uncertainty, while statistics gives the toolkit for estimating structure from data.

This primer is scoped for the rest of the curriculum. It stays at the level most ML practitioners need for learning theory, probabilistic modeling, Bayesian inference, and information-theoretic losses. Measure-theoretic probability is intentionally out of scope; deeper treatments are deferred to references such as Murphy and Bishop.

Three viewpoints will appear throughout:

- the **descriptive viewpoint**, where distributions summarize variability in data;
- the **inferential viewpoint**, where unknown parameters are estimated from samples; and
- the **modeling viewpoint**, where probability distributions become trainable ML models.

## Assumptions and Notation

We write random variables in uppercase, such as $X$ and $Y$, and observed values in lowercase, such as $x$ and $y$. A random vector is written in bold uppercase, such as $\mathbf{X} \in \mathbb{R}^d$, with realization $\mathbf{x}$.

Probabilities are written as $P(A)$ for events $A$, and densities or mass functions are written as $p(x)$ when the context is clear. Expectation under a distribution $p$ is written as $\mathbb{E}_p[\cdot]$ or simply $\mathbb{E}[\cdot]$.

If $\mathbf{x}_1, \ldots, \mathbf{x}_n$ are observed data points, we assume they are independent and identically distributed unless stated otherwise.

## Probability Spaces Without Measure-Theoretic Machinery

Probability theory begins with three ingredients:

- a sample space $\Omega$ of possible outcomes;
- a collection of events $\mathcal{F}$ we allow ourselves to ask about; and
- a probability rule $P$ assigning each event a number in $[0,1]$.

For this course, the main operational facts are:

1. $P(A) \geq 0$ for every event $A$;
2. $P(\Omega) = 1$;
3. if $A$ and $B$ are disjoint, then $P(A \cup B) = P(A) + P(B)$.

These facts support almost everything we use in ML. For example, classification uncertainty is expressed through probabilities over labels, and probabilistic models assign probabilities or densities to observations.

> **Example.** In binary spam classification, the sample space can be written as
> $$
> \Omega = \{(\mathbf{x}, y) : \mathbf{x} \text{ is an email representation},\ y \in \{\text{spam}, \text{not spam}\}\}.
> $$
> The event $A = \{y = \text{spam}\}$ has probability equal to the spam rate in the population.

## Random Variables and Distributions

> **Definition.** A random variable is a function from outcomes in the sample space to numerical values.

This definition matters because it separates two levels:

- the underlying uncertain outcome $\omega \in \Omega$;
- the numerical quantity $X(\omega)$ derived from that outcome.

In ML, random variables can represent labels, features, gradients, losses, or latent variables.

### Discrete random variables and PMFs

If $X$ takes values in a countable set, its probability mass function is

$$
p_X(x) = P(X = x).
$$

The PMF satisfies

$$
\sum_x p_X(x) = 1.
$$

> **Example.** A Bernoulli random variable $Y \in \{0,1\}$ with parameter $\theta \in [0,1]$ has
> $$
> P(Y = y) = \theta^y (1-\theta)^{1-y}, \qquad y \in \{0,1\}.
> $$
> In ML, this is the basic model for binary labels and binary features.

### Continuous random variables and PDFs

If $X$ is continuous, it is described by a probability density function $p_X(x)$ such that

$$
P(a \leq X \leq b) = \int_a^b p_X(x)\,dx.
$$

The PDF satisfies

$$
\int_{-\infty}^{\infty} p_X(x)\,dx = 1.
$$

For a continuous random variable, the probability at a single point is zero:

$$
P(X = x) = 0.
$$

What matters is probability over intervals or regions.

> **Example.** If $X \sim \mathcal{N}(\mu, \sigma^2)$, then
> $$
> p_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}}
> \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right).
> $$
> Gaussian models appear throughout ML because sums of many small effects are often approximately Gaussian, and because Gaussian algebra is analytically convenient.

## Expectation, Variance, and Covariance

Expectation summarizes average behavior.

> **Definition.** For a discrete random variable $X$,
> $$
> \mathbb{E}[X] = \sum_x x\,p_X(x).
> $$
> For a continuous random variable $X$,
> $$
> \mathbb{E}[X] = \int_{-\infty}^{\infty} x\,p_X(x)\,dx.
> $$

More generally, if $g$ is a function of $X$, then

$$
\mathbb{E}[g(X)] = \sum_x g(x)p_X(x)
$$

or

$$
\mathbb{E}[g(X)] = \int g(x)p_X(x)\,dx,
$$

depending on whether $X$ is discrete or continuous.

### Variance

> **Definition.** The variance of $X$ is
> $$
> \mathrm{Var}(X) = \mathbb{E}\left[(X - \mathbb{E}[X])^2\right].
> $$

Variance measures spread around the mean. Expanding the square gives the useful identity

$$
\mathrm{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2.
$$

> **Proof Sketch.**
> $$
> \mathbb{E}\left[(X-\mu)^2\right]
> = \mathbb{E}[X^2 - 2\mu X + \mu^2]
> = \mathbb{E}[X^2] - 2\mu\mathbb{E}[X] + \mu^2
> = \mathbb{E}[X^2] - \mu^2,
> $$
> where $\mu = \mathbb{E}[X]$.

### Covariance

> **Definition.** If $X$ and $Y$ are random variables, their covariance is
> $$
> \mathrm{Cov}(X,Y) = \mathbb{E}\left[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])\right].
> $$

If $\mathbf{X} \in \mathbb{R}^d$ is a random vector with mean $\boldsymbol{\mu} = \mathbb{E}[\mathbf{X}]$, its covariance matrix is

$$
\boldsymbol{\Sigma}
=
\mathrm{Cov}(\mathbf{X})
=
\mathbb{E}\left[(\mathbf{X}-\boldsymbol{\mu})(\mathbf{X}-\boldsymbol{\mu})^\top\right].
$$

Each entry $\Sigma_{ij}$ records how coordinates $X_i$ and $X_j$ vary together.

### Correlation

> **Definition.** If $\mathrm{Var}(X) > 0$ and $\mathrm{Var}(Y) > 0$, the correlation coefficient is
> $$
> \mathrm{Corr}(X,Y) = \frac{\mathrm{Cov}(X,Y)}{\sqrt{\mathrm{Var}(X)\mathrm{Var}(Y)}}.
> $$

Correlation rescales covariance into a unitless quantity between $-1$ and $1$.

### ML connection: covariance, correlation, PCA, and feature analysis

Covariance and correlation are not just descriptive statistics. They affect model design and data preprocessing.

- In PCA, eigenvectors of the covariance matrix define directions of maximal variance.
- In exploratory feature analysis, large pairwise correlations suggest redundancy, multicollinearity, or shared latent structure.
- In Gaussian models, the covariance matrix controls uncertainty geometry through ellipsoidal level sets.

> **Example.** If two features in a regression dataset are highly correlated, then the design matrix contains nearly redundant columns. This can destabilize parameter estimates and motivates PCA, feature selection, or regularization.

## Joint, Marginal, and Conditional Distributions

Many ML problems are about relationships between multiple random variables.

### Joint distribution

The joint distribution of $X$ and $Y$ is written as $p(x,y)$ or $P(X=x, Y=y)$. It describes uncertainty about both variables together.

### Marginal distribution

The marginal distribution is obtained by summing or integrating out the other variable:

$$
p_X(x) = \sum_y p(x,y)
$$

in the discrete case, and

$$
p_X(x) = \int p(x,y)\,dy
$$

in the continuous case.

### Conditional distribution

If $P(Y=y) > 0$, then

$$
p(x \mid y) = \frac{p(x,y)}{p(y)}.
$$

This is the distribution of $X$ after observing $Y=y$.

Conditional distributions are central in supervised learning. A discriminative classifier models $p(y \mid \mathbf{x})$, while a generative classifier may model $p(\mathbf{x} \mid y)$ together with $p(y)$.

### Product rule and independence

Rearranging the conditional definition gives the product rule:

$$
p(x,y) = p(x \mid y)p(y) = p(y \mid x)p(x).
$$

> **Definition.** $X$ and $Y$ are independent if
> $$
> p(x,y) = p(x)p(y)
> $$
> for all relevant $x$ and $y$.

Independence is a strong modeling assumption. Naive Bayes uses a conditional-independence approximation because it simplifies inference and often works surprisingly well in high-dimensional text problems.

## Bayes' Theorem

Bayes' theorem is a direct consequence of the product rule:

$$
p(y \mid x) = \frac{p(x \mid y)p(y)}{p(x)}.
$$

When $y$ ranges over finitely many classes,

$$
p(x) = \sum_{y'} p(x \mid y')p(y').
$$

So Bayes' theorem can also be written as

$$
p(y \mid x)
=
\frac{p(x \mid y)p(y)}
{\sum_{y'} p(x \mid y')p(y')}.
$$

The terms are interpreted as follows:

- $p(y)$: prior belief about the class;
- $p(x \mid y)$: likelihood of observing features $x$ under class $y$;
- $p(y \mid x)$: posterior belief after observing $x$.

### Motivating example: spam classification

Suppose the class variable $Y \in \{\text{spam}, \text{ham}\}$ and $X$ is a binary feature indicating whether the word "free" appears in an email. Then

$$
P(\text{spam} \mid X=1)
=
\frac{P(X=1 \mid \text{spam})P(\text{spam})}
{P(X=1)}.
$$

If the word "free" is much more likely in spam than in ham, then the likelihood ratio raises the posterior probability of spam. This is the basic logic behind naive Bayes text classification.

> **ML Interpretation.** Bayes' theorem formalizes updating predictions as evidence arrives. It underlies generative classification, Bayesian neural networks, state estimation, and probabilistic graphical models.

## Common Distributions for ML

This section focuses on a small set of distributions that recur throughout the curriculum.

### Bernoulli distribution

If $Y \in \{0,1\}$ and $P(Y=1)=\theta$, then

$$
P(Y=y) = \theta^y(1-\theta)^{1-y}.
$$

Its mean and variance are

$$
\mathbb{E}[Y] = \theta, \qquad \mathrm{Var}(Y) = \theta(1-\theta).
$$

Bernoulli variables model binary outcomes such as click/no-click, failure/success, or class labels.

### Categorical distribution

If $Y$ takes values in $\{1,\ldots,K\}$ with probabilities $\pi_1,\ldots,\pi_K$, where $\pi_k \geq 0$ and $\sum_{k=1}^K \pi_k = 1$, then

$$
P(Y=k) = \pi_k.
$$

This is the multiclass analogue of the Bernoulli distribution. Softmax classifiers produce estimated categorical probabilities.

### Poisson distribution

If $X$ counts events in a fixed interval with rate $\lambda > 0$, then

$$
P(X=k) = \frac{e^{-\lambda}\lambda^k}{k!}, \qquad k=0,1,2,\ldots
$$

Its mean and variance are both $\lambda$.

Poisson models are used for count data such as arrivals, token counts, or event frequencies.

### Gaussian distribution

The univariate Gaussian $\mathcal{N}(\mu,\sigma^2)$ has density

$$
p(x)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right).
$$

The multivariate Gaussian $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ in $\mathbb{R}^d$ has density

$$
p(\mathbf{x})
=
\frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}}
\exp\left(
-\frac{1}{2}
(\mathbf{x}-\boldsymbol{\mu})^\top
\boldsymbol{\Sigma}^{-1}
(\mathbf{x}-\boldsymbol{\mu})
\right).
$$

Gaussian models are important because:

- they are analytically tractable;
- covariance encodes directional uncertainty;
- least-squares regression can be interpreted as Gaussian likelihood maximization.

## Statistics From Samples

In statistics, parameters of a population distribution are unknown and data are observed. We use samples to estimate those parameters.

If $x_1,\ldots,x_n$ are scalar observations, the sample mean is

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i.
$$

If $\mathbf{x}_1,\ldots,\mathbf{x}_n \in \mathbb{R}^d$, the sample covariance matrix is

$$
\mathbf{S}
=
\frac{1}{n}\sum_{i=1}^n (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^\top,
$$

where

$$
\bar{\mathbf{x}} = \frac{1}{n}\sum_{i=1}^n \mathbf{x}_i.
$$

For ML, this version with factor $1/n$ is often more convenient than the unbiased factor $1/(n-1)$ because it aligns directly with likelihood calculations.

## Likelihood, Log-Likelihood, and Maximum Likelihood Estimation

Suppose a model has parameter $\theta$ and data $x_1,\ldots,x_n$. The likelihood is the probability of the observed data as a function of $\theta$:

$$
L(\theta)
=
p(x_1,\ldots,x_n \mid \theta).
$$

If the data are i.i.d., then

$$
L(\theta) = \prod_{i=1}^n p(x_i \mid \theta).
$$

Because products are cumbersome, we often optimize the log-likelihood:

$$
\ell(\theta)
=
\log L(\theta)
=
\sum_{i=1}^n \log p(x_i \mid \theta).
$$

> **Definition.** A maximum likelihood estimator is any parameter value
> $$
> \hat{\theta}_{\mathrm{MLE}} \in \arg\max_\theta \ell(\theta).
> $$

### Fully worked derivation: Bernoulli MLE

Let $y_1,\ldots,y_n \in \{0,1\}$ be i.i.d. Bernoulli$(\theta)$ observations. The likelihood is

$$
L(\theta)
=
\prod_{i=1}^n \theta^{y_i}(1-\theta)^{1-y_i}.
$$

Taking logs,

$$
\ell(\theta)
=
\sum_{i=1}^n \left[y_i \log \theta + (1-y_i)\log(1-\theta)\right].
$$

Group the terms:

$$
\ell(\theta)
=
\left(\sum_{i=1}^n y_i\right)\log\theta
+
\left(n-\sum_{i=1}^n y_i\right)\log(1-\theta).
$$

Differentiate with respect to $\theta$:

$$
\frac{d\ell}{d\theta}
=
\frac{\sum_{i=1}^n y_i}{\theta}
-
\frac{n-\sum_{i=1}^n y_i}{1-\theta}.
$$

Set the derivative equal to zero:

$$
\frac{\sum_{i=1}^n y_i}{\theta}
=
\frac{n-\sum_{i=1}^n y_i}{1-\theta}.
$$

Multiply both sides by $\theta(1-\theta)$:

$$
\left(\sum_{i=1}^n y_i\right)(1-\theta)
=
\theta\left(n-\sum_{i=1}^n y_i\right).
$$

Expand both sides:

$$
\sum_{i=1}^n y_i - \theta\sum_{i=1}^n y_i
=
n\theta - \theta\sum_{i=1}^n y_i.
$$

The terms $-\theta\sum_i y_i$ cancel, leaving

$$
\sum_{i=1}^n y_i = n\theta.
$$

Hence

$$
\hat{\theta}_{\mathrm{MLE}}
=
\frac{1}{n}\sum_{i=1}^n y_i.
$$

So the Bernoulli MLE is the empirical fraction of ones.

To verify this is a maximum, compute the second derivative:

$$
\frac{d^2\ell}{d\theta^2}
=
-\frac{\sum_{i=1}^n y_i}{\theta^2}
-\frac{n-\sum_{i=1}^n y_i}{(1-\theta)^2}
< 0
$$

for $\theta \in (0,1)$, so the log-likelihood is concave.

### Fully worked derivation: Gaussian MLE

Let $x_1,\ldots,x_n$ be i.i.d. samples from $\mathcal{N}(\mu,\sigma^2)$ with both $\mu$ and $\sigma^2$ unknown.

The likelihood is

$$
L(\mu,\sigma^2)
=
\prod_{i=1}^n
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right).
$$

The log-likelihood is

$$
\ell(\mu,\sigma^2)
=
\sum_{i=1}^n
\left[
-\frac{1}{2}\log(2\pi\sigma^2)
-\frac{(x_i-\mu)^2}{2\sigma^2}
\right].
$$

Collect terms:

$$
\ell(\mu,\sigma^2)
=
-\frac{n}{2}\log(2\pi)
-\frac{n}{2}\log(\sigma^2)
-\frac{1}{2\sigma^2}\sum_{i=1}^n (x_i-\mu)^2.
$$

First optimize with respect to $\mu$. Differentiate:

$$
\frac{\partial \ell}{\partial \mu}
=
-\frac{1}{2\sigma^2}
\sum_{i=1}^n 2(x_i-\mu)(-1)
=
\frac{1}{\sigma^2}\sum_{i=1}^n (x_i-\mu).
$$

Set this equal to zero:

$$
\sum_{i=1}^n (x_i-\mu) = 0.
$$

Expand:

$$
\sum_{i=1}^n x_i - n\mu = 0.
$$

Therefore

$$
\hat{\mu}_{\mathrm{MLE}}
=
\frac{1}{n}\sum_{i=1}^n x_i
=
\bar{x}.
$$

Now optimize with respect to $\sigma^2$. Treat $\mu$ as fixed for the derivative:

$$
\frac{\partial \ell}{\partial \sigma^2}
=
-\frac{n}{2\sigma^2}
+
\frac{1}{2(\sigma^2)^2}\sum_{i=1}^n (x_i-\mu)^2.
$$

Set the derivative equal to zero:

$$
-\frac{n}{2\sigma^2}
+
\frac{1}{2(\sigma^2)^2}\sum_{i=1}^n (x_i-\mu)^2
= 0.
$$

Multiply by $2(\sigma^2)^2$:

$$
-n\sigma^2 + \sum_{i=1}^n (x_i-\mu)^2 = 0.
$$

So

$$
\hat{\sigma}^2
=
\frac{1}{n}\sum_{i=1}^n (x_i-\mu)^2.
$$

Substitute $\hat{\mu}_{\mathrm{MLE}} = \bar{x}$:

$$
\hat{\sigma}^2_{\mathrm{MLE}}
=
\frac{1}{n}\sum_{i=1}^n (x_i-\bar{x})^2.
$$

Thus the Gaussian MLEs are the sample mean and the sample variance with denominator $n$.

> **ML Interpretation.** Squared-error regression corresponds to Gaussian noise modeling. If
> $$
> Y_i = f_\theta(\mathbf{x}_i) + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0,\sigma^2),
> $$
> then maximizing the likelihood over $\theta$ is equivalent to minimizing the sum of squared residuals.

## Maximum A Posteriori Estimation

MLE uses only the likelihood. MAP estimation adds prior information.

If $\theta$ is a parameter with prior density $p(\theta)$ and data are $x_{1:n}$, Bayes' theorem gives

$$
p(\theta \mid x_{1:n})
\propto
p(x_{1:n} \mid \theta)p(\theta).
$$

> **Definition.** A maximum a posteriori estimator is
> $$
> \hat{\theta}_{\mathrm{MAP}}
> \in
> \arg\max_\theta
> \log p(x_{1:n} \mid \theta) + \log p(\theta).
> $$

So MAP is MLE plus a log-prior term.

### ML interpretation of MAP as regularization

MAP often looks like penalized optimization.

- A Gaussian prior on parameters leads to an $\ell_2$ penalty.
- A Laplace prior leads to an $\ell_1$ penalty.

This is why ridge regression and lasso have Bayesian interpretations.

> **Example.** If $w \sim \mathcal{N}(0,\tau^2)$ and the likelihood corresponds to squared-error regression, then maximizing the posterior over $w$ is equivalent to minimizing
> $$
> \sum_{i=1}^n (y_i - \mathbf{x}_i^\top w)^2 + \lambda \|w\|_2^2
> $$
> for an appropriate $\lambda$.

## Conjugate Priors: First Introduction

A conjugate prior is a prior distribution that yields a posterior in the same family after observing data.

This matters because Bayesian updating stays analytically tractable.

### Bernoulli likelihood with Beta prior

Let $Y_1,\ldots,Y_n \mid \theta \overset{\text{i.i.d.}}{\sim} \mathrm{Bernoulli}(\theta)$ and suppose

$$
\theta \sim \mathrm{Beta}(\alpha,\beta),
$$

with density

$$
p(\theta) \propto \theta^{\alpha-1}(1-\theta)^{\beta-1}.
$$

The likelihood is

$$
p(y_{1:n} \mid \theta)
\propto
\theta^{\sum_i y_i}(1-\theta)^{n-\sum_i y_i}.
$$

Multiply likelihood and prior:

$$
p(\theta \mid y_{1:n})
\propto
\theta^{\sum_i y_i + \alpha - 1}
(1-\theta)^{n-\sum_i y_i + \beta - 1}.
$$

So the posterior is

$$
\theta \mid y_{1:n}
\sim
\mathrm{Beta}\left(\alpha + \sum_i y_i,\ \beta + n - \sum_i y_i\right).
$$

### Categorical likelihood with Dirichlet prior

For multiclass probabilities $\boldsymbol{\pi} = (\pi_1,\ldots,\pi_K)$, a Dirichlet prior is conjugate to the categorical likelihood. Posterior parameters update by adding observed class counts.

This is the multiclass extension of Beta-Bernoulli updating.

### Why conjugacy matters for later modules

Conjugate priors appear in:

- naive Bayes smoothing;
- Bayesian linear regression;
- latent-variable models with tractable updates;
- expectation-maximization and variational approximations, where conjugacy often simplifies the algebra.

## Sampling, Estimation, and Generalization Intuition

Statistics in ML is not only about fitting one dataset. It is about relating the observed sample to the unseen population.

If a model fits training data perfectly but relies on accidental noise, it will generalize poorly. Probability supplies the language for data-generating assumptions, and statistics supplies the estimators and diagnostics used to study generalization.

At this stage, the key ideas are:

- samples fluctuate;
- estimators inherit uncertainty from the sample;
- stronger models can fit more patterns, including spurious ones;
- probabilistic assumptions make losses and estimators interpretable.

These themes lead directly into statistical learning theory, probabilistic modeling, and information theory.

## Category-Theoretic Insertion Point

Probability is not the place to force abstract language, but one structural insight is useful: conditional distributions compose. A pipeline that maps raw data to features and then to posterior beliefs can be read as a sequence of transformations between structured spaces.

That observation is enough for Module 00. A more formal treatment belongs later in Module 16.

## Scope Notes and Limitations

- Measure-theoretic probability is out of scope here.
- We have emphasized i.i.d. sampling because it is the default starting assumption in ML.
- Conjugate priors are presented for intuition, not as a claim that exact Bayesian inference is always practical in modern models.
- Correlation is not causation; statistical dependence does not by itself identify causal structure.

## References for Deeper Study

- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*
- Christopher M. Bishop, *Pattern Recognition and Machine Learning*
- David J. C. MacKay, *Information Theory, Inference, and Learning Algorithms*
