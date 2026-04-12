---
title: "Logistic Regression"
module: "03-linear-models"
lesson: "logistic-regression"
doc_type: "notes"
topic: "classification"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability"
  - "00-math-toolkit/information-theory"
  - "00-math-toolkit/multivariable-calculus"
  - "02-statistical-learning/statistical-learning-foundations"
updated: "2026-04-11"
owner: "curriculum-team"
tags:
  - "logistic-regression"
  - "classification"
  - "cross-entropy"
  - "maximum-likelihood"
  - "softmax"
---

## Purpose

These notes introduce logistic regression as the canonical linear model for classification.
The central idea is simple: keep a linear score in the features, but map that score into a valid probability model for class labels.
This lets us connect geometry, likelihood, information theory, and optimization in one model family.

## Learning objectives

After working through this note, you should be able to:

- define binary logistic regression and multiclass softmax regression;
- explain why the sigmoid and softmax functions produce valid class probabilities;
- derive the negative log-likelihood and identify it with cross-entropy loss;
- connect cross-entropy minimization to KL divergence from the empirical label distribution to the model;
- compute and interpret logistic regression gradients;
- describe the decision boundary induced by a linear score; and
- explain why logistic regression is a probabilistic classification model rather than a least-squares classifier.

## 1. Motivation

In linear regression, we predict a real-valued outcome with

$$
\hat{y}(x) = w^\top x + b.
$$

That is appropriate when the target is numeric.
For classification, however, the target is a label.
In binary classification we often write

$$
y \in \{0,1\}.
$$

We would like a model that outputs

$$
\mathbb{P}(Y=1 \mid X=x),
$$

because probabilities support thresholding, uncertainty estimates, and log-likelihood training.

The difficulty is that the linear score

$$
z(x) = w^\top x + b
$$

can take any real value, while a probability must lie in $[0,1]$.
Logistic regression solves this by keeping the linear score but passing it through a nonlinear link.

## 2. Assumptions and notation

Let the training sample be

$$
S = \{(x_i, y_i)\}_{i=1}^n,
$$

where each feature vector satisfies $x_i \in \mathbb{R}^d$.
For binary classification, let $y_i \in \{0,1\}$.

It is convenient to absorb the bias into the parameter vector.
Define the augmented feature vector

$$
\tilde{x}_i = \begin{bmatrix}1 \\ x_i\end{bmatrix} \in \mathbb{R}^{d+1},
$$

and the augmented parameter vector

$$
\theta = \begin{bmatrix}b \\ w\end{bmatrix} \in \mathbb{R}^{d+1}.
$$

Then the score is

$$
z_i = \theta^\top \tilde{x}_i.
$$

When matrix notation is useful, let

$$
X \in \mathbb{R}^{n \times (d+1)}
$$

be the design matrix whose $i$th row is $\tilde{x}_i^\top$, and let

$$
y = (y_1,\dots,y_n)^\top \in \mathbb{R}^n.
$$

## 3. Binary logistic regression

### 3.1 Sigmoid link

The logistic sigmoid is the function

$$
\sigma(z) = \frac{1}{1+e^{-z}}.
$$

It maps $\mathbb{R}$ into $(0,1)$, so it can be used to parameterize a Bernoulli probability:

$$
p_\theta(x) = \mathbb{P}(Y=1 \mid X=x;\theta) = \sigma(\theta^\top \tilde{x}).
$$

Then

$$
\mathbb{P}(Y=0 \mid X=x;\theta) = 1 - \sigma(\theta^\top \tilde{x}).
$$

This yields a Bernoulli model with parameter determined by a linear score.

### 3.2 Log-odds interpretation

The sigmoid is special because it linearizes the log-odds:

$$
\frac{p_\theta(x)}{1-p_\theta(x)} = e^{\theta^\top \tilde{x}},
$$

so

$$
\log \frac{p_\theta(x)}{1-p_\theta(x)} = \theta^\top \tilde{x}.
$$

This says logistic regression is linear in log-odds, not linear in probability.

> **Interpretation.** If feature $x_j$ increases by one unit while other coordinates are fixed, then the log-odds change by $\theta_j$.

### 3.3 Exponential-family motivation

The Bernoulli distribution belongs to the exponential family:

$$
p(y \mid \eta)
= \exp\bigl(y\eta - A(\eta)\bigr),
\quad y \in \{0,1\},
$$

where the natural parameter is

$$
\eta = \log\frac{p}{1-p}
$$

and the log-partition function is

$$
A(\eta) = \log(1+e^\eta).
$$

If we model the natural parameter as a linear function,

$$
\eta(x) = \theta^\top \tilde{x},
$$

then

$$
\mathbb{E}[Y \mid X=x]
= \frac{e^{\eta(x)}}{1+e^{\eta(x)}}
= \sigma(\theta^\top \tilde{x}).
$$

This is the generalized-linear-model viewpoint, but the important lesson here is narrower:
the sigmoid appears because Bernoulli likelihood and a linear natural parameter fit together exactly.

### 3.4 Maximum-entropy motivation

The same form can be motivated from maximum entropy.
Among all Bernoulli distributions with a prescribed mean

$$
\mathbb{E}[Y \mid X=x] = \mu(x),
$$

the maximum-entropy distribution is Bernoulli with that mean.
If we choose a linear model for the canonical statistic through the log-odds constraint, then the resulting conditional distribution has the logistic form above.

This is why logistic regression is often described as both:

- an exponential-family conditional model; and
- a maximum-entropy classifier under linear feature constraints.

## 4. Likelihood and cross-entropy

### 4.1 Bernoulli likelihood

For one example $(x_i,y_i)$, the conditional probability of the observed label is

$$
p_\theta(y_i \mid x_i)
= \sigma(z_i)^{y_i}\bigl(1-\sigma(z_i)\bigr)^{1-y_i},
$$

where $z_i = \theta^\top \tilde{x}_i$.

Assuming the examples are conditionally independent given $\theta$, the likelihood is

$$
L(\theta)
= \prod_{i=1}^n \sigma(z_i)^{y_i}\bigl(1-\sigma(z_i)\bigr)^{1-y_i}.
$$

Taking logs gives the log-likelihood

$$
\ell(\theta)
= \sum_{i=1}^n
\left[
y_i \log \sigma(z_i)
 (1-y_i)\log\bigl(1-\sigma(z_i)\bigr)
\right].
$$

The maximum-likelihood estimator is

$$
\hat{\theta}
\in \arg\max_{\theta} \ell(\theta).
$$

### 4.2 Negative log-likelihood as binary cross-entropy

It is standard to minimize the negative average log-likelihood:

$$
\mathcal{L}(\theta)
= -\frac{1}{n}\ell(\theta)
= -\frac{1}{n}\sum_{i=1}^n
\left[
y_i \log p_i + (1-y_i)\log(1-p_i)
\right],
$$

where $p_i = \sigma(z_i)$.

This is the **binary cross-entropy loss**.

For one example, if we encode the empirical label distribution as

$$
q_i = (y_i, 1-y_i),
$$

and the model distribution as

$$
\hat{q}_i = (p_i, 1-p_i),
$$

then the loss is

$$
H(q_i,\hat{q}_i)
= -y_i\log p_i - (1-y_i)\log(1-p_i).
$$

### 4.3 Connection to KL divergence

From information theory,

$$
H(q_i,\hat{q}_i) = H(q_i) + D_{\mathrm{KL}}(q_i \| \hat{q}_i).
$$

For a hard label $y_i \in \{0,1\}$, the empirical distribution $q_i$ is one-hot, so $H(q_i)=0$.
Therefore

$$
H(q_i,\hat{q}_i) = D_{\mathrm{KL}}(q_i \| \hat{q}_i).
$$

Averaging over the dataset gives

$$
\mathcal{L}(\theta)
= \frac{1}{n}\sum_{i=1}^n D_{\mathrm{KL}}(q_i \| \hat{q}_i).
$$

So logistic regression training fits the model by making the predicted Bernoulli distribution close, in KL divergence, to the empirical label distribution on each example.

> **Important caveat.** In supervised learning we do not know the true conditional distribution $p(y \mid x)$ exactly. The KL statement above is with respect to the empirical target distribution induced by the observed labels.

## 5. Gradient and optimization

The key simplification is that the derivative of the sigmoid satisfies

$$
\sigma'(z) = \sigma(z)\bigl(1-\sigma(z)\bigr).
$$

Using this identity, the gradient of the negative average log-likelihood becomes

$$
\nabla_\theta \mathcal{L}(\theta)
= \frac{1}{n}X^\top(p-y),
$$

where

$$
p =
\begin{bmatrix}
\sigma(z_1) \\
\vdots \\
\sigma(z_n)
\end{bmatrix}.
$$

The full derivation appears in [logistic-gradient.md](../derivations/logistic-gradient.md).

This gradient has a clean interpretation:

- if the model predicts too large a probability for class $1$, then $p_i-y_i > 0$ and the update pushes the score downward along $\tilde{x}_i$;
- if the model predicts too small a probability, then $p_i-y_i < 0$ and the update pushes the score upward.

Gradient descent uses

$$
\theta^{(t+1)} = \theta^{(t)} - \alpha \nabla_\theta \mathcal{L}(\theta^{(t)}),
$$

with step size $\alpha > 0$.

## 6. Decision boundaries

Logistic regression is nonlinear as a probability model but linear as a classifier boundary.
If we classify by thresholding at probability $1/2$, then

$$
\hat{y}(x)=1
\quad \Longleftrightarrow \quad
\sigma(\theta^\top \tilde{x}) \geq \frac{1}{2}.
$$

Because $\sigma$ is monotone increasing,

$$
\sigma(\theta^\top \tilde{x}) \geq \frac{1}{2}
\quad \Longleftrightarrow \quad
\theta^\top \tilde{x} \geq 0.
$$

So the decision boundary is

$$
\theta^\top \tilde{x} = 0,
$$

which is a hyperplane in feature space.

> **Example.** In two dimensions, the boundary is a line. Logistic regression can curve probabilities smoothly across the plane, but the $0.5$ classification contour remains linear.

## 7. Multiclass logistic regression and softmax

### 7.1 Score vector and softmax

Suppose now that

$$
y \in \{1,\dots,K\}.
$$

Let

$$
W \in \mathbb{R}^{K \times (d+1)}
$$

be a parameter matrix, and let the class-$k$ score be

$$
s_k(x) = w_k^\top \tilde{x}.
$$

The softmax model defines

$$
\mathbb{P}(Y=k \mid X=x;W)
= \frac{e^{s_k(x)}}{\sum_{j=1}^K e^{s_j(x)}}.
$$

These probabilities are positive and sum to one.

### 7.2 Exponential-family and maximum-entropy motivation

The categorical distribution is also an exponential-family distribution.
If the class scores are linear in the features, normalizing the exponentials yields the softmax form.

There is also a maximum-entropy view:
among distributions over $K$ classes satisfying linear expectation constraints on features, the maximum-entropy distribution has Gibbs form

$$
p_k(x) \propto e^{s_k(x)}.
$$

Normalization then gives softmax.

This is the multiclass analogue of the logistic sigmoid construction.

### 7.3 Multiclass cross-entropy

Encode the true class by a one-hot vector

$$
y_i \in \{0,1\}^K,
\quad \sum_{k=1}^K y_{ik}=1.
$$

If the predicted probability vector is

$$
p_i = (p_{i1},\dots,p_{iK}),
$$

then the negative average log-likelihood is

$$
\mathcal{L}(W)
= -\frac{1}{n}\sum_{i=1}^n \sum_{k=1}^K y_{ik}\log p_{ik}.
$$

This is multiclass cross-entropy.
Using

$$
H(y_i,p_i) = H(y_i) + D_{\mathrm{KL}}(y_i \| p_i),
$$

and the fact that $H(y_i)=0$ for one-hot labels, we again obtain an average KL divergence to the empirical class distribution.

## 8. Worked example: one-dimensional binary classifier

Suppose the score is

$$
z(x) = -3 + 2x.
$$

Then

$$
\mathbb{P}(Y=1 \mid X=x) = \sigma(-3+2x).
$$

At $x=1$,

$$
z(1) = -1,
\qquad
\sigma(-1) \approx 0.269.
$$

At $x=2$,

$$
z(2) = 1,
\qquad
\sigma(1) \approx 0.731.
$$

The decision boundary satisfies

$$
-3 + 2x = 0,
$$

so the threshold is

$$
x = 1.5.
$$

This example shows the characteristic logistic behavior:

- linear score;
- nonlinear probability;
- linear classification boundary.

## 9. Common misconceptions

- **Logistic regression is not regression in the ordinary least-squares sense.** The name comes from the functional form and historical GLM terminology, but the task is classification.
- **The output probability is not linear in the features.** The log-odds are linear.
- **Using squared loss for binary labels is possible, but it is not the likelihood-matched objective.** Cross-entropy is the natural objective for Bernoulli and categorical models.
- **Multiclass softmax regression is still a linear model in score space.** The nonlinear part is the normalization into probabilities.

## 10. GLM intuition without expanding scope

Logistic regression is one example of a generalized linear model:

- a distribution from the exponential family;
- a linear predictor $\theta^\top \tilde{x}$; and
- a link between the mean and the linear predictor.

That broader view is useful, but for this module the main lesson is concrete:
classification can be handled by pairing a linear score with a probabilistic link and training by maximum likelihood.

## 11. Category theory insertion point

The logistic pipeline can be read as a composition

$$
\mathbb{R}^d \xrightarrow{\ \tilde{x}\mapsto \theta^\top \tilde{x}\ } \mathbb{R}
\xrightarrow{\ \sigma\ } (0,1),
$$

or, in the multiclass case,

$$
\mathbb{R}^d \to \mathbb{R}^K \to \Delta^{K-1},
$$

where $\Delta^{K-1}$ is the probability simplex.
This is a useful structural observation about composition of maps, but it does not replace the statistical derivation.

## 12. Summary

Logistic regression is the canonical classification analogue of linear regression.
Its defining features are:

- a linear score in the input features;
- a sigmoid or softmax map that turns scores into probabilities;
- maximum-likelihood training, equivalent to cross-entropy minimization; and
- a linear decision boundary in feature space.

The model is simple enough to derive by hand and strong enough to illustrate deep ideas that recur throughout ML: exponential families, cross-entropy, KL divergence, and gradient-based optimization.
