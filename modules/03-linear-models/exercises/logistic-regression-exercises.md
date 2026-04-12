---
title: "Logistic Regression Exercises"
module: "03-linear-models"
lesson: "logistic-regression-exercises"
doc_type: "exercise"
topic: "classification"
status: "draft"
prerequisites:
  - "00-math-toolkit/multivariable-calculus"
  - "00-math-toolkit/information-theory"
  - "03-linear-models/logistic-regression"
updated: "2026-04-11"
owner: "curriculum-team"
tags:
  - "logistic-regression"
  - "softmax"
  - "cross-entropy"
  - "gradient"
  - "decision-boundary"
---

## Purpose

These exercises reinforce the mechanics and interpretation of logistic regression in both binary and multiclass settings.
They mix derivation, geometry, information theory, and implementation.

## Exercise 1. Sigmoid basics

Let

$$
\sigma(z) = \frac{1}{1+e^{-z}}.
$$

1. Show that $0 < \sigma(z) < 1$ for all $z \in \mathbb{R}$.
2. Show that $\sigma(-z) = 1-\sigma(z)$.
3. Differentiate $\sigma(z)$ and prove that

$$
\sigma'(z) = \sigma(z)(1-\sigma(z)).
$$

4. Explain why monotonicity of $\sigma$ implies that thresholding at probability $1/2$ is equivalent to thresholding the linear score at $0$.

## Exercise 2. Log-odds interpretation

Consider the binary logistic model

$$
p_\theta(x) = \sigma(\theta^\top \tilde{x}).
$$

1. Show that

$$
\log\frac{p_\theta(x)}{1-p_\theta(x)} = \theta^\top \tilde{x}.
$$

2. Explain in words what coefficient $\theta_j$ means for the log-odds when the $j$th feature increases by one unit and all other features are held fixed.
3. Why is this interpretation different from the coefficient interpretation in ordinary least squares regression?

## Exercise 3. Bernoulli likelihood and cross-entropy

For one observation $(x_i,y_i)$ with $y_i \in \{0,1\}$, let

$$
p_i = \sigma(\theta^\top \tilde{x}_i).
$$

1. Write the Bernoulli likelihood $p_\theta(y_i \mid x_i)$.
2. Write the log-likelihood contribution of this example.
3. Show that the negative log-likelihood equals

$$
-y_i\log p_i - (1-y_i)\log(1-p_i).
$$

4. Explain why this is called binary cross-entropy.

## Exercise 4. Cross-entropy and KL divergence

For a binary label, define the empirical target distribution

$$
q_i = (y_i,1-y_i)
$$

and the predicted distribution

$$
\hat{q}_i = (p_i,1-p_i).
$$

1. Write the cross-entropy $H(q_i,\hat{q}_i)$.
2. Use the identity

$$
H(q_i,\hat{q}_i) = H(q_i) + D_{\mathrm{KL}}(q_i \| \hat{q}_i)
$$

to show that, for a hard label, binary cross-entropy equals a KL divergence.
3. Explain what part of this statement changes if the target is a soft label rather than a hard label.

## Exercise 5. Step-by-step binary gradient derivation

Let

$$
\mathcal{L}(\theta)
= -\sum_{i=1}^n
\left[
y_i\log p_i + (1-y_i)\log(1-p_i)
\right],
\qquad
p_i = \sigma(\theta^\top \tilde{x}_i).
$$

1. Differentiate the loss with respect to $p_i$.
2. Differentiate $p_i$ with respect to $\theta$.
3. Apply the chain rule carefully to derive

$$
\nabla_\theta \mathcal{L}(\theta)
= \sum_{i=1}^n (p_i-y_i)\tilde{x}_i.
$$

4. Rewrite the result in matrix form.

## Exercise 6. Decision boundary geometry

Assume binary logistic regression predicts class $1$ whenever $p_\theta(x) \geq 1/2$.

1. Show that the decision boundary is the set of points satisfying

$$
\theta^\top \tilde{x} = 0.
$$

2. In two dimensions, describe this boundary geometrically.
3. Give one reason why logistic regression can still output well-calibrated probabilities even though the classification boundary is linear.

## Exercise 7. Softmax mechanics

Let

$$
p_k(x) = \frac{e^{w_k^\top \tilde{x}}}{\sum_{j=1}^K e^{w_j^\top \tilde{x}}}.
$$

1. Show that $p_k(x) > 0$ for every class $k$.
2. Show that

$$
\sum_{k=1}^K p_k(x) = 1.
$$

3. Explain why adding the same constant to every score leaves the softmax probabilities unchanged.
4. Why does this imply that the parameterization has a redundancy?

## Exercise 8. Softmax gradient

For one-hot target vector $y \in \{0,1\}^K$ and predicted probability vector $p \in \mathbb{R}^K$, the multiclass loss is

$$
\mathcal{L}(W) = -\sum_{k=1}^K y_k \log p_k.
$$

1. Show that

$$
\frac{\partial \mathcal{L}}{\partial s_r} = p_r - y_r,
$$

where $s_r = w_r^\top \tilde{x}$ is the score for class $r$.
2. Use the chain rule to show that

$$
\nabla_{w_r}\mathcal{L} = (p_r-y_r)\tilde{x}.
$$

3. Write the full matrix gradient for a dataset of $n$ examples.

## Exercise 9. Optimization interpretation

Suppose the model predicts probability $0.95$ for class $1$ on an example whose true label is $0$.

1. Compute the per-example binary cross-entropy loss.
2. Is the contribution to the gradient residual $p_i-y_i$ positive or negative?
3. Explain qualitatively how one gradient step changes the score on this example.

## Exercise 10. Binary implementation task

Implement binary logistic regression from scratch using only `numpy`.

Requirements:

1. add a bias term to the feature matrix;
2. implement sigmoid, binary cross-entropy, and its gradient;
3. train with gradient descent;
4. report the training loss over iterations; and
5. visualize the decision boundary on a two-dimensional synthetic dataset.

Then answer:

6. What happens if the learning rate is too large?
7. How does the boundary change when you increase class overlap in the data?

## Exercise 11. Multiclass implementation task

Implement softmax regression from scratch on a three-class synthetic dataset.

Requirements:

1. use one-hot targets;
2. implement a numerically stable softmax by subtracting the rowwise maximum score before exponentiating;
3. optimize multiclass cross-entropy with gradient descent; and
4. visualize the learned decision regions in the plane.

Then answer:

5. Why is the score-shift trick numerically useful?
6. How do the predicted class probabilities behave near the class boundaries?

## Exercise 12. Logistic regression versus least squares classification

Suppose someone fits a linear model to binary labels using squared loss instead of cross-entropy.

1. State one reason the squared-loss predictor is not naturally a probability model.
2. State one reason cross-entropy is better matched to Bernoulli likelihood.
3. Give one situation in which logistic regression is still limited, even though it is the correct canonical linear classifier.

## Exercise 13. Exponential-family and maximum-entropy reflection

Write a short response to the following prompt.

Why is logistic regression best understood not merely as "a sigmoid on top of a linear function," but as a probabilistic classifier justified by both exponential-family structure and maximum-entropy reasoning?
