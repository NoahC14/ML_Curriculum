---
title: "Gradient Derivation for Logistic Regression"
module: "03-linear-models"
lesson: "logistic-gradient"
doc_type: "derivation"
topic: "optimization"
status: "draft"
prerequisites:
  - "00-math-toolkit/multivariable-calculus"
  - "00-math-toolkit/information-theory"
  - "03-linear-models/logistic-regression"
updated: "2026-04-11"
owner: "curriculum-team"
tags:
  - "logistic-regression"
  - "gradient"
  - "cross-entropy"
  - "softmax"
  - "maximum-likelihood"
---

## Purpose

This derivation shows, step by step, how the gradient of the logistic-regression log-likelihood reduces to a simple residual form.
That simplification is one of the main reasons logistic regression is such an important teaching model.

## 1. Setup and notation

Let

$$
\tilde{x}_i \in \mathbb{R}^{d+1},
\qquad
\theta \in \mathbb{R}^{d+1},
\qquad
z_i = \theta^\top \tilde{x}_i.
$$

For binary logistic regression, define

$$
p_i = \sigma(z_i) = \frac{1}{1+e^{-z_i}}.
$$

The observed label satisfies $y_i \in \{0,1\}$.

The conditional model is

$$
p_\theta(y_i \mid x_i)
= p_i^{y_i}(1-p_i)^{1-y_i}.
$$

## 2. Binary log-likelihood

The full log-likelihood over $n$ examples is

$$
\ell(\theta)
= \sum_{i=1}^n
\left[
y_i \log p_i + (1-y_i)\log(1-p_i)
\right].
$$

We will derive the gradient of both $\ell(\theta)$ and the negative average log-likelihood

$$
\mathcal{L}(\theta) = -\frac{1}{n}\ell(\theta).
$$

## 3. Useful derivative of the sigmoid

Start from

$$
\sigma(z) = (1+e^{-z})^{-1}.
$$

Differentiate:

$$
\sigma'(z)
= -(1+e^{-z})^{-2}(-e^{-z})
= \frac{e^{-z}}{(1+e^{-z})^2}.
$$

Now rewrite this in terms of $\sigma(z)$ itself:

$$
\sigma(z) = \frac{1}{1+e^{-z}},
\qquad
1-\sigma(z) = \frac{e^{-z}}{1+e^{-z}}.
$$

Therefore

$$
\sigma'(z) = \sigma(z)\bigl(1-\sigma(z)\bigr).
$$

## 4. Per-example gradient

Consider the contribution of one example:

$$
\ell_i(\theta) = y_i \log p_i + (1-y_i)\log(1-p_i).
$$

Using the chain rule,

$$
\nabla_\theta \ell_i
= y_i \frac{1}{p_i}\nabla_\theta p_i
+ (1-y_i)\frac{1}{1-p_i}\nabla_\theta(1-p_i).
$$

Since

$$
\nabla_\theta(1-p_i) = -\nabla_\theta p_i,
$$

we get

$$
\nabla_\theta \ell_i
= \left(
\frac{y_i}{p_i}
- \frac{1-y_i}{1-p_i}
\right)\nabla_\theta p_i.
$$

Next compute $\nabla_\theta p_i$.
Because $p_i = \sigma(z_i)$ and $z_i = \theta^\top \tilde{x}_i$,

$$
\nabla_\theta p_i
= \sigma'(z_i)\nabla_\theta z_i
= p_i(1-p_i)\tilde{x}_i.
$$

Substitute this into the previous expression:

$$
\nabla_\theta \ell_i
= \left(
\frac{y_i}{p_i}
- \frac{1-y_i}{1-p_i}
\right)p_i(1-p_i)\tilde{x}_i.
$$

Now simplify the scalar factor:

$$
\left(
\frac{y_i}{p_i}
- \frac{1-y_i}{1-p_i}
\right)p_i(1-p_i)
= y_i(1-p_i) - (1-y_i)p_i.
$$

Expand:

$$
y_i(1-p_i) - (1-y_i)p_i
= y_i - y_ip_i - p_i + y_ip_i
= y_i - p_i.
$$

Therefore

$$
\nabla_\theta \ell_i = (y_i-p_i)\tilde{x}_i.
$$

This is the central simplification.

## 5. Dataset gradient

Summing over examples,

$$
\nabla_\theta \ell(\theta)
= \sum_{i=1}^n (y_i-p_i)\tilde{x}_i.
$$

Let

$$
X =
\begin{bmatrix}
\tilde{x}_1^\top \\
\vdots \\
\tilde{x}_n^\top
\end{bmatrix}
\in \mathbb{R}^{n \times (d+1)},
\qquad
p =
\begin{bmatrix}
p_1 \\
\vdots \\
p_n
\end{bmatrix}
\in \mathbb{R}^n.
$$

Then

$$
\nabla_\theta \ell(\theta) = X^\top(y-p).
$$

For the negative average log-likelihood,

$$
\mathcal{L}(\theta) = -\frac{1}{n}\ell(\theta),
$$

we obtain

$$
\nabla_\theta \mathcal{L}(\theta)
= \frac{1}{n}X^\top(p-y).
$$

This is the form usually implemented in practice.

## 6. Alternative derivation by rewriting the log-likelihood

The same result appears even faster if we first rewrite the objective in terms of the score $z_i$.
Since

$$
\log \sigma(z_i) = z_i - \log(1+e^{z_i}),
$$

and

$$
\log(1-\sigma(z_i)) = -\log(1+e^{z_i}),
$$

the per-example log-likelihood becomes

$$
\ell_i(\theta)
= y_i z_i - \log(1+e^{z_i}).
$$

Differentiate directly:

$$
\nabla_\theta \ell_i
= y_i \nabla_\theta z_i
- \frac{e^{z_i}}{1+e^{z_i}}\nabla_\theta z_i.
$$

Because

$$
\frac{e^{z_i}}{1+e^{z_i}} = \sigma(z_i) = p_i
$$

and

$$
\nabla_\theta z_i = \tilde{x}_i,
$$

we again obtain

$$
\nabla_\theta \ell_i = (y_i-p_i)\tilde{x}_i.
$$

This second route is often the cleanest algebraically.

## 7. Softmax gradient for multiclass logistic regression

Now let there be $K$ classes.
For one example, let

$$
\tilde{x} \in \mathbb{R}^{d+1},
\qquad
W \in \mathbb{R}^{K \times (d+1)},
\qquad
s = W\tilde{x} \in \mathbb{R}^K.
$$

The softmax probabilities are

$$
p_k = \frac{e^{s_k}}{\sum_{j=1}^K e^{s_j}},
\qquad k=1,\dots,K.
$$

Let the label be represented by a one-hot vector

$$
y \in \{0,1\}^K,
\qquad
\sum_{k=1}^K y_k = 1.
$$

The per-example negative log-likelihood is

$$
\mathcal{L}(W)
= -\sum_{k=1}^K y_k \log p_k.
$$

### 7.1 Derivative with respect to the scores

Write

$$
\log p_k = s_k - \log\left(\sum_{j=1}^K e^{s_j}\right).
$$

Then

$$
\mathcal{L}(W)
= -\sum_{k=1}^K y_k s_k
+ \sum_{k=1}^K y_k
\log\left(\sum_{j=1}^K e^{s_j}\right).
$$

Because $\sum_{k=1}^K y_k = 1$, this simplifies to

$$
\mathcal{L}(W)
= -\sum_{k=1}^K y_k s_k
+ \log\left(\sum_{j=1}^K e^{s_j}\right).
$$

Differentiate with respect to score coordinate $s_r$:

$$
\frac{\partial \mathcal{L}}{\partial s_r}
= -y_r + \frac{e^{s_r}}{\sum_{j=1}^K e^{s_j}}
= -y_r + p_r
= p_r - y_r.
$$

So, in vector form,

$$
\nabla_s \mathcal{L} = p - y.
$$

### 7.2 Derivative with respect to the parameters

Each score has the form

$$
s_k = w_k^\top \tilde{x}.
$$

Therefore

$$
\frac{\partial s_k}{\partial w_r}
=
\begin{cases}
\tilde{x}, & k=r, \\
0, & k\neq r.
\end{cases}
$$

By the chain rule,

$$
\nabla_{w_r}\mathcal{L}
= (p_r-y_r)\tilde{x}.
$$

Stacking the row gradients gives

$$
\nabla_W \mathcal{L} = (p-y)\tilde{x}^\top.
$$

For a dataset with design matrix $X \in \mathbb{R}^{n \times (d+1)}$, target matrix

$$
Y \in \mathbb{R}^{n \times K},
$$

and predicted probability matrix

$$
P \in \mathbb{R}^{n \times K},
$$

the negative average log-likelihood gradient is

$$
\nabla_W \mathcal{L}
= \frac{1}{n}(P-Y)^\top X.
$$

Equivalently,

$$
\nabla_{W^\top} \mathcal{L}
= \frac{1}{n}X^\top(P-Y),
$$

depending on whether one stores classes in rows or columns.

## 8. Interpretation

The binary and multiclass gradients share the same pattern:

$$
\text{gradient} = \text{features}^\top \times (\text{predicted probabilities} - \text{targets}).
$$

This matters conceptually because:

- the residual is probabilistic rather than squared-error residual;
- the update vanishes when predicted probabilities match the targets;
- the same pattern reappears later in neural-network output-layer gradients.

## 9. Scope note

The derivation above is exact for unregularized maximum likelihood.
If an $\ell_2$ penalty is added, for example

$$
\mathcal{L}_{\lambda}(\theta)
= \mathcal{L}(\theta) + \frac{\lambda}{2}\|w\|_2^2,
$$

then the gradient acquires the additional term $\lambda w$ on the non-bias coordinates.

## 10. Final result

For binary logistic regression,

$$
\nabla_\theta \mathcal{L}(\theta)
= \frac{1}{n}X^\top(p-y).
$$

For softmax regression,

$$
\nabla_W \mathcal{L}(W)
= \frac{1}{n}(P-Y)^\top X.
$$

Both formulas are consequences of the same structure:
log-likelihood from an exponential-family classifier with linear scores.
