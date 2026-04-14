---
title: "Statistical Learning Foundations"
module: "02-statistical-learning"
lesson: "statistical-learning-foundations"
doc_type: "notes"
topic: "statistical-learning-theory"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
  - "00-math-toolkit/probability"
  - "01-optimization/convexity-and-optimization"
updated: "2026-04-11"
owner: "curriculum-team"
tags:
  - "statistical-learning"
  - "empirical-risk-minimization"
  - "generalization"
  - "bias-variance"
  - "regularization"
  - "cross-validation"
---

## Purpose

These notes introduce the core language of supervised statistical learning: data-generating distributions, loss functions, empirical risk minimization, generalization, overfitting, bias-variance tradeoffs, capacity control, and regularization.
The goal is to make clear why learning is not only an optimization problem but also an inference problem under finite data.

## Learning objectives

After working through this note, you should be able to:

- define true risk, empirical risk, hypothesis class, and empirical risk minimization;
- explain why low training error does not by itself imply good predictive performance;
- distinguish approximation error, estimation error, and irreducible noise;
- state and interpret the squared-loss bias-variance decomposition;
- explain why model capacity influences overfitting and generalization;
- give an intuitive account of VC dimension and Rademacher complexity;
- motivate regularization from both optimization and Bayesian viewpoints; and
- describe how validation and cross-validation are used to choose model complexity.

## 1. Supervised learning as a probabilistic problem

We observe examples

$$
S = \{(x_i, y_i)\}_{i=1}^n
$$

drawn independently from an unknown distribution $P$ on $\mathcal{X} \times \mathcal{Y}$.
A predictor is a function

$$
f : \mathcal{X} \to \mathcal{Y},
$$

chosen from a hypothesis class $\mathcal{F}$.

To evaluate a predictor, we choose a loss function

$$
\ell(f(x), y),
$$

which measures the penalty for predicting $f(x)$ when the true outcome is $y$.
Examples include:

- squared loss: $\ell(\hat{y}, y) = (\hat{y} - y)^2$ for regression;
- absolute loss: $\ell(\hat{y}, y) = |\hat{y} - y|$;
- zero-one loss: $\ell(\hat{y}, y) = \mathbf{1}\{\hat{y} \neq y\}$ for classification; and
- log loss for probabilistic classification.

The crucial point is that the data are random.
Even if the learning algorithm is deterministic given $S$, the learned predictor depends on the random sample.

## 2. True risk and empirical risk

The quantity we actually care about is **true risk** or **population risk**:

$$
R(f) = \mathbb{E}_{(X,Y) \sim P}[\ell(f(X), Y)].
$$

This expectation is taken over fresh draws from the same data-generating process.
It answers the predictive question directly: how well will the model perform on new examples?

Because $P$ is unknown, we cannot usually compute $R(f)$ exactly.
Instead we use the **empirical risk**

$$
\widehat{R}_n(f) = \frac{1}{n}\sum_{i=1}^n \ell(f(x_i), y_i).
$$

Empirical risk is observable because it depends only on the training sample.
The most common learning principle is:

$$
\widehat{f}_n \in \arg\min_{f \in \mathcal{F}} \widehat{R}_n(f),
$$

which is called **empirical risk minimization** (ERM).

This is where optimization enters.
Once $\mathcal{F}$ and $\ell$ are fixed, learning becomes the problem of solving an optimization problem over parameters or functions.

### Example: polynomial regression

Suppose $\mathcal{F}$ is the set of degree-$d$ polynomials

$$
f_w(x) = \sum_{j=0}^d w_j x^j.
$$

With squared loss, ERM becomes least squares over the coefficient vector $w$.
For small $d$, the class may be too rigid to represent the target relation.
For large $d$, the class may interpolate noise.
The statistical question is not only whether the optimization problem can be solved, but whether the resulting predictor generalizes.

## 3. Generalization

The **generalization gap** of a predictor $f$ is

$$
R(f) - \widehat{R}_n(f).
$$

When this gap is small, training performance is a good proxy for future performance.
When the gap is large, the training set is giving an overly optimistic view.

Learning theory asks for conditions under which

$$
R(f) \approx \widehat{R}_n(f)
$$

uniformly over functions in $\mathcal{F}$.
This is difficult because the learner chooses $\widehat{f}_n$ after seeing the data.
We therefore need more than pointwise concentration for a single fixed function; we need control over the whole function class.

### A useful decomposition

For ERM, the excess population risk can be decomposed as

$$
R(\widehat{f}_n) - \inf_{f \in \mathcal{F}} R(f)
= \bigl(R(\widehat{f}_n) - \widehat{R}_n(\widehat{f}_n)\bigr)
+ \bigl(\widehat{R}_n(\widehat{f}_n) - \widehat{R}_n(f_{\mathcal{F}}^\star)\bigr)
+ \bigl(\widehat{R}_n(f_{\mathcal{F}}^\star) - R(f_{\mathcal{F}}^\star)\bigr),
$$

where

$$
f_{\mathcal{F}}^\star \in \arg\min_{f \in \mathcal{F}} R(f).
$$

The middle term is non-positive because $\widehat{f}_n$ minimizes empirical risk.
This tells us that the main challenge is controlling the discrepancy between empirical and population risk.

## 4. Overfitting and underfitting

**Underfitting** occurs when the hypothesis class is too restrictive or the optimization is too constrained.
Both training error and test error remain large because the model cannot capture the systematic structure of the data.

**Overfitting** occurs when a predictor adapts too closely to idiosyncrasies of the training sample.
Training error becomes very small, but test error increases because the model has learned noise or sample-specific variation.

An overfit model often has:

- low empirical risk;
- high sensitivity to perturbations of the sample; and
- poor out-of-sample performance.

The standard visual signature is:

- training error decreases as model flexibility increases; but
- validation or test error first decreases and then increases.

This is one reason we do not optimize model complexity solely by training error.

## 5. Approximation, estimation, and noise

A useful conceptual split is:

$$
\text{prediction error} = \text{approximation error} + \text{estimation error} + \text{irreducible noise}.
$$

- **Approximation error** measures how far the best function in $\mathcal{F}$ is from the true regression relationship.
- **Estimation error** measures how far the learned predictor is from the best function in $\mathcal{F}$ because only finite data are available.
- **Irreducible noise** comes from randomness in $Y$ even when $X$ is known.

Increasing model capacity usually reduces approximation error but increases estimation error.
This is the structural basis of the bias-variance tradeoff.

## 6. Bias-variance tradeoff

For squared loss regression, the expected prediction error can be decomposed into three pieces:

$$
\mathbb{E}\bigl[(Y - \widehat{f}(x))^2 \mid X=x\bigr]
= \sigma^2(x)
+ \operatorname{Bias}(\widehat{f}(x))^2
+ \operatorname{Var}(\widehat{f}(x)),
$$

where

$$
\sigma^2(x) = \operatorname{Var}(Y \mid X=x)
$$

is irreducible noise, and the expectation is over the random training sample and response at input $x$.

Interpretation:

- high bias means the average fitted predictor misses the true signal systematically;
- high variance means the fitted predictor changes too much across different training samples; and
- irreducible noise cannot be removed by the learner because it is inherent in the data-generating process.

Simple models often have high bias and low variance.
Highly flexible models often have low bias and high variance.

See [bias-variance-decomposition.md](../derivations/bias-variance-decomposition.md) for the full derivation under squared loss.

## 7. Capacity and why complexity matters

The class $\mathcal{F}$ matters because generalization depends on how expressive it is.
If $\mathcal{F}$ is too large, it can fit many random labelings or accidental fluctuations in the sample.

### VC dimension intuition

For classification, the **VC dimension** is the largest number of points that can be shattered by the hypothesis class.
To say that $\mathcal{F}$ shatters a set means that for every possible binary labeling of those points, there exists some $f \in \mathcal{F}$ that realizes that labeling exactly.

Interpretation:

- low VC dimension means limited combinatorial flexibility;
- high VC dimension means the class can implement many distinct decision patterns; and
- when VC dimension is large relative to sample size, memorization becomes easier.

Example intuition:

- thresholds on the line have VC dimension $1$;
- intervals on the line have VC dimension $2$; and
- affine separators in $\mathbb{R}^d$ have VC dimension $d+1$.

VC dimension does not tell the whole story, but it captures a core fact: expressive classes require more data to justify confidence in generalization.

### Rademacher complexity intuition

Rademacher complexity measures how well a function class correlates with random $\pm 1$ noise on a given sample.
If a class can fit random signs easily, it is too adaptable to the sample.

This gives a more data-dependent notion of capacity than VC dimension.
The practical intuition is:

- a class with small Rademacher complexity cannot chase arbitrary fluctuations;
- a class with large Rademacher complexity can align with random noise; and
- regularization or norm constraints often reduce Rademacher complexity.

We will not prove uniform convergence bounds here.
What matters for practice is the causal chain:

$$
\text{higher capacity} \Rightarrow \text{more flexible fits} \Rightarrow \text{larger overfitting risk unless controlled by data or regularization.}
$$

## 8. Regularization

Regularization modifies ERM to prefer simpler or more stable predictors.
A common form is

$$
\widehat{f}_{n,\lambda} \in \arg\min_{f \in \mathcal{F}}
\widehat{R}_n(f) + \lambda \Omega(f),
$$

where $\Omega(f)$ is a complexity penalty and $\lambda \geq 0$ controls its strength.

### Optimization viewpoint

From the optimization side, regularization:

- makes objectives better conditioned in many models;
- discourages extreme parameter values;
- reduces sensitivity to noise in the sample; and
- can turn ill-posed problems into well-posed ones.

Examples:

- ridge regression uses $\Omega(w) = \|w\|_2^2$;
- lasso uses $\Omega(w) = \|w\|_1$;
- early stopping behaves like an implicit regularizer in iterative methods.

Ridge regression is a useful example because adding $\lambda \|w\|_2^2$ shrinks coefficients and stabilizes the inverse problem when features are nearly collinear.

### Bayesian viewpoint

Regularization can also be understood as posterior inference under a prior.
For parameterized models $f_w$, minimizing

$$
\sum_{i=1}^n \ell(f_w(x_i), y_i) + \lambda \Omega(w)
$$

is often equivalent to maximum a posteriori (MAP) estimation:

$$
\arg\max_w p(w \mid S) \propto p(S \mid w)p(w).
$$

Examples:

- an $\ell_2$ penalty corresponds to a Gaussian prior on parameters;
- an $\ell_1$ penalty corresponds to a Laplace prior;
- stronger regularization corresponds to a more concentrated prior around simple parameter values.

The optimization view emphasizes numerical stability and complexity control.
The Bayesian view emphasizes prior beliefs and posterior shrinkage.
In many models, these are two descriptions of the same mathematical mechanism.

## 9. Model selection and cross-validation

A learner often must choose:

- model degree;
- regularization strength $\lambda$;
- feature representation; or
- other hyperparameters.

Choosing these on the training set alone biases the estimate toward overly flexible models.
We therefore separate fitting from evaluation.

### Validation split

In a basic train-validation-test workflow:

1. fit the model on the training set;
2. choose hyperparameters using validation performance; and
3. report final performance once on a held-out test set.

The validation set is for selection.
The test set is for final assessment.

### K-fold cross-validation

When data are limited, a single validation split can be unstable.
In **K-fold cross-validation**:

1. partition the training data into $K$ folds;
2. for each fold $k$, train on the other $K-1$ folds and evaluate on fold $k$;
3. average the validation scores across folds; and
4. choose the hyperparameter setting with the best average score.

This produces a more stable estimate of predictive performance for model selection than a single split.
Typical choices are $K=5$ or $K=10$.

Practical cautions:

- all preprocessing that learns from data must be fit inside each training fold;
- the test set must remain untouched until the end; and
- time series and grouped data require specialized validation schemes.

The notebook [SL-01-bias-variance-demo.ipynb](../notebooks/SL-01-bias-variance-demo.ipynb) demonstrates validation and cross-validation for polynomial regression.

## 10. Dataset design and evaluation

Generalization is not only about algorithms.
It also depends on whether the dataset supports the claim being made.

Questions to ask:

- Does the train-test split match the deployment distribution?
- Is the sample size large enough relative to model capacity?
- Are labels noisy, biased, or systematically missing?
- Are there duplicated or near-duplicated examples across splits?
- Does the evaluation metric match the actual decision problem?

Learning theory gives abstractions such as risk and capacity.
Dataset design determines whether those abstractions are being applied to the right problem.

## 11. Category theory insertion point

Category theory is not needed to define ERM, but it can clarify structure.
A learning pipeline can be viewed as a composition of morphisms:

$$
\text{data} \to \text{features} \to \text{hypothesis fitting} \to \text{evaluation}.
$$

This viewpoint is useful when asking which transformations preserve information relevant to prediction and how regularization constrains admissible morphisms.
For this module, that perspective is supplementary rather than foundational.

## 12. Unity Theory insertion point

Any Unity Theory framing should remain explicitly interpretive.
One admissible perspective is to treat generalization as a constrained transfer from local empirical observations to stable relational structure.
That may be philosophically suggestive, but the canonical machinery here remains probability, optimization, and statistical complexity.

## Summary

Statistical learning theory sits between optimization and probability.
Optimization explains how a predictor is fit.
Probability explains why that fitted predictor may or may not work on future data.

The core lessons are:

- ERM minimizes training loss, not population loss directly;
- generalization requires controlling the gap between empirical and true risk;
- capacity and sample size jointly determine overfitting risk;
- bias and variance capture a central prediction tradeoff under squared loss; and
- regularization and cross-validation are practical tools for controlling complexity.

## References

- Trevor Hastie, Robert Tibshirani, and Jerome Friedman, *The Elements of Statistical Learning*, 2nd ed.
- Shai Shalev-Shwartz and Shai Ben-David, *Understanding Machine Learning: From Theory to Algorithms*.
