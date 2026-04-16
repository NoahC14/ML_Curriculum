---
title: "Statistical Learning Exercises"
module: "02-statistical-learning"
lesson: "statistical-learning-exercises"
doc_type: "exercise"
topic: "statistical-learning-theory"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability"
  - "01-optimization/convexity-and-optimization"
  - "02-statistical-learning/statistical-learning-foundations"
updated: "2026-04-11"
owner: "curriculum-team"
tags:
  - "statistical-learning"
  - "generalization"
  - "bias-variance"
  - "cross-validation"
  - "regularization"
---

## Purpose

These exercises reinforce the basic concepts of supervised statistical learning: empirical risk, generalization, overfitting, capacity, regularization, and model selection.

## Exercise 1. True risk versus empirical risk

**Taxonomy**

- `difficulty`: `foundational`
- `type`: `analysis`
- `tags`: `true-risk`, `empirical-risk`, `generalization`

Let $\mathcal{F}$ be a class of real-valued predictors and let $\ell$ be squared loss.

1. Write the definitions of true risk and empirical risk for a predictor $f \in \mathcal{F}$.
2. Explain in words why $\widehat{R}_n(f)$ is observable but $R(f)$ is not.
3. Give one reason why minimizing $\widehat{R}_n(f)$ may fail to minimize $R(f)$.

## Exercise 2. ERM as optimization

Consider linear regression with predictors $f_w(x) = w^\top x$ and squared loss.

1. Write the ERM objective for a dataset $\{(x_i,y_i)\}_{i=1}^n$.
2. Express the objective in matrix form using a design matrix $X$ and target vector $y$.
3. State one property of this objective that makes it accessible to optimization methods from Module 01.

## Exercise 3. Reading an overfitting curve

Suppose model complexity increases along the horizontal axis, while training and validation error are plotted on the vertical axis.

1. Sketch the usual qualitative shape of the training error curve.
2. Sketch the usual qualitative shape of the validation error curve.
3. Identify a region associated with underfitting and a region associated with overfitting.
4. Explain why the minimizer of validation error is not typically the minimizer of training error.

## Exercise 4. Approximation and estimation

Give a short explanation of each term:

1. approximation error;
2. estimation error; and
3. irreducible noise.

Then answer:

4. What usually happens to approximation error when the hypothesis class becomes richer?
5. What can happen to estimation error when the hypothesis class becomes richer but sample size stays fixed?

## Exercise 5. Squared-loss bias-variance decomposition

Assume

$$
f^\star(x) = \mathbb{E}[Y \mid X=x].
$$

1. Starting from $(Y - \widehat{f}_S(x))^2$, add and subtract $f^\star(x)$.
2. Expand the square and identify the cross term.
3. Explain why that cross term has conditional expectation zero.
4. Define the mean predictor $\bar{f}(x) = \mathbb{E}_S[\widehat{f}_S(x)]$.
5. Complete the decomposition into noise, squared bias, and variance.

## Exercise 6. Capacity intuition

Answer the following conceptual questions.

1. What does it mean for a hypothesis class to shatter a set of points?
2. Why does a large VC dimension suggest greater overfitting risk when the sample is small?
3. In informal terms, what is Rademacher complexity measuring?
4. Why can norm constraints or parameter shrinkage reduce effective capacity even when the model family stays the same?

## Exercise 7. Regularization from two viewpoints

Consider ridge regression:

$$
\min_w \frac{1}{n}\sum_{i=1}^n (y_i - w^\top x_i)^2 + \lambda \|w\|_2^2.
$$

1. Explain how the penalty changes the optimization problem.
2. Explain why the penalty can improve generalization even if it increases training error.
3. State the Bayesian prior corresponding to the $\ell_2$ penalty.
4. Explain in one paragraph why the optimization and Bayesian interpretations are compatible rather than contradictory.

## Exercise 8. Cross-validation workflow

You must choose the polynomial degree for a regression model.

1. Describe a 5-fold cross-validation procedure for this task.
2. At what step is the degree chosen?
3. After the degree is chosen, what data should be used to fit the final model?
4. Why should the test set not participate in the cross-validation loop?

## Exercise 9. Leakage diagnosis

A practitioner standardizes all features using the full dataset, then runs K-fold cross-validation on the transformed data.

1. Explain why this causes data leakage.
2. Describe the correct procedure.
3. Explain why pipelines are useful in this setting.

## Exercise 10. Applied notebook analysis

Run [SL-01-bias-variance-demo.ipynb](../notebooks/SL-01-bias-variance-demo.ipynb).

1. Which polynomial degrees achieve the smallest training error?
2. Which degrees achieve the smallest cross-validation error?
3. Compare the low-degree and high-degree fitted curves. Which one shows higher bias? Which one shows higher variance?
4. Change the sample size and noise level. How do the train, test, and cross-validation curves respond?

## Exercise 11. Short proof-style question

Suppose $\mathcal{F}_1 \subseteq \mathcal{F}_2$.

1. Show that

$$
\inf_{f \in \mathcal{F}_2} \widehat{R}_n(f)
\leq
\inf_{f \in \mathcal{F}_1} \widehat{R}_n(f).
$$

2. Explain why the analogous statement for true risk does not imply that the larger class always gives better test performance after fitting on finite data.

## Exercise 12. Reflection

Write a short response to the following prompt:

Why is statistical learning theory best understood as the interaction of probability, optimization, and model class design rather than as optimization alone?
