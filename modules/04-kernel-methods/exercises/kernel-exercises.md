---
title: "Kernel Methods Exercises"
module: "04-kernel-methods"
lesson: "kernel-exercises"
doc_type: "exercises"
topic: "feature-maps-kernels-svms"
status: "draft"
prerequisites:
  - "04-kernel-methods/kernel-methods"
  - "04-kernel-methods/kernel-trick"
  - "04-kernel-methods/svm-dual"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "kernel-methods"
  - "exercises"
  - "svm"
  - "rkhs"
---

## Purpose

These exercises reinforce the main ideas of the module: explicit feature maps, kernelized prediction, the SVM dual, and RKHS intuition.
They mix algebra, geometry, and computational interpretation.

## Exercise 1: explicit quadratic feature map

Let $x=(x_1,x_2)^\top$ and define

$$
\phi(x) =
\begin{bmatrix}
1 \\
\sqrt{2}x_1 \\
\sqrt{2}x_2 \\
x_1^2 \\
\sqrt{2}x_1x_2 \\
x_2^2
\end{bmatrix}.
$$

1. Compute $\langle \phi(x), \phi(z) \rangle$ explicitly.
2. Show that the result equals $(x^\top z + 1)^2$.
3. Explain why a linear classifier in $\phi(x)$ becomes quadratic in the original coordinates.

## Exercise 2: PSD Gram matrix

Let $K(x,z)=\langle \phi(x),\phi(z)\rangle$ for some feature map into an inner-product space.

1. For a sample $x_1,\dots,x_n$, define the Gram matrix $G_{ij}=K(x_i,x_j)$.
2. Show that $G$ is symmetric.
3. Prove that $c^\top G c \geq 0$ for every $c \in \mathbb{R}^n$.
4. State in words why positive semidefiniteness is the algebraic signature of a valid kernel.

## Exercise 3: kernelized ridge predictor

Assume a regularized least-squares predictor in feature space has the form

$$
f(x) = \langle w, \phi(x) \rangle
$$

with

$$
w = \sum_{i=1}^n \alpha_i \phi(x_i).
$$

1. Write the prediction rule $f(x)$ using only kernel evaluations.
2. Write the vector of training predictions using the Gram matrix.
3. Explain how this differs conceptually from explicit polynomial feature engineering.

## Exercise 4: hard-margin geometry

For a separating hyperplane $w^\top x+b=0$:

1. define the functional margin and the geometric margin;
2. show how rescaling $(w,b)$ changes the functional margin;
3. show that the geometric margin is invariant to this rescaling; and
4. explain why maximizing margin can improve robustness to perturbations.

## Exercise 5: derive the dual constraints

Starting from the soft-margin SVM primal problem,

$$
\min_{w,b,\xi}
\frac{1}{2}\|w\|_2^2 + C\sum_{i=1}^n \xi_i
$$

subject to

$$
y_i(w^\top x_i+b)\geq 1-\xi_i,
\qquad
\xi_i \geq 0,
$$

derive:

1. the Lagrangian;
2. the stationarity condition for $w$;
3. the equality constraint involving $\alpha_i$ and $y_i$;
4. the box constraint $0 \leq \alpha_i \leq C$.

Do not skip intermediate steps.

## Exercise 6: support vectors

Suppose $\alpha^\star$ solves the SVM dual.

1. Show that if $\alpha_i^\star = 0$, then example $i$ does not affect the prediction rule.
2. Explain why only points with positive dual weight are called support vectors.
3. Interpret the cases $0 < \alpha_i^\star < C$ and $\alpha_i^\star = C$ in terms of margin behavior and slack.

## Exercise 7: compare kernels conceptually

For each of the following kernels, give one situation where it may be appropriate and one limitation:

- linear kernel;
- polynomial kernel;
- RBF kernel;
- sigmoid kernel.

Your answer should mention both geometry and computational tradeoffs.

## Exercise 8: RKHS intuition

Answer the following without invoking full functional analysis.

1. What does it mean informally for a kernel to define a hypothesis space of functions?
2. Why does the expression $f(x)=\langle f, K(x,\cdot)\rangle_{\mathcal{H}_K}$ matter?
3. How does the RKHS norm play a role similar to parameter norm regularization in finite-dimensional models?

## Exercise 9: empirical comparison

Using a two-dimensional nonlinear classification dataset such as concentric circles or two moons:

1. fit a linear SVM;
2. fit polynomial and RBF kernel SVMs;
3. compare training accuracy, test accuracy, and number of support vectors;
4. visualize the decision boundaries; and
5. explain which kernel best matches the data geometry.

## Exercise 10: limitations and scaling

Write a short note addressing the following.

1. Why can kernel methods become expensive as $n$ grows?
2. What role does the kernel matrix play in that scaling issue?
3. Why might a deep model be preferred over a kernel method on some modern large-scale problems?
4. Give one reason kernels remain pedagogically important even when they are not the default industrial choice.
