---
title: "Kernel Methods"
module: "04-kernel-methods"
lesson: "kernel-methods"
doc_type: "notes"
topic: "feature-maps-kernels-svms-rkhs"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
  - "01-optimization/convexity-and-optimization"
  - "02-statistical-learning/statistical-learning-foundations"
  - "03-linear-models/logistic-regression"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "kernel-methods"
  - "feature-maps"
  - "kernel-trick"
  - "svm"
  - "rkhs"
  - "margin-based-learning"
---

## Purpose

These notes develop the bridge from linear models to nonlinear learning.
The main idea is that a model may stay linear in a transformed feature space even when it becomes nonlinear in the original input coordinates.
Kernel methods make that transformation implicit, which lets us reason about rich hypothesis classes through inner products and convex optimization.

## Learning objectives

After working through this note, you should be able to:

- define a feature map and explain how nonlinear decision functions can arise from linear models in transformed coordinates;
- define a kernel function and explain when it can replace explicit feature construction;
- derive the kernelized prediction rule from inner products in feature space;
- explain maximum-margin classification and why support vectors determine the solution;
- connect the SVM dual problem to pairwise kernel evaluations;
- give an intuitive account of what it means for a kernel to define an RKHS; and
- compare common kernels and describe when each may or may not fit the geometry of a dataset.

## 1. From linear models to nonlinear decision boundaries

A linear predictor in the original input space $\mathcal{X} \subseteq \mathbb{R}^d$ has the form

$$
f(x) = w^\top x + b,
$$

with $w \in \mathbb{R}^d$ and $b \in \mathbb{R}$.
The decision boundary for binary classification is the hyperplane

$$
\{x \in \mathbb{R}^d : w^\top x + b = 0\}.
$$

This is expressive when the problem is approximately linear in the observed coordinates, but it fails on patterns such as concentric circles or XOR-like arrangements.

One remedy is to enlarge the representation by choosing a feature map

$$
\phi : \mathcal{X} \to \mathcal{H},
$$

where $\mathcal{H}$ is another inner-product space, often $\mathbb{R}^p$ for some larger $p$.
We then fit a linear model in feature space:

$$
f(x) = \langle w, \phi(x) \rangle_{\mathcal{H}} + b.
$$

Although this predictor is linear in $\phi(x)$, it can be nonlinear in the original coordinates $x$.

### Example: polynomial features

Suppose $x = (x_1, x_2)^\top \in \mathbb{R}^2$ and define

$$
\phi(x) =
\begin{bmatrix}
1 \\
\sqrt{2}x_1 \\
\sqrt{2}x_2 \\
x_1^2 \\
\sqrt{2}x_1x_2 \\
x_2^2
\end{bmatrix}
\in \mathbb{R}^6.
$$

Then a linear predictor in $\phi(x)$ becomes a quadratic function of $(x_1,x_2)$.
The resulting decision boundary in input space is no longer a hyperplane.

This is the first conceptual move of kernel methods:

- keep the optimization problem linear in parameters; and
- let nonlinearity enter through the representation.

## 2. Inner products are the key computational object

Many linear learning algorithms depend on training examples only through inner products.
If a solution in feature space can be written as a linear combination of mapped training points,

$$
w = \sum_{i=1}^n \alpha_i \phi(x_i),
$$

then prediction at a new point $x$ becomes

$$
f(x)
= \left\langle \sum_{i=1}^n \alpha_i \phi(x_i), \phi(x) \right\rangle_{\mathcal{H}} + b
= \sum_{i=1}^n \alpha_i \langle \phi(x_i), \phi(x) \rangle_{\mathcal{H}} + b.
$$

The mapped vectors never appear alone.
Only their pairwise inner products matter.

This motivates the central definition.

### Definition: kernel

A **kernel function** is a function

$$
K : \mathcal{X} \times \mathcal{X} \to \mathbb{R}
$$

for which there exists an inner-product space $\mathcal{H}$ and feature map $\phi : \mathcal{X} \to \mathcal{H}$ such that

$$
K(x,z) = \langle \phi(x), \phi(z) \rangle_{\mathcal{H}}
$$

for all $x,z \in \mathcal{X}$.

When such a representation exists, we can replace explicit features with kernel evaluations.
This replacement is the **kernel trick**.

## 3. Kernel matrices and positive semidefiniteness

Given training inputs $x_1,\dots,x_n$, define the kernel matrix

$$
K \in \mathbb{R}^{n \times n},
\qquad
K_{ij} = K(x_i,x_j).
$$

If $K(x,z) = \langle \phi(x), \phi(z) \rangle_{\mathcal{H}}$, then the matrix $K$ is symmetric and positive semidefinite.
Indeed, for any $c \in \mathbb{R}^n$,

$$
c^\top K c
= \sum_{i=1}^n \sum_{j=1}^n c_i c_j K(x_i,x_j)
= \sum_{i=1}^n \sum_{j=1}^n c_i c_j \langle \phi(x_i), \phi(x_j) \rangle
= \left\| \sum_{i=1}^n c_i \phi(x_i) \right\|_{\mathcal{H}}^2
\geq 0.
$$

So valid kernels generate Gram matrices, exactly as in ordinary linear algebra.

The converse direction is deeper: under mild conditions, symmetric positive-semidefinite kernels are precisely the kernels that come from some feature map.
For this module, the working rule is:

- if a kernel behaves like an inner product through a PSD Gram matrix, then it can support linear methods in an implicit feature space.

## 4. The kernel trick

Suppose an algorithm can be written entirely in terms of inner products between training points and test points.
Then the substitutions

$$
\langle x_i, x_j \rangle \rightsquigarrow K(x_i,x_j),
\qquad
\langle x_i, x \rangle \rightsquigarrow K(x_i,x)
$$

produce a nonlinear version of the method without explicit construction of $\phi(x)$.

This is useful in two cases:

1. the feature map is high-dimensional but finite; or
2. the feature map is infinite-dimensional, yet kernel evaluations remain simple.

For example, the radial basis function kernel

$$
K_{\mathrm{RBF}}(x,z)
= \exp\left(-\frac{\|x-z\|_2^2}{2\sigma^2}\right)
$$

corresponds to an infinite-dimensional feature space, but each pairwise evaluation is still easy to compute.

See [kernel-trick.md](../derivations/kernel-trick.md) for the formal derivation of the kernelized prediction rule.

## 5. Margin-based classification

Kernel methods are often taught together with support vector machines because the SVM optimization problem depends on data through inner products and therefore kernelizes cleanly.

For binary labels $y_i \in \{-1,+1\}$, a linear classifier predicts

$$
f(x) = \operatorname{sign}(w^\top x + b).
$$

If the data are linearly separable, we can ask for a separator that not only classifies correctly but does so with a large margin.

### Functional and geometric margins

For one example $(x_i,y_i)$, the **functional margin** is

$$
\widehat{\gamma}_i = y_i(w^\top x_i + b).
$$

Its sign records correctness, but its magnitude depends on rescaling $(w,b)$.
To remove that scaling ambiguity, define the **geometric margin**

$$
\gamma_i = \frac{y_i(w^\top x_i + b)}{\|w\|_2}.
$$

The hard-margin SVM chooses $(w,b)$ to maximize the minimum geometric margin over the training set.
After the standard normalization, this becomes the convex program

$$
\min_{w,b} \frac{1}{2}\|w\|_2^2
\quad
\text{subject to}
\quad
y_i(w^\top x_i + b) \geq 1
\quad \text{for } i=1,\dots,n.
$$

The objective prefers a small norm, which corresponds to a large margin under the normalization.

### Soft margin

Real datasets are rarely perfectly separable.
Introduce slack variables $\xi_i \geq 0$ and penalty parameter $C > 0$:

$$
\min_{w,b,\xi}
\frac{1}{2}\|w\|_2^2 + C\sum_{i=1}^n \xi_i
$$

subject to

$$
y_i(w^\top x_i + b) \geq 1 - \xi_i,
\qquad
\xi_i \geq 0.
$$

This balances two goals:

- large margin through $\|w\|_2^2$; and
- low classification error through the slack penalties.

The equivalent unconstrained loss uses the hinge penalty

$$
\ell_{\mathrm{hinge}}(f(x_i), y_i)
= \max(0, 1 - y_i f(x_i)).
$$

## 6. Why support vectors matter

In the dual SVM problem, the learned weight vector takes the form

$$
w^\star = \sum_{i=1}^n \alpha_i^\star y_i x_i.
$$

In the kernelized case this becomes

$$
w^\star = \sum_{i=1}^n \alpha_i^\star y_i \phi(x_i).
$$

Only examples with $\alpha_i^\star > 0$ contribute to the classifier.
These are the **support vectors**.

Geometrically, support vectors lie on or inside the margin boundary.
Statistically, they are the points the classifier finds hardest to ignore.
Computationally, they determine the prediction rule

$$
f(x)
= \sum_{i=1}^n \alpha_i^\star y_i K(x_i, x) + b^\star.
$$

This sparsity is one of the appealing features of SVMs.

See [svm-dual.md](../derivations/svm-dual.md) for the full Lagrangian derivation.

## 7. RKHS intuition: what a kernel buys you

The phrase **reproducing kernel Hilbert space** can sound more forbidding than the idea actually needed here.
For this module, the key intuition is functional rather than measure-theoretic.

### Informal picture

A kernel $K$ does not only define pairwise similarities.
It also determines a space of functions of the form

$$
f(\cdot) = \sum_{i=1}^n \alpha_i K(x_i, \cdot),
$$

completed in an appropriate norm.
That function space is the RKHS associated with $K$, usually denoted $\mathcal{H}_K$.

Two facts matter most:

1. every point $x$ determines a representer $K(x,\cdot) \in \mathcal{H}_K$;
2. evaluation at a point is an inner product:
   $$
   f(x) = \langle f, K(x,\cdot) \rangle_{\mathcal{H}_K}.
   $$

The second fact is the **reproducing property**.

### Why this matters in ML

The RKHS viewpoint tells us that:

- kernels define both a similarity notion and a hypothesis space;
- the norm $\|f\|_{\mathcal{H}_K}$ measures complexity in that hypothesis space; and
- regularized learning often searches for a function that fits the data while keeping RKHS norm small.

So a kernel is not merely a computational shortcut.
It encodes what kinds of functions are considered smooth, simple, or plausible.

### A useful mental model

Think of ordinary linear regression as choosing a vector in Euclidean space.
Kernel regression and SVMs instead choose a function from a function space whose geometry is induced by the kernel.

For this reason, different kernels express different inductive biases:

- polynomial kernels prefer algebraic interactions of bounded degree;
- RBF kernels prefer locally smooth decision surfaces; and
- linear kernels keep the original global hyperplane geometry.

We stop here on purpose.
The full functional-analytic theory of RKHSs belongs in more advanced study, but this intuition is enough to understand what kernels buy in machine learning.

## 8. Common kernels

### Linear kernel

$$
K(x,z) = x^\top z.
$$

This recovers ordinary linear methods.
It is the right baseline when the original representation is already informative.

### Polynomial kernel

$$
K(x,z) = (x^\top z + c)^p,
\qquad c \geq 0,\ p \in \mathbb{N}.
$$

This captures interaction terms up to degree $p$ without explicitly enumerating all polynomial features.
It is useful when low-degree nonlinear interactions matter and global smoothness is acceptable.

### RBF kernel

$$
K(x,z) = \exp(-\gamma \|x-z\|_2^2),
\qquad \gamma > 0.
$$

This is a local similarity kernel.
It is highly flexible and often performs well on medium-sized tabular problems, but it requires careful tuning of $\gamma$ and $C$.

### Sigmoid kernel

$$
K(x,z) = \tanh(\kappa x^\top z + c).
$$

This has historical interest because of its relation to neural nonlinearities, but its validity depends on parameter choices and it is used less often in modern practice.

## 9. Practical guidance

Kernel methods are strongest when:

- the dataset is small or medium-sized;
- the geometry of pairwise similarity matters;
- nonlinear structure is present but not so large-scale that deep models are required; and
- convex optimization and interpretable margins are desirable.

Their main limitations are:

- training and prediction can scale poorly with the number of examples because the kernel matrix is $n \times n$;
- kernel choice and hyperparameter tuning are problem dependent; and
- the implicit feature space can obscure interpretability in the original coordinates.

In modern ML, kernels remain important both as practical methods and as conceptual tools.
They make explicit the idea that representation and similarity jointly determine learnability.

## 10. Category theory and Unity Theory insertion points

### Category theory insertion point

At a light structural level, a feature map can be read as a morphism from an input representation object to a richer feature object, while the kernel packages the pullback of inner-product structure along that map.
This is useful only as an interpretive aid; the computational content still comes from linear algebra, convexity, and duality.

### Unity Theory insertion point

Kernel methods provide a clean companion example of how different coordinate descriptions can preserve task-relevant relational structure.
This should be read as an interpretive remark about representation, not as a replacement for the standard SVM and RKHS mathematics.

## 11. Summary

Kernel methods extend linear learning by moving the geometry rather than abandoning linear structure.
The core chain of ideas is:

1. choose or imply a feature map $\phi$;
2. observe that many algorithms need only inner products;
3. replace those inner products with kernel evaluations;
4. solve a convex optimization problem, often in dual variables; and
5. interpret the result as a function in a kernel-defined hypothesis space.

This module is the mathematical gateway from explicit finite-dimensional features to implicit feature spaces and function-space viewpoints that recur throughout machine learning.
