---
title: "Neural Networks from First Principles"
module: "06-neural-networks"
lesson: "neural-networks-first-principles"
doc_type: "notes"
topic: "perceptrons-mlps-activations-losses"
status: "draft"
prerequisites:
  - "00-math-toolkit/multivariable-calculus"
  - "00-math-toolkit/linear-algebra"
  - "01-optimization/convexity-and-optimization"
  - "02-statistical-learning/statistical-learning-foundations"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "neural-networks"
  - "perceptron"
  - "mlp"
  - "backpropagation"
  - "activation-functions"
  - "loss-functions"
---

## Purpose

These notes build multilayer neural networks from the smallest possible ingredients:
affine maps, nonlinearities, compositions, and scalar losses.
The goal is to make the transition from linear models to deep models mathematically explicit rather than mysterious.

## Learning objectives

After working through this note, you should be able to:

- define a perceptron and explain how it differs from a linear regression or logistic regression model;
- write the forward pass of a multilayer perceptron in vector and layer notation;
- explain why nonlinear activations are necessary;
- compare common activation functions in terms of smoothness, saturation, and gradient flow;
- state the universal approximation theorem at an appropriate level and explain its geometric meaning;
- choose a loss function that matches a prediction task; and
- describe the loss-landscape viewpoint that motivates optimization and initialization choices.

## 1. Why neural networks are compositions

A neural network is a parameterized composition of simple functions.
At layer level, the pattern is

$$
\text{linear map} \;\longrightarrow\; \text{add bias} \;\longrightarrow\; \text{apply nonlinearity}.
$$

If $x \in \mathbb{R}^{d_0}$ is an input vector, a depth-$L$ feedforward network computes

$$
a^{[0]} = x,
$$

and for layers $\ell = 1,\dots,L$,

$$
z^{[\ell]} = W^{[\ell]} a^{[\ell-1]} + b^{[\ell]},
\qquad
a^{[\ell]} = \phi^{[\ell]}(z^{[\ell]}).
$$

Here:

- $W^{[\ell]} \in \mathbb{R}^{d_\ell \times d_{\ell-1}}$ is the weight matrix;
- $b^{[\ell]} \in \mathbb{R}^{d_\ell}$ is the bias vector;
- $z^{[\ell]} \in \mathbb{R}^{d_\ell}$ is the pre-activation;
- $a^{[\ell]} \in \mathbb{R}^{d_\ell}$ is the post-activation;
- $\phi^{[\ell]}$ is applied coordinatewise unless otherwise stated.

The final layer may use a different activation from the hidden layers.
For regression we often use the identity map in the output layer.
For binary classification we often use a sigmoid output.
For multiclass classification we often use a softmax output.

## 2. The perceptron as the first neural unit

The perceptron is the simplest historical neural model.
Given input $x \in \mathbb{R}^d$, parameters $w \in \mathbb{R}^d$ and $b \in \mathbb{R}$, it computes

$$
f(x) = \mathrm{sign}(w^\top x + b).
$$

Geometrically, the expression $w^\top x + b = 0$ defines a hyperplane.
The perceptron classifies by asking on which side of that hyperplane the point lies.

This is already a useful model, but it has a hard limitation:
it can represent only linearly separable decision rules.
XOR is the standard counterexample.
No single affine separator in $\mathbb{R}^2$ can classify XOR correctly.

> **ML Interpretation.** The perceptron is a thresholded linear model.
> It is not yet "deep learning," but it supplies the basic neuron picture: weighted input aggregation followed by a nonlinearity.

## 3. Why one layer is not enough

Suppose we stack affine maps without nonlinearities:

$$
z^{[1]} = W^{[1]}x + b^{[1]},
\qquad
z^{[2]} = W^{[2]}z^{[1]} + b^{[2]}.
$$

Substitute the first equation into the second:

$$
z^{[2]}
= W^{[2]}(W^{[1]}x + b^{[1]}) + b^{[2]}
= (W^{[2]}W^{[1]})x + (W^{[2]}b^{[1]} + b^{[2]}).
$$

So the composition of affine maps is still affine.
Without nonlinear activations, depth does not buy new expressive power.

This is the first structural reason neural networks need activation functions.
Nonlinearity prevents the entire network from collapsing into one linear transformation.

## 4. Multilayer perceptrons

A multilayer perceptron (MLP) is a feedforward network with one or more hidden layers.
For a two-hidden-layer network,

$$
\begin{aligned}
a^{[0]} &= x, \\
z^{[1]} &= W^{[1]}a^{[0]} + b^{[1]}, \qquad a^{[1]} = \phi^{[1]}(z^{[1]}), \\
z^{[2]} &= W^{[2]}a^{[1]} + b^{[2]}, \qquad a^{[2]} = \phi^{[2]}(z^{[2]}), \\
z^{[3]} &= W^{[3]}a^{[2]} + b^{[3]}, \qquad \hat{y} = \psi(z^{[3]}).
\end{aligned}
$$

The output activation $\psi$ depends on the task:

- regression: $\psi(z)=z$;
- binary classification: $\psi(z)=\sigma(z)$;
- multiclass classification: $\psi(z)=\mathrm{softmax}(z)$.

An MLP is therefore a nested composition

$$
\hat{y}(x)
=
\psi \circ T^{[L]} \circ \phi^{[L-1]} \circ T^{[L-1]} \circ \cdots \circ \phi^{[1]} \circ T^{[1]}(x),
$$

where each $T^{[\ell]}(u) = W^{[\ell]}u + b^{[\ell]}$ is affine.

## 5. Computational graph viewpoint

A feedforward network can be drawn as a directed acyclic graph whose nodes are intermediate quantities and whose edges represent functional dependence.

For example, in a two-hidden-layer network the graph includes nodes for:

- input $x$;
- each affine intermediate $z^{[1]}, z^{[2]}, z^{[3]}$;
- each activation $a^{[1]}, a^{[2]}, \hat{y}$;
- the scalar loss $\mathcal{L}(\hat{y}, y)$.

This viewpoint matters because backpropagation is not a mysterious neural trick.
It is reverse-mode automatic differentiation on that computational graph.

The graph also helps separate:

- forward evaluation, which computes values from input to loss;
- backward evaluation, which propagates sensitivities from loss back to parameters.

## 6. Activation functions and gradient behavior

Activation choice affects both expressivity and trainability.
Below, $u \in \mathbb{R}$ denotes a scalar pre-activation.

### 6.1 Sigmoid

$$
\sigma(u) = \frac{1}{1+e^{-u}}.
$$

Its derivative is

$$
\sigma'(u) = \sigma(u)\bigl(1-\sigma(u)\bigr).
$$

Properties:

- output range is $(0,1)$;
- smooth and monotone;
- saturates for large positive or negative $u$;
- derivative is at most $1/4$, so gradients can shrink across many layers.

Sigmoid is still natural in binary output layers because its range matches Bernoulli probabilities.
It is less common in deep hidden layers because saturation slows learning.

### 6.2 Hyperbolic tangent

$$
\tanh(u) = \frac{e^u - e^{-u}}{e^u + e^{-u}},
\qquad
\tanh'(u) = 1 - \tanh^2(u).
$$

Properties:

- output range is $(-1,1)$;
- zero-centered, which can help optimization relative to sigmoid;
- still saturates for large $|u|$;
- suffers vanishing gradients when units spend much of training in saturated regions.

### 6.3 ReLU

$$
\mathrm{ReLU}(u) = \max(0,u).
$$

A convenient derivative is

$$
\mathrm{ReLU}'(u) =
\begin{cases}
1, & u > 0, \\
0, & u < 0.
\end{cases}
$$

At $u=0$, the classical derivative is undefined, but a subgradient convention is used in practice.

Properties:

- piecewise linear and cheap to compute;
- does not saturate on the positive side;
- can produce sparse activations;
- can create "dead" neurons when many pre-activations stay negative, causing zero local gradient.

### 6.4 Leaky ReLU

For fixed $\alpha \in (0,1)$,

$$
\mathrm{LeakyReLU}(u) =
\begin{cases}
u, & u \geq 0, \\
\alpha u, & u < 0.
\end{cases}
$$

Properties:

- preserves a small negative-side gradient;
- reduces the dead-ReLU problem;
- remains piecewise linear and inexpensive;
- introduces an extra slope hyperparameter or learned variant if using PReLU.

### 6.5 GELU

The Gaussian error linear unit is commonly written as

$$
\mathrm{GELU}(u) = u \Phi(u),
$$

where $\Phi$ is the standard normal cdf.

Properties:

- smooth rather than piecewise linear;
- keeps small negative activations with reduced weight rather than zeroing them abruptly;
- works well in many transformer architectures;
- is more expensive to evaluate exactly, so approximations are often used.

### 6.6 Comparison table

| Activation | Range | Saturation | Derivative behavior | Typical use |
| --- | --- | --- | --- | --- |
| Sigmoid | $(0,1)$ | both tails | bounded by $1/4$ | binary outputs |
| Tanh | $(-1,1)$ | both tails | tends to $0$ in tails | some hidden layers, older RNNs |
| ReLU | $[0,\infty)$ | negative side only | $0$ or $1$ away from origin | default hidden layers in many MLPs/CNNs |
| Leaky ReLU | $\mathbb{R}$ | no full dead negative side | $\alpha$ or $1$ | hidden layers when dead ReLUs are a concern |
| GELU | $\mathbb{R}$ | soft attenuation | smooth, input-dependent | transformer-style hidden layers |

> **Common misconception.** A more nonlinear activation is not automatically better.
> Activation choice is constrained by optimization, initialization, architecture, and hardware considerations.

## 7. Universal approximation intuition

One classical universal approximation theorem says, informally, that a feedforward network with a single hidden layer and a sufficiently rich non-polynomial activation can approximate any continuous function on a compact subset of $\mathbb{R}^d$ arbitrarily well.

A representative statement is:

> **Universal Approximation Theorem (informal).**
> Let $K \subset \mathbb{R}^d$ be compact, and let $f : K \to \mathbb{R}$ be continuous.
> For suitable activations such as sigmoid-type or other non-polynomial activations, there exists a one-hidden-layer network
> $$
> x \mapsto \sum_{j=1}^m \alpha_j \phi(w_j^\top x + b_j)
> $$
> that approximates $f$ uniformly on $K$ to arbitrary accuracy as $m$ becomes large enough.

This theorem is important, but it should not be over-read.

It does **not** say:

- that the needed width is reasonable;
- that optimization will find the approximating parameters;
- that the approximation generalizes well from finite data; or
- that shallow networks are always preferable to deep ones.

### Geometric intuition

A hidden unit creates a soft feature detector based on the affine score $w^\top x + b$.
Combining many such detectors partitions the input space into regions and assembles piecewise or smoothly varying responses over those regions.

For ReLU networks, this geometric picture is especially concrete:

- each hidden unit introduces a hinge along a hyperplane;
- collections of hidden units carve the space into polyhedral regions;
- the network acts like different affine maps in different regions.

Depth matters because a deep network can reuse intermediate features compositionally.
Instead of building one enormous flat expansion, it can build edges from pixels, parts from edges, objects from parts, and decisions from objects.

## 8. Output layers and loss functions

The network itself produces a prediction $\hat{y}$ or logits $z^{[L]}$.
Training requires a scalar loss

$$
\mathcal{L}(\hat{y}, y),
$$

where $y$ is the target.

### 8.1 Mean squared error

For regression with target $y \in \mathbb{R}^k$,

$$
\mathcal{L}_{\mathrm{MSE}}(\hat{y}, y)
= \frac{1}{2}\|\hat{y} - y\|_2^2.
$$

Why the factor $1/2$?
It makes the gradient cleaner:

$$
\nabla_{\hat{y}} \mathcal{L}_{\mathrm{MSE}} = \hat{y} - y.
$$

MSE corresponds to a Gaussian noise model under a maximum-likelihood interpretation.

### 8.2 Binary cross-entropy

For binary classification with predicted probability $\hat{y} \in (0,1)$ and label $y \in \{0,1\}$,

$$
\mathcal{L}_{\mathrm{BCE}}(\hat{y}, y)
= -\bigl[y\log \hat{y} + (1-y)\log(1-\hat{y})\bigr].
$$

This is the negative log-likelihood of a Bernoulli model.
When $\hat{y}=\sigma(z)$, the derivative with respect to the logit simplifies:

$$
\frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y.
$$

That simplification is one reason the sigmoid-plus-cross-entropy pairing is so important.

### 8.3 Multiclass cross-entropy

For $C$ classes, let $z \in \mathbb{R}^C$ be logits and

$$
\hat{y}_c = \frac{e^{z_c}}{\sum_{j=1}^C e^{z_j}}
$$

the softmax probabilities.
With one-hot target vector $y \in \{0,1\}^C$,

$$
\mathcal{L}_{\mathrm{CE}}(\hat{y}, y)
= -\sum_{c=1}^C y_c \log \hat{y}_c.
$$

Again the logit gradient simplifies to

$$
\nabla_z \mathcal{L}_{\mathrm{CE}} = \hat{y} - y.
$$

### 8.4 Margin and surrogate losses

Although cross-entropy is standard in deep classification, it is not the only choice.
Hinge-style and other surrogate losses can also be used, especially when margin behavior is the main concern.

The larger lesson is structural:
the loss function specifies what it means for the network to be wrong.
Architecture and optimization only make sense relative to that choice.

## 9. Empirical risk for neural networks

Given a dataset

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n,
$$

and network parameters

$$
\theta = \{W^{[\ell]}, b^{[\ell]}\}_{\ell=1}^L,
$$

the empirical risk is

$$
J(\theta)
=
\frac{1}{n}\sum_{i=1}^n \mathcal{L}(f_\theta(x_i), y_i).
$$

Training a neural network means solving

$$
\min_\theta J(\theta),
$$

usually by stochastic gradient methods rather than closed-form algebra.

Unlike ordinary least squares, this objective is generally nonconvex because of layerwise compositions and nonlinear activations.

## 10. Loss landscapes

The loss landscape is the graph of $J(\theta)$ over parameter space.
For modern networks the parameter space is extremely high-dimensional, so we cannot view the full landscape directly.
Still, the concept is useful.

It explains why initialization, normalization, and optimizer design matter:

- sharp local curvature can destabilize large steps;
- flat or nearly flat regions can slow progress;
- saturating units can create weak gradients;
- symmetries in parameterization can create many equivalent solutions.

Two-dimensional slices and interpolation plots are common visualization tools.
Useful references for this viewpoint include:

- Goodfellow, Bengio, and Courville, Chapter 8, for optimization pathologies;
- Li et al., *Visualizing the Loss Landscape of Neural Nets* (2018), for filter-normalized landscape plots;
- practical notebook visualizations in later modules that inspect trajectories and curvature proxies.

> **Caveat.** "The loss landscape" is not a single picture that explains all deep learning behavior.
> Visualizations depend on projection choices, parameter symmetries, normalization, and scale.

## 11. Why backpropagation is the central derivation

The forward map of a network is just repeated function composition.
The real algorithmic question is how to compute

$$
\nabla_\theta J(\theta)
$$

efficiently.

Naively differentiating each parameter independently would repeat enormous amounts of work.
Backpropagation avoids that waste by reusing intermediate derivatives from the computational graph.

The key pattern is:

1. evaluate all intermediate activations in the forward pass;
2. differentiate the scalar loss with respect to the final layer outputs;
3. propagate these sensitivities backward layer by layer via the chain rule.

The full derivation appears in [backpropagation.md](../derivations/backpropagation.md).

## 12. Limits of the first-principles picture

The basic MLP story is foundational, but incomplete.
Real deep learning systems must also address:

- parameter initialization;
- normalization and scale control;
- regularization;
- batching and vectorized implementations;
- optimization with momentum or adaptive methods;
- architectural priors such as convolution, attention, and recurrence.

Those topics build on this note rather than replacing it.

## 13. Summary

Neural networks are compositions of affine maps and nonlinear activations trained by minimizing a scalar loss.
The perceptron gives the first linear-threshold picture, but hidden layers with nonlinear activations produce richer representations.
Universal approximation explains expressive potential at a high level, while activation and loss choices determine whether that potential is trainable in practice.
The remaining central step is to derive backpropagation carefully.
