---
title: "Initialization and Normalization"
module: "06-neural-networks"
lesson: "initialization-and-normalization"
doc_type: "notes"
topic: "gradient-scale-control"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
  - "00-math-toolkit/multivariable-calculus"
  - "01-optimization/convexity-and-optimization"
  - "06-neural-networks/backpropagation"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "neural-networks"
  - "initialization"
  - "normalization"
  - "xavier"
  - "he-initialization"
  - "batchnorm"
  - "layernorm"
---

## Purpose

These notes explain why neural-network training depends strongly on scale.
Initialization and normalization are not cosmetic implementation choices.
They control whether activations and gradients remain in numerically useful ranges as signals move forward and backward through the network.

## Learning objectives

After working through this note, you should be able to:

- explain why zero initialization fails for multilayer networks;
- describe how poor scale choices cause vanishing or exploding activations and gradients;
- state the basic intuition behind Xavier and He initialization;
- distinguish input normalization, batch normalization, and layer normalization; and
- connect these techniques to optimization stability rather than treating them as isolated tricks.

## 1. Why initialization matters

Backpropagation repeatedly multiplies by weight matrices and activation derivatives.
If those factors are badly scaled, the network can fail before learning begins.

Common failure modes include:

- hidden activations growing so large that nonlinearities saturate;
- hidden activations shrinking toward zero across layers;
- gradients exploding during the backward pass;
- gradients vanishing before they reach early layers;
- symmetric neurons evolving identically because they started identically.

Initialization is therefore the first attempt to preserve signal scale at depth.

## 2. Why zero initialization is wrong for hidden layers

Suppose all entries of $W^{[\ell]}$ are initialized to zero in a hidden layer.
Then every neuron in that layer computes the same pre-activation and the same output.
Because the backward pass also sees the same local values, all those neurons receive identical gradients.

Therefore they remain identical after every gradient update.
The network has many parameters, but they do not diversify into distinct features.

> **Conclusion.** Hidden units must break symmetry.
> Random initialization is not just convenient; it is structurally necessary.

Biases are different.
They are often initialized to zero or small constants without creating the same symmetry problem, because the weights already differentiate the units.

## 3. Variance propagation heuristic

To motivate standard initialization rules, consider a hidden layer

$$
z_j = \sum_{i=1}^{d_{\mathrm{in}}} W_{ji} a_i + b_j.
$$

Assume for a rough heuristic that:

- the inputs $a_i$ are mean-zero and independent with variance $\operatorname{Var}(a_i)=v_a$;
- the weights $W_{ji}$ are independent, mean-zero, and independent of the inputs;
- biases are small enough to ignore in the variance calculation.

Then

$$
\operatorname{Var}(z_j)
\approx
\sum_{i=1}^{d_{\mathrm{in}}} \operatorname{Var}(W_{ji} a_i)
=
d_{\mathrm{in}} \operatorname{Var}(W_{ji}) v_a.
$$

If we want the pre-activation variance to stay of the same order as the input variance, we want roughly

$$
d_{\mathrm{in}} \operatorname{Var}(W_{ji}) \approx 1.
$$

This suggests

$$
\operatorname{Var}(W_{ji}) \approx \frac{1}{d_{\mathrm{in}}}.
$$

The backward pass yields a similar condition involving the outgoing width.
Balancing both directions motivates the classical schemes below.

## 4. Xavier or Glorot initialization

For activations that are roughly symmetric around zero, such as $\tanh$, Xavier initialization aims to preserve variance across layers in both forward and backward propagation.

A common variance target is

$$
\operatorname{Var}(W_{ji}) \approx \frac{2}{d_{\mathrm{in}} + d_{\mathrm{out}}}.
$$

Equivalent sampling rules include:

- normal initialization with matching variance;
- uniform initialization over a symmetric interval with the same variance.

The main idea is not the exact constant.
It is scale matching between incoming and outgoing widths.

## 5. He initialization

ReLU-like activations zero out a substantial fraction of inputs.
That changes variance propagation.
He initialization compensates by increasing the variance:

$$
\operatorname{Var}(W_{ji}) \approx \frac{2}{d_{\mathrm{in}}}.
$$

This works well for ReLU and related piecewise linear activations because only part of the signal is active after the nonlinearity.

### Why the factor of 2 appears

Very roughly, if a ReLU passes about half the mass and zeros the rest, then activation variance after the nonlinearity is reduced.
Doubling the weight variance counteracts that reduction at initialization.

This is a heuristic moment calculation, not an exact theorem for every architecture or dataset.

## 6. Saturation and gradient flow

Initialization interacts with the activation function.

If weights are too large:

- sigmoids and tanh units saturate in the tails;
- local derivatives become small;
- backpropagated gradients shrink rapidly.

If weights are too small:

- the network can behave almost linearly near zero;
- activations may collapse toward a narrow range;
- gradient signals can become too weak to separate useful features.

Good initialization tries to place most units in the responsive regime of the chosen activation.

## 7. Input normalization

Before discussing internal normalization, start with data.
If input features have wildly different scales, optimization becomes harder because one coordinate direction can dominate the gradient geometry.

Common preprocessing choices include:

- centering each feature to mean zero;
- scaling each feature to unit variance;
- rescaling images to fixed ranges such as $[0,1]$ or standardized channel statistics.

This does not replace internal normalization, but it reduces avoidable conditioning problems at the first layer.

## 8. Batch normalization

Batch normalization applies normalization to intermediate activations using mini-batch statistics.
For one feature coordinate, a simplified training-time formula is

$$
\hat{z} = \frac{z - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \varepsilon}},
\qquad
y = \gamma \hat{z} + \beta,
$$

where:

- $\mu_{\mathcal{B}}$ and $\sigma_{\mathcal{B}}^2$ are batch mean and variance;
- $\varepsilon > 0$ is a numerical stabilizer;
- $\gamma$ and $\beta$ are learned scale and shift parameters.

### Why it helps

Batch normalization often helps because it:

- keeps intermediate scales more stable;
- permits larger learning rates in many settings;
- reduces sensitivity to initialization;
- can add a mild stochastic regularization effect because batch statistics fluctuate.

### Important caveat

It is common to hear that batch normalization works because it reduces "internal covariate shift."
That slogan is historically influential but incomplete.
A safer interpretation is that normalization improves optimization geometry and gradient behavior in practice.

### Training versus inference

During training, batch statistics are used.
During inference, running averages or other stored statistics are typically used instead.
This difference matters when batch sizes are small or data are nonstationary.

## 9. Layer normalization

Layer normalization normalizes across features within a single example rather than across examples in a batch.
For hidden vector $z \in \mathbb{R}^d$,

$$
\hat{z}_i
=
\frac{z_i - \mu(z)}{\sqrt{\sigma^2(z) + \varepsilon}},
$$

with learned per-feature scale and shift afterward.

This has two advantages:

- it does not depend on batch size;
- it is natural in sequence models and transformers, where batch statistics may be awkward or unstable.

### Batch norm versus layer norm

Batch normalization couples examples within a batch.
Layer normalization acts independently on each example.

As a result:

- batch norm is often effective in convolutional settings with reasonably large batches;
- layer norm is standard in transformers and many sequence models.

## 10. Other normalization and scale-control ideas

Several related ideas appear later in the curriculum:

- weight decay controls parameter magnitude through the objective;
- residual connections help preserve gradient pathways;
- gradient clipping limits exploding updates, especially in recurrent settings;
- weight normalization and RMS-style normalizers reparameterize or rescale model components differently from batch and layer normalization.

These techniques are related because they all try to keep optimization in a tractable numerical regime.

## 11. Initialization and normalization as a systems issue

The theory here is approximate.
Real networks violate the independence assumptions used in variance heuristics.
Still, the qualitative message is robust:

- depth amplifies scale problems;
- activation choice changes the right scale;
- normalization changes the effective geometry of optimization;
- practical training depends on managing signal propagation, not just choosing a loss and pressing "train."

## 12. Worked reasoning example

Suppose a 20-layer ReLU network is initialized with weight variance much larger than $2/d_{\mathrm{in}}$.
Then typical pre-activations will have large magnitude.
On the positive side, activations can grow layer after layer.
On the negative side, many units may be cut to zero.

In the backward pass, the transposed weight multiplications can amplify gradients strongly.
Training may become unstable, with loss spikes or numerical overflow.

Now suppose instead the variance is far smaller than $2/d_{\mathrm{in}}$.
Then activations may shrink toward zero and the network behaves too close to a weak linear map near the origin.
Gradient signals become small and early-layer learning stalls.

The point is not that one formula solves everything.
The point is that initialization determines the starting dynamical regime.

## 13. Summary

Initialization breaks symmetry and sets the starting scale of activations and gradients.
Normalization keeps those quantities in a trainable range as optimization proceeds.
Xavier initialization is motivated by variance preservation for approximately symmetric activations, while He initialization adapts that reasoning to ReLU-like units.
Batch normalization and layer normalization then stabilize internal representations in different ways.
