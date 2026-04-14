---
title: "CNN Exercises"
module: "08-cnn-vision"
lesson: "cnn-exercises"
doc_type: "exercise"
topic: "convolution-equivariance-architectures-transfer-learning"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
  - "06-neural-networks/neural-networks-first-principles"
  - "07-deep-learning-systems/training-deep-networks"
  - "08-cnn-vision/convolutional-networks"
  - "08-cnn-vision/cnn-architectures"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "cnn"
  - "computer-vision"
  - "equivariance"
  - "resnet"
  - "transfer-learning"
---

## Purpose

These exercises reinforce the algebraic, geometric, and architectural ideas behind convolutional networks.
They mix direct calculation, interpretation, and design reasoning.

## Exercise 1: 1D convolution as a matrix

Let $x \in \mathbb{R}^6$ and let the 1D kernel be

$$
w = [w_0, w_1, w_2].
$$

Assume valid cross-correlation with stride $1$ and no padding.

1. Write the output vector explicitly in coordinates.
2. Construct the matrix $T_w$ such that $y = T_w x$.
3. Explain why $T_w$ is Toeplitz.
4. Compare the number of free parameters in $T_w$ with those in a general dense map from $\mathbb{R}^6$ to $\mathbb{R}^4$.

## Exercise 2: output shapes under stride and padding

Consider a 2D input of shape $32 \times 32$.

1. Compute the output shape of a `5x5` convolution with stride `1` and padding `0`.
2. Compute the output shape of a `5x5` convolution with stride `1` and padding `2`.
3. Compute the output shape of a `3x3` convolution with stride `2` and padding `1`.
4. For each case, explain what spatial information is lost or preserved.

## Exercise 3: parameter sharing

Suppose an RGB image of shape `64 x 64 x 3` is fed into:

- a dense layer with `128` hidden units; and
- a convolution layer with `128` filters of size `3x3`.

1. Compute the number of learned weights in the dense layer.
2. Compute the number of learned weights in the convolution layer.
3. Explain in words what structural assumption justifies this large reduction in parameters.
4. Give one situation where that assumption may be less appropriate.

## Exercise 4: equivariance to translation

Let $\tau_\delta$ denote translation by one pixel to the right, ignoring boundary complications.

1. State what it means for a feature map $f$ to be translation equivariant.
2. Show algebraically why convolution satisfies this property.
3. Explain why max pooling with stride `2` does not preserve exact equivariance.
4. Distinguish equivariance from invariance using one vision example.

## Exercise 5: receptive-field growth

Assume all convolutions are `3x3`, stride `1`, and use padding that preserves spatial size.

1. What is the receptive-field size after one layer?
2. What is the receptive-field size after two layers?
3. What is the receptive-field size after four layers?
4. Why can multiple small kernels be preferable to one large kernel with a similar receptive field?

## Exercise 6: pooling and hierarchy

1. Compare max pooling and average pooling in terms of what signal each preserves.
2. Explain why repeated pooling can help classification.
3. Explain why repeated pooling can hurt segmentation.
4. Give one argument for replacing pooling with strided convolution in a modern architecture.

## Exercise 7: architecture comparison

Use the architecture note to compare LeNet, AlexNet, VGG, and ResNet.

1. Which architecture first made large-scale supervised CNN training dominant in modern vision?
2. Which architecture is most closely associated with repeated small `3x3` kernels?
3. Which architecture introduced residual connections as the central design idea?
4. Order the four models by approximate parameter count from smallest to largest.
5. For each model, state the main design principle in one sentence.

## Exercise 8: residual blocks and optimization

Consider a residual block

$$
h_{\ell+1} = h_\ell + F_\ell(h_\ell).
$$

1. Compute the Jacobian $\frac{\partial h_{\ell+1}}{\partial h_\ell}$.
2. Explain why the identity term helps gradient flow.
3. Compare this with a plain block $h_{\ell+1} = F_\ell(h_\ell)$.
4. Why does this matter more as depth increases?

## Exercise 9: transfer learning design

You have a pretrained ResNet-50 and a target dataset of `2,000` labeled medical images.

1. Describe a feature-extraction approach.
2. Describe a fine-tuning approach.
3. Which approach would you try first if compute is limited?
4. Which approach becomes more attractive if validation performance plateaus and you can afford more training?
5. Give two reasons transfer learning is often effective in vision.

## Exercise 10: task formulation

For each task below, state the output object and one suitable evaluation metric.

1. Image classification
2. Object detection
3. Semantic segmentation
4. Keypoint estimation

Then explain why a classifier head is not sufficient for segmentation even if the backbone is shared.
