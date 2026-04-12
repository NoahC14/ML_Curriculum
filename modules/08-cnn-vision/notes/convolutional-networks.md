---
title: "Convolutional Networks"
module: "08-cnn-vision"
lesson: "convolutional-networks"
doc_type: "notes"
topic: "convolution-equivariance-pooling-vision-tasks"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
  - "01-optimization/OPT-02-stochastic-and-momentum"
  - "02-statistical-learning/statistical-learning-foundations"
  - "06-neural-networks/neural-networks-first-principles"
  - "07-deep-learning-systems/training-deep-networks"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "cnn"
  - "computer-vision"
  - "convolution"
  - "equivariance"
  - "pooling"
  - "transfer-learning"
---

## Purpose

These notes introduce convolutional neural networks as structured linear operators designed for spatial data.
The central idea is that images are not just vectors in a large Euclidean space.
They carry locality, translation structure, and multi-scale patterns.
Convolution exploits that structure through local receptive fields, parameter sharing, and translation equivariance.

The goal is to understand CNNs at two levels at once:

- algebraically, as sparse linear maps that can be written as matrix multiplication; and
- structurally, as architectures whose inductive bias reflects the symmetries of images.

## Learning objectives

After working through this note, you should be able to:

- define 1D and 2D discrete convolution and distinguish it from cross-correlation as used in most deep-learning libraries;
- rewrite convolution as multiplication by a structured matrix;
- compute output shapes under stride, padding, dilation, and pooling;
- explain parameter sharing, local connectivity, and receptive fields;
- state why convolution is translation equivariant and why pooling only approximates invariance;
- describe how spatial hierarchies emerge across layers; and
- formulate common vision tasks such as classification, localization, detection, and segmentation in learning terms.

## 1. Why images require structure-aware models

An image with height $H$, width $W$, and channels $C$ can be flattened into a vector in $\mathbb{R}^{HWC}$.
In principle, a multilayer perceptron can process that vector.
In practice, a dense layer from $\mathbb{R}^{HWC}$ to $\mathbb{R}^m$ ignores two facts:

- nearby pixels tend to be more related than distant pixels; and
- the same local pattern may appear at many spatial locations.

For a $224 \times 224 \times 3$ image, the input dimension is $150{,}528$.
A dense layer with $1{,}000$ hidden units would therefore use about $150$ million weights before any depth is added.
That parameterization is not only expensive.
It also fails to encode the translational regularity of vision.

Convolution addresses both issues by restricting interactions to local neighborhoods and reusing the same filter across space.

## 2. Discrete 1D convolution

Let $x \in \mathbb{R}^n$ be a 1D signal and let $w \in \mathbb{R}^k$ be a filter.
The mathematical discrete convolution is

$$
\bigl(x * w\bigr)_t = \sum_{j=0}^{k-1} x_{t-j}\,w_j.
$$

Most machine-learning libraries implement cross-correlation instead:

$$
\bigl(x \star w\bigr)_t = \sum_{j=0}^{k-1} x_{t+j}\,w_j.
$$

The only difference is whether the filter is reversed.
Since the filter is learned, both operators are equivalent up to reindexing of parameters.
Deep-learning practice therefore calls the implemented cross-correlation operation a convolution layer.

### 2.1 Output length with stride and padding

Suppose:

- the input length is $n$;
- the kernel length is $k$;
- zero-padding of size $p$ is applied on each side; and
- stride is $s$.

Then the output length is

$$
n_{\text{out}} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1.
$$

If dilation $d$ is used, then the effective kernel size becomes

$$
k_{\text{eff}} = d(k-1) + 1,
$$

and the output length is

$$
n_{\text{out}} = \left\lfloor \frac{n + 2p - k_{\text{eff}}}{s} \right\rfloor + 1.
$$

### 2.2 Matrix view

For valid 1D convolution with stride $1$, the map $x \mapsto x \star w$ is linear in $x$.
Hence there exists a matrix $T_w$ such that

$$
x \star w = T_w x.
$$

For example, with $n=5$ and $k=3$,

$$
\begin{bmatrix}
\bigl(x \star w\bigr)_0 \\
\bigl(x \star w\bigr)_1 \\
\bigl(x \star w\bigr)_2
\end{bmatrix}
=
\begin{bmatrix}
w_0 & w_1 & w_2 & 0 & 0 \\
0 & w_0 & w_1 & w_2 & 0 \\
0 & 0 & w_0 & w_1 & w_2
\end{bmatrix}
\begin{bmatrix}
x_0 \\ x_1 \\ x_2 \\ x_3 \\ x_4
\end{bmatrix}.
$$

This matrix is Toeplitz: each descending diagonal is constant.
The key lesson is that convolution is not "less linear" than a dense layer.
It is a linear layer with strong structure.

## 3. Two-dimensional convolution

Let $X \in \mathbb{R}^{H \times W}$ be an image and let $K \in \mathbb{R}^{r \times s}$ be a kernel.
Ignoring padding for the moment, the 2D cross-correlation output is

$$
Y[i,j] = \sum_{a=0}^{r-1} \sum_{b=0}^{s-1} X[i+a, j+b] K[a,b].
$$

For multi-channel inputs with $C_{\text{in}}$ channels, a single output channel uses a kernel tensor

$$
K \in \mathbb{R}^{C_{\text{in}} \times r \times s},
$$

and computes

$$
Y[i,j] = \sum_{c=1}^{C_{\text{in}}} \sum_{a=0}^{r-1} \sum_{b=0}^{s-1}
X[c, i+a, j+b] K[c,a,b].
$$

If we produce $C_{\text{out}}$ output channels, then the kernel bank has shape

$$
C_{\text{out}} \times C_{\text{in}} \times r \times s.
$$

The total number of learned weights is therefore

$$
C_{\text{out}} C_{\text{in}} r s,
$$

plus one bias per output channel if biases are used.

### 3.1 Output shape in 2D

With input size $H \times W$, kernel size $r \times s$, padding $(p_h, p_w)$, and stride $(s_h, s_w)$,

$$
H_{\text{out}} = \left\lfloor \frac{H + 2p_h - r}{s_h} \right\rfloor + 1,
\qquad
W_{\text{out}} = \left\lfloor \frac{W + 2p_w - s}{s_w} \right\rfloor + 1.
$$

In the common square case with kernel size $k$, padding $p$, and stride $s$,

$$
H_{\text{out}} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1,
\qquad
W_{\text{out}} = \left\lfloor \frac{W + 2p - k}{s} \right\rfloor + 1.
$$

### 3.2 Matrix view in 2D

If we flatten every receptive field into a row and flatten the kernel into a column, 2D convolution also becomes matrix multiplication.
This is the algebra behind the common `im2col` implementation trick.

Let:

- $P \in \mathbb{R}^{N_{\text{patch}} \times (C_{\text{in}}rs)}$ be the matrix whose rows are flattened local patches; and
- $\theta \in \mathbb{R}^{C_{\text{in}}rs}$ be the flattened kernel for one output channel.

Then

$$
\operatorname{vec}(Y) = P\theta.
$$

For multiple output channels, we stack kernels into a matrix $\Theta$ and obtain

$$
\operatorname{vec}(Y_{\text{all channels}}) = P\Theta.
$$

The resulting linear map is sparse and highly structured.
That structure is the algebraic manifestation of locality and weight sharing.

## 4. Parameter sharing and local receptive fields

Convolutional layers differ from dense layers in two main ways.

### 4.1 Local connectivity

Each output unit depends only on a small neighborhood of the input rather than on all pixels.
This makes sense in images because edges, corners, and textures are local phenomena.

### 4.2 Parameter sharing

The same kernel weights are reused at every spatial location.
If a detector for a vertical edge is useful in the upper left corner, it is also useful in the lower right corner.

This is why a convolution layer can use far fewer parameters than a dense layer with a similar input size.
For example:

- a dense layer from $32 \times 32 \times 3$ to $64$ hidden units uses $(3072)(64) = 196{,}608$ weights; while
- a convolution layer with $64$ filters of size $3 \times 3$ uses $(64)(3)(3)(3) = 1{,}728$ weights.

The reduction is not accidental.
It is exactly the cost of imposing translational structure.

## 5. Equivariance and the structural meaning of convolution

Let $\tau_\delta$ denote translation of an image by displacement $\delta$.
A map $f$ is translation equivariant if

$$
f(\tau_\delta X) = \tau_\delta f(X)
$$

for every admissible translation $\delta$.

Ignoring boundary effects and stride for the moment, convolution has exactly this property:

$$
(\tau_\delta X) \star K = \tau_\delta (X \star K).
$$

This matters because objects do not cease to be objects when they shift slightly in the image plane.
The network should transform predictably under translation rather than relearn the same feature at every position.

### 5.1 Why pooling is not equivariance

Pooling is often used to reduce spatial resolution.
For example, max pooling over a $2 \times 2$ window with stride $2$ computes a local summary of nearby activations.

Pooling does not preserve exact translation equivariance because subsampling discards location information.
Instead, it induces a degree of local invariance or robustness:

- small shifts that keep the dominant activation inside the same pooling cell may leave the pooled output unchanged;
- larger shifts can change the pooled output abruptly.

Hence the standard slogan is:

- convolution encourages equivariance; and
- pooling encourages limited invariance.

## 6. Spatial hierarchy and receptive-field growth

Stacking convolutional layers grows the effective receptive field.
The first layer may detect edges.
The next layer can combine edges into corners or simple motifs.
Deeper layers can respond to textures, parts, and eventually object-level patterns.

This hierarchy is not magic.
It arises because each deeper unit composes many local operations from earlier layers.

For example, if we stack two $3 \times 3$ stride-1 convolutions with suitable padding, then a unit in the second layer depends on a $5 \times 5$ region of the original image.
Three such layers give a $7 \times 7$ receptive field.

This observation explains one of the design choices behind deeper CNNs:
many small kernels can achieve a large receptive field while inserting nonlinearities between layers.

## 7. Pooling, downsampling, and alternatives

Pooling reduces spatial dimensions and aggregates local evidence.
Common forms include:

- max pooling, which keeps the strongest activation in each region;
- average pooling, which keeps the mean response; and
- global average pooling, which averages each channel over the full spatial grid.

Pooling has several effects:

- it reduces memory and compute;
- it enlarges the effective receptive field of later units;
- it introduces some robustness to local perturbations; and
- it can discard fine spatial detail.

Modern architectures sometimes replace explicit pooling with strided convolutions.
The design choice is not philosophical.
It is a tradeoff between learned downsampling, computational cost, and information retention.

## 8. Nonlinearities and normalization in CNNs

A pure stack of convolutional layers without nonlinearities is still just one linear map.
To represent complex decision boundaries, CNNs interleave convolutions with nonlinearities such as ReLU, GELU, or variants.

Normalization layers such as batch normalization are particularly natural in CNNs because statistics can be computed channelwise over batch and spatial positions.
This often stabilizes optimization and allows deeper models to train reliably.

The common convolutional block

$$
\text{conv} \rightarrow \text{normalization} \rightarrow \text{nonlinearity}
$$

should therefore be seen as the vision analogue of the multilayer building blocks studied in Module 06 and the training-stability tools studied in Module 07.

## 9. Vision task formulations

CNNs are not tied to one prediction task.
The same feature extractor can support several output heads.

### 9.1 Image classification

Input:

$$
X \in \mathbb{R}^{C \times H \times W}.
$$

Output:

$$
\hat{y} \in \Delta^{K-1},
$$

where $\Delta^{K-1}$ is the probability simplex over $K$ classes.

Typical loss: cross-entropy.

The model learns a global representation and predicts one label for the whole image.

### 9.2 Localization and detection

In localization, the output includes both a class prediction and a bounding box.
In object detection, the model predicts multiple objects, each with:

- class;
- bounding box; and
- confidence score.

The loss combines classification and box-regression terms.

### 9.3 Semantic segmentation

The output is a label for each pixel:

$$
\hat{Y} \in \{1,\dots,K\}^{H \times W}
$$

or a per-pixel class distribution.

This task requires preserving spatial detail rather than collapsing everything into one global descriptor.

### 9.4 Instance segmentation and keypoints

Instance segmentation separates different objects of the same class.
Keypoint estimation predicts structured landmark locations such as joints or facial points.

These tasks illustrate an important principle:
the convolutional backbone extracts spatial features, while the task head determines what geometric object is predicted.

## 10. Transfer learning in vision

Large image datasets such as ImageNet allow CNNs to learn broadly useful visual features.
A model pretrained on such data can then be adapted to a smaller target task.

Two common strategies are:

- feature extraction: freeze the backbone and train only a new classifier head; and
- fine-tuning: initialize from pretrained weights and continue training some or all layers on the target task.

Transfer learning is especially effective in vision because early and intermediate convolutional features often capture reusable patterns such as edges, textures, and motifs.

## 11. Category-theoretic insertion point

At a lightweight structural level, a convolutional layer can be read as a morphism between spaces of feature fields that respects translation structure.
The important point is not formalism for its own sake.
The point is that the architecture is constrained to commute, approximately, with a symmetry action.

That is why equivariance is more than a slogan.
It is a preserved relation between transformations of input space and transformations of feature space.

## 12. Limitations and scope notes

CNNs are powerful, but their inductive bias is not universally correct.

- Translation equivariance is helpful for many natural-image tasks, but less so when absolute position itself is semantically important.
- Pooling and downsampling can destroy fine detail needed for dense prediction.
- Classical CNNs handle long-range dependencies less directly than self-attention mechanisms.
- Learned invariances are only approximate in the presence of padding, finite images, and nonlinear subsampling.

For this reason, modern vision systems often combine convolutional backbones with residual design, normalization, attention, or transformer-style components rather than relying on plain early CNN stacks alone.

## Summary

Convolutional networks are best understood as structured linear operators embedded inside deep nonlinear systems.
Their power comes from three linked ideas:

- local receptive fields;
- parameter sharing; and
- approximate translation-equivariant processing.

Those ideas reduce parameter count, improve sample efficiency, and create layered spatial hierarchies that make image learning tractable.
The next note studies how these principles evolved through landmark architectures such as LeNet, AlexNet, VGG, and ResNet.
