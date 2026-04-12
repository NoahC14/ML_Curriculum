---
title: "CNN Architectures"
module: "08-cnn-vision"
lesson: "cnn-architectures"
doc_type: "notes"
topic: "lenet-alexnet-vgg-resnet-transfer-learning"
status: "draft"
prerequisites:
  - "06-neural-networks/neural-networks-first-principles"
  - "07-deep-learning-systems/training-deep-networks"
  - "08-cnn-vision/convolutional-networks"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "cnn"
  - "lenet"
  - "alexnet"
  - "vgg"
  - "resnet"
  - "transfer-learning"
---

## Purpose

These notes compare the major architectural ideas that shaped modern convolutional vision systems.
The emphasis is on design principles rather than nostalgia:
depth, receptive-field growth, normalization, residual pathways, and transfer.

## Learning objectives

After working through this note, you should be able to:

- describe the motivating design idea behind LeNet, AlexNet, VGG, and ResNet;
- compare these architectures in terms of depth, parameter count, and optimization behavior;
- explain why residual connections changed the practical depth frontier;
- identify which parts of a pretrained vision model are typically reused during transfer learning; and
- distinguish historical CNN backbones from broader modern vision families.

## 1. Architecture progression as design refinement

The history of CNNs is not just a list of named models.
It is a sequence of answers to recurring questions:

- How should local visual features be composed?
- How deep can a model become before optimization fails?
- How can receptive field be expanded without uncontrolled parameter growth?
- Which computations should be shared across tasks through pretraining?

The major architectures below matter because each sharpened one of these questions.

## 2. LeNet: early local-feature hierarchies

LeNet-5 was developed for handwritten digit recognition.
Its importance is conceptual rather than merely historical.
It showed that alternating convolution-like feature extraction and subsampling could outperform feature engineering on image data.

### 2.1 Characteristic design

LeNet-style networks use:

- small grayscale inputs;
- a few convolution layers;
- subsampling or pooling;
- one or more fully connected layers at the end.

A canonical LeNet-5 variant contains roughly:

- `Conv(1 -> 6, 5x5)`
- subsampling
- `Conv(6 -> 16, 5x5)`
- subsampling
- fully connected classifier layers

Depending on the exact implementation, LeNet-5 has about `60K` parameters.

### 2.2 Design rationale

LeNet established three durable CNN ideas:

- local receptive fields reduce parameter count;
- repeated filters can be reused across the image; and
- higher layers can aggregate simpler lower-level patterns.

Its limitations are equally instructive.
The network is shallow, built for small inputs, and not designed for the scale or diversity of modern datasets.

## 3. AlexNet: scale, ReLU, and GPU-era deep vision

AlexNet marked the deep-learning breakthrough on ImageNet.
It was not the first CNN, but it was the first to combine sufficient dataset scale, GPU training, and effective architectural choices to dominate large-scale image classification.

### 3.1 Characteristic design

AlexNet uses:

- five convolutional layers;
- three fully connected layers;
- ReLU nonlinearities;
- overlapping max pooling;
- dropout in the classifier head; and
- aggressive data augmentation for the time.

The original network has about `61M` parameters.
Most of them reside in the fully connected layers.

### 3.2 Design rationale

AlexNet mattered for several reasons:

- ReLU made optimization substantially easier than saturating activations such as sigmoid or tanh in deep nets.
- Larger-scale supervised pretraining on ImageNet produced general visual features useful beyond one dataset.
- GPU implementation made deep convolutional training computationally feasible.

The lesson is not "copy AlexNet."
The lesson is that architecture, optimization, hardware, and data scale interact.

## 4. VGG: depth through repeated small kernels

VGG networks explored a simple but powerful thesis:
many stacked `3x3` convolutions can replace larger kernels while increasing depth and expressivity.

### 4.1 Characteristic design

VGG-style networks use:

- blocks of repeated `3x3` convolutions with stride `1`;
- periodic max pooling for downsampling; and
- a deep fully connected classifier head in the original versions.

Two common variants are:

- VGG-16 with about `138M` parameters;
- VGG-19 with about `144M` parameters.

### 4.2 Why small kernels help

Two consecutive `3x3` convolutions achieve a receptive field similar to a `5x5` convolution.
Three `3x3` convolutions achieve a receptive field similar to a `7x7` convolution.

But stacking small kernels has advantages:

- fewer parameters than one large kernel at the same channel width;
- more nonlinearities between receptive-field expansions; and
- a highly regular architecture that is easy to scale and reason about.

### 4.3 Limitations

VGG is conceptually clean but computationally heavy.
The parameter count is very large, memory use is substantial, and optimization becomes harder as depth increases.
These weaknesses motivated architectures with better parameter efficiency and more stable gradient transport.

## 5. ResNet: residual learning and depth at scale

ResNet introduced residual blocks, which changed deep vision from "depth is fragile" to "depth is manageable."

### 5.1 Residual block idea

Instead of learning a direct mapping $H(x)$, a residual block learns a correction $F(x)$ and returns

$$
y = x + F(x).
$$

The identity skip path gives gradients a direct transport route and makes it easier for deeper networks to avoid harmful degradation.

### 5.2 Characteristic design

The standard ResNet family includes:

- ResNet-18 with about `11.7M` parameters;
- ResNet-34 with about `21.8M` parameters;
- ResNet-50 with about `25.6M` parameters;
- ResNet-101 with about `44.5M` parameters.

ResNet-50 and deeper variants use bottleneck blocks, typically reducing then expanding channel dimension inside a residual unit to control computation.

### 5.3 Design rationale

Residual connections matter for both optimization and representation.

- If the best transformation near some layer is close to identity, the block can learn a small residual rather than an entirely new mapping.
- Gradient flow improves because the Jacobian contains an identity contribution.
- Depth becomes useful rather than merely harder to train.

ResNet therefore shifted the design principle from "how can we stack more layers?" to "how can we compose many layers without destroying signal?"

## 6. Comparison table

| Architecture | Approx. depth | Approx. parameters | Main design idea | Practical lesson |
| --- | --- | ---: | --- | --- |
| LeNet-5 | 5-7 learned layers depending on counting convention | `~60K` | local filters plus subsampling | CNNs exploit image locality efficiently |
| AlexNet | 8 learned layers | `~61M` | deeper CNN with ReLU, pooling, dropout, GPU training | scale and optimization matter |
| VGG-16 | 16 learned layers | `~138M` | repeated small `3x3` kernels | depth and regular blocks improve features |
| ResNet-50 | 50 learned layers | `~25.6M` | residual blocks with skip connections | depth must be paired with stable gradient transport |

The absolute counts vary slightly by implementation, classifier width, and dataset-specific output dimension.
The relative design lessons are the important part.

## 7. Transfer learning as an architectural use case

CNN backbones became especially influential because pretrained features transfer well.
A typical transfer-learning pipeline separates a model into:

- backbone or feature extractor; and
- task-specific head.

For example, a pretrained ResNet can be adapted by replacing its final classifier layer with a new linear head for a small custom dataset.

### 7.1 Why transfer works

Early convolutional layers often learn generic detectors:

- edges;
- corners;
- color contrasts;
- simple textures.

Intermediate layers often capture reusable parts and motifs.
Only later layers become strongly specialized to a particular label space.

This is why freezing early layers and training only the head can work surprisingly well when data is limited.

### 7.2 Feature extraction versus fine-tuning

Feature extraction:

- keep backbone weights fixed;
- train only the new head;
- useful when the target dataset is small or compute is limited.

Fine-tuning:

- initialize from pretrained weights;
- unfreeze some or all backbone layers;
- continue training at a smaller learning rate.

Fine-tuning is usually stronger when the target task differs moderately from pretraining but enough target data exists to benefit from adaptation.

## 8. Brief modern mentions

After ResNet, CNN research continued along several directions.

### 8.1 Efficiency-focused families

Architectures such as MobileNet and EfficientNet target better accuracy-compute tradeoffs.
Typical tools include:

- depthwise separable convolution;
- width, depth, and resolution scaling rules; and
- architecture choices designed for deployment constraints.

### 8.2 Dense and multi-path designs

Architectures such as DenseNet and Inception explore richer connectivity patterns.
The guiding idea is that information flow and feature reuse can be improved without only making networks deeper.

### 8.3 CNNs in the transformer era

Vision transformers changed the landscape, but CNNs remain important because:

- they encode strong spatial inductive bias;
- they are effective in low-data and deployment-sensitive regimes; and
- hybrid architectures often still use convolutional stems or multiscale CNN components.

The right conclusion is not that CNNs are obsolete.
It is that CNNs are one major point in the broader design space of equivariant, hierarchical visual representation learning.

## 9. Category-theoretic insertion point

Residual blocks are a natural place to discuss compositional structure.
Instead of viewing a deep network as a brittle chain of transformations, residual design adds a canonical identity path alongside learned morphisms.
That viewpoint clarifies why "do nothing" remains available as a stable default and why compositional depth does not force representational instability at every stage.

## 10. Scope notes

These notes focus on core design principles rather than an exhaustive architecture survey.
Important omissions include:

- detection-specific families such as Faster R-CNN and YOLO;
- segmentation architectures such as U-Net;
- normalization and activation variants across vision models; and
- the full modern landscape of self-supervised and multimodal pretraining.

Those topics are important, but the foundational lesson of Module 08 is to internalize why convolutional architectures were designed the way they were and why their features transfer.

## Summary

LeNet introduced local-feature hierarchies.
AlexNet demonstrated that deep convolutional models could scale with data and hardware.
VGG showed the power of systematic depth with small kernels.
ResNet made very deep vision models trainable through residual pathways.

Together these architectures teach the main CNN design principles:

- exploit spatial locality;
- grow receptive fields gradually;
- stabilize optimization as depth increases; and
- reuse learned representations through transfer learning.
