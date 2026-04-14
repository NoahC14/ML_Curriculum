---
title: "Mapping Canonical ML Concepts into Categorical Language"
module: "00-math-toolkit"
lesson: "ml-categorical-mapping"
doc_type: "notes"
topic: "category-theory"
status: "draft"
prerequisites:
  - "00-math-toolkit/category-theory-primer"
  - "00-math-toolkit/linear-algebra"
  - "00-math-toolkit/multivariable-calculus"
  - "00-math-toolkit/probability-statistics"
updated: "2026-04-09"
owner: "curriculum-team"
tags:
  - "category-theory"
  - "ml-foundations"
  - "structural-clarification"
  - "supervised-learning"
---

## Motivation

This note builds a translation layer from standard machine learning vocabulary into category-theoretic language. The goal is not to redescribe all of ML as category theory. The goal is narrower: identify where categorical language is exact, where it is only a useful structural analogy, and where it should not be used as the main explanatory tool.

That distinction matters. Many statements such as "a model is a morphism" are correct only after we specify a category, identify the relevant structure, and state what composition means. Other statements such as "training is a functor" are often too loose to be taken literally without substantial additional formalization.

This note therefore follows the boundary guide from Card 0.3:

- canonical ML content is stated first in ordinary language;
- categorical restatements are used in `Structural note` mode;
- loose analogies are labeled as such rather than overstated.

## Scope and Notation

We use the following symbols throughout:

- $X$ for an input space;
- $Y$ for an output or label space;
- $\mathcal{D} \subseteq X \times Y$ for a supervised dataset;
- $H$ or $Z$ for intermediate representation spaces;
- $\mathcal{H} \subseteq Y^X$ for a hypothesis class of functions $h : X \to Y$;
- $\Theta$ for a parameter space;
- $f_\theta : X \to Y$ for a model indexed by parameters $\theta \in \Theta$;
- $\ell : Y \times Y \to \mathbb{R}_{\ge 0}$ for a pointwise loss;
- $R(\theta)$ or $\widehat{R}(\theta)$ for population or empirical risk.

Unless stated otherwise, the background category is either:

- `Set`, when we only need ordinary functions; or
- `Vect_\mathbb{R}`, when maps are required to be linear.

Whenever a proposed correspondence needs extra structure beyond these categories, that dependence is stated explicitly.

## What Counts as a Good Mapping

A disciplined mapping should answer four questions:

1. What are the objects?
2. What are the morphisms?
3. What is composition?
4. Is the claim exact, conditional on extra setup, or only an analogy?

If one of these questions has no clean answer, categorical language may still be suggestive, but it is not yet a formal equivalence.

## Core Mapping Table

| Canonical ML concept | Standard description | Categorical reading | Status | What makes the reading valid | Main limitation |
| --- | --- | --- | --- | --- | --- |
| Input space $X$ | Set or structured space of examples | Object of `Set`, or of a richer category if structure matters | Exact | A category is fixed and $X$ is an object in it | The object says nothing yet about probability, geometry, or semantics |
| Label space $Y$ | Target space for prediction | Object of `Set`; sometimes object of `Vect_\mathbb{R}` for regression | Exact | Same as above | Classification labels usually live naturally as sets, not vector spaces |
| Dataset $\mathcal{D}$ | Finite sample of pairs $(x_i,y_i)$ | Usually a finite object together with maps into $X$ and $Y$, or a relation $D \to X \times Y$ | Conditional structural reading | We model the dataset as a finite indexed set $D$ with maps $x : D \to X$, $y : D \to Y$ | A raw dataset is not automatically a single canonical object; sampling noise and empirical measure are extra structure |
| Hypothesis $h$ | Predictor $h : X \to Y$ | Morphism $h : X \to Y$ in `Set`; linear hypothesis is a morphism in `Vect_\mathbb{R}` | Exact | The hypothesis belongs to the chosen morphism class | Most interesting model classes restrict admissible morphisms beyond arbitrary functions |
| Hypothesis class $\mathcal{H}$ | Family of allowed predictors | Hom-set subset, object of a function-space category, or parameterized family $\Theta \to Y^X$ | Conditional structural reading | We specify whether $\mathcal{H}$ is a subset of $\mathrm{Hom}(X,Y)$ or is represented by parameters | Function-space objects may not exist in the ambient category; parameterization can hide non-identifiability |
| Feature map $\phi$ | Map from raw input to representation | Morphism $\phi : X \to Z$ | Exact | Treated as an ordinary function or structure-preserving map | Whether it preserves relevant invariants is an additional claim, not part of being a morphism alone |
| Model pipeline $c \circ \phi$ | Feature extractor followed by prediction head | Composite morphism $X \xrightarrow{\phi} Z \xrightarrow{c} Y$ | Exact | Domains and codomains match | Composition alone does not explain learnability or performance |
| Data augmentation $a$ | Transformation of inputs | Endomorphism $a : X \to X$ | Exact | The augmentation is a function on the input space | Label preservation requires an extra commuting condition; not every augmentation is label preserving |
| Invariance condition | Prediction unchanged under allowed transformations | Equation $h \circ a = h$ for chosen endomorphisms $a$ | Exact structural statement | A transformation family on $X$ is specified | Which transformations should count as symmetries is problem dependent |
| Equivariance | Model commutes with transformations on input and output | Commutative square between actions on $X$ and $Y$ | Exact structural statement | Both actions are defined and the square is stated precisely | This often needs group actions or representation theory, not only basic category language |
| Loss $\ell$ | Penalty comparing prediction and target | Morphism $Y \times Y \to \mathbb{R}_{\ge 0}$ in `Set` | Exact at function level | Products and codomain are fixed | The scalar codomain hides statistical meaning and optimization geometry |
| Empirical risk $\widehat{R}$ | Average loss over sample | Composite built from dataset indexing, prediction, target, loss, and averaging maps | Conditional structural reading | We represent the dataset explicitly as a finite index object | Averaging usually needs algebraic structure such as addition and scalar division, not plain category theory alone |
| Parameterization $\theta \mapsto f_\theta$ | Family of models indexed by parameters | Map $\Theta \to \mathrm{Hom}(X,Y)$ when such a hom-object exists, or external parameter assignment | Conditional structural reading | We work in a category with suitable function-space structure, or keep the hom-set external | Many categories used in beginner ML notes are not automatically Cartesian closed |
| Optimizer step $s_D$ | One parameter update on dataset $D$ | Endomorphism $s_D : \Theta \to \Theta$ | Exact at the level of iterative update | A concrete update rule is fixed | The construction of $s_D$ uses calculus, linear algebra, and stochastic approximations outside elementary category theory |
| Training loop | Repeated application of optimizer steps | Composition of endomorphisms on $\Theta$ | Exact structural statement | Each step is a map with matching domain/codomain | Convergence and stability are not categorical consequences |
| Evaluation metric | Map from predictions and targets to a scalar summary | Morphism from an object of prediction-target pairs to a score object | Conditional structural reading | The evaluation protocol is encoded as functions on a finite sample or distribution | Statistical uncertainty, confidence intervals, and hypothesis testing require more than compositional language |
| Natural transformation between model families | Coherent comparison between two constructions across many spaces | Family of maps between functors, if each architecture assignment is actually functorial | Usually analogy unless formalized carefully | We must define source and target categories, both functors, and naturality squares | In ordinary ML practice, "architecture A versus B" is usually not specified functorially |

## Glossary of Correspondences and Limitations

### Objects

Objects can represent input spaces, label spaces, feature spaces, latent spaces, parameter spaces, or dataset index sets. This is usually the least controversial categorical translation.

The limitation is that an object only specifies membership in a category. It does not automatically encode probability measures, topology, smooth structure, geometry, or semantics unless those are built into the chosen category.

### Morphisms

Morphisms are the main translation target for predictors, feature maps, decoders, augmentations, projections, and update steps. This is exact when the ML construction is genuinely a map of the relevant kind.

The limitation is that "morphism" only means "allowed structure-preserving map in the current category." If the category is `Set`, every function qualifies. If the category is `Vect_\mathbb{R}`, only linear maps qualify. Neural networks with nonlinear activations are not morphisms in `Vect_\mathbb{R}`; they are ordinary functions in `Set`, smooth maps in a smooth category, or maps in some other richer setting.

### Composition

Composition is the strongest and most reliable bridge between ML and category theory. Preprocessing, representation learning, prediction heads, decoders, and optimizer iterations all compose.

The limitation is that composition captures architecture and workflow shape, not empirical success. Two systems with the same compositional type can behave very differently in optimization and generalization.

### Products

Products describe situations where a single source produces multiple outputs together, such as paired feature extractors or joint access to input and label spaces.

The limitation is that not every concatenation used in ML is literally a universal product in the category under discussion. Sometimes it is only implemented as vector concatenation in coordinates.

### Coproducts

Coproducts can model branchwise definitions, multimodal disjoint unions, or case distinctions.

The limitation is that many "multimodal" systems do not use true disjoint unions at all. They often use synchronized tuples, which are closer to products than coproducts.

### Functors

Functorial language is appropriate when an ML construction assigns:

- to each structured domain an output object; and
- to each admissible transformation a corresponding map on outputs;

while preserving identities and composition.

> [!NOTE]
> **Structural note.** A data representation procedure becomes a literal functor only when it transports both objects and admissible transformations coherently. An embedding method applied to one fixed dataset is usually just a map, not a functor.

The limitation is severe: many useful ML constructions are defined on one dataset, one architecture, or one parameter space, so there is no source category rich enough to support a nontrivial functorial statement.

### Natural Transformations

Natural transformations can express coherent conversions between two functorial ML constructions.

> [!NOTE]
> **Structural note.** For example, if two preprocessing pipelines are each defined for every input space and behave functorially with respect to admissible data transformations, then a family of conversion maps between their outputs may form a natural transformation.

The limitation is that this is rarely available "for free." A natural transformation is not the same as "there exists a conversion layer between two models." Naturality is a global coherence condition, not just a pointwise existence statement.

### Non-categorical but adjacent notions

Some central ML notions are not well captured by elementary category language alone:

- probabilities and expectations need measure-theoretic or probabilistic structure;
- gradients and Hessians need differential structure;
- convexity and convergence need geometric and analytic structure;
- stochastic optimization needs randomness, filtration, or Markovian machinery;
- generalization bounds need statistics and learning theory.

This does not make category theory irrelevant. It means it operates as a structural layer alongside, not instead of, these tools.

## Worked Example 1: Linear Regression

### Canonical setup

Let $X = \mathbb{R}^d$, let $Y = \mathbb{R}$, and consider linear predictors

$$
h_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b,
$$

where $\mathbf{w} \in \mathbb{R}^d$ and $b \in \mathbb{R}$. For a dataset

$$
\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n,
$$

the mean squared error is

$$
\widehat{R}(\mathbf{w},b) = \frac{1}{n} \sum_{i=1}^n \big(h_{\mathbf{w},b}(\mathbf{x}_i) - y_i\big)^2.
$$

### Categorical translation

One clean translation uses affine maps in `Set`, since the bias term prevents strict linearity in `Vect_\mathbb{R}` unless we homogenize coordinates.

- $X$ and $Y$ are objects.
- Each predictor $h_{\mathbf{w},b} : X \to Y$ is a morphism in `Set`.
- If we augment inputs by one coordinate,
  $$
  \widetilde{\mathbf{x}} = (\mathbf{x},1) \in \mathbb{R}^{d+1},
  $$
  then linear regression becomes a linear map
  $$
  \widetilde{h}_{\widetilde{\mathbf{w}}} : \mathbb{R}^{d+1} \to \mathbb{R}
  $$
  in `Vect_\mathbb{R}`.

The empirical risk can be decomposed as

$$
D \xrightarrow{(x,y)} X \times Y
\xrightarrow{h \times \mathrm{id}_Y} Y \times Y
\xrightarrow{\ell_{\mathrm{sq}}} \mathbb{R}_{\ge 0}
\xrightarrow{\mathrm{avg}} \mathbb{R}_{\ge 0},
$$

where:

- $D = \{1,\ldots,n\}$ is the finite index set of examples;
- $x(i) = \mathbf{x}_i$ and $y(i) = y_i$;
- $\ell_{\mathrm{sq}}(\hat{y},y) = (\hat{y}-y)^2$.

### What is exact here?

- Predictors as morphisms: exact.
- The pipeline defining empirical risk from dataset indices: exact once $D$ and the averaging map are specified.
- Linear regression as a morphism in `Vect_\mathbb{R}`: exact only after using augmented coordinates or restricting to zero bias.

### What is not captured?

Normal equations, convexity, and statistical properties of the least-squares estimator are not consequences of the categorical description. Those require linear algebra, optimization, and statistics.

## Worked Example 2: Multiclass Classification with Feature Maps

### Canonical setup

Let:

- $X$ be an input space;
- $Y = \{1,\ldots,K\}$ be a finite label set;
- $\phi : X \to \mathbb{R}^m$ be a feature map;
- $g : \mathbb{R}^m \to \mathbb{R}^K$ be a score map;
- $\operatorname*{argmax} : \mathbb{R}^K \to Y$ pick the predicted class.

Then the classifier is

$$
h = \operatorname*{argmax} \circ g \circ \phi : X \to Y.
$$

### Categorical translation

This is a direct compositional factorization in `Set`:

$$
X \xrightarrow{\phi} \mathbb{R}^m
\xrightarrow{g} \mathbb{R}^K
\xrightarrow{\operatorname*{argmax}} Y.
$$

If an augmentation $a : X \to X$ preserves class labels, the label map $y : X \to Y$ should satisfy

$$
y \circ a = y.
$$

If the classifier itself is invariant to that augmentation, then

$$
h \circ a = h.
$$

### Structural reading

> [!NOTE]
> **Structural note.** The equations $y \circ a = y$ and $h \circ a = h$ are exact commutative-diagram statements. They clarify what it means for an augmentation to preserve semantic labels and for a classifier to respect that preservation.

### Where the analogy begins

One sometimes hears that a feature extractor is a functor from "raw examples" to "representations." That is usually only an analogy in this basic setting. The map $\phi : X \to \mathbb{R}^m$ is certainly a morphism, but to call it a functor we would need:

- a source category of structured data domains and admissible transformations;
- a target category of representation spaces and corresponding maps;
- a rule sending every admissible transformation on inputs to a compatible transformation on representations.

Without that full assignment, $\phi$ is a morphism, not a functor.

## Worked Example 3: Neural Network Training

### Canonical setup

Let $f_\theta : X \to Y$ be a neural network with parameters $\theta \in \Theta = \mathbb{R}^p$. For a fixed dataset $\mathcal{D}$ and learning rate $\eta > 0$, one gradient descent step is

$$
s_{\mathcal{D}}(\theta) = \theta - \eta \nabla_\theta \widehat{R}(\theta).
$$

After $T$ steps,

$$
\theta_T = s_{\mathcal{D}}^{\,T}(\theta_0),
$$

where $s_{\mathcal{D}}^{\,T}$ denotes $T$-fold composition of the update map with itself.

### Categorical translation

At the most reliable level:

- the parameter space $\Theta$ is an object;
- one training step is an endomorphism $s_{\mathcal{D}} : \Theta \to \Theta$;
- the training loop is composition
  $$
  \Theta \xrightarrow{s_{\mathcal{D}}} \Theta \xrightarrow{s_{\mathcal{D}}} \Theta \xrightarrow{s_{\mathcal{D}}} \cdots.
  $$

Within the network itself, the forward pass is another composite:

$$
X \xrightarrow{f_1} H_1 \xrightarrow{f_2} H_2 \to \cdots \to H_L \xrightarrow{f_{L+1}} Y.
$$

### What is exact?

- Forward propagation as composition of layerwise maps: exact.
- The update rule as an endomorphism on parameter space: exact, once the optimizer and dataset are fixed.
- Iterative training as repeated composition: exact.

### What is only a partial structural account?

The gradient computation is not explained by category theory alone. It depends on differential structure, chain-rule calculations, and numerical choices. Likewise, claims about implicit bias, generalization, or training stability do not follow from the fact that updates compose.

### Caution on stronger claims

Statements such as "backpropagation is a natural transformation" or "optimization is a functor" require much more formal setup than is usually present in an introductory ML discussion. They may be research-level formalisms in specialized settings, but in this note they should be treated as at most prospective directions, not as established equivalences.

## A Short Checklist for Future Categorical Readings of ML

Before using categorical language for an ML construction, ask:

1. Which category is actually being used?
2. Are the proposed morphisms really morphisms in that category?
3. Is the statement about one fixed map, a compositional pipeline, or a genuinely functorial assignment?
4. If a functor or natural transformation is claimed, what are the source and target categories, and what coherence laws are being asserted?
5. Which important parts of the ML argument still depend on optimization, probability, statistics, or numerics rather than category theory?

If these questions cannot be answered concretely, the categorical language should be presented as heuristic only.

## Shared Figure References

The categorical mapping note should reuse the shared guide [diagram-notation.md](../../../shared/style-guides/diagram-notation.md) rather than inventing local figure conventions.

- Core pipeline factorization: [feature-pipeline.svg](../../../shared/figures/category-theory/rendered/feature-pipeline.svg)
- Empirical-risk composition: [empirical-risk-pipeline.svg](../../../shared/figures/category-theory/rendered/empirical-risk-pipeline.svg)
- Label preservation and invariance: [augmentation-label-square.svg](../../../shared/figures/category-theory/rendered/augmentation-label-square.svg)
- Equivariant representation transport: [representation-equivariance.svg](../../../shared/figures/category-theory/rendered/representation-equivariance.svg)
- Functorial and naturality claims: [functor-composition.svg](../../../shared/figures/category-theory/rendered/functor-composition.svg) and [naturality-square.svg](../../../shared/figures/category-theory/rendered/naturality-square.svg)

## Summary

The safest and most useful categorical translations in early ML are:

- spaces as objects;
- predictors, feature maps, augmentations, and update rules as morphisms;
- pipelines and training iterations as compositions;
- invariance and equivariance as commuting diagrams;
- some multimodal and paired constructions as products or coproduct-like patterns.

The least safe translations are the strongest-sounding ones:

- calling a single feature map a functor;
- calling an arbitrary model comparison a natural transformation;
- treating category theory as if it replaced optimization, probability, or learning theory.

Used carefully, categorical language clarifies how ML systems are wired. Used incautiously, it creates the illusion of structure without the burden of formalization. This note is meant to support the first use and block the second.
