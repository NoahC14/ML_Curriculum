---
title: "Category Theory Primer for Machine Learning"
module: "00-math-toolkit"
lesson: "category-theory-primer"
doc_type: "notes"
topic: "category-theory"
status: "draft"
prerequisites:
  - "basic-set-theory"
  - "00-math-toolkit/linear-algebra"
updated: "2026-04-09"
owner: "curriculum-team"
tags:
  - "category-theory"
  - "ml-foundations"
  - "composition"
  - "diagrams"
---

## Motivation

Category theory enters this course as a working language for composition. It helps us say, with precision, that a preprocessing stage feeds a model, a decoder follows an encoder, or a parameter update follows a gradient computation. That is useful in machine learning because many systems are built by wiring together simpler pieces.

This note is deliberately a **primer**. The goal is not to make category theory the main mathematical language of early ML. The goal is to make a few structural ideas available early so that later topics such as linear maps, computation graphs, equivariance, transfer, and modular model design can be discussed more cleanly.

We will proceed in the following order:

- first with standard mathematical examples such as `Set`, `Vect`, and `Grp`;
- then with a small toolkit of definitions and diagrams;
- then with concrete ML examples.

Throughout, we use category theory in **structural clarification** mode: it sharpens how parts compose, but it does not replace linear algebra, probability, optimization, or implementation details.

## Assumptions and Notation

We assume comfort with sets, functions, matrices, and basic linear maps. If $X$ and $Y$ are sets, a function $f : X \to Y$ assigns each $x \in X$ a unique output $f(x) \in Y$.

In categorical language:

- objects play the role of spaces, sets, types, or state spaces;
- morphisms play the role of structure-respecting maps between objects;
- composition means applying one morphism after another when domains and codomains match.

We will write

$$
f : A \to B, \qquad g : B \to C, \qquad g \circ f : A \to C.
$$

The expression $g \circ f$ means "first apply $f$, then apply $g$."

## Categories: Objects and Morphisms

> **Definition.** A **category** $\mathcal{C}$ consists of:
>
> - a collection of objects;
> - for any objects $A$ and $B$, a collection of morphisms $f : A \to B$;
> - for any composable pair $f : A \to B$ and $g : B \to C$, a composite morphism $g \circ f : A \to C$;
> - for every object $A$, an identity morphism $\mathrm{id}_A : A \to A$.
>
> These data satisfy two axioms:
>
> 1. **Associativity:** if $f : A \to B$, $g : B \to C$, and $h : C \to D$, then
>    $$
>    h \circ (g \circ f) = (h \circ g) \circ f.
>    $$
> 2. **Identity laws:** if $f : A \to B$, then
>    $$
>    f \circ \mathrm{id}_A = f
>    \qquad \text{and} \qquad
>    \mathrm{id}_B \circ f = f.
>    $$

The definition is short, but it packages a common pattern: we have entities, allowed maps between them, and a rule for chaining maps.

### Standard Examples

> **Example (`Set`).** The category `Set` has sets as objects and functions as morphisms. Composition is ordinary function composition, and identities are ordinary identity functions.

`Set` is the simplest place to build intuition. Many ML pipelines can first be thought of as functions between sets, even before we ask whether they preserve extra structure.

> **Example (`Vect_\mathbb{R}`).** The category `Vect_\mathbb{R}` has real vector spaces as objects and linear maps as morphisms.

This category matters for ML because matrices represent linear maps between finite-dimensional vector spaces. A layer such as

$$
\mathbf{x} \mapsto \mathbf{W}\mathbf{x}
$$

is naturally a morphism in `Vect_\mathbb{R}`.

> **Example (`Grp`).** The category `Grp` has groups as objects and group homomorphisms as morphisms.

This example is useful because it shows that morphisms are not always arbitrary functions. They are maps that preserve the structure the category cares about.

## Identity and Composition

Identity morphisms formalize the idea of "doing nothing but staying in the same space." In ML, an identity map appears when a residual block carries forward an unchanged activation, or when a pipeline stage is intentionally skipped.

Composition formalizes staged computation. If

$$
f : X \to Z
\qquad \text{and} \qquad
g : Z \to Y,
$$

then $g \circ f : X \to Y$ is the pipeline that first computes $f$ and then computes $g$.

> **Example.** Suppose:
>
> - $n : \mathbb{R}^d \to \mathbb{R}^d$ normalizes a feature vector;
> - $h : \mathbb{R}^d \to \mathbb{R}^k$ produces a hidden representation;
> - $o : \mathbb{R}^k \to \mathbb{R}^c$ maps the representation to class scores.
>
> Then the overall predictor is
> $$
> o \circ h \circ n : \mathbb{R}^d \to \mathbb{R}^c.
> $$

The associativity axiom says that it does not matter how we parenthesize this chain. That seems obvious, but it is precisely what lets large systems be reasoned about in modular blocks.

## Commutative Diagrams

A diagram is a picture of objects and morphisms. It is **commutative** if every path with the same start and end gives the same composite morphism.

> **Example.** Let $f : X \to Y$, $g : Y \to Z$, and $h : X \to Z$. The diagram
>
> $$
> \begin{CD}
> X @>f>> Y \\
> @VhVV   @VVgV \\
> Z @= Z
> \end{CD}
> $$
>
> commutes when
> $$
> h = g \circ f.
> $$

In practice, a commutative diagram says that two ways of computing something agree.

### Why Diagrams Matter in ML

Diagrams help express:

- a feature pipeline that can be factored into simpler stages;
- an encoder followed by a classifier;
- a data transformation that should preserve labels;
- a transfer-learning map that should be consistent with a downstream task map.

In each case, the point is not the picture itself. The point is the equality of composites that the picture encodes.

## Products and Coproducts

Products and coproducts are early examples of **universal properties**, which characterize an object by how maps into or out of it behave.

### Product

In `Set`, the product of sets $A$ and $B$ is the Cartesian product $A \times B$, together with projection maps

$$
\pi_A : A \times B \to A,
\qquad
\pi_B : A \times B \to B.
$$

> **Universal Property of the Product.** Given any object $X$ and morphisms
> $$
> f : X \to A, \qquad g : X \to B,
> $$
> there is a unique morphism
> $$
> \langle f, g \rangle : X \to A \times B
> $$
> such that
> $$
> \pi_A \circ \langle f, g \rangle = f,
> \qquad
> \pi_B \circ \langle f, g \rangle = g.
> $$

Intuitively, to define a map into a product, it is enough to define each component.

> **ML Interpretation.** Feature concatenation often behaves like a product construction. If one map extracts text features and another extracts metadata features from the same example $x$, then pairing them yields a combined representation.

### Coproduct

The coproduct is dual to the product. In `Set`, it is the disjoint union $A \sqcup B$ with inclusion maps

$$
\iota_A : A \to A \sqcup B,
\qquad
\iota_B : B \to A \sqcup B.
$$

> **Universal Property of the Coproduct.** Given morphisms
> $$
> f : A \to X, \qquad g : B \to X,
> $$
> there is a unique morphism
> $$
> [f, g] : A \sqcup B \to X
> $$
> such that
> $$
> [f, g] \circ \iota_A = f,
> \qquad
> [f, g] \circ \iota_B = g.
> $$

Intuitively, to define a map out of a coproduct, it is enough to say what happens on each branch.

> **ML Interpretation.** A multimodal or mixture-style system can look coproduct-like: one branch handles images, another handles tabular metadata, and a downstream rule merges the branchwise outputs into one common target space.

## Universal Properties: Intuition First

A universal property does not describe an object by its internal ingredients alone. It describes the object by the role it plays in relation to all other objects.

That shift is important. It means:

- a product is not just a particular construction;
- it is the most efficient object receiving maps that jointly encode $A$ and $B$;
- any other object with the same mapping behavior is equivalent in the categorical sense.

For this primer, the main lesson is that universal properties define structures by how they compose with other maps. This is part of why category theory is useful for ML architecture: it emphasizes interfaces and composition laws, not just implementation details.

## Functors

> **Definition.** A **functor** $F : \mathcal{C} \to \mathcal{D}$ sends:
>
> - each object $A$ of $\mathcal{C}$ to an object $F(A)$ of $\mathcal{D}$;
> - each morphism $f : A \to B$ of $\mathcal{C}$ to a morphism $F(f) : F(A) \to F(B)$ of $\mathcal{D}$,
>
> in a way that preserves identities and composition:
>
> $$
> F(\mathrm{id}_A) = \mathrm{id}_{F(A)},
> \qquad
> F(g \circ f) = F(g) \circ F(f).
> $$

A functor translates one compositional world into another without breaking the wiring rules.

> **Example.** The forgetful functor
> $$
> U : \mathrm{Grp} \to \mathrm{Set}
> $$
> sends a group to its underlying set and a group homomorphism to its underlying function.

> **ML Interpretation.** A representation map often acts like a structure-preserving translation. For example, a pipeline might send raw objects into vector representations while preserving which transformations of the input count as meaningful comparisons. We should be careful here: not every embedding pipeline is literally a functor. The point is that functorial language becomes useful when we want consistent transport of structure across stages.

## Natural Transformations

Natural transformations compare functors.

> **Definition.** Let $F, G : \mathcal{C} \to \mathcal{D}$ be functors. A **natural transformation**
> $$
> \eta : F \Rightarrow G
> $$
> assigns to each object $A$ a morphism
> $$
> \eta_A : F(A) \to G(A)
> $$
> such that for every morphism $f : A \to B$ in $\mathcal{C}$, the diagram
> $$
> \begin{CD}
> F(A) @>F(f)>> F(B) \\
> @V\eta_A VV      @VV\eta_B V \\
> G(A) @>G(f)>> G(B)
> \end{CD}
> $$
> commutes.

This means that converting by $\eta$ before or after transporting along $f$ gives the same result.

> **ML Interpretation.** Suppose two feature-construction procedures assign representations to each input space, and $\eta_A$ converts the first representation into the second. Naturality says this conversion is compatible with how data transformations propagate through the pipeline. In practice, this idea appears when we demand consistency across model families, resolutions, or augmentations.

For a first pass, it is enough to remember that natural transformations are "coherent families of maps between functors."

## Monoidal Intuition

Many ML systems combine components in parallel as well as in sequence. Category theory captures that idea with monoidal structure.

> **Informal Definition.** A **monoidal category** is a category equipped with:
>
> - a binary operation $\otimes$ for putting objects and morphisms side by side;
> - a unit object $I$;
> - coherence laws ensuring parallel composition behaves consistently with sequential composition.

For this primer, we do not need the full formal axioms. We need the intuition.

### Why Monoidal Structure Appears in ML

- pairing two feature streams side by side;
- processing two modalities in parallel;
- combining independent transformations on separate coordinates;
- building tensor-product style constructions in linear algebra.

In finite-dimensional vector spaces, tensor products provide a standard monoidal operation. In sets, Cartesian product often plays the parallel-composition role.

> **Example.** If one branch computes an image embedding and another computes a text embedding, a fused representation may be formed by pairing or combining them. Sequential composition says "do this, then that." Monoidal intuition says "do these in parallel, then combine."

## Five ML Constructions as Compositional Diagrams

The following examples are intentionally simple. Their job is to show how categorical language clarifies composition.

### 1. Feature Pipeline

Let:

- $n : X \to X'$ be normalization;
- $\phi : X' \to H$ be feature extraction;
- $c : H \to Y$ be a classifier.

Then the predictor is

$$
c \circ \phi \circ n : X \to Y.
$$

This is the basic categorical pattern: a model is a composite morphism.

### 2. Training Loop as Repeated Update Composition

Let:

- $L : \Theta \times \mathcal{D} \to \mathbb{R}$ be a loss;
- $g : \Theta \times \mathcal{D} \to T\Theta$ compute a gradient-like update direction;
- $u : \Theta \times T\Theta \to \Theta$ apply a parameter update.

Fixing the dataset $D$, one training step can be viewed as a map

$$
s_D : \Theta \to \Theta.
$$

Multiple training epochs are then compositions

$$
s_D \circ s_D \circ \cdots \circ s_D.
$$

The categorical point is not that optimization becomes trivial. It is that an iterative procedure is a compositional process on a parameter space.

### 3. Encoder-Decoder

Let:

- $e : X \to Z$ be an encoder;
- $d : Z \to X$ be a decoder.

Then the reconstruction map is

$$
d \circ e : X \to X.
$$

The latent space $Z$ is useful precisely because a complicated end-to-end map factors through it.

### 4. Data Augmentation Chain

Suppose:

- $a_1 : X \to X$ is a crop;
- $a_2 : X \to X$ is a color jitter;
- $a_3 : X \to X$ is a blur.

Then the total augmentation is

$$
a_3 \circ a_2 \circ a_1 : X \to X.
$$

If labels are unchanged under these augmentations, we also want consistency with a label map $y : X \to \mathcal{Y}$. Diagrammatically, we want the square

$$
\begin{CD}
X @>a>> X \\
@VyVV   @VVyV \\
\mathcal{Y} @= \mathcal{Y}
\end{CD}
$$
to commute, meaning $y \circ a = y$.

### 5. Transfer Learning

Let:

- $f_{\mathrm{src}} : X_{\mathrm{src}} \to H$ be a representation learned on a source task;
- $t : H \to H'$ adapt that representation;
- $c_{\mathrm{tgt}} : H' \to Y_{\mathrm{tgt}}$ be a target-task head.

Then the transferred predictor is

$$
c_{\mathrm{tgt}} \circ t \circ f_{\mathrm{src}}.
$$

The map $t$ is the adaptation layer between the source representation and the target decision rule. Later, Module 16 will revisit such constructions in a more formal way.

## Remarks on Limits

Category theory is useful here because it highlights:

- what composes with what;
- which equalities of composites matter;
- which structures should be preserved.

It does **not** by itself tell us:

- how to optimize a loss;
- whether a model generalizes;
- how to estimate uncertainty;
- which approximation is numerically stable.

Those questions still require the standard mathematics of ML.

## Summary

This primer introduced:

- categories as objects and morphisms with identities and composition;
- standard examples `Set`, `Vect`, and `Grp`;
- commutative diagrams as equalities of composite maps;
- products and coproducts as first universal constructions;
- functors as structure-preserving translations between categories;
- natural transformations as coherent maps between functors;
- monoidal intuition as parallel composition.

For early ML, the central habit is simple: when you see a pipeline, latent factorization, update rule, or invariance condition, ask what the objects are, what the morphisms are, and which diagrams should commute.
