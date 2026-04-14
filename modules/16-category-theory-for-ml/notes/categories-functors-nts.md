---
title: "Categories, Functors, and Natural Transformations for ML"
module: "16-category-theory-for-ml"
lesson: "categories-functors-nts"
doc_type: "notes"
topic: "category-theory-core"
status: "draft"
prerequisites:
  - "00-math-toolkit/category-theory-primer"
  - "00-math-toolkit/ml-categorical-mapping"
  - "02-statistical-learning"
  - "06-neural-networks"
  - "08-cnn-vision"
updated: "2026-04-13"
owner: "curriculum-team"
tags:
  - "category-theory"
  - "functors"
  - "natural-transformations"
  - "supervised-learning"
  - "representation-learning"
---

# Categories, Functors, and Natural Transformations for ML

## Purpose

Module 00 introduced categories, functors, and natural transformations as a primer. This note returns to those ideas formally and asks a stricter question:

> Which ML constructions are genuinely categorical, and which are only suggestively so?

The payoff is a disciplined language for compositional pipelines, representation maps, and coherent comparisons between model families.

## 1. Categories Revisited

> **Definition.** A **category** $\mathcal{C}$ consists of:
>
> - a class of objects;
> - for each pair of objects $A,B$, a class $\mathrm{Hom}_{\mathcal{C}}(A,B)$ of morphisms $f : A \to B$;
> - for each composable pair $f : A \to B$ and $g : B \to C$, a composition law $g \circ f : A \to C$;
> - for each object $A$, an identity morphism $\mathrm{id}_A : A \to A$.
>
> These satisfy associativity and the identity laws.

In this module, objects are usually spaces with intended structure and morphisms are the maps that preserve the structure we care about. The choice of morphisms is part of the modeling decision.

### Example: A Supervised Pipeline Category

Fix a family of admissible data spaces. Let an object be a triple
$$
(X, Y, \ell),
$$
where $X$ is an input space, $Y$ is a label space, and $\ell : X \times Y \to \mathbb{R}_{\ge 0}$ is a task loss. A morphism
$$
(u,v) : (X,Y,\ell) \to (X',Y',\ell')
$$
is a pair of maps $u : X \to X'$ and $v : Y \to Y'$ preserving the task interface in the chosen sense.

This is only one possible category of supervised tasks, but it illustrates the main point: categorical claims require explicit objects and morphisms.

## 2. Functors

> **Definition.** A **functor** $F : \mathcal{C} \to \mathcal{D}$ assigns:
>
> - to each object $A$ of $\mathcal{C}$ an object $F(A)$ of $\mathcal{D}$;
> - to each morphism $f : A \to B$ a morphism $F(f) : F(A) \to F(B)$;
>
> such that
> $$
> F(\mathrm{id}_A) = \mathrm{id}_{F(A)},
> \qquad
> F(g \circ f) = F(g) \circ F(f).
> $$

A functor transports both entities and compositional structure.

### ML Reading: Representation as Structure Transport

Suppose $\mathcal{A}$ is a category whose objects are image domains equipped with admissible augmentations and whose morphisms are label-preserving transformations. Let $\mathrm{Vect}_{\mathbb{R}}$ denote real vector spaces and linear maps.

A representation pipeline may define a functor
$$
F : \mathcal{A} \to \mathrm{Vect}_{\mathbb{R}}
$$
by sending:

- each input domain $X$ to a feature space $F(X)$;
- each admissible transformation $f : X \to X'$ to a linear transport map
  $$
  F(f) : F(X) \to F(X').
  $$

When this is legitimate, it formalizes the claim that the representation treats data transformations coherently rather than sample by sample.

## 3. Natural Transformations

> **Definition.** Let $F,G : \mathcal{C} \to \mathcal{D}$ be functors. A **natural transformation**
> $$
> \eta : F \Rightarrow G
> $$
> assigns to each object $A$ a morphism
> $$
> \eta_A : F(A) \to G(A)
> $$
> such that for every morphism $f : A \to B$ in $\mathcal{C}$,
> $$
> \eta_B \circ F(f) = G(f) \circ \eta_A.
> $$

The condition says that converting from the $F$-construction to the $G$-construction is compatible with every admissible map in the source category.

### Naturality Square

$$
\begin{CD}
F(A) @>F(f)>> F(B) \\
@V\eta_A VV        @VV\eta_B V \\
G(A) @>G(f)>> G(B)
\end{CD}
$$

The square commutes if both routes agree.

## 4. Supervised Learning as Compositional Structure

A supervised predictor is usually a composite
$$
X \xrightarrow{\phi} H \xrightarrow{h} Y,
$$
where $\phi$ is a feature map and $h$ is a task head. Categorically, this is the first payoff: a model is often a composite morphism, not a monolith.

### Structural Note: Training Is Usually Extra Structure

If $\Theta$ is parameter space and $S_D : \Theta \to \Theta$ is one optimizer step for dataset $D$, then iterative training is repeated composition of an endomorphism:
$$
S_D^t = \underbrace{S_D \circ \cdots \circ S_D}_{t\text{ times}}.
$$

This is a categorical pattern, but convergence and generalization remain analytical questions.

## 5. Representation Learning Case Study

Consider a category $\mathcal{T}$ of input domains with admissible transformations such as crops, translations, or graph isomorphisms. Suppose two representation schemes produce functors
$$
F,G : \mathcal{T} \to \mathrm{Vect}_{\mathbb{R}}.
$$

For each domain $X$, let
$$
\eta_X : F(X) \to G(X)
$$
be an adapter from one representation family to another.

If
$$
\eta_{X'} \circ F(f) = G(f) \circ \eta_X
$$
for every admissible transformation $f : X \to X'$, then $\eta$ is a natural transformation. This means the adapter is not merely objectwise available; it is coherent across transformations.

This cleanly distinguishes:

- a one-off projection layer between two encoders;
- a family of conversions that works uniformly across all admissible domains and transformations.

## 6. Exactness Versus Analogy

| Claim | Usually exact? | Comment |
| --- | --- | --- |
| "A feedforward network is a composite of maps." | Yes | Literal composition. |
| "A symmetry-respecting representation defines a functor." | Sometimes | Only after source and target categories are explicit. |
| "An adapter family is a natural transformation." | Sometimes | Requires a commuting square for every source morphism. |
| "Optimization is a functor." | Usually not at this level | Too vague without more setup. |

## Summary

The formal lesson is short but powerful:

- categories give the ambient compositional setting;
- functors transport structure across domains;
- natural transformations compare constructions coherently;
- supervised and representation-learning pipelines become easier to analyze once we separate exact structure from analogy.
