---
title: "Products, Limits, and Colimits for ML Interfaces"
module: "16-category-theory-for-ml"
lesson: "products-limits-colimits"
doc_type: "notes"
topic: "universal-properties"
status: "draft"
prerequisites:
  - "00-math-toolkit/category-theory-primer"
  - "03-linear-models"
  - "05-probabilistic-modeling"
  - "13-graph-learning"
updated: "2026-04-13"
owner: "curriculum-team"
tags:
  - "category-theory"
  - "products"
  - "limits"
  - "colimits"
  - "multimodal-learning"
---

# Products, Limits, and Colimits for ML Interfaces

## Purpose

Universal properties shift attention away from coordinates and toward interfaces. In ML, that is useful whenever a construction is best understood by how information enters or exits it rather than by one particular implementation.

We restrict attention to products, coproducts, pullbacks, and pushout intuition.

## 1. Products

> **Definition.** A **product** of objects $A$ and $B$ in a category $\mathcal{C}$ is an object $A \times B$ with morphisms
> $$
> \pi_A : A \times B \to A,
> \qquad
> \pi_B : A \times B \to B,
> $$
> such that for every object $X$ and morphisms
> $$
> f : X \to A,
> \qquad
> g : X \to B,
> $$
> there exists a unique morphism
> $$
> \langle f,g \rangle : X \to A \times B
> $$
> with
> $$
> \pi_A \circ \langle f,g \rangle = f,
> \qquad
> \pi_B \circ \langle f,g \rangle = g.
> $$

### ML Example: Paired Features in Supervised Learning

If an example has metadata $m(x)$ and learned features $z(x)$, then the joint feature map
$$
x \mapsto (m(x), z(x))
$$
is product-shaped. The universal property says that any such joint constructor is determined by its two coordinate maps.

Products therefore model jointly available information.

## 2. Coproducts

> **Definition.** A **coproduct** of $A$ and $B$ is an object $A \sqcup B$ with morphisms
> $$
> \iota_A : A \to A \sqcup B,
> \qquad
> \iota_B : B \to A \sqcup B,
> $$
> such that for every object $X$ and morphisms
> $$
> f : A \to X,
> \qquad
> g : B \to X,
> $$
> there exists a unique morphism
> $$
> [f,g] : A \sqcup B \to X
> $$
> with
> $$
> [f,g] \circ \iota_A = f,
> \qquad
> [f,g] \circ \iota_B = g.
> $$

### ML Example: Branchwise Processing

If a system accepts either a text query or an image query and maps both into a shared retrieval space $R$, then one branch map
$$
f : \mathrm{Text} \to R
$$
and another
$$
g : \mathrm{Image} \to R
$$
determine a unique case-handling map from a coproduct-like input type.

Coproducts model one of several alternatives, not jointly observed information.

## 3. Pullbacks

> **Definition.** Given morphisms
> $$
> f : A \to C,
> \qquad
> g : B \to C,
> $$
> a **pullback** is an object $P$ with morphisms
> $$
> p_A : P \to A,
> \qquad
> p_B : P \to B
> $$
> such that
> $$
> f \circ p_A = g \circ p_B,
> $$
> and universal with respect to this property.

Intuitively, $P$ records compatible pairs from $A$ and $B$ over the common target $C$.

### ML Example: Joining Two Sources by a Shared Key

Let:

- $A$ be image examples with patient identifiers;
- $B$ be lab measurements with patient identifiers;
- $C$ be the identifier space.

Maps
$$
f : A \to C,
\qquad
g : B \to C
$$
send each record to its identifier. The pullback consists of compatible pairs
$$
(a,b) \in A \times B
\quad \text{such that} \quad
f(a)=g(b).
$$

This is the structural core of a relational join.

## 4. Pushout Intuition

> **Definition.** Dually, given morphisms
> $$
> f : C \to A,
> \qquad
> g : C \to B,
> $$
> a **pushout** is an object $Q$ receiving maps from $A$ and $B$ that identify points coming from the common source $C$, and is universal with that property.

We use pushouts only at the level of intuition in this module.

### ML Reading

Pushout-shaped reasoning appears when two constructions share a common subinterface and we want to glue them along that shared part, for example:

- merging two schema-level descriptions with common labels;
- combining architecture components that share a standardized interface;
- unifying task ontologies along an agreed overlap.

## 5. Representation-Learning Case Study

Consider a multimodal learner using image views and text captions for the same identity set $I$.

Let:

- $u : \mathrm{Img} \to I$ assign each image to its sample id;
- $v : \mathrm{Txt} \to I$ assign each caption to its sample id.

Then the pullback
$$
\mathrm{Img} \times_I \mathrm{Txt}
$$
collects compatible image-text pairs. A learner on this object sees actual aligned pairs rather than arbitrary pairs.

This sharpens the distinction between exact supervised alignment and heuristic similarity-based pairing.

## Summary

Products, coproducts, pullbacks, and pushout intuition provide a structural vocabulary for:

- jointly available features;
- branchwise case handling;
- compatibility-based data alignment; and
- interface-level module composition.
