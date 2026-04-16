---
title: "Category Theory Exercises for ML"
module: "16-category-theory-for-ml"
lesson: "category-theory-exercises"
doc_type: "exercises"
topic: "category-theory-consolidation"
status: "draft"
prerequisites:
  - "16-category-theory-for-ml/categories-functors-nts"
  - "16-category-theory-for-ml/products-limits-colimits"
  - "16-category-theory-for-ml/monoidal-categories"
  - "16-category-theory-for-ml/operads-intuition"
  - "16-category-theory-for-ml/diagrammatic-reasoning"
updated: "2026-04-13"
owner: "curriculum-team"
tags:
  - "category-theory"
  - "ml-exercises"
  - "structural-reasoning"
---

# Category Theory Exercises for ML

These exercises emphasize structural reasoning. Most problems ask you to specify objects, morphisms, and commuting conditions rather than repeat definitions.

## Exercise 1: Supervised pipeline as composition

**Taxonomy**

- `difficulty`: `foundational`
- `type`: `analysis`
- `tags`: `composition`, `pipeline`, `morphisms`

Let a classifier factor as
$$
X \xrightarrow{\phi} H \xrightarrow{h} Y.
$$

1. State the composite predictor as a single morphism.
2. Suppose a preprocessing map $n : X' \to X$ is added before $\phi$. Write the new composite.
3. Explain in two or three sentences what associativity guarantees about grouping these stages.

## Exercise 2: Exact or analogical?

For each claim below, label it as `exact`, `possibly exact with extra setup`, or `merely suggestive`, and justify your answer briefly.

1. A feedforward network is a composite of morphisms.
2. A learned adapter between two encoders is automatically a natural transformation.
3. A symmetry-respecting representation may define a functor.
4. Gradient descent is a functor.

## Exercise 3: Naturality square for representations

Let $F,G : \mathcal{C} \to \mathrm{Vect}_{\mathbb{R}}$ be two representation functors and let $\eta : F \Rightarrow G$.

1. Write the naturality condition for a morphism $f : A \to B$.
2. Explain why objectwise linear maps $\eta_A$ are not enough by themselves.
3. Give one plausible ML interpretation of the failure of naturality.

## Exercise 4: Product or coproduct?

For each situation below, decide whether product or coproduct structure is the better first abstraction. Justify the choice.

1. A model consumes both a patient embedding and a lab summary for the same patient.
2. A retrieval system accepts either an image query or a text query.
3. A multimodal classifier concatenates audio and video features observed for the same clip.

## Exercise 5: Pullback reasoning

Suppose $A$ is a set of medical images, $B$ is a set of clinical records, and both map to a patient-id set $I$.

1. Write the pullback condition explicitly.
2. Describe what elements of the pullback represent.
3. Explain why nearest-neighbor pairing by embedding similarity is not literally the same construction.

## Exercise 6: Monoidal structure and message passing

Consider one layer of message passing on a graph.

1. Identify one part of the construction that is sequential.
2. Identify one part that is parallel.
3. Explain why ordinary function composition alone is not the most informative structural description.

## Exercise 7: Attention as structured composition

Write a short structural description of self-attention using the stages:

- query-key-value construction;
- score computation;
- normalization;
- value aggregation.

Then answer:

1. Which stages are most naturally sequential?
2. Where does parallelism appear in multi-head attention?
3. Why does the categorical description not remove the need for matrix algebra?

## Exercise 8: Diagrammatic invariance

Let $a : X \to X$ be an augmentation and $y : X \to Y$ a label map.

1. Write the commutative square expressing label preservation.
2. Explain what it would mean for the square to fail.
3. Give one concrete augmentation in vision or graph learning for which commutativity is intended and one for which it may fail.

## Exercise 9: Agent pipeline as operadic composition

An agent pipeline contains a planner, retriever, tool executor, and verifier.

1. Describe one stage as a multi-input operation with typed inputs and one output.
2. Explain why this is closer to operadic than purely unary categorical language.
3. State one advantage of making the slot structure explicit.

## Exercise 10: Structural critique

Pick one of the following case studies:

- supervised learning;
- representation learning;
- message passing;
- attention;
- agent pipelines.

Write a short response addressing all four prompts:

1. What is the canonical ML construction?
2. What categorical structure is exact here?
3. What part of the categorical reading is only conditional or partial?
4. What important part of the analysis still depends on non-categorical mathematics?
