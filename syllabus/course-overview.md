# Course Overview

## Course Thesis
This course is a repository-based machine learning and AI curriculum for learners who want standard ML taught with full mathematical and computational accountability. It follows the canonical sequence used in strong technical programs, from linear algebra, probability, and optimization through statistical learning, deep learning, generative modeling, reinforcement learning, and modern AI systems. What distinguishes it from a typical applied course is not a different subject matter but a different standard of integration: core results are derived rather than only stated, implementations are tied to the mathematics they realize, empirical behavior is interpreted in light of those derivations, and structural ideas from category theory are introduced when they clarify composition, abstraction, invariance, and transfer. Unity Theory appears only as a clearly marked companion layer for interpretation and research reflection, never as a replacement for conventional ML exposition.

## Target Learners

### Primary learner persona
The primary learner is an advanced undergraduate, graduate student, researcher, or technical professional who wants a mathematically serious path through modern ML without sacrificing implementation skill. This learner is dissatisfied with curricula that stop at recipes, API usage, or intuition-only explanations, and wants to understand why methods work, where they fail, and how different parts of the field connect.

### Prerequisite expectations
Learners should be comfortable with:
- single-variable and multivariable calculus
- linear algebra at the level of vectors, matrices, eigendecompositions, and linear maps
- probability and basic statistics
- writing and debugging Python code

Helpful but not required background:
- introductory real analysis
- numerical optimization
- prior exposure to proofs
- prior exposure to abstraction, including algebraic or categorical language

This course is not aimed at absolute beginners in programming or mathematics, and it is not optimized for learners seeking the fastest path to training production models with minimal theory.

## Mathematical Rigor Target
The course targets moderate-to-high mathematical rigor. Major algorithms and model families should be accompanied by explicit assumptions, notation, derivations, and dimensionally clear intermediate steps whenever a competent learner would plausibly need them. Proofs should appear where they deepen understanding of a central result, but the default style is proof sketch plus worked derivation plus computational interpretation rather than theorem accumulation for its own sake. Learners should expect regular algebraic manipulation, probability calculations, optimization arguments, and occasional proof-based exercises, but not a pure-math presentation detached from implementation or experiment.

## Scope
In scope:
- canonical machine learning foundations, including linear algebra, probability, optimization, and statistical learning
- classical supervised and unsupervised methods, including linear models, kernel methods, and probabilistic models
- neural networks and deep learning, including training dynamics, architectures, and systems considerations
- modern AI topics such as transformers, generative models, reinforcement learning, graph learning, causality, and evaluation
- repository-native artifacts such as notes, derivations, notebooks, labs, exercises, projects, and references
- carefully placed category-theoretic framing that clarifies structure without displacing standard mathematics

Not in scope:
- a bootcamp-style survey focused primarily on tooling, prompting, or framework APIs
- a pure mathematics course in abstract algebra, category theory, or logic for their own sake
- Unity Theory as the default language of instruction for canonical ML topics
- speculative claims presented as established ML doctrine
- coverage breadth that comes at the expense of derivational clarity, implementation quality, or conceptual coherence

## Role of Category Theory
Category theory has a bounded structural role. Early in the curriculum, especially in the mathematical foundations, it appears in primer mode through concrete ideas such as objects, morphisms, composition, products, functors, and diagrams tied to sets, vector spaces, datasets, and ML pipelines. Later, in its dedicated advanced module, it becomes a more formal language for comparing architectures, reasoning about compositional systems, and framing transfer or invariance. It is used to clarify structure, not to replace linear algebra, probability, optimization, or statistical learning theory.

## Role of Unity Theory
Unity Theory has a bounded companion role. It may be used for interpretive correspondence, conceptual synthesis, or research-facing reflection around themes such as identity, relation, embodiment, symmetry, coherence, and transformation. When it appears, it should be explicitly labeled as interpretive, exploratory, or speculative as appropriate. Core lessons must remain legible to a conventional technical audience using standard ML language alone. A learner should be able to complete the canonical course without accepting Unity Theory claims, while still benefiting from those companion materials if they want a broader philosophical or research-generative frame.

## How This Course Differs From Standard ML Curricula
- It preserves the standard ML spine rather than replacing it with an idiosyncratic framework.
- It demands more mathematical internalization than most application-first courses.
- It ties derivation, implementation, and empirical interpretation together as a default pattern.
- It uses category theory as disciplined structural enrichment rather than ornamental abstraction.
- It keeps Unity Theory clearly separated from canonical instruction while still making room for companion synthesis.
