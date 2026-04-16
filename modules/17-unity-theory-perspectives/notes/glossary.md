---
title: "Unity Theory Glossary for ML Readers"
module: "17-unity-theory-perspectives"
lesson: "glossary"
doc_type: "notes"
topic: "unity-theory-glossary"
status: "draft"
prerequisites:
  - "17-unity-theory-perspectives/scope-memo"
  - "00-math-toolkit/unity-theory-primer-interface"
  - "01-optimization/unity/optimization-companion"
updated: "2026-04-13"
owner: "curriculum-team"
tags:
  - "unity-theory"
  - "glossary"
  - "representation"
  - "companion-material"
---

# Unity Theory Glossary for ML Readers

## Purpose

This glossary defines the core Unity Theory terms used in the companion layer of the curriculum.
Its role is operational rather than metaphysical: each term is defined so that an ML reader can tell what kind of canonical object, process, or comparison the term is pointing to.

> [!NOTE]
> **Interpretive note.** These entries are disciplined correspondences for companion notes. They are not part of standard ML terminology, and they do not replace canonical definitions from linear algebra, probability, optimization, representation learning, or category theory.

## How to read this glossary

For each term:

1. read the canonical anchor first;
2. use the Unity Theory term only for the narrow interpretation stated here; and
3. check the "does not mean" clause before reusing the term elsewhere.

This glossary is deliberately conservative.
If a later note needs a stronger or more speculative meaning, that note should say so explicitly instead of silently stretching the definition.

## Term map

| Term | Canonical anchor | Main question it helps ask |
| --- | --- | --- |
| identity | state, representation, equivalence class, task-stable object | What counts as the same thing for the task? |
| relation | map, dependency, interaction, compatibility condition | Through which structured connections does the object matter? |
| embodiment | concrete realization in data, coordinates, architecture, or parameters | In what finite form is the structure actually realized? |
| multiplicity | many views, decompositions, coordinates, hypotheses, or trajectories | How can one target object be expressed in more than one valid way? |
| coherence | selective stability, consistency, invariance, compatibility | Which changes preserve task-relevant structure? |
| informational action | update or decision driven by available information under constraints | How does the system use information to produce admissible change? |

## Identity

### Definition

`Identity` is the task-indexed criterion by which a learning system treats two presentations, states, or representations as counting as the same object for the purpose at hand.

In canonical terms, identity is usually realized by one of the following:

- an equivalence relation induced by admissible transformations;
- a representation that preserves the observables needed for prediction or control; or
- a state description that remains behaviorally interchangeable for the downstream task.

The key point is that identity is not "whatever exists in the world independent of modeling."
It is the specific sameness criterion operationalized by the model, dataset, and task.

### ML-grounded example

In image classification, two images of the same cat under small translations or lighting changes may be treated as the same task-relevant object.
A representation is preserving identity when those nuisance changes do not alter the class-relevant information used by the classifier.

### Does not mean

Identity does not mean:

- metaphysical essence;
- byte-for-byte equality of inputs;
- identical coordinates in every embedding space; or
- guaranteed human-level object persistence.

Two representations may encode the same task identity while differing by invertible change of basis, scaling, or other benign reparameterization.

### Companion-note cross-references

- Primer interface: [unity-theory-primer-interface.md](../../00-math-toolkit/unity/unity-theory-primer-interface.md)
- Scope memo: [scope-memo.md](./scope-memo.md)
- Optimization companion: [optimization-companion.md](../../01-optimization/unity/optimization-companion.md)

## Relation

### Definition

`Relation` is the structured linkage through which an ML object participates in computation, comparison, dependence, or update.

Depending on context, the canonical anchor may be:

- a function or morphism between spaces;
- a similarity or kernel evaluation;
- a conditional dependence in a probabilistic model;
- an edge pattern in a graph; or
- a compatibility condition expressed by a commutative or approximately commutative diagram.

The term is useful because a representation with no task-relevant relations attached to it is not yet informative for learning.

### ML-grounded example

In a graph neural network, the hidden state of a node matters partly because of its relations to neighboring nodes through message passing.
The representation is not interpreted in isolation; it is interpreted through the update and aggregation relations that connect it to the rest of the graph.

### Does not mean

Relation does not mean:

- any vague association whatever;
- a mystical connection outside the model;
- a guarantee of causation; or
- that every mathematical map deserves special interpretive emphasis.

The term should be used only when the relevant map, dependency, or interaction can be named concretely.

### Companion-note cross-references

- Primer interface: [unity-theory-primer-interface.md](../../00-math-toolkit/unity/unity-theory-primer-interface.md)
- Scope memo: [scope-memo.md](./scope-memo.md)

## Embodiment

### Definition

`Embodiment` is the concrete finite realization of an abstract structure in a particular dataset, coordinate system, model architecture, sensor channel, optimization path, or parameterization.

Canonical ML objects are never encountered as pure abstractions alone.
They are always embodied in specific design choices:

- a latent variable is realized in a specific network and loss;
- a signal is recorded through a specific measurement process;
- a classifier is trained on a specific sample from a population; and
- a learned representation is carried by particular parameters and computational constraints.

Embodiment is the reminder that implementation choices are not incidental wrappers around a content-free abstract object.

### ML-grounded example

A latent code in a variational autoencoder and a hidden representation in a contrastive encoder may both aim to capture similar semantic structure, but they embody that structure differently because the objectives, architectures, and data pipelines differ.

### Does not mean

Embodiment does not mean:

- that one implementation is the uniquely correct one;
- that abstract structure is unreal or useless;
- merely "having physical hardware"; or
- that every implementation detail is equally important to interpretation.

The point is narrower: any learned structure reaches the user through a specific realization with specific biases and limitations.

### Companion-note cross-references

- Primer interface: [unity-theory-primer-interface.md](../../00-math-toolkit/unity/unity-theory-primer-interface.md)
- Primer discussion of latent codes: [unity-theory-primer-interface.md](../../00-math-toolkit/unity/unity-theory-primer-interface.md)
- Scope memo: [scope-memo.md](./scope-memo.md)

## Multiplicity

### Definition

`Multiplicity` is the structured fact that one task-relevant object, pattern, or objective may admit more than one valid description, decomposition, trajectory, or local realization.

In canonical ML language, multiplicity appears as:

- multiple coordinate systems or bases for the same vector space;
- multiple feature decompositions supporting the same prediction task;
- multiple parameter settings that realize similar input-output behavior; or
- multiple candidate hypotheses or trajectories consistent with limited data.

Multiplicity is not mere ambiguity.
It is plurality constrained by shared structure.

### ML-grounded example

Two neural networks with different hidden units may learn internally different feature bases while still achieving similar downstream classification accuracy.
The task-relevant object is not tied to a single literal coordinate decomposition of the hidden representation.

### Does not mean

Multiplicity does not mean:

- that all descriptions are equally useful;
- that identifiability problems disappear;
- that model selection no longer matters; or
- permission to use imprecise language about "many perspectives" without naming the structure held fixed.

Multiplicity must always be paired with the question: many realizations of what, under which constraints?

### Companion-note cross-references

- Primer interface: [unity-theory-primer-interface.md](../../00-math-toolkit/unity/unity-theory-primer-interface.md)
- Primer discussion of basis choice: [unity-theory-primer-interface.md](../../00-math-toolkit/unity/unity-theory-primer-interface.md)
- Scope memo: [scope-memo.md](./scope-memo.md)

## Coherence

### Definition

`Coherence` is selective stability or compatibility of task-relevant structure across a specified family of transformations, views, updates, or inferential constraints.

The canonical anchors include:

- invariance or equivariance under an admissible transformation family;
- consistency between coupled model components;
- stability of predictions or representations under perturbations that should preserve task identity; and
- approximate commutation or compatibility of the maps that define a learning system.

Coherence is therefore always relative to a named standard.
One must say coherent with respect to which transformations, observables, objectives, or constraints.

### ML-grounded example

If a vision model should preserve labels under small translations, then stable predictions under those translations are one form of coherence.
If a sequence model should preserve long-range state information, then the hidden-state dynamics are coherent only to the extent that the relevant temporal dependencies remain usable across updates.

### Does not mean

Coherence does not mean:

- immunity to all perturbations;
- global correctness;
- guaranteed generalization;
- visual smoothness in an embedding plot; or
- a substitute for proper evaluation metrics.

A model can be coherent relative to one transformation family and still fail badly under dataset shift, spurious correlations, or a badly chosen objective.

### Companion-note cross-references

- Primer interface: [unity-theory-primer-interface.md](../../00-math-toolkit/unity/unity-theory-primer-interface.md)
- Primer discussion of invariance: [unity-theory-primer-interface.md](../../00-math-toolkit/unity/unity-theory-primer-interface.md)
- Scope memo: [scope-memo.md](./scope-memo.md)
- Optimization companion: [optimization-companion.md](../../01-optimization/unity/optimization-companion.md)

## Informational Action

### Definition

`Informational action` is an update, intervention, or decision that uses available information to produce admissible change under explicit constraints.

This term is the most process-oriented entry in the glossary.
Its canonical anchors are things like:

- gradient-based parameter updates;
- belief or posterior updates from new evidence;
- policy actions chosen from a state estimate under environment constraints; and
- control or search steps that convert local information into directed behavior.

The term is useful when the important question is not just what a representation is, but how the system acts on the basis of partial information and within restricted feasible moves.

### ML-grounded example

In projected gradient descent, the gradient supplies local information about objective decrease, while the projection enforces feasibility.
The resulting step is informational action in a narrow technical sense: it turns available first-order information into an admissible update.

In reinforcement learning, a policy uses state information to choose an action under dynamics and reward constraints.
That is another form of informational action, though the canonical explanation still comes from MDPs, value functions, and policy optimization.

### Does not mean

Informational action does not mean:

- free-floating agency;
- consciousness or intent;
- that every update rule deserves anthropomorphic language; or
- a replacement for the mathematics of optimization, inference, or control.

If the underlying information source, action space, or constraint set cannot be named precisely, the term is being used too loosely.

### Companion-note cross-references

- Module 17 overview: [README.md](../README.md)
- Optimization companion correspondence table: [optimization-companion.md](../../01-optimization/unity/optimization-companion.md)
- Optimization companion discussion: [optimization-companion.md](../../01-optimization/unity/optimization-companion.md)

## Usage constraints across companion notes

The glossary is intended to prevent drift.
When these terms are reused elsewhere in Module 17 or in `unity/` companion notes across the course, the following rules apply:

1. Name the canonical object first.
2. State the Unity mapping explicitly rather than assuming the reader already accepts it.
3. Keep the interpretation narrower than the theorem, derivation, or experiment it comments on.
4. Do not infer stronger philosophical claims from a successful local correspondence.
5. If a note needs `transformation` as a separate term, use the Module 00 primer definition and treat it as the change relative to which identity or coherence is being assessed.

## Minimal memory aid

If a reader wants the shortest possible summary:

- `identity` asks what counts as the same;
- `relation` asks through which structured links it matters;
- `embodiment` asks where that structure is concretely realized;
- `multiplicity` asks how it can appear in more than one valid form;
- `coherence` asks what remains stable under admissible change; and
- `informational action` asks how available information is converted into constrained update or decision.
