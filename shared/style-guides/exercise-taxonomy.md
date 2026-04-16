---
title: "Exercise Taxonomy"
module: "shared"
doc_type: "reference"
topic: "exercise-design"
status: "draft"
updated: "2026-04-15"
owner: "curriculum-team"
tags:
  - "assessment"
  - "exercise-taxonomy"
  - "difficulty-tiers"
  - "exercise-types"
---

# Exercise Taxonomy

## Purpose

This guide standardizes exercise difficulty, exercise format, and tagging conventions across all modules.
It extends the assessment philosophy in [Assessment Strategy](../../syllabus/assessment-strategy.md) into a concrete authoring standard for exercise banks.

## Design goals

- preserve the canonical-first ML sequence;
- keep mathematics, implementation, and interpretation visibly connected;
- make exercise banks comparable across modules without forcing identical local content; and
- support both instructor-led use and self-study.

## Difficulty tiers

Every exercise should be classified into one of the following tiers.

### Foundational

Purpose: confirm fluency with definitions, notation, standard calculations, and core conceptual distinctions.

Typical signs:
- short proofs or proof fragments;
- direct derivations with limited branching;
- dimensional checks;
- small implementation tasks with strong scaffolding; and
- brief interpretation prompts tied to a standard result.

Expected learner behavior:
- reproduce canonical results;
- execute a known procedure correctly; and
- explain the local meaning of an equation, algorithm step, or modeling choice.

Assessment alignment:
- closest to Tier 1 in [Assessment Strategy](../../syllabus/assessment-strategy.md).

### Intermediate

Purpose: require the learner to integrate mathematics, implementation, and interpretation across multiple steps.

Typical signs:
- multi-step derivations;
- structured debugging or notebook analysis;
- model-comparison exercises;
- empirical interpretation tasks; and
- short synthesis prompts that combine theory and code.

Expected learner behavior:
- connect equations to algorithmic behavior;
- explain what evidence would support or weaken a claim; and
- make bounded technical choices with justification.

Assessment alignment:
- closest to Tier 2 in [Assessment Strategy](../../syllabus/assessment-strategy.md).

### Challenge

Purpose: test transfer, judgment, and disciplined open-ended reasoning.

Typical signs:
- proof-plus-implementation tasks;
- comparative studies;
- open-ended design questions with bounded scope;
- critique of assumptions, limitations, or evaluation choices; and
- mini-project style prompts inside an exercise bank.

Expected learner behavior:
- adapt familiar tools to a new setting;
- defend modeling or systems choices;
- identify limitations and failure modes; and
- communicate scope clearly.

Assessment alignment:
- closest to Tier 3 in [Assessment Strategy](../../syllabus/assessment-strategy.md).

## Exercise types

Every exercise should have one primary type.
Secondary types may be mentioned in the tag list when useful, but the main type should stay singular for consistency.

### Proof

Use for theorem proofs, proof sketches, counterexamples, and assumption checks where formal validity is the main goal.

### Derivation

Use for algebraic, probabilistic, variational, or optimization-driven calculations where the learner must show intermediate steps and arrive at a formal result.

### Implementation

Use for coding tasks, notebook labs, reproducibility checks, experiment setup, or reusable component work where executable artifacts are central.

### Analysis

Use for metric interpretation, failure analysis, ablation reading, qualitative comparison, ethics or evaluation reasoning, and other evidence-based explanatory work.

### Open-ended

Use for bounded design prompts, mini-projects, critique memos, or research-facing tasks where more than one technically acceptable answer is possible.

## Required per-exercise taxonomy block

At minimum, one or more exercises in each module should include an explicit taxonomy block, and new exercise banks should default to this convention.
Place the block immediately after the exercise heading.

Template:

```md
**Taxonomy**

- `difficulty`: `foundational`
- `type`: `derivation`
- `tags`: `gradient`, `convexity`, `optimization`
```

Rules:
- `difficulty` must be one of `foundational`, `intermediate`, or `challenge`.
- `type` must be one of `proof`, `derivation`, `implementation`, `analysis`, or `open-ended`.
- `tags` should be short, lower-case, and topic-specific.
- Use `1` to `5` exercise-level tags.
- Keep document-level front matter tags broad and exercise-level tags local.

## Tagging conventions

### Document-level tags

Use front matter `tags` for the whole file:
- module topic names such as `optimization` or `transformers`;
- major methods such as `ridge-regression`, `policy-gradient`, or `diffusion`; and
- broad curriculum labels such as `exercises`.

### Exercise-level tags

Use the taxonomy block for local classification:
- concepts such as `generalization-gap`, `bellman-equation`, or `causal-graph`;
- artifacts such as `notebook`, `proof-sketch`, or `error-analysis`; and
- themes such as `robustness`, `equivariance`, or `representation`.

Prefer descriptive topical tags over administrative tags.
Avoid tags that only restate the module slug.

## Authoring rules

- Every exercise file should contain a mix of difficulties unless the file is explicitly specialized.
- Every module should expose at least two exercise types across its exercise bank.
- At least one exercise per module should be tagged explicitly as a consistency check.
- Challenge exercises should remain bounded enough to grade or self-assess with a rubric.
- Unity Theory exercises, when present, should still use the same taxonomy but be clearly marked as companion or interpretive in the prompt itself.

## Suggested distribution within a module

Recommended default balance for a standard exercise bank:
- `40%` foundational;
- `40%` intermediate; and
- `20%` challenge.

Recommended type coverage:
- at least one `proof` or `derivation` task;
- at least one `implementation` or computational task where the module supports coding;
- at least one `analysis` or interpretation task; and
- optional `open-ended` tasks for synthesis-heavy modules.

Modules may vary when the content warrants it:
- Module `12` can lean more heavily on implementation and analysis;
- Module `15` can lean more heavily on analysis and open-ended evaluation work; and
- Module `17` can lean more heavily on analysis and bounded synthesis, but should not replace canonical ML exercise coverage elsewhere.

## Retroactive consistency check

As part of Card `11.4`, at least one exercise artifact in each module has been tagged using this taxonomy.
Those tags are a live compatibility check for the current repo structure, not a promise about final coverage density.
