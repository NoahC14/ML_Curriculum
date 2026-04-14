# AGENTS.md

This is the repo-root agent entrypoint. Keep it aligned with `.codex/AGENTS.md`, and keep repo-local skills under `.codex/skills/`.

## Purpose
This repository is building a repo-based machine learning and AI curriculum with:
- a canonical ML spine;
- strong mathematical foundations;
- disciplined category theory integration; and
- clearly bounded Unity Theory companion material.

Until the repo is split into the full directory tree, treat `ml_ai_course_kanban_v2.md` as the source of truth for:
- course scope and audience;
- module sequence;
- repository information architecture;
- milestone order; and
- backlog structure.

## Project north star
Build a rigorous, extensible curriculum that:
1. teaches standard machine learning and AI in a form recognizable to a conventional technical audience;
2. makes the mathematics explicit enough for deep internalization;
3. uses category theory to clarify composition, abstraction, invariance, and transfer;
4. uses Unity Theory only as a clearly marked interpretive or research layer; and
5. produces reusable repo artifacts such as syllabus docs, notes, derivations, notebooks, labs, exercises, projects, references, and planning assets.

## Non-negotiable design principles
- Preserve the canonical-first ML sequence.
- Keep mathematics, implementation, and empirical interpretation connected.
- Treat category theory as structural enrichment, not replacement mathematics.
- Treat Unity Theory as companion material, not default textbook language.
- Prefer modular docs, reusable templates, and stable naming.
- Match the kanban architecture unless the plan itself is being revised.

## Operating workflow
For substantial tasks:
1. identify the exact deliverable and target path;
2. read `ml_ai_course_kanban_v2.md` plus nearby repo files;
3. choose the matching skill under `.codex/skills/`;
4. implement the smallest coherent change that advances the backlog;
5. run the relevant checks; and
6. self-review for scope fidelity, duplication, and file placement.

## Planned repository layout
The target structure from the kanban plan is:
- `README.md`
- `syllabus/`
  - `course-overview.md`
  - `learning-objectives.md`
  - `reading-list.md`
  - `pacing-guide.md`
  - `assessment-strategy.md`
- `modules/`
  - `00-math-toolkit/`
  - `01-optimization/`
  - `02-statistical-learning/`
  - `03-linear-models/`
  - `04-kernel-methods/`
  - `05-probabilistic-modeling/`
  - `06-neural-networks/`
  - `07-deep-learning-systems/`
  - `08-cnn-vision/`
  - `09-sequence-models/`
  - `10-transformers-llms/`
  - `11-generative-models/`
  - `12-reinforcement-learning/`
  - `13-graph-learning/`
  - `14-causality-reasoning/`
  - `15-ethics-safety-evaluation/`
  - `16-category-theory-for-ml/`
  - `17-unity-theory-perspectives/`
- `shared/`
  - `figures/`
  - `templates/`
  - `datasets/`
  - `bibliography/`
  - `style-guides/`
- `tooling/`
  - `environment/`
  - `scripts/`
  - `linting/`
  - `ci/`
- `projects/`
  - `beginner/`
  - `intermediate/`
  - `advanced/`
  - `research/`
- `kanban/`
  - `backlog.md`
  - `epics.md`
  - `work-cards.md`
- `.codex/AGENTS.md`
- `.codex/skills/`

If the current repo is still sparse, create new artifacts against this layout unless the plan is being deliberately revised.

## Module conventions
Every module should include, when applicable:
- `README.md`
- `notes/`
- `derivations/`
- `notebooks/`
- `src/`
- `exercises/`
- `solutions/`
- `projects/`
- `references/`
- `unity/`

Each module should specify:
- purpose;
- prerequisites;
- learning objectives;
- lesson breakdown;
- key definitions and results;
- computational labs;
- exercises and assessments;
- category theory insertion points; and
- Unity Theory insertion points.

## Writing standards
- Use clear mathematical prose.
- Define symbols before using them.
- Prefer short subsections and explicit transitions.
- Pair formal development with at least one concrete ML example.
- Distinguish definitions, propositions, theorems, remarks, examples, and exercises.
- Avoid hype, overclaiming, and vague claims of obviousness.

## Mathematical standards
For mathematically serious artifacts, include when appropriate:
- assumptions and notation;
- definitions;
- derivations or proof sketches;
- worked examples;
- computational interpretation;
- ML relevance; and
- limitations or scope notes.

When writing derivations:
- show intermediate steps when the audience could plausibly need them;
- make dimensions, domains, and codomains explicit when they matter;
- state whether a result is exact, heuristic, asymptotic, or approximate; and
- keep standard derivations intact even when adding structural or interpretive framing.

## Category theory placement
Use category theory in two stages:
1. Primer mode in `modules/00-math-toolkit/`
   - objects, morphisms, identity, composition, diagrams, products, coproducts, functors, and natural transformations in concrete form.
2. Consolidation mode in `modules/16-category-theory-for-ml/`
   - more formal constructions, architecture comparisons, pipeline reasoning, and research ideation.

Outside those anchors, use category theory only where it clarifies composition, abstraction, invariance, or transfer.

## Unity Theory integration policy
- Unity Theory is companion material.
- Keep canonical ML exposition first.
- Label Unity sections as interpretive, exploratory, or speculative when appropriate.
- If a formal correspondence is claimed, state the mapping precisely.
- Prefer `unity/` notes, boxed sidebars, or Module 17 for extended Unity Theory treatment.

## Kanban conventions
When generating or revising work cards:
- preserve the epic structure from `ml_ai_course_kanban_v2.md` unless intentionally refactoring the backlog;
- keep cards atomic and testable;
- separate architecture, writing, implementation, and review tasks;
- separate canonical curriculum work from speculative research work; and
- include explicit deliverables, acceptance criteria, and dependencies.

## Skills in this repo
Check these before doing bespoke work:
- `.codex/skills/curriculum-architecture/`
  Use for module sequencing, syllabus structure, learning objectives, and repo architecture.
- `.codex/skills/math-rigor-and-proof/`
  Use for mathematically serious notes, derivations, and proof-driven exposition.
- `.codex/skills/kanban-cards-and-planning/`
  Use for backlog design, milestone planning, and work-card decomposition.
- `.codex/skills/labs-and-tooling/`
  Use for notebooks, reusable code, environment setup, scripts, CI, and lab infrastructure.
- `.codex/skills/unity-category-integration/`
  Use for disciplined category theory and Unity Theory framing.
- `.codex/skills/repo-quality-and-review/`
  Use for consistency passes, cleanup, placement checks, and review.

## Done criteria
A task is not complete until:
- the requested artifact exists in the expected location;
- the artifact matches the kanban-aligned repo conventions;
- internal references and naming are consistent;
- obvious placeholders are resolved or explicitly marked; and
- relevant code, notebook, or formatting checks were run when applicable.

## What to avoid
- Do not collapse the course into one giant omnibus file when modular structure is better.
- Do not mix speculative material into canonical lessons without clear labeling.
- Do not create duplicate templates unless the existing one is insufficient.
- Do not invent citations or present unsupported claims as established results.
- Do not silently drift from the planned repo architecture.
