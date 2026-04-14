---
name: labs-and-tooling
description: Use when creating or revising notebook labs, reusable module code, environment setup, scripts, datasets plumbing, linting, smoke tests, or CI for the ML/AI curriculum repo. Do not use for high-level syllabus design or pure mathematical exposition.
---

# Purpose
Build reproducible, teachable technical infrastructure for the curriculum repo.

# Typical outputs
This skill should help produce:
- module `notebooks/`
- module `src/`
- `tooling/environment/`
- `tooling/scripts/`
- `tooling/linting/`
- `tooling/ci/`
- `shared/datasets/`
- reusable notebook or lab templates

# Core rules
- Separate reusable code from notebook exploration.
- Keep lab dependencies explicit and reproducible.
- Prefer small, inspectable implementations over clever abstractions.
- Make notebooks runnable top-to-bottom on a fresh setup.
- Add smoke tests or validation for changed notebooks and code when practical.
- Document dataset assumptions, download steps, and licensing constraints.
- Reuse shared utilities instead of copy-pasting module code.

# Lab design checklist
- clear learning goal
- minimal setup friction
- visible intermediate outputs or visualizations
- one concrete ML concept per lab
- brief interpretation of results
- extension prompts or follow-up exercises

# Placement rules
- Module-specific labs live under that module's `notebooks/` and `src/`.
- Cross-module utilities belong under `shared/` or `tooling/`.
- Environment, linting, and CI changes should support the kanban-aligned repo architecture rather than one-off local paths.

# Verification rules
Before finishing, verify:
- a fresh learner can run the artifact with documented steps;
- notebook and code paths are consistent;
- the implementation supports the lesson objective;
- checks are light but real; and
- no speculative framing replaces the canonical technical task.
