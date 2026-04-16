---
title: "Solutions Policy"
module: "shared"
doc_type: "reference"
topic: "solution-release-policy"
status: "draft"
updated: "2026-04-15"
owner: "curriculum-team"
tags:
  - "assessment"
  - "solutions-policy"
  - "honor-code"
  - "release-timing"
---

# Solutions Policy

## Purpose

This document defines what solution material is public, what is instructor-gated, when releases occur, and how honor-code expectations apply across the curriculum.

The policy is designed to balance two competing goals:
- self-learners need enough feedback to use the repo independently; and
- active course offerings need protection against trivial answer-copying.

## Default two-tier policy

### Public materials

The public repo should normally include:
- exercise prompts;
- deliverable specifications;
- bounded hints;
- rubric summaries or grading criteria when useful for self-assessment;
- selected short answer checks for clearly formative foundational exercises; and
- selected solution sketches released after the relevant delay window.

### Instructor-gated materials

Instructor-only or access-controlled materials should normally include:
- full worked solutions for current summative exercises;
- grading notes and common-error annotations;
- hidden test cases for implementation tasks;
- exemplar submissions or reports used for live grading calibration; and
- any answer key whose immediate publication would materially weaken assessment integrity.

Instructor-gated materials may live outside the public repo or in a controlled distribution channel.
Public placeholders in `solutions/` directories should indicate the existence of gated material when relevant without exposing it.

## Release timing

### Foundational exercises

Default release:
- hints may be public immediately;
- short answer checks or concise solution sketches may be released after a reasonable attempt window; and
- in a live course run, full worked solutions should wait until the submission deadline passes.

### Intermediate exercises

Default release:
- prompts and bounded hints remain public;
- full worked solutions should normally be delayed until the module closes for the current offering; and
- self-study cohorts may receive public solution sketches once the exercise is no longer part of an active graded window.

### Challenge exercises

Default release:
- prompts remain public;
- public post-release materials should usually be partial solution sketches, rubric-oriented guidance, or discussion of strong approaches rather than full canonical answers; and
- complete worked solutions should usually remain instructor-gated unless the exercise has been retired from live assessment use.

## Public versus gated outputs by artifact type

### Proof and derivation tasks

Public by default:
- prompts;
- assumptions and notation reminders;
- brief hints; and
- after release, outline-level solution sketches for formative work.

Gated by default:
- full line-by-line proofs for active summative tasks; and
- grading notes on common logical gaps.

### Implementation tasks

Public by default:
- prompts;
- API expectations;
- sample input-output contracts;
- reproducibility requirements; and
- lightweight debugging hints.

Gated by default:
- complete reference implementations for active assignments;
- hidden tests; and
- benchmark or grading harness details that would trivialize the exercise.

### Analysis and open-ended tasks

Public by default:
- prompts;
- rubric criteria;
- examples of acceptable evidence standards; and
- after release, discussion of strong answer characteristics.

Gated by default:
- full exemplar memos or reports during active course use when those exemplars would become answer templates.

## Solution artifact conventions

- `exercises/` should always contain the public prompt.
- `solutions/` may contain public solution material only when release timing permits.
- If a full solution is gated, the public `solutions/README.md` should say whether the exercise has `public`, `delayed-public`, or `instructor-gated` solution status.
- Public solutions should distinguish between `solution sketch` and `full worked solution`.
- Retired exercises may be moved from `instructor-gated` to `delayed-public` or `public` when there is no active integrity risk.

## Honor code expectations

All learners are expected to:
- attempt the exercise independently before consulting solution material;
- cite collaborators, external references, and AI assistance when course policy requires it;
- avoid submitting copied public solutions as original work;
- avoid sharing instructor-gated materials outside the authorized course context; and
- preserve the distinction between formative checking and summative assessment.

### AI-use expectation

When the curriculum is used in a graded setting, learners should follow the local course policy for AI assistance.
Unless a local module policy says otherwise, AI tools may be used for debugging, notation clarification, and study support, but not to generate final submissions without disclosure.

## Policy for self-learners

Self-learners should have a workable path through the repo without private access.
To support that path:
- every module should expose public prompts and at least minimal hints;
- a subset of foundational exercises should eventually receive public solution sketches;
- challenge tasks may rely more on rubrics and discussion prompts than on full public answers; and
- notebooks and projects should remain interpretable even when full instructor solutions are withheld.

## Policy for live offerings

For active instructor-led use:
- current graded exercises should default to `delayed-public` or `instructor-gated`;
- reused exercises should be rotated or revised across offerings when practical;
- full worked solutions should be released only after grading closes or the exercise is retired; and
- summative implementation tasks should use hidden tests or withheld reference outputs when needed.

## Recommended status labels

When documenting an exercise bank or solution directory, use one of:
- `public`
- `delayed-public`
- `instructor-gated`

These labels describe access to solution material, not access to the prompt.

## Review checklist

Before publishing solution material, verify:
- whether the exercise is formative or summative;
- whether the current offering still depends on answer secrecy;
- whether the public release would still leave self-learners with enough feedback;
- whether hidden tests, grading notes, or exemplar reports need to remain gated; and
- whether the release status is stated clearly in the relevant `solutions/` documentation.
