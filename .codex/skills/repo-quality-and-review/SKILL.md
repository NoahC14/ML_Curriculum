---
name: repo-quality-and-review
description: Use when reviewing artifacts for consistency, checking repository conventions, validating file placement, tightening prose, reducing duplication, or performing pre-merge quality control for the ML/AI curriculum repo. Do not use for first-pass ideation.
---

# Purpose
Keep the repository coherent, navigable, and professionally maintainable.

# Review checklist
Check for:
- alignment with `ml_ai_course_kanban_v2.md` and `.codex/AGENTS.md`
- correct file placement under `syllabus/`, `modules/`, `shared/`, `tooling/`, `projects/`, `kanban/`, or `.codex/`
- consistent module numbering and naming
- complete module skeletons when module directories exist
- clear separation between canonical content and companion or Unity material
- duplication across docs
- broken internal references
- missing prerequisites or objectives in modules
- missing acceptance criteria in planning docs
- stale starter-language or wrong `.codex/skills/` paths

# Writing cleanup checklist
- tighten section titles
- remove repetition
- standardize terminology
- preserve mathematical precision
- separate core content from sidebars

# Repository hygiene
Prefer:
- reusable templates
- predictable filenames
- short README files in dense directories
- modular docs over giant omnibus files
- kanban cards that point to concrete artifacts

# Done criteria
A reviewed artifact should:
- read cleanly;
- match the repo's design intent;
- be easy for another agent or human to extend;
- preserve the canonical-first course posture; and
- be ready for the next backlog step without hidden cleanup debt.
