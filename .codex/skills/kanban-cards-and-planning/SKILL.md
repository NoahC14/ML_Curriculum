---
name: kanban-cards-and-planning
description: Use when creating roadmap documents, backlog structures, vibe-kanban work cards, execution plans, milestones, or task decompositions for building the ML/AI curriculum repository. Do not use for lesson writing or proof polishing.
---

# Purpose
Turn broad curriculum goals into concrete, buildable, reviewable work items.

# Baseline planning source
When `ml_ai_course_kanban_v2.md` exists, use it as the default backlog baseline. Extend or refine its epic/card structure instead of inventing a parallel planning system unless the user asks for a re-plan.

# Card schema
Each work card should contain, whenever appropriate:
- Title
- Type
- Purpose
- Inputs / context
- Deliverable
- Acceptance criteria
- Dependencies
- Suggested owner or agent type
- Notes / risks

# Card taxonomy
Prefer card types that match the repo plan:
- Planning
- Governance
- Engineering
- Content
- Lab
- Companion
- Curriculum
- Publishing
- Editing

# Decomposition rules
- Make cards atomic and testable.
- Separate design from implementation.
- Separate writing from review.
- Separate canonical curriculum work from speculative research work.
- Prefer cards that change one coherent artifact or one coherent decision.
- When possible, tie each card to a concrete repo path under `syllabus/`, `modules/`, `shared/`, `tooling/`, `projects/`, or `kanban/`.

# Good card titles
Use strong verb-first titles, such as:
- Draft Module 00 lesson map
- Write linear algebra primer section on eigendecomposition
- Design category theory sidebar template
- Review supervised learning assessment coverage

# Planning rules
For major initiatives, structure as:
- epics or features
- PBIs or workstreams
- tasks
- review or QA tasks

# Acceptance criteria rules
Acceptance criteria should be objective. Prefer:
- file created at path X
- section includes items A/B/C
- review checklist completed
- examples included
- dependencies linked

# Final check
Before finishing, ensure:
- the cards can be executed independently when possible;
- dependencies are explicit;
- there is a visible path from repo architecture to finished curriculum; and
- the plan still matches the kanban-aligned course sequence and milestones.
