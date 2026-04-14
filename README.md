# ML/AI Course

[![CI](https://github.com/NoahC14/ML_Curriculum/actions/workflows/ci.yml/badge.svg)](https://github.com/NoahC14/ML_Curriculum/actions/workflows/ci.yml)

Repository-based machine learning and AI curriculum with:
- a canonical ML spine;
- strong mathematical foundations;
- disciplined category theory integration; and
- clearly bounded Unity Theory companion material.

## Current status
The repository scaffold is in place so syllabus, module, tooling, project, and kanban work can land in stable locations from the start.

## Source of truth
- `kanban/ml_ai_course_kanban_v2.md` for course scope, module order, and backlog
- `AGENTS.md` for durable repo guidance
- `.codex/` for repo-local agent skills and instructions

## Repository layout
- `syllabus/` course-level overview, objectives, pacing, readings, and assessment policy
- `modules/` module-by-module curriculum artifacts, with a common internal structure for each module
- `shared/` templates, style guides, figures, datasets, and bibliography
- `tooling/` environment setup, scripts, linting, and CI
- `projects/` beginner through research project tracks
- `kanban/` backlog, epics, and work-card views derived from the course plan

## Naming conventions
Repository naming is locked in `shared/style-guides/naming-conventions.md`. Use that document before adding new modules, notes, notebooks, labs, or project artifacts.

## Next build steps
- populate `syllabus/` from Epic 0
- expand reusable templates in `shared/templates/`
- begin Module 00 and repo standards work from Epics 1 and 2
- continue splitting planning content from `kanban/ml_ai_course_kanban_v2.md` into stable kanban artifacts
