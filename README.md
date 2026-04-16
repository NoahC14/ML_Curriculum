# ML/AI Course

[![CI](https://github.com/NoahC14/ML_Curriculum/actions/workflows/ci.yml/badge.svg)](https://github.com/NoahC14/ML_Curriculum/actions/workflows/ci.yml)

Repository-based machine learning and AI curriculum with:
- a canonical ML spine;
- strong mathematical foundations;
- disciplined category theory integration; and
- clearly bounded Unity Theory companion material.

## Start here
If you are new to the repository, use this flow:
1. Read `syllabus/onboarding.md`.
2. Complete the prerequisite self-assessment.
3. Bootstrap the Python environment from `tooling/environment/README.md`.
4. Open `modules/00-math-toolkit/notebooks/linear-algebra-warmup.ipynb` as your first notebook.
5. Choose a study path from `syllabus/pacing-guide.md`.

This should get a mathematically prepared learner from clone to a working first notebook in under `30` minutes on a standard laptop.

## Learner onboarding
- `syllabus/onboarding.md` for the full setup walkthrough, repo navigation guide, FAQ, and path-specific first steps
- `syllabus/course-overview.md` for audience, scope, and rigor expectations
- `syllabus/pacing-guide.md` for single-semester, two-semester, and self-study routes

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

## First-study path
For most learners, the recommended first session is:
1. Clone the repo and create a virtual environment with `tooling/scripts/bootstrap.ps1` or `tooling/scripts/bootstrap.sh`.
2. Activate the environment and confirm the baseline with `python tooling/scripts/run_notebook_smoke_test.py`.
3. Start in `modules/00-math-toolkit/README.md`.
4. Open `modules/00-math-toolkit/notebooks/linear-algebra-warmup.ipynb`.
5. Use `syllabus/pacing-guide.md` to decide whether you are following the single-semester, two-semester, or self-study sequence.

## Naming conventions
Repository naming is locked in `shared/style-guides/naming-conventions.md`. Use that document before adding new modules, notes, notebooks, labs, or project artifacts.

## Next build steps
- populate `syllabus/` from Epic 0
- expand reusable templates in `shared/templates/`
- begin Module 00 and repo standards work from Epics 1 and 2
- continue splitting planning content from `kanban/ml_ai_course_kanban_v2.md` into stable kanban artifacts
