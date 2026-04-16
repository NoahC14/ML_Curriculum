# Learner Onboarding Guide

## Purpose
This guide helps a new learner go from repository clone to a working first notebook with minimal friction. It is written for the primary course audience defined in `syllabus/course-overview.md`: mathematically prepared learners who want a rigorous, self-directed path through machine learning and AI.

## Who this guide is for
This onboarding flow assumes you are:
- comfortable installing Python packages and using a terminal;
- ready to work through mathematics, code, and derivations in parallel; and
- choosing between a single-semester, two-semester, or self-study path.

If that does not describe you yet, use the self-assessment below before committing to the full sequence.

## Prerequisite self-assessment
Rate each item as `ready`, `needs review`, or `not yet`.

### Mathematics
- I can multiply matrices, interpret a linear map, and reason about rank, eigenvalues, and orthogonality.
- I can compute partial derivatives and gradients and follow a multivariable chain rule derivation.
- I can work with expectations, variances, conditional probability, and common distributions.
- I can read notation-heavy technical prose without getting blocked by every symbol.

### Programming
- I can create and activate a Python virtual environment.
- I can run Python scripts from the command line.
- I can open and run cells in Jupyter notebooks.
- I can read basic NumPy and scikit-learn code.

### Workflow
- I know how to clone a Git repository and navigate directories in a shell.
- I can tolerate some ambiguity in a repo that is still being expanded.
- I am willing to pause and review prerequisite math before moving forward.

## How to interpret your result
- Mostly `ready`: start with Module `00` and follow the pacing guide that matches your time budget.
- Mixed `ready` and `needs review`: still start with Module `00`, but plan to spend extra time in the math toolkit before moving to optimization.
- Multiple `not yet` items in math: delay the full curriculum and first strengthen linear algebra, calculus, probability, and Python fundamentals.
- Multiple `not yet` items in workflow: learn basic Git, terminal navigation, and notebook usage first. This repo assumes those skills.

## Fast-start checklist
Use this checklist if your goal is to run the first notebook in under `30` minutes.

1. Clone the repository and open a terminal in the repo root.
2. Confirm you have Python `3.11` or `3.12`.
3. Run the appropriate bootstrap script:
   Windows PowerShell:
   ```powershell
   ./tooling/scripts/bootstrap.ps1
   ```
   macOS / Linux / WSL:
   ```bash
   ./tooling/scripts/bootstrap.sh
   ```
4. Activate the new virtual environment.
   Windows PowerShell:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   macOS / Linux / WSL:
   ```bash
   source .venv/bin/activate
   ```
5. Run the baseline notebook smoke test:
   ```bash
   python tooling/scripts/run_notebook_smoke_test.py
   ```
6. Launch Jupyter and open:
   - `modules/00-math-toolkit/notebooks/linear-algebra-warmup.ipynb`
7. Select the `Python (ml-curriculum)` kernel if prompted.

If this flow works, your environment is ready for the early modules.

## Environment setup walkthrough
The canonical environment instructions live in `tooling/environment/README.md`. The short version is below.

### Baseline platform assumptions
- Python `3.11` or `3.12`
- CPU-only environment by default
- Windows, macOS, Linux, or WSL

### Recommended path
Use the repo bootstrap scripts because they:
- create a local `.venv`;
- install the project in editable mode;
- register a Jupyter kernel named `Python (ml-curriculum)`.

### Optional deep-learning extra
If you want PyTorch installed immediately, use:

Windows PowerShell:
```powershell
./tooling/scripts/bootstrap.ps1 -IncludeDeepLearning
```

macOS / Linux / WSL:
```bash
./tooling/scripts/bootstrap.sh .venv --deep-learning
```

### Manual installation
Use manual setup only if you already manage Python environments comfortably.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
python -m ipykernel install --user --name ml-curriculum --display-name "Python (ml-curriculum)"
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` before the last three commands.

## Repository navigation guide
Use the repository in this order instead of trying to read everything at once.

### `README.md`
Start here for the global orientation and the shortest route into the course.

### `syllabus/`
Use these files to understand the course before diving into content.
- `course-overview.md`: audience, rigor, scope, and design intent
- `learning-objectives.md`: what the curriculum expects you to know and produce
- `pacing-guide.md`: which study path matches your schedule
- `reading-list.md`: core references
- `assessment-strategy.md`: how mastery is meant to be demonstrated

### `modules/`
This is the main body of the course. Each module follows a common scaffold:
- `README.md` for module purpose, prerequisites, and study plan
- `notes/` for exposition
- `derivations/` for mathematical development
- `notebooks/` for computational work
- `exercises/` for practice
- `solutions/` for solution material when available
- `src/` for reusable code
- `projects/` for module-specific project work
- `references/` for papers and texts
- `unity/` for clearly marked companion material

### `tooling/`
Use this when you need setup, validation, or reproducibility help.
- `environment/README.md`: environment setup
- `scripts/`: bootstrap and validation scripts
- `linting/` and `ci/`: repository quality and smoke-test infrastructure

### `projects/`
Use these after you have completed at least a few core modules and want applied practice at beginner, intermediate, advanced, or research depth.

### `kanban/`
Use this only if you want the planning source of truth, backlog history, or repository build context. Learners do not need it for normal study.

## Suggested first steps by study path
Choose your study path from `syllabus/pacing-guide.md`, then use the matching first-session plan below.

### Single-semester core ML path
- Read `syllabus/course-overview.md` and `syllabus/pacing-guide.md`.
- Complete Module `00` selectively, with emphasis on linear algebra, calculus, probability, and notation refresh.
- Run `modules/00-math-toolkit/notebooks/linear-algebra-warmup.ipynb`.
- Move next to `modules/01-optimization/README.md`.

### Two-semester full-sequence path
- Read the course overview and the full pacing guide carefully.
- Work through Module `00` in order rather than compressing it.
- Use the category primer in Module `00` as required background, not optional enrichment.
- Keep a weekly written summary from the start so later structural modules are easier to integrate.

### Self-study path
- Treat `syllabus/pacing-guide.md` as binding unless you have strong prior preparation.
- Start with Module `00` and plan review checkpoints exactly where the pacing guide places them.
- Run the notebook smoke test before opening your first notebook so environment problems do not compound with learning problems.
- Write a one-page note after each checkpoint on assumptions, definitions, and failure modes you still find unclear.

### If you want the quickest first win
Do this in your first sitting:
- bootstrap the environment;
- run the smoke test;
- open `linear-algebra-warmup.ipynb`;
- read `modules/00-math-toolkit/README.md`;
- stop after you can rerun the notebook and explain what each section is checking.

## First-week recommendation
For most learners, the best first week is:
- one pass through the prerequisite self-assessment;
- environment setup and smoke test;
- `modules/00-math-toolkit/README.md`;
- one notebook from Module `00`;
- one short written recap of what felt easy, shaky, or missing.

That first-week recap should determine whether you stay on schedule or slow down before Module `01`.

## FAQ
### Do I need category theory before starting?
No. The course assumes no prior category theory. The primer appears inside Module `00`, and the formal consolidation waits until Module `16`.

### Do I need Unity Theory to complete the course?
No. Unity Theory is companion-only material. The canonical ML curriculum remains complete and legible without it.

### Which notebook should I run first?
Start with `modules/00-math-toolkit/notebooks/linear-algebra-warmup.ipynb`. It is part of the baseline smoke-test path and sits in the intended first module.

### Should I install the deep-learning extra immediately?
Only if you already know you want PyTorch from the start. The default CPU environment is enough for the early modules and keeps setup simpler.

### What if the smoke test fails?
Read the traceback carefully, confirm the active environment is `.venv`, and verify that the Jupyter kernel was registered as `Python (ml-curriculum)`. If the failure is notebook-specific, try running that notebook alone to isolate the issue.

### Do I need to read every file in a module before moving on?
No. Start with the module `README.md`, then use notes, derivations, and notebooks as the primary path. Exercises and projects are reinforcement, not the entrypoint.

### I know applied ML already. Can I skip Module `00`?
You can compress it, but do not skip it blindly. Use the self-assessment and at least run the warmup notebook before assuming your foundations are strong enough.

### Is this repo beginner-friendly?
Not for absolute beginners. It is designed for advanced undergraduates, graduate students, researchers, and technical professionals who already have basic math and Python fluency.

## Validation status
This guide is aligned to the current repository structure, the environment instructions in `tooling/environment/README.md`, the audience profile in `syllabus/course-overview.md`, and the study-path assumptions in `syllabus/pacing-guide.md`.

The onboarding flow should still be tested on a fresh machine by a learner unfamiliar with the repo to fully satisfy the acceptance note for Card `12.3`.
