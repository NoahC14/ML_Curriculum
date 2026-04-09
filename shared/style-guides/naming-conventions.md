# Naming Conventions

This document locks the repository naming scheme early so content can grow without later path churn or link breakage.

## Core rules
- Use lowercase kebab-case for directory names and markdown file names.
- Preserve two-digit numeric prefixes for ordered modules.
- Keep names descriptive but short enough to remain readable in links and imports.
- Prefer stable semantic names over temporary milestone labels.
- Avoid spaces, underscores, and ad hoc abbreviations in repository paths.

## Top-level directories
- Keep the top-level layout limited to `syllabus/`, `modules/`, `shared/`, `tooling/`, `projects/`, and `kanban/` unless the course plan is intentionally revised.
- Add new top-level areas only through an explicit architecture decision.

## Modules
- Module directories must follow the pattern `NN-module-name/`, where `NN` is a zero-padded sequence number.
- Preserve the canonical numbering from `modules/00-math-toolkit/` through `modules/17-unity-theory-perspectives/`.
- Do not rename module directories after content lands unless link migrations are planned across the repository.

## Module internal structure
- Every module keeps the same subdirectory names: `notes/`, `derivations/`, `notebooks/`, `src/`, `exercises/`, `solutions/`, `projects/`, `references/`, and `unity/`.
- Keep `README.md` at the module root and in each internal subdirectory as the landing page for that scope.
- Put canonical ML material in the standard module folders first. Put Unity Theory companion material in `unity/`.

## Markdown documents
- Use `README.md` for directory landing pages.
- Use kebab-case descriptive names for standalone documents, such as `gradient-descent-notes.md` or `bias-variance-tradeoff.md`.
- Reserve suffixes like `-template`, `-guide`, `-overview`, and `-strategy` for repo-wide documents when they clarify document purpose.

## Notebooks
- Use kebab-case names that begin with a concise topic or lab identifier.
- Prefer patterns such as `linear-regression-from-scratch.ipynb`, `cnn-feature-maps.ipynb`, or `rl-policy-evaluation.ipynb`.
- Keep exploratory scratch notebooks out of canonical module paths unless they are promoted and cleaned.

## Source code
- Match file and directory names to the dominant language conventions where needed, but keep parent curriculum paths in kebab-case.
- Group reusable teaching code by concept, not by one-off lesson date or author initials.

## Exercises and solutions
- Keep exercise and solution file names aligned one-to-one when both exist.
- Prefer stable names such as `exercise-01-linear-algebra-review.md` and `exercise-01-linear-algebra-review-solution.md`.

## References and bibliography
- Use short, source-oriented names for local notes and curated summaries.
- Avoid inventing citation keys inside file names when a readable topic name is sufficient.

## Change control
- If a new artifact does not fit these conventions, update this document before introducing a competing naming pattern.
- When in doubt, choose the name that will still make sense after several later modules depend on it.
