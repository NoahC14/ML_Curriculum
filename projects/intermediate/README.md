# Intermediate Projects

This track contains Tier 3 synthesis projects for the deep-learning portion of the curriculum. Each project asks students to connect model theory, implementation choices, diagnostics, and written analysis in a compact empirical study.

## Track goals

Intermediate projects should require students to:

- implement or adapt a modern deep-learning pipeline rather than only run a canned notebook;
- justify design decisions about architecture, optimization, regularization, and evaluation;
- use diagnostics from `shared/src/training_diagnostics.py` where relevant;
- write a short technical analysis that distinguishes empirical evidence from conjecture; and
- work within realistic compute limits on a single GPU or a reduced-scale CPU setup.

## Expected submission format

Each project should produce:

- a reproducible notebook or script-based experiment pipeline;
- a short report or notebook writeup interpreting the results;
- figures or tables supporting design decisions; and
- brief notes on limitations, failure modes, and compute constraints.

## Shared grading emphasis

Across the track, grading should reflect the course assessment philosophy:

- `problem framing and theory connection`;
- `implementation correctness and code quality`;
- `experimental discipline and diagnostics`; and
- `written analysis and technical judgment`.

## Project menu

| Project | Main modules | Estimated time | Standard compute | Reduced-scale path |
| --- | --- | --- | --- | --- |
| `cnn-architecture-comparison/` | Modules 07-08 | `12-16 hours` | single consumer GPU preferred | small subset, shallower CNNs, fewer epochs |
| `sequence-diagnostics-study/` | Modules 07 and 09 | `10-14 hours` | CPU or single GPU | shorter sequences, smaller hidden states |
| `transformer-text-generation/` | Modules 07 and 10 | `14-20 hours` | single GPU preferred | character-level model, shorter context, fewer layers |
| `vae-vs-gan-comparison/` | Modules 07 and 11 | `14-18 hours` | single GPU preferred | grayscale data, low-resolution images, limited latent size |

## Recommended use

- assign one project as a required module capstone; or
- let students choose one project plus one brief proposal memo explaining why it matches their strengths and interests.

Each project directory includes a project brief and a rubric.
