# Pacing Guide

## Purpose
This guide turns the 18-module curriculum into three usable study modes:
- a single-semester path that preserves a conventional core ML course;
- a two-semester path that covers the full curriculum at teachable depth; and
- a self-study path with weekly time estimates and explicit checkpoints.

The canonical ML spine remains primary in every path. Category theory and Unity Theory companion layers stay optional in compressed schedules and become fully scheduled only in the longer paths.

## Planning assumptions
- A standard semester is `14` instructional weeks plus `1` assessment or buffer week.
- A two-semester sequence is `30` total weeks across two `15`-week terms.
- Self-study assumes `8-12` hours per week and no instructor intervention.
- Time estimates below are based on current repo artifact volume: note-heavy and derivation-heavy modules require more time than scaffold-light synthesis modules.

## Module workload profile
The current repo footprint supports the following rough workload classes.

| Module | Workload | Current signals | Notes |
| --- | --- | --- | --- |
| `00-math-toolkit` | Heavy | `6` notes, `2` derivations, `8` notebooks, `5` exercise files | Treat as a front-loaded bridge or spread across multiple weeks. |
| `01-optimization` | Medium | `1` note, `1` derivation, `5` notebooks | Computationally dense despite lighter prose volume. |
| `02-statistical-learning` | Medium | `2` notes, `1` derivation, `3` notebooks | Core prerequisite for nearly everything after Module `03`. |
| `03-linear-models` | Medium | `2` notes, `3` derivations, `2` notebooks | Good anchor for early assessment. |
| `04-kernel-methods` | Medium | `1` note, `2` derivations, `1` notebook | Best kept after linear models and optimization. |
| `05-probabilistic-modeling` | Medium-heavy | `3` notes, `1` derivation, `2` notebooks | Needed before later generative framing. |
| `06-neural-networks` | Medium | `2` notes, `1` derivation, `3` notebooks | Must precede the deep learning stack. |
| `07-deep-learning-systems` | Medium-heavy | `1` note, `4` notebooks | Practice-heavy and should not be rushed. |
| `08-cnn-vision` | Medium | `2` notes, `3` notebooks | Can run as a focused architecture application. |
| `09-sequence-models` | Medium | `1` note, `2` notebooks | Best taught directly before transformers. |
| `10-transformers-llms` | Medium-heavy | `2` notes, `1` derivation, `2` notebooks | Depends on sequence and neural-network fluency. |
| `11-generative-models` | Heavy | `4` notes, `2` derivations, `3` notebooks | Better in longer formats than compressed ones. |
| `12-reinforcement-learning` | Medium-heavy | `2` notes, `2` derivations, `2` notebooks | Needs protected time for Bellman and policy-gradient work. |
| `13-graph-learning` | Medium | `2` notes, `2` notebooks | Assumes neural nets and benefits from transformer exposure. |
| `14-causality-reasoning` | Medium-light | `2` notes, `1` exercise file | Conceptually dense even with lighter artifact count. |
| `15-ethics-safety-evaluation` | Light | `1` exercise file | Integrate with model-evaluation checkpoints throughout. |
| `16-category-theory-for-ml` | Heavy companion | `7` notes, `1` exercise file | Keep late and optional in short formats. |
| `17-unity-theory-perspectives` | Medium companion | `4` notes, `1` exercise file | Always companion-only. |

## Path A: Single-Semester Core ML

### Design goal
Deliver a credible one-semester ML course for mathematically prepared learners. The required path covers canonical foundations through deep learning and transformers. Companion layers remain optional and clearly marked.

### Required modules
- `00-math-toolkit` as an accelerated bridge with category-theory primer subsections optional
- `01-optimization`
- `02-statistical-learning`
- `03-linear-models`
- `04-kernel-methods`
- `05-probabilistic-modeling`
- `06-neural-networks`
- `07-deep-learning-systems`
- `10-transformers-llms`
- `15-ethics-safety-evaluation` integrated across the term and revisited at the end

### Optional extension modules
- `08-cnn-vision` if the course wants one concrete modality case study
- `09-sequence-models` as a bridge week before transformers if students need more recurrence background
- `11-generative-models` as an end-of-term survey rather than full treatment
- `12-14` only as capstone topics, not core
- `16-17` only as companion reading or discussion sections

### Week-by-week schedule

| Week | Modules | Focus | Estimated learner time |
| --- | --- | --- | --- |
| 1 | `00` | linear algebra, calculus, probability refresh; notation contract | `10-12` hours |
| 2 | `00` + optional `00` category primer | information theory, conditioning, ML mappings | `10-12` hours |
| 3 | `01` | gradients, convexity, GD, SGD | `8-10` hours |
| 4 | `02` | risk, generalization, regularization, evaluation | `8-10` hours |
| 5 | `03` | linear regression, ridge, logistic regression | `8-10` hours |
| 6 | `04` | kernels, margins, dual views | `7-9` hours |
| 7 | `05` | Bayesian inference, EM, graphical models | `8-10` hours |
| 8 | checkpoint | midterm synthesis: compare linear, kernel, and probabilistic views | `6-8` hours |
| 9 | `06` | MLPs, backpropagation, initialization | `8-10` hours |
| 10 | `07` | normalization, regularization, diagnostics, residual intuition | `8-10` hours |
| 11 | `08` optional or project buffer | CNN case study or supervised project sprint | `6-10` hours |
| 12 | `09` optional bridge or `10` start | recurrence limits, attention motivation, or direct transformer launch | `8-10` hours |
| 13 | `10` | self-attention, positional encoding, pretraining objectives | `8-10` hours |
| 14 | `15` + `10` | evaluation, safety, distribution shift, reporting | `7-9` hours |
| 15 | final integration | final project, oral defense, or technical report | `8-12` hours |

### Feasibility notes
- This path is complete because every required module has its prerequisites satisfied in order.
- `08`, `09`, and `11-17` are not orphaned; they are explicitly marked as optional extensions after the required spine.
- If only one optional deep-learning application week is available, prefer `08` for vision-heavy cohorts and `09` for language-heavy cohorts.
- Companion layers should be assigned as optional reading responses, not assessed as required doctrine.

## Path B: Two-Semester Full Sequence

### Design goal
Cover the full 18-module architecture with enough room for derivations, labs, projects, and the companion layers in their intended late-course position.

### Semester 1: Foundations and classical ML

| Week | Modules | Focus | Estimated learner time |
| --- | --- | --- | --- |
| 1 | `00` | linear algebra for ML | `8-10` hours |
| 2 | `00` | multivariable calculus and matrix chain rule | `8-10` hours |
| 3 | `00` | probability, statistics, information theory | `8-10` hours |
| 4 | `00` | category primer, ML mappings, first checkpoint | `8-10` hours |
| 5 | `01` | convexity, gradients, constrained optimization | `8-10` hours |
| 6 | `01` + `02` | stochastic optimization into statistical learning | `8-10` hours |
| 7 | `02` | risk, bias-variance, evaluation design | `8-10` hours |
| 8 | `03` | linear models and GLM intuition | `8-10` hours |
| 9 | `03` | regularization and likelihood-based views | `8-10` hours |
| 10 | `04` | kernels, SVMs, RKHS intuition | `7-9` hours |
| 11 | `05` | Bayesian inference and graphical models | `8-10` hours |
| 12 | `05` | EM, latent variables, discriminative versus generative comparison | `8-10` hours |
| 13 | `15` integrated | ethics, fairness, robustness, evaluation checkpoints | `6-8` hours |
| 14 | review | comparative project or midyear exam | `6-8` hours |
| 15 | buffer | catch-up, project polish, assessment feedback | `4-6` hours |

### Semester 2: Deep learning, advanced AI, and companion layers

| Week | Modules | Focus | Estimated learner time |
| --- | --- | --- | --- |
| 1 | `06` | perceptrons, MLPs, backpropagation | `8-10` hours |
| 2 | `07` | training systems, diagnostics, scaling | `8-10` hours |
| 3 | `08` | CNNs and vision pipelines | `8-10` hours |
| 4 | `09` | recurrent models and sequence pathologies | `8-10` hours |
| 5 | `10` | self-attention and transformer block mechanics | `8-10` hours |
| 6 | `10` | pretraining, adaptation, evaluation | `8-10` hours |
| 7 | `11` | VAEs, GANs, diffusion overview | `9-11` hours |
| 8 | `12` | Bellman equations, dynamic programming, policy gradients | `8-10` hours |
| 9 | `13` | message passing, spectral methods, graph transformers | `8-10` hours |
| 10 | `14` | interventions, counterfactuals, reasoning systems | `7-9` hours |
| 11 | `15` | safety, robustness, and evaluation synthesis | `6-8` hours |
| 12 | `16` | formal category-theory consolidation for ML | `8-10` hours |
| 13 | `16` | diagrammatic case studies and research framing | `8-10` hours |
| 14 | `17` | Unity Theory companion synthesis, clearly optional | `5-7` hours |
| 15 | final integration | advanced project, research memo, or oral exam | `8-12` hours |

### Feasibility notes
- This path includes all modules with no omissions.
- `15` appears in both semesters because evaluation and safety should not be isolated to the final week only.
- `16` comes after graph learning, transformers, and RL so the categorical module can consolidate concrete experience instead of replacing it.
- `17` stays last and companion-only by policy.

## Path C: Self-Study Sequence

### Design goal
Provide a realistic solo-learning schedule with built-in review points, explicit hours, and enough slack for difficult derivations or notebook debugging.

### Recommended cadence
- Standard weeks: `8-10` hours
- Heavy weeks: `10-12` hours
- Review weeks: `4-6` hours
- Default weekly split:
  - `3-4` hours reading notes and derivations
  - `3-4` hours notebook or coding work
  - `1-2` hours exercises and self-assessment
  - `1` hour recap and written summary

### Weekly schedule

| Week | Modules | Focus | Estimated time |
| --- | --- | --- | --- |
| 1 | `00` | linear algebra refresh | `8-10` hours |
| 2 | `00` | multivariable calculus | `8-10` hours |
| 3 | `00` | probability and statistics | `8-10` hours |
| 4 | `00` | information theory and numerical conditioning | `8-10` hours |
| 5 | `00` | category primer and ML mappings | `8-10` hours |
| 6 | checkpoint | cumulative problem set and notebook rerun | `4-6` hours |
| 7 | `01` | convexity, gradients, Hessians | `8-10` hours |
| 8 | `01` | SGD, momentum, second-order methods | `8-10` hours |
| 9 | `02` | risk, generalization, model evaluation | `8-10` hours |
| 10 | `03` | linear regression and normal equations | `8-10` hours |
| 11 | `03` | logistic regression and regularization | `8-10` hours |
| 12 | checkpoint | mini-project on classical supervised learning | `6-8` hours |
| 13 | `04` | kernel methods and SVMs | `8-10` hours |
| 14 | `05` | Bayesian inference and naive Bayes | `8-10` hours |
| 15 | `05` | graphical models and EM | `8-10` hours |
| 16 | review | discriminative versus generative comparison | `4-6` hours |
| 17 | `06` | forward pass and multilayer networks | `8-10` hours |
| 18 | `06` | backpropagation and training loop | `8-10` hours |
| 19 | `07` | normalization, regularization, diagnostics | `8-10` hours |
| 20 | `08` | convolutions and image pipelines | `8-10` hours |
| 21 | checkpoint | small vision experiment and writeup | `6-8` hours |
| 22 | `09` | RNNs, LSTMs, teacher forcing | `8-10` hours |
| 23 | `10` | self-attention and transformer mechanics | `8-10` hours |
| 24 | `10` | pretraining, scaling, adaptation | `8-10` hours |
| 25 | `11` | VAEs and GANs | `10-12` hours |
| 26 | `11` | diffusion and comparative generative analysis | `10-12` hours |
| 27 | review | language or generative mini-project | `6-8` hours |
| 28 | `12` | MDPs, Bellman equations, dynamic programming | `8-10` hours |
| 29 | `12` | policy gradients and actor-critic methods | `8-10` hours |
| 30 | `13` | graph structure and message passing | `8-10` hours |
| 31 | `14` | causality, interventions, counterfactuals | `8-10` hours |
| 32 | `15` | fairness, robustness, evaluation design | `6-8` hours |
| 33 | checkpoint | advanced-topic synthesis and reading memo | `4-6` hours |
| 34 | `16` | categories, functors, natural transformations | `8-10` hours |
| 35 | `16` | limits, monoidal structure, pipeline diagrams | `8-10` hours |
| 36 | `17` | Unity Theory companion readings and scope checks | `5-7` hours |
| 37 | capstone | choose a project track and draft final report | `8-10` hours |
| 38 | capstone | final implementation, revision, and retrospective | `8-10` hours |

### Self-assessment checkpoints
At the end of Weeks `6`, `12`, `21`, `27`, and `33`, pause and verify:
- you can solve at least one derivation without looking at notes;
- you can rerun or adapt at least one notebook from the preceding block;
- you can explain the previous block's main assumptions and failure modes in writing;
- you can state which companion material was optional and whether you used it.

If a checkpoint fails, repeat the previous week before advancing. Self-study is least forgiving when optimization, statistical learning, or backpropagation are only partially internalized.

## Companion-layer policy by path

| Module | Single semester | Two semesters | Self-study |
| --- | --- | --- | --- |
| `00` category primer | Optional subsections | Required | Required |
| `16` category theory for ML | Optional reading only | Required | Required late-stage module |
| `17` Unity Theory perspectives | Optional companion only | Optional but scheduled | Optional companion only |

## Recommended assessment anchors
- Single semester: one midterm synthesis, one final project or report.
- Two semesters: one project per semester plus a final advanced synthesis.
- Self-study: one checkpoint writeup every `5-6` weeks and one capstone artifact at the end.

## Revision triggers
Revisit this guide when:
- module `README.md` files gain more explicit prerequisite chains;
- module artifact counts change materially, especially in `14`, `15`, `16`, or `17`;
- new labs or projects increase the expected weekly time above the current ranges.
