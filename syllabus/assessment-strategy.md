# Assessment Strategy

## Purpose
This document defines the shared assessment philosophy, exercise tiers, grading approach, and repository-native deliverable formats for the curriculum.

## Assessment philosophy
Assessment should reward durable understanding rather than short-lived benchmark chasing. Each module should evaluate:
- `formal reasoning`: can the learner define assumptions, carry out derivations, and justify claims?
- `computational execution`: can the learner build, test, and analyze working ML artifacts?
- `interpretive judgment`: can the learner explain behavior, limitations, and tradeoffs clearly?

The default course balance is:
- `40%` proof and derivation work;
- `40%` coding and empirical work;
- `20%` intuition, communication, and synthesis work.

Modules may document bounded deviations in their `README.md` when the content demands it. Typical examples:
- Module 12 may weight environment-based coding work above notebook-style derivations.
- Module 15 may weight evaluation memos and case analysis above theorem-heavy proofs.
- Module 17 may weight synthesis and scope-discipline over implementation volume.

## Repository-native assessment design
Assessments should create reusable repo artifacts rather than disposable submissions. Acceptable outputs include:
- worked derivations in `derivations/`;
- executable notebooks in `notebooks/`;
- reusable code in `src/`;
- exercises in `exercises/` and instructor materials in `solutions/`;
- project briefs, reports, and experiments in `projects/`; and
- curated references or companion notes in `references/` and `unity/`.

Every module should default to the following internal structure:
- `README.md`
- `notes/`
- `derivations/`
- `notebooks/`
- `src/`
- `exercises/`
- `solutions/`
- `projects/`
- `references/`
- `unity/`

When a module needs an exception, the `README.md` should name the substitute artifact and justify it. For example, RL may include environment wrappers or simulators in place of some notebooks.

## Minimum assessment inventory per module
Unless an exception is documented, each module should include:
- at least `2` derivation or proof exercises;
- at least `2` coding exercises or labs;
- at least `1` intuition, reflection, or critique task;
- at least `1` graded synthesis task combining mathematics, implementation, and interpretation.

Recommended module-level cadence:
- `weekly`: low-stakes exercises or notebook checks;
- `per module`: one medium-stakes synthesis task;
- `per part`: one project, comparative study, or cumulative checkpoint.

## Exercise tiers

### Tier 1: Core competency
Purpose: verify essential fluency with definitions, notation, and standard implementations.

Typical formats:
- short proofs or derivation fragments;
- guided coding tasks;
- reading checks;
- low-stakes quizzes.

Expected outcome:
- the learner can reproduce standard results and complete a canonical workflow with support.

### Tier 2: Integrative practice
Purpose: connect mathematical formulation, implementation, and empirical interpretation.

Typical formats:
- notebook labs with analysis questions;
- multi-step derivations;
- model-comparison exercises;
- structured error-analysis writeups.

Expected outcome:
- the learner can connect equations, code, and evidence without relying on rote templates.

### Tier 3: Transfer and synthesis
Purpose: test whether the learner can adapt methods, critique assumptions, and communicate limitations.

Typical formats:
- open-ended projects;
- proof-plus-implementation assignments;
- comparative empirical studies;
- design memos or technical essays.

Expected outcome:
- the learner can reason independently, make defensible technical choices, and explain tradeoffs.

## Assessment formats
Use a mix of the following across the curriculum:
- `proof exercises`: derivations, proof sketches, counterexamples, and assumption checks;
- `coding labs`: notebook-based or script-based implementation tasks with reproducible outputs;
- `interpretation tasks`: short explanations of geometry, statistics, or systems behavior;
- `empirical studies`: ablations, baselines, metric comparisons, and error analyses;
- `projects`: scoped builds, replications, or research-style investigations;
- `reading responses`: brief critical reflections on papers, chapters, or evaluation claims.

## Grading philosophy
Grades should emphasize demonstrated reasoning and revision capacity rather than penalizing ambitious attempts. The default policy is:
- correctness matters, but partial credit is substantial when assumptions, setup, and logic are sound;
- implementation grades should reward reproducibility, debugging discipline, and analysis, not just final metrics;
- interpretation grades should reward precision, scope control, and evidence-based claims;
- late-stage synthesis work should reward explicit discussion of limitations and failure modes;
- solution releases should generally follow a delayed-release model so learners first attempt the work independently.

Recommended weighting by tier within a module:
- Tier 1: `30%`
- Tier 2: `40%`
- Tier 3: `30%`

## Rubrics

### Rubric A: Proof and derivation exercises
Use for theorem proofs, proof sketches, derivation notebooks, and formal argumentation.

Criteria:
- `mathematical setup` (`25%`): assumptions, notation, dimensions, and definitions are stated correctly.
- `logical progression` (`35%`): steps follow coherently, with no major gaps in argument.
- `technical correctness` (`25%`): algebra, calculus, probability, and theorem use are correct.
- `interpretation` (`15%`): the learner explains why the result matters for ML practice.

Performance anchors:
- `excellent`: correct setup, coherent derivation, and clear ML interpretation.
- `competent`: mostly correct reasoning with minor gaps or notation issues.
- `needs revision`: major logical omissions, unsupported steps, or incorrect assumptions.

### Rubric B: Coding labs and implementation tasks
Use for notebooks, scripts, reusable components, and experiment pipelines.

Criteria:
- `correctness and completeness` (`30%`): the task runs and satisfies the stated functional goals.
- `code quality` (`20%`): structure, readability, naming, and basic testing or validation are adequate.
- `experimental discipline` (`25%`): configuration, metrics, baselines, and reproducibility are handled responsibly.
- `analysis` (`25%`): results are interpreted accurately, including limitations and failure cases.

Performance anchors:
- `excellent`: working implementation, reproducible results, and technically sound analysis.
- `competent`: implementation mostly works with modest issues in rigor or interpretation.
- `needs revision`: broken workflow, missing controls, or unsupported conclusions.

### Rubric C: Synthesis, critique, and project work
Use for comparative studies, design memos, technical essays, and end-of-module projects.

Criteria:
- `problem framing` (`20%`): goals, assumptions, and scope are explicit.
- `integration` (`30%`): mathematics, implementation, and evidence are connected rather than treated separately.
- `judgment` (`30%`): tradeoffs, limitations, and alternatives are discussed credibly.
- `communication` (`20%`): the artifact is organized, precise, and appropriately documented for the repo.

Performance anchors:
- `excellent`: disciplined scope, strong evidence, and mature technical judgment.
- `competent`: useful integration with some thin spots in justification or communication.
- `needs revision`: weak framing, disconnected evidence, or imprecise claims.

## Solution and feedback policy
Default practice:
- provide fast feedback on Tier 1 work so misconceptions do not compound;
- provide rubric-based feedback on Tier 2 and Tier 3 work;
- release full solutions only after the relevant submission window unless the artifact is explicitly formative;
- retain selected exemplar solutions or reports in the repo only after removing learner-specific identifiers.

## Category theory and Unity Theory assessment policy
- Category theory should be assessed primarily on clarificatory use: composition, abstraction, invariance, and transfer.
- Learners should not be penalized for avoiding category-theory language when they can give an equally precise standard explanation, except in Module 16 or explicitly designated tasks.
- Unity Theory tasks should be marked as companion or exploratory.
- No Unity Theory assessment should replace canonical ML competency checks.

## Acceptance checklist for future module authors
Before a module is considered assessment-ready, verify that it has:
- a documented mapping from objectives to exercises;
- the default directory structure or a justified exception;
- the minimum assessment inventory or a justified exception;
- at least one rubric-bearing graded task;
- clear solution-release timing; and
- explicit separation between canonical assessment and companion material.
