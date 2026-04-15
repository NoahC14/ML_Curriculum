# Project: Novel Algorithm Design Using Compositional Principles

## Problem statement
Propose, implement, or partially prototype a learning algorithm, training loop, or model component
motivated by compositional principles. The project should explain what the proposed method preserves,
how components compose, and why the design may improve modularity, transfer, controllability, or
sample efficiency.

Potential directions include:

- compositional routing or modular experts;
- structure-preserving latent transitions;
- functor-inspired transfer between related tasks;
- pipeline designs that make invariance constraints explicit;
- compositional objectives for multi-stage training.

## Research emphasis
This prompt is intended for students who want to move from structural analysis toward method design.
Ambition is welcome, but the project must stay honest about whether the method is a proof of concept,
an engineering prototype, or a fully evaluated contribution.

## Suggested guiding questions
- What exact failure mode or gap in current methods motivates the new design?
- What structural principle is being enforced or exploited?
- Which parts of the proposal are formal, heuristic, or speculative?
- What is the strongest evaluation that is feasible within the course timeline?

## Suggested readings
- Module 01 on optimization if the proposal changes the learning procedure.
- Modules 06-13 depending on the application area.
- Module 16 on category theory for ML.
- Two to four primary papers directly relevant to the chosen design space.

## Recommended deliverables
- a `6-12` page paper;
- pseudocode or a reference implementation;
- baseline comparisons or ablations where feasible; and
- a section distinguishing proven properties, design intuitions, and unanswered questions.

## Evaluation criteria
- the design goal is concrete and technically motivated;
- the proposal specifies how composition enters the method;
- experiments or analytic arguments are strong enough for the scale of the claim;
- novelty claims are calibrated against prior work; and
- the paper names the limits of the prototype honestly.

## Scope notes
- A negative result is acceptable if the design and evaluation are rigorous.
- If full implementation is unrealistic, the project should narrow to a formalization sketch plus
  one small proof-of-concept experiment.
