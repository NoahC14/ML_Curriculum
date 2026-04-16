# Project: Categorical Analysis of a Novel Architecture

## Problem statement
Choose a modern architecture or architecture family not treated exhaustively in the core modules,
then analyze it through standard ML tools and a disciplined categorical lens. The project should
ask whether categorical language clarifies the architecture's composition, invariances, bottlenecks,
or transfer behavior beyond what a purely component-level description provides.

Appropriate targets include:

- graph transformers;
- state space sequence models;
- diffusion transformers;
- retrieval-augmented generation pipelines;
- multimodal encoder-decoder systems.

## Why this project belongs in the advanced track
This prompt is open-ended enough to support original framing, but it still expects canonical ML
analysis first: architecture, objective, training dynamics, empirical behavior, and comparison to
alternatives.

## Suggested guiding questions
- What are the architecture's core computational stages and data transformations?
- Which parts of the system can be modeled as compositions of maps, products, coproduct-like merges,
  or natural comparisons between representations?
- Does the structural view expose an invariance, failure mode, or design tradeoff that standard
  prose descriptions hide?
- How does the chosen architecture compare with at least one simpler baseline or predecessor?

## Suggested readings
- Module 07 on deep learning systems.
- Module 10 on transformers and LLM foundations.
- Module 13 on graph learning, if the architecture is relational.
- Module 16 project and companion materials on category theory for ML.
- One foundational paper for the chosen architecture and one strong follow-up paper.

## Recommended deliverables
- a `6-10` page report;
- at least one architecture diagram rewritten in the student's own notation;
- one comparison table connecting standard ML terminology to structural terminology;
- a small empirical study, replication, or ablation when practical; and
- a short section on what the categorical view does not explain.

## Evaluation criteria
- the architecture description is technically accurate and current relative to the selected papers;
- categorical language is used for structural clarification, not as decoration;
- empirical or comparative evidence is sufficient to support the claims;
- limitations and non-equivalences are stated plainly; and
- the submission could support a seminar talk or workshop-style extended abstract.

## Scope notes
- Avoid trying to formalize every subcomponent. A partial but precise structural map is stronger
  than a sweeping but vague categorical rewrite.
- If the chosen architecture is very recent, prefer one narrow claim that can be defended well.
