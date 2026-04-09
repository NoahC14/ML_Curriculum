---
name: curriculum-architecture
description: Use when creating or revising the high-level course structure, module sequence, lesson decomposition, prerequisite graph, learning objectives, assessment architecture, or repository information architecture for the ML/AI curriculum. Do not use for local line edits, isolated proof polishing, or single-card kanban formatting.
---

# Purpose
Design a coherent curriculum architecture for a mathematically rigorous ML/AI course that preserves the canonical learning sequence while making disciplined space for category theory and Unity Theory.

# Baseline source
When `ml_ai_course_kanban_v2.md` exists, treat it as the default baseline for:
- module ordering;
- part grouping;
- repo information architecture; and
- milestone sequencing.

# Outcomes
This skill should help produce:
- top-level syllabus structures;
- module outlines;
- lesson sequencing;
- prerequisite graphs;
- assessment strategies; and
- mappings from conceptual goals to repo directories.

# Core rules
1. Preserve the canonical ML spine.
2. Distinguish foundational, intermediate, advanced, and companion layers.
3. Treat category theory as a structural language, not replacement mathematics.
4. Treat Unity Theory as an interpretive and research-generative layer, not replacement doctrine.
5. Prefer modular decomposition over monolithic files.

# Canonical course template
Use the kanban-aligned structure unless the architecture is being revised on purpose:
- Part I. Foundations
  - Module 00: Mathematical Toolkit for ML
  - Module 01: Optimization
- Part II. Classical Machine Learning
  - Module 02: Statistical Learning Foundations
  - Module 03: Linear Models
  - Module 04: Kernel Methods and Margin-Based Learning
  - Module 05: Probabilistic Modeling
- Part III. Neural Networks and Deep Learning
  - Module 06: Neural Networks from First Principles
  - Module 07: Deep Learning Systems
  - Module 08: Convolutional Neural Networks and Vision
  - Module 09: Sequence Models
  - Module 10: Transformers and LLM Foundations
  - Module 11: Generative Models
- Part IV. Advanced AI
  - Module 12: Reinforcement Learning
  - Module 13: Graph Learning
  - Module 14: Causality and Reasoning
  - Module 15: Ethics, Safety, and Evaluation
- Part V. Structural and Philosophical Extensions
  - Module 16: Category Theory for Machine Learning
  - Module 17: Unity Theory Perspectives on AI and Learning

# Repository architecture targets
When shaping repo structure, target:
- `syllabus/`
- `modules/`
- `shared/`
- `tooling/`
- `projects/`
- `kanban/`

Each module directory should be able to support:
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

# What to produce for each module
Every module should specify:
- title;
- purpose;
- prerequisites;
- learning objectives;
- lecture or lesson map;
- key mathematics;
- computational labs;
- exercises and assessments;
- category theory insertion points;
- Unity Theory insertion points; and
- expected repo outputs.

# Sequencing heuristics
- Start from standard mathematical maturity, not from abstraction.
- Place the category theory primer inside Module 00 using concrete examples from sets, functions, vector spaces, and ML pipelines.
- Revisit categorical ideas formally in Module 16 once learners have enough concrete ML experience.
- Keep Unity Theory late, boxed, or clearly companion-oriented unless the task is explicitly about scope policy.
- Follow the build order in the kanban plan: scaffolding first, then foundations, then core ML, then advanced topics, then companion layers, then publishing/polish.

# Quality check
Before finishing, verify:
- the sequence is teachable;
- dependencies are realistic;
- no module is overloaded;
- abstraction supports practice rather than obscuring it; and
- the architecture maps cleanly onto the planned repo directories.
