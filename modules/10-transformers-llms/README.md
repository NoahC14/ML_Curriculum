# Module 10: Transformers and LLM Foundations

## Status
Scaffolded placeholder derived from `ml_ai_course_kanban_v2.md`.

## Purpose
Build the mathematical and systems foundations for transformers and large language model workflows.

## Planned focus
- self-attention derivation
- positional encoding
- encoder-decoder structure
- pretraining objectives
- scaling laws intuition
- fine-tuning, instruction tuning, and RLHF overview

## Shared toolkit usage

Transformer labs in this module should reuse `shared/src/training_diagnostics.py` for:

- learning-rate schedule inspection during warmup and decay;
- gradient-norm monitoring across attention and MLP blocks; and
- activation-distribution and confusion-matrix checks for downstream classifier heads.
