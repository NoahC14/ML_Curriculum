---
title: "Ethics, Safety, and Evaluation Exercises"
module: "15-ethics-safety-evaluation"
lesson: "ethics-safety-exercises"
doc_type: "exercise"
topic: "ethics-safety-evaluation"
status: "draft"
prerequisites:
  - "02-statistical-learning/model-evaluation"
  - "10-transformers-llms/pretraining-and-scaling"
updated: "2026-04-15"
owner: "curriculum-team"
tags:
  - "ethics"
  - "safety"
  - "evaluation"
  - "distribution-shift"
  - "fairness"
---

# Ethics, Safety, and Evaluation Exercises

These exercises emphasize evaluation design, failure analysis, and scope discipline.
They are intentionally weighted toward `analysis` and `open-ended` work rather than theorem-heavy derivations.

## Exercise 1. Metric choice under class imbalance

**Taxonomy**

- `difficulty`: `foundational`
- `type`: `analysis`
- `tags`: `class-imbalance`, `evaluation-metrics`, `precision-recall`

A toxicity classifier is deployed on a platform where positive examples are rare.

1. Explain why accuracy alone can be misleading in this setting.
2. Compare precision, recall, and false-positive rate for this deployment context.
3. State one reason an ROC-style summary can still hide an operational problem.
4. Recommend one primary metric and justify the choice.

## Exercise 2. Distribution shift triage memo

1. A model performs well on a benchmark validation set but degrades after deployment in a new region.
2. List three plausible forms of distribution shift.
3. For each one, name one diagnostic you would run.
4. Explain how you would distinguish data-quality failure from model-capacity failure.

## Exercise 3. Fairness tradeoff case analysis

Two demographic groups have different base rates for the target label.

1. Explain why equalizing one fairness metric can worsen another.
2. Choose two fairness criteria and describe the tradeoff concretely.
3. State what extra domain information is needed before recommending a deployment decision.

## Exercise 4. Red-team exercise design

Design a bounded red-team protocol for a text-generation model used in a customer-support setting.

Your response should specify:
1. the failure class you are probing;
2. the prompt families you would test;
3. the evidence you would collect;
4. one stop condition for escalation; and
5. one limitation of your protocol.
