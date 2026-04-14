---
title: "Causality Exercises"
module: "14-causality-reasoning"
lesson: "causality-exercises"
doc_type: "exercise"
topic: "causal-inference-and-reasoning"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability"
  - "05-probabilistic-modeling/graphical-models"
  - "14-causality-reasoning/causal-inference"
  - "14-causality-reasoning/reasoning-architectures"
updated: "2026-04-13"
owner: "curriculum-team"
tags:
  - "causality"
  - "interventions"
  - "counterfactuals"
  - "confounding"
  - "reasoning"
---

## Purpose

These exercises reinforce the distinction between observation and intervention, introduce simple SCM calculations, and ask for careful reflection on current reasoning architectures.

## Exercise 1. Observation versus intervention

Suppose $Z$ is a confounder for treatment $X$ and outcome $Y$, with graph

$$
Z \to X \to Y,
\qquad
Z \to Y.
$$

1. Write $p(y \mid x)$ as a sum over $z$.
2. Write $p(y \mid \operatorname{do}(x))$ as a sum over $z$.
3. Identify the only formal difference between the two formulas.
4. Explain in two or three sentences why that difference matters.

## Exercise 2. Medical-treatment adjustment

Let $S$ denote illness severity, $T$ treatment, and $Y$ recovery.
Assume $S \to T$, $S \to Y$, and $T \to Y$.

1. State whether $S$ is a confounder, mediator, or collider.
2. Write the back-door adjustment formula for $p(y \mid \operatorname{do}(t))$.
3. Explain why comparing treated and untreated patients directly may give a biased estimate of treatment effect.
4. Give one practical difficulty in applying the adjustment formula with real data.

## Exercise 3. A/B testing and randomization

Consider an online experiment where $A$ is assignment to a new interface and $Y$ is conversion.

1. Explain why randomized assignment changes the causal interpretation of the estimate.
2. State which quantity a well-run randomized A/B test is designed to estimate: $p(y \mid a)$ or $p(y \mid \operatorname{do}(a))$.
3. Give one reason an apparently randomized experiment can still produce misleading conclusions in practice.
4. Name one advantage of randomized experiments over observational adjustment.

## Exercise 4. Mediators and total effect

Suppose $X \to M \to Y$ and there are no other arrows.

1. State whether $M$ is a confounder, mediator, or collider.
2. If the goal is the total effect of $X$ on $Y$, should you condition on $M$? Explain briefly.
3. If the goal is instead to isolate the direct effect of $X$ not through $M$, why might conditioning on $M$ become relevant?
4. Describe one ML example where a mediator naturally appears.

## Exercise 5. Collider bias

Suppose $X \to C \leftarrow Y$.

1. State whether $X$ and $Y$ are associated through this path before conditioning on $C$.
2. State what can happen after conditioning on $C$.
3. Give one intuitive example of collider bias from selection or admissions data.
4. Explain why "add every available variable as a control" is not a valid causal rule.

## Exercise 6. Structural equations

Consider the SCM

$$
Z := U_Z,
\qquad
X := 2Z + U_X,
\qquad
Y := 3X + Z + U_Y,
$$

where $U_Z, U_X, U_Y$ are mutually independent with zero mean.

1. Draw the causal graph.
2. State whether $Z$ confounds the effect of $X$ on $Y$.
3. Compute $\mathbb{E}[Y \mid \operatorname{do}(X=x)]$.
4. Compute $\mathbb{E}[Y \mid X=x]$ in words: which extra dependence makes it differ from the interventional quantity?

## Exercise 7. Counterfactual reasoning

Answer each prompt in two or three sentences.

1. What is the conceptual difference between an interventional query and a counterfactual query?
2. Why do counterfactual queries require stronger assumptions than ordinary prediction?
3. What are the three SCM steps often summarized as abduction, action, and prediction?
4. Give one fairness or policy question that is naturally counterfactual.

## Exercise 8. Causal discovery limitations

1. Explain why observational data alone usually identifies an equivalence class of graphs rather than one unique causal graph.
2. State one additional assumption that can help orient edges.
3. State one reason that assumption may fail in practice.
4. Explain why causal discovery should be treated as assumption-sensitive scientific inference rather than generic tabular ML.

## Exercise 9. Reasoning architectures taxonomy

Classify each system below into the most appropriate bucket:

- plain end-to-end predictor;
- predictor plus intermediate text trace;
- predictor plus search over traces;
- predictor plus external tools;
- symbolic-neural hybrid.

Systems:

1. A transformer that outputs one final answer directly.
2. A language model prompted to "think step by step" before answering.
3. A system that samples several solution traces and chooses the majority answer.
4. A model that writes Python, runs it, and uses the output in its answer.
5. A perception model coupled to a symbolic planner.

## Exercise 10. Chain-of-thought caution

Write a short paragraph addressing all four points:

1. Why can chain-of-thought improve performance?
2. Why is a chain-of-thought trace not automatically a faithful explanation?
3. Why might a verifier or external tool improve reliability?
4. What claim about reasoning would you avoid making on the basis of benchmark gains alone?

## Exercise 11. Causality meets reasoning

Answer each prompt in three or four sentences.

1. Why is causal reasoning important for decision-making systems rather than only for scientific explanation?
2. How do counterfactual questions resemble branching reasoning problems?
3. Why might a tool-using or planner-style architecture be better suited than a one-pass predictor for some causal-analysis tasks?
4. Give one example where a high-accuracy predictive model would still be unsuitable for intervention planning.

## Exercise 12. Modeling reflection

Choose one domain: medicine, education, recommendation systems, policy, or scientific discovery.
Write a short reflection answering:

1. What is one predictive question in the domain?
2. What is one causal question in the domain?
3. What variables are likely confounders?
4. Would an experiment be feasible, unethical, or expensive?
5. What kind of reasoning architecture or tool support would help analyze the problem well?
