---
title: "Causal Inference"
module: "14-causality-reasoning"
lesson: "causal-inference"
doc_type: "notes"
topic: "structural-causal-models-interventions-counterfactuals"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability"
  - "02-statistical-learning/statistical-learning-foundations"
  - "05-probabilistic-modeling/graphical-models"
updated: "2026-04-13"
owner: "curriculum-team"
tags:
  - "causality"
  - "scm"
  - "interventions"
  - "counterfactuals"
  - "confounding"
  - "do-calculus"
---

## Purpose

These notes introduce the minimum causal language needed by a machine learning student who already knows probability and graphical models.
The goal is not to turn this module into a full causal-inference course.
It is to make the distinction between statistical association and intervention mathematically precise, and to show where causal reasoning changes what can be learned from data.

## Learning objectives

After working through this note, you should be able to:

- state formally why $p(y \mid x)$ is not generally the same quantity as $p(y \mid \operatorname{do}(x))$;
- define a structural causal model and identify its observable and interventional implications;
- explain confounding, mediators, and colliders in graph-based terms;
- compute simple interventional quantities from a known causal graph;
- describe what counterfactual queries ask for and why they require stronger assumptions than ordinary prediction;
- explain the back-door adjustment in words and symbols; and
- state why causal discovery from observational data alone is generally limited.

## 1. Why prediction is not yet causation

In standard supervised learning, we often estimate a conditional distribution such as

$$
p(y \mid x).
$$

This quantity answers a predictive question:
if we observe that $X=x$, what should we believe about $Y$?

Causal inference asks a different question:
what would happen to $Y$ if we intervened and set $X$ to $x$?
That quantity is written

$$
p(y \mid \operatorname{do}(x)).
$$

The symbols look similar, but they are not interchangeable.
Conditioning updates beliefs after observation.
Intervening changes the data-generating mechanism itself.

### Example: treatment assignment

Let:

- $T \in \{0,1\}$ denote whether a patient receives a treatment;
- $Y \in \{0,1\}$ denote recovery; and
- $S$ denote baseline illness severity.

In many observational datasets, sicker patients are more likely to receive treatment.
Then

$$
p(Y=1 \mid T=1)
$$

may be lower than

$$
p(Y=1 \mid T=0)
$$

even when the treatment is genuinely helpful, because the treated and untreated groups differ systematically in severity.

The causal estimand is instead

$$
p(Y=1 \mid \operatorname{do}(T=1)),
\qquad
p(Y=1 \mid \operatorname{do}(T=0)).
$$

Their difference is an average treatment effect:

$$
\operatorname{ATE}
=
\mathbb{E}[Y \mid \operatorname{do}(T=1)]
-
\mathbb{E}[Y \mid \operatorname{do}(T=0)].
$$

## 2. Structural causal models

A structural causal model (SCM) makes the causal story explicit.
One standard formulation contains:

- exogenous noise variables $U_1,\dots,U_n$;
- endogenous variables $X_1,\dots,X_n$; and
- structural assignments

$$
X_i := f_i(\operatorname{Pa}_i, U_i),
$$

where $\operatorname{Pa}_i$ denotes the parent variables of $X_i$ in the causal graph.

The corresponding directed graph is acyclic in the standard basic setting.
It looks like a Bayesian network, but the semantics are stronger:
edges are not only factorization devices, but claims about direct causal dependence.

### Observable distribution induced by an SCM

If the graph is acyclic, the observational joint distribution factorizes as

$$
p(x_1,\dots,x_n)
=
\prod_{i=1}^n p(x_i \mid \operatorname{pa}_i).
$$

At the observational level, this is formally similar to a directed graphical model.
The difference is that an SCM also tells us what happens under intervention, because the structural assignments can be modified.

## 3. Interventions as model surgery

An intervention $\operatorname{do}(X=x)$ means:
replace the structural equation for $X$ by the constant assignment $X:=x$ and remove the influence of its original parents.

This is often called graph surgery.

If the original SCM contains

$$
X := f_X(\operatorname{Pa}_X, U_X),
$$

then after the intervention we use

$$
X := x.
$$

All downstream variables are then recomputed using the altered system.

### A/B testing example

Suppose:

- $A \in \{0,1\}$ is whether a user sees a new interface;
- $C$ is user intent or baseline engagement;
- $Y$ is conversion.

In observational logs,

$$
p(y \mid a)
$$

can be distorted if high-intent users are more likely to self-select into one variant, or if rollout policy targets a specific user segment.

In a randomized experiment, assignment is generated independently of the pre-treatment user state.
Then the randomized estimate targets

$$
p(y \mid \operatorname{do}(a)).
$$

Randomization matters because it breaks the arrow from latent user intent to assignment.

## 4. Conditioning versus intervention

The cleanest place to see the difference is a simple confounding graph:

$$
Z \to X \to Y,
\qquad
Z \to Y.
$$

Here $Z$ is a confounder because it influences both treatment $X$ and outcome $Y$.

The observational conditional is

$$
p(y \mid x)
=
\sum_z p(y \mid x,z)\,p(z \mid x).
$$

The interventional distribution is

$$
p(y \mid \operatorname{do}(x))
=
\sum_z p(y \mid x,z)\,p(z).
$$

The only difference is the weighting distribution over $Z$, but that difference is exactly the issue.

- In $p(y \mid x)$, the confounder distribution is updated after observing $X=x$.
- In $p(y \mid \operatorname{do}(x))$, the intervention cuts incoming arrows into $X$, so $Z$ keeps its original marginal distribution.

This is the formal version of "correlation is not causation."

## 5. Back-door adjustment

The back-door criterion gives a tractable way to identify interventional quantities.

Informally, a set of variables $Z$ is a valid adjustment set for estimating the effect of $X$ on $Y$ if:

- no element of $Z$ is a descendant of $X$; and
- conditioning on $Z$ blocks every back-door path from $X$ to $Y$.

When such a set exists, the intervention is identified by

$$
p(y \mid \operatorname{do}(x))
=
\sum_z p(y \mid x,z)\,p(z).
$$

For continuous $Z$, replace the sum by an integral.

### Medical example

Let:

- $S$ be illness severity;
- $T$ be treatment; and
- $Y$ be recovery.

If $S$ causes both $T$ and $Y$, and there are no other unblocked back-door paths, then

$$
p(y \mid \operatorname{do}(t))
=
\sum_s p(y \mid t,s)\,p(s).
$$

Operationally:

1. estimate recovery as a function of treatment and severity;
2. reweight by the population distribution of severity, not by the severity distribution inside the treated group.

## 6. Confounders, mediators, and colliders

These roles matter because conditioning can help or harm depending on graph structure.

### Confounder

In

$$
Z \to X \to Y,
\qquad
Z \to Y,
$$

$Z$ is a confounder.
Adjusting for $Z$ is typically needed.

### Mediator

In

$$
X \to M \to Y,
$$

$M$ is a mediator.
If the goal is the total effect of $X$ on $Y$, adjusting for $M$ usually blocks part of the effect we want to measure.

### Collider

In

$$
X \to C \leftarrow Y,
$$

$C$ is a collider.
Without conditioning, the path through $C$ is blocked.
Conditioning on $C$ or its descendants can induce a spurious association between $X$ and $Y$.

That is why "control for everything available" is not a valid causal strategy.

## 7. Do-calculus intuition

Pearl's do-calculus gives symbolic rules for transforming expressions involving interventions into expressions involving only observational distributions when the graph justifies it.

For this module, the exact rules are less important than the operational idea:

- use the graph to decide which paths remain active after intervention;
- exchange observations and interventions only when graphical conditions justify it; and
- reduce a causal query to a statistical estimand only if identifiability conditions hold.

Back-door adjustment is the most important first example of this broader logic.
The reason causal inference is harder than ordinary prediction is that not every interventional query is identifiable from observational data.

## 8. Counterfactuals

Observational and interventional queries ask about populations or regimes.
Counterfactuals ask about alternate worlds for a specific unit.

Example:

> This patient received treatment and recovered. Would this same patient have recovered without treatment?

That is a query of the form

$$
Y_{T=0}(u)
$$

for the same latent background state $u$.

In SCM language, counterfactual reasoning usually follows three steps:

1. abduction: infer a posterior over latent variables from observed facts;
2. action: modify the structural equation by the intervention;
3. prediction: propagate the altered system forward.

Counterfactuals are powerful because they support explanations, fairness questions, and policy analysis.
They are also stronger than ordinary prediction because they depend on assumptions about the unit-level structural mechanism, not only on a joint distribution.

## 9. Causal discovery and its limits

Causal discovery asks whether the graph itself can be inferred from data.
This is much harder than estimating parameters in a known graph.

### Why the problem is underdetermined

Distinct causal graphs can imply the same observational distribution.
Even with infinite data, observational equivalence classes remain.

For example, the three graphs

$$
X \to Y \to Z,
\qquad
X \leftarrow Y \to Z,
\qquad
X \leftarrow Y \leftarrow Z
$$

can share the same skeleton and many of the same conditional-independence patterns.

### What extra assumptions buy

Causal discovery algorithms typically need additional assumptions such as:

- causal sufficiency or explicit treatment of hidden confounders;
- faithfulness or stability of conditional independences;
- functional-form restrictions such as additive-noise asymmetries; or
- interventional data from perturbations or randomized experiments.

These assumptions are often reasonable in some domains and fragile in others.
Students should treat causal discovery as assumption-sensitive scientific modeling, not as a push-button feature of generic tabular ML.

## 10. Relation to machine learning

Causal language changes several ML questions:

- domain shift: invariant causal mechanisms may transfer better than raw correlations;
- policy learning: recommendation and decision systems need effects of actions, not only predictions under passive observation;
- interpretability: explanations often implicitly ask counterfactual questions;
- fairness: some fairness criteria depend on causal pathways, not just marginal parity.

At the same time, most standard predictive ML pipelines are not causal models by default.
A high-accuracy predictor can still answer the wrong question if deployment involves intervention.

## 11. Structural insertion points

### Category theory insertion point

The natural structural lesson is modest:
an SCM composes local mechanisms into a global generative system, and an intervention is a controlled replacement of one morphism in that composition.
That framing can clarify modularity, but the substance of the module remains probabilistic and graph-theoretic.

### Unity Theory insertion point

No Unity Theory vocabulary is required here.
If a companion note is later added, the proper role would be interpretive only:
causal analysis studies which transformations preserve or alter system identity under intervention.

## 12. Summary

- An observational conditional $p(y \mid x)$ is not generally an interventional distribution $p(y \mid \operatorname{do}(x))$.
- SCMs strengthen graphical models by assigning causal semantics to edges and by supporting intervention and counterfactual queries.
- Confounders often require adjustment, mediators should not be adjusted for when estimating total effects, and colliders can create bias when conditioned on.
- Back-door adjustment is the first key identifiability tool.
- Counterfactuals ask unit-level alternate-world questions and require stronger assumptions.
- Causal discovery from observational data alone is limited and assumption-sensitive.
