---
title: "Policy Gradient Theorem Derivation"
module: "12-reinforcement-learning"
lesson: "policy-gradient-theorem"
doc_type: "derivation"
topic: "trajectory-gradients-and-baselines"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability"
  - "01-optimization/convexity-and-optimization"
  - "12-reinforcement-learning/notes/policy-gradients"
updated: "2026-04-13"
owner: "curriculum-team"
tags:
  - "reinforcement-learning"
  - "policy-gradient"
  - "reinforce"
  - "actor-critic"
---

## Goal

Derive the policy gradient theorem step by step.
The main technical ideas are:

- write the RL objective as an expectation over trajectories;
- differentiate the trajectory probability with the log-derivative trick;
- use causality to remove irrelevant past rewards; and
- rewrite the result in terms of the action-value function.

## 1. Setup

Let $\pi_\theta(a\mid s)$ be a differentiable stochastic policy with parameters $\theta\in\mathbb{R}^p$.
Fix an initial-state distribution $\rho_0$ and transition kernel $P(s'\mid s,a)$.

A trajectory is

$$
\tau = (s_0,a_0,r_1,s_1,a_1,r_2,\dots).
$$

Its probability density under the policy is

$$
p_\theta(\tau)
=
\rho_0(s_0)\prod_{t=0}^{\infty}\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t).
$$

The performance objective is

$$
J(\theta)
=
\mathbb{E}_{\tau\sim p_\theta}[G_0]
=
\int p_\theta(\tau)\,G_0(\tau)\,d\tau,
$$

where

$$
G_t = \sum_{k=0}^{\infty}\gamma^k R_{t+k+1}.
$$

## 2. Differentiate the objective

Differentiate under the integral sign:

$$
\nabla_\theta J(\theta)
=
\int \nabla_\theta p_\theta(\tau)\,G_0(\tau)\,d\tau.
$$

Use the identity

$$
\nabla_\theta p_\theta(\tau)
=
p_\theta(\tau)\nabla_\theta \log p_\theta(\tau),
$$

to obtain

$$
\nabla_\theta J(\theta)
=
\int p_\theta(\tau)\nabla_\theta \log p_\theta(\tau)\,G_0(\tau)\,d\tau
=
\mathbb{E}_{\tau\sim p_\theta}
\bigl[\nabla_\theta \log p_\theta(\tau)\,G_0\bigr].
$$

This is the score-function estimator.

## 3. Expand the trajectory score

Take logs:

$$
\log p_\theta(\tau)
=
\log \rho_0(s_0)
+
\sum_{t=0}^{\infty}\log \pi_\theta(a_t\mid s_t)
+
\sum_{t=0}^{\infty}\log P(s_{t+1}\mid s_t,a_t).
$$

The initial-state distribution and environment dynamics do not depend on $\theta$.
Therefore

$$
\nabla_\theta \log p_\theta(\tau)
=
\sum_{t=0}^{\infty}\nabla_\theta \log \pi_\theta(a_t\mid s_t).
$$

Substitute into the gradient:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\left(
\sum_{t=0}^{\infty}\nabla_\theta \log \pi_\theta(A_t\mid S_t)
\right)G_0
\right].
$$

Expanding the sum gives

$$
\nabla_\theta J(\theta)
=
\sum_{t=0}^{\infty}
\mathbb{E}
\bigl[
\nabla_\theta \log \pi_\theta(A_t\mid S_t)\,G_0
\bigr].
$$

## 4. Use causality

Write the full return as

$$
G_0
=
\sum_{k=0}^{t-1}\gamma^k R_{k+1}
+
\gamma^t G_t.
$$

Call the first term the past-return component $C_t$.
Then

$$
\mathbb{E}\bigl[\nabla_\theta \log \pi_\theta(A_t\mid S_t)\,G_0\bigr]
=
\mathbb{E}\bigl[\nabla_\theta \log \pi_\theta(A_t\mid S_t)\,C_t\bigr]
+
\mathbb{E}\bigl[\nabla_\theta \log \pi_\theta(A_t\mid S_t)\,\gamma^t G_t\bigr].
$$

The first term is zero.
Condition on $S_t$:

$$
\mathbb{E}\bigl[\nabla_\theta \log \pi_\theta(A_t\mid S_t)\,C_t\bigr]
=
\mathbb{E}\left[
C_t\,
\mathbb{E}\bigl[\nabla_\theta \log \pi_\theta(A_t\mid S_t)\mid S_t\bigr]
\right].
$$

Now

$$
\mathbb{E}\bigl[\nabla_\theta \log \pi_\theta(A_t\mid S_t)\mid S_t=s\bigr]
=
\sum_a \pi_\theta(a\mid s)\nabla_\theta \log \pi_\theta(a\mid s).
$$

Using $\nabla \log \pi = \nabla \pi / \pi$,

$$
\sum_a \pi_\theta(a\mid s)\nabla_\theta \log \pi_\theta(a\mid s)
=
\sum_a \nabla_\theta \pi_\theta(a\mid s)
=
\nabla_\theta \sum_a \pi_\theta(a\mid s)
=
\nabla_\theta 1
= 0.
$$

Hence the past-return term vanishes and

$$
\nabla_\theta J(\theta)
=
\sum_{t=0}^{\infty}
\mathbb{E}
\bigl[
\gamma^t \nabla_\theta \log \pi_\theta(A_t\mid S_t)\,G_t
\bigr].
$$

In episodic derivations the factor $\gamma^t$ is often absorbed into the trajectory distribution or omitted by convention.
The key structural point is that the score at time $t$ multiplies only future return from time $t$ onward.

## 5. Introduce the action-value function

Condition on $(S_t,A_t)$:

$$
\mathbb{E}[G_t\mid S_t=s,A_t=a] = Q^{\pi_\theta}(s,a).
$$

Therefore

$$
\nabla_\theta J(\theta)
=
\sum_{t=0}^{\infty}
\mathbb{E}
\bigl[
\gamma^t \nabla_\theta \log \pi_\theta(A_t\mid S_t)\,
Q^{\pi_\theta}(S_t,A_t)
\bigr].
$$

This is already a policy-gradient identity.

## 6. Rewrite using discounted state occupancy

Define the discounted state occupancy measure

$$
d^{\pi_\theta}(s)
=
\sum_{t=0}^{\infty}\gamma^t \mathbb{P}(S_t=s\mid\pi_\theta).
$$

Then the previous expression can be rewritten as

$$
\nabla_\theta J(\theta)
=
\sum_s d^{\pi_\theta}(s)\sum_a
\pi_\theta(a\mid s)\nabla_\theta \log \pi_\theta(a\mid s)\,
Q^{\pi_\theta}(s,a).
$$

Use

$$
\pi_\theta(a\mid s)\nabla_\theta \log \pi_\theta(a\mid s)
=
\nabla_\theta \pi_\theta(a\mid s),
$$

to obtain

$$
\boxed{
\nabla_\theta J(\theta)
=
\sum_s d^{\pi_\theta}(s)\sum_a
\nabla_\theta \pi_\theta(a\mid s)\,Q^{\pi_\theta}(s,a)
}.
$$

This is the policy gradient theorem.

An equivalent expectation form is

$$
\boxed{
\nabla_\theta J(\theta)
=
\mathbb{E}_{S\sim d^{\pi_\theta},\,A\sim\pi_\theta}
\bigl[
\nabla_\theta \log \pi_\theta(A\mid S)\,Q^{\pi_\theta}(S,A)
\bigr]
}.
$$

## 7. Baseline invariance

Let $b(s)$ be any function of the state alone.
Then

$$
\mathbb{E}_{A\sim\pi_\theta(\cdot\mid s)}
\bigl[
\nabla_\theta \log \pi_\theta(A\mid s)\,b(s)
\bigr]
=
b(s)\sum_a \pi_\theta(a\mid s)\nabla_\theta \log \pi_\theta(a\mid s)
=0.
$$

So

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\bigl[
\nabla_\theta \log \pi_\theta(A\mid S)\,(Q^{\pi_\theta}(S,A)-b(S))
\bigr].
$$

Choosing $b(s)=V^{\pi_\theta}(s)$ yields the advantage form

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\bigl[
\nabla_\theta \log \pi_\theta(A\mid S)\,A^{\pi_\theta}(S,A)
\bigr].
$$

This explains why actor-critic methods use value baselines.

## 8. From theorem to algorithms

The theorem itself is exact, but practical algorithms replace $Q^{\pi_\theta}$ or $A^{\pi_\theta}$ with sampled estimators:

- REINFORCE uses Monte Carlo returns;
- actor-critic uses learned value estimates and TD errors;
- PPO uses clipped surrogate optimization built on advantage estimates.

## 9. Scope note

This derivation uses the discounted objective and differentiable stochastic policies.
Average-reward formulations, deterministic policy gradients, and off-policy corrections require additional machinery.
