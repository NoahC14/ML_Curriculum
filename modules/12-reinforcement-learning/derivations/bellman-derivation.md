---
title: "Bellman Equations Derivation"
module: "12-reinforcement-learning"
lesson: "bellman-derivation"
doc_type: "derivation"
topic: "value-and-action-value-recursions"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability"
  - "12-reinforcement-learning/notes/rl-foundations"
updated: "2026-04-13"
owner: "curriculum-team"
tags:
  - "reinforcement-learning"
  - "bellman-equation"
  - "value-function"
  - "action-value"
---

## Goal

Derive the Bellman expectation equations and Bellman optimality equations for both state-value and action-value functions.
The central point is that the return decomposes into immediate reward plus discounted future return.

## 1. Setup

Let $\mathcal{M}=(\mathcal{S},\mathcal{A},P,R,\gamma)$ be a discounted MDP with $\gamma\in[0,1)$.
Fix a policy $\pi(a\mid s)$.

Define the return

$$
G_t = \sum_{k=0}^{\infty}\gamma^k R_{t+k+1}.
$$

Then

$$
G_t = R_{t+1} + \gamma G_{t+1}.
$$

The value functions are

$$
V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s],
\qquad
Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].
$$

## 2. Derivation of the Bellman expectation equation for $V^\pi$

Start from the definition:

$$
V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s].
$$

Insert the recursive decomposition of the return:

$$
V^\pi(s)
=
\mathbb{E}_\pi[R_{t+1}+\gamma G_{t+1}\mid S_t=s].
$$

Use linearity of expectation:

$$
V^\pi(s)
=
\mathbb{E}_\pi[R_{t+1}\mid S_t=s]
+
\gamma \mathbb{E}_\pi[G_{t+1}\mid S_t=s].
$$

Now condition on the action chosen at state $s$:

$$
V^\pi(s)
=
\sum_{a}\pi(a\mid s)
\mathbb{E}_\pi[R_{t+1}+\gamma G_{t+1}\mid S_t=s,A_t=a].
$$

Next condition on the next state and reward:

$$
V^\pi(s)
=
\sum_a \pi(a\mid s)
\sum_{s',r}
p(s',r\mid s,a)
\mathbb{E}_\pi[r+\gamma G_{t+1}\mid S_t=s,A_t=a,S_{t+1}=s',R_{t+1}=r].
$$

Inside the inner expectation, $r$ is fixed, so

$$
\mathbb{E}_\pi[r+\gamma G_{t+1}\mid \cdots]
=
r+\gamma \mathbb{E}_\pi[G_{t+1}\mid S_{t+1}=s'].
$$

By the definition of the state-value function at the next state,

$$
\mathbb{E}_\pi[G_{t+1}\mid S_{t+1}=s'] = V^\pi(s').
$$

Therefore

$$
\boxed{
V^\pi(s)
=
\sum_a \pi(a\mid s)\sum_{s',r}
p(s',r\mid s,a)\bigl[r+\gamma V^\pi(s')\bigr]
}.
$$

This is the Bellman expectation equation for $V^\pi$.

## 3. Derivation of the Bellman expectation equation for $Q^\pi$

Start from

$$
Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].
$$

Again substitute the recursive return:

$$
Q^\pi(s,a)
=
\mathbb{E}_\pi[R_{t+1}+\gamma G_{t+1}\mid S_t=s,A_t=a].
$$

Condition on the next state and reward:

$$
Q^\pi(s,a)
=
\sum_{s',r}
p(s',r\mid s,a)
\mathbb{E}_\pi[r+\gamma G_{t+1}\mid S_t=s,A_t=a,S_{t+1}=s',R_{t+1}=r].
$$

As before,

$$
Q^\pi(s,a)
=
\sum_{s',r}
p(s',r\mid s,a)
\left[r+\gamma \mathbb{E}_\pi[G_{t+1}\mid S_{t+1}=s']\right].
$$

At time $t+1$, the policy samples an action $A_{t+1}\sim\pi(\cdot\mid s')$, so

$$
\mathbb{E}_\pi[G_{t+1}\mid S_{t+1}=s']
=
\sum_{a'}\pi(a'\mid s')Q^\pi(s',a').
$$

Hence

$$
\boxed{
Q^\pi(s,a)
=
\sum_{s',r}
p(s',r\mid s,a)
\left[
r+\gamma \sum_{a'}\pi(a'\mid s')Q^\pi(s',a')
\right]
}.
$$

This is the Bellman expectation equation for $Q^\pi$.

## 4. Bellman optimality equation for $V^*$

Define

$$
V^*(s)=\sup_\pi V^\pi(s).
$$

At an optimal state, the agent should choose the action with maximal long-run value.
So for any state $s$,

$$
V^*(s)
=
\max_a
\mathbb{E}[R_{t+1}+\gamma G_{t+1}\mid S_t=s,A_t=a,\pi^*].
$$

Condition on the next state and reward:

$$
V^*(s)
=
\max_a \sum_{s',r}
p(s',r\mid s,a)
\left[r+\gamma V^*(s')\right].
$$

Therefore

$$
\boxed{
V^*(s)
=
\max_a \sum_{s',r}
p(s',r\mid s,a)\bigl[r+\gamma V^*(s')\bigr]
}.
$$

## 5. Bellman optimality equation for $Q^*$

Define

$$
Q^*(s,a)=\sup_\pi Q^\pi(s,a).
$$

After taking action $a$ in state $s$, optimal control continues from the next state.
Thus

$$
Q^*(s,a)
=
\sum_{s',r}
p(s',r\mid s,a)
\left[r+\gamma \max_{a'}Q^*(s',a')\right].
$$

So the Bellman optimality equation for the action-value function is

$$
\boxed{
Q^*(s,a)
=
\sum_{s',r}
p(s',r\mid s,a)
\left[r+\gamma \max_{a'}Q^*(s',a')\right]
}.
$$

## 6. Why the equations matter

These recursive equations are the foundation of nearly all classical RL algorithms:

- policy evaluation solves or approximates the Bellman expectation equation;
- value iteration repeatedly applies the Bellman optimality operator;
- SARSA approximates the Bellman expectation recursion for a behavior policy;
- Q-learning approximates the Bellman optimality recursion from sampled transitions.

## 7. Scope note

The derivations above assume bounded rewards and a discounted infinite horizon so that the return is well defined.
Average-reward and finite-horizon settings use related but slightly different equations.
