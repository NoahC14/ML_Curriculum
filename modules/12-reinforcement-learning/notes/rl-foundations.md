---
title: "Reinforcement Learning Foundations"
module: "12-reinforcement-learning"
lesson: "rl-foundations"
doc_type: "notes"
topic: "mdps-bellman-td-control"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability"
  - "01-optimization/convexity-and-optimization"
  - "06-neural-networks/neural-networks-first-principles"
updated: "2026-04-13"
owner: "curriculum-team"
tags:
  - "reinforcement-learning"
  - "mdp"
  - "bellman-equation"
  - "dynamic-programming"
  - "temporal-difference"
  - "q-learning"
  - "sarsa"
---

## Purpose

These notes introduce the canonical mathematical objects of reinforcement learning: Markov decision processes, return, value functions, Bellman equations, dynamic programming, and temporal-difference control.
The emphasis is on the standard RL toolkit used throughout the literature, with enough mathematical detail to support later deep RL methods.

## Learning objectives

After working through this note, you should be able to:

- define an MDP and explain the role of states, actions, rewards, transitions, and discounting;
- distinguish finite-horizon, infinite-horizon, episodic, and continuing settings;
- define policy, return, state-value, and action-value functions;
- derive the Bellman expectation equations and Bellman optimality equations for both $V$ and $Q$;
- explain the difference between model-based and model-free RL;
- formalize the exploration-exploitation tradeoff;
- describe dynamic programming, temporal-difference learning, SARSA, and Q-learning; and
- explain why bootstrapping is both powerful and potentially unstable.

## 1. Motivation

In supervised learning, data arrive as pairs $(x,y)$ sampled from a fixed distribution.
In reinforcement learning, the learner interacts with an environment over time.
Its actions influence the future states it sees, and therefore influence the future data distribution.

This creates three core difficulties:

- feedback is delayed, because actions affect long-term reward rather than one-step labels;
- data are non-i.i.d., because the policy changes which states are visited; and
- the learner must balance exploiting known good actions against exploring uncertain ones.

The canonical framework for these problems is the Markov decision process.

## 2. Assumptions and notation

We work with a discounted Markov decision process

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma),
$$

where:

- $\mathcal{S}$ is the state space;
- $\mathcal{A}$ is the action space;
- $P(s' \mid s,a)$ is the transition kernel;
- $R(s,a)$ or $R(s,a,s')$ denotes expected immediate reward; and
- $\gamma \in [0,1)$ is the discount factor.

At time $t$, the agent observes a state $S_t$, chooses an action $A_t$, receives a reward $R_{t+1}$, and transitions to a new state $S_{t+1}$.

We assume the Markov property:

$$
\mathbb{P}(S_{t+1}=s', R_{t+1}=r \mid S_0,A_0,\dots,S_t,A_t)
=
\mathbb{P}(S_{t+1}=s', R_{t+1}=r \mid S_t,A_t).
$$

This says the present state-action pair contains all relevant predictive information about the next step.

## 3. Policies and return

A policy is a rule for selecting actions.
For a stochastic policy $\pi$,

$$
\pi(a \mid s) = \mathbb{P}(A_t=a \mid S_t=s).
$$

A deterministic policy is a map $\pi:\mathcal{S}\to\mathcal{A}$.

The discounted return from time $t$ is

$$
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}.
$$

Discounting serves several roles:

- it makes distant rewards less influential;
- it keeps infinite-horizon sums finite when rewards are bounded; and
- it encodes a preference for earlier reward.

When $\gamma=0$, only immediate reward matters.
As $\gamma$ approaches $1$, long-term planning becomes more important.

## 4. Value functions

For a fixed policy $\pi$, the state-value function is

$$
V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t=s].
$$

The action-value function is

$$
Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t=s, A_t=a].
$$

These quantify long-run utility under a policy.
The difference is conditioning:

- $V^\pi(s)$ evaluates a state before the policy acts;
- $Q^\pi(s,a)$ evaluates a specific action in that state.

The advantage function is

$$
A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s),
$$

which measures how much better or worse action $a$ is than the policy's average behavior at state $s$.

## 5. Bellman expectation equations

Because

$$
G_t = R_{t+1} + \gamma G_{t+1},
$$

we can express values recursively.
For a fixed policy $\pi$,

$$
V^\pi(s)
=
\sum_{a\in\mathcal{A}} \pi(a\mid s)
\sum_{s',r} p(s',r\mid s,a)\bigl[r + \gamma V^\pi(s')\bigr].
$$

This is the Bellman expectation equation for $V^\pi$.
Likewise,

$$
Q^\pi(s,a)
=
\sum_{s',r} p(s',r\mid s,a)
\left[
r + \gamma \sum_{a'} \pi(a'\mid s')Q^\pi(s',a')
\right].
$$

These equations say value is immediate reward plus discounted continuation value.

## 6. Optimal value functions

The goal of control is to find a policy maximizing expected return.
Define

$$
V^*(s) = \sup_\pi V^\pi(s),
\qquad
Q^*(s,a) = \sup_\pi Q^\pi(s,a).
$$

The Bellman optimality equation for the optimal state-value function is

$$
V^*(s)
=
\max_{a\in\mathcal{A}}
\sum_{s',r} p(s',r\mid s,a)\bigl[r + \gamma V^*(s')\bigr].
$$

For the optimal action-value function,

$$
Q^*(s,a)
=
\sum_{s',r} p(s',r\mid s,a)
\left[
r + \gamma \max_{a'} Q^*(s',a')
\right].
$$

Any greedy policy with respect to $Q^*$,

$$
\pi^*(s) \in \arg\max_a Q^*(s,a),
$$

is optimal.

## 7. Exploration versus exploitation

The agent faces a sequential decision problem under uncertainty.
Exploitation means choosing the action that currently appears best.
Exploration means choosing actions that may look suboptimal now but improve future knowledge.

A useful formalization uses regret.
Let $a_t$ be the chosen action and $a_t^*$ the reward-maximizing action under the true environment at time $t$.
Then cumulative regret over $T$ rounds is

$$
\operatorname{Regret}(T)
=
\sum_{t=1}^T
\mathbb{E}\bigl[r_t(a_t^*) - r_t(a_t)\bigr].
$$

In bandits and RL alike, purely greedy behavior can lock the agent into a poor estimate.
Common exploration strategies include:

- $\varepsilon$-greedy: with probability $\varepsilon$, choose a random action;
- softmax or Boltzmann exploration: sample actions proportional to exponentiated value estimates;
- optimism in the face of uncertainty: initialize values high to encourage trial; and
- explicit uncertainty methods such as posterior sampling or upper-confidence bounds.

The correct tradeoff depends on the environment, the sample budget, and the function approximator.

## 8. Model-based versus model-free RL

The distinction is about whether the learner uses an explicit transition-reward model.

In model-based RL, the agent estimates or is given $P$ and $R$, then plans with that model.
Examples include policy evaluation by matrix methods, value iteration, and tree-search planners.

In model-free RL, the agent learns values or policies directly from sampled experience without explicitly estimating the full environment dynamics.
Examples include Monte Carlo methods, TD learning, SARSA, Q-learning, policy gradients, and actor-critic methods.

The tradeoff is standard:

- model-based methods can be more sample efficient because they reuse a learned model for planning;
- model-free methods avoid model misspecification but may require more interaction data.

## 9. Dynamic programming

Dynamic programming assumes access to the full MDP model.
The two central procedures are policy evaluation and policy improvement.

### 9.1 Policy evaluation

Given $\pi$, solve the Bellman expectation equation for $V^\pi$.
An iterative form is

$$
V_{k+1}(s)
=
\sum_a \pi(a\mid s)\sum_{s',r} p(s',r\mid s,a)
\bigl[r + \gamma V_k(s')\bigr].
$$

This is repeated application of the Bellman expectation operator.

### 9.2 Policy improvement

Given a value function estimate, define a greedy policy

$$
\pi_{\text{new}}(s) \in \arg\max_a
\sum_{s',r} p(s',r\mid s,a)\bigl[r+\gamma V^\pi(s')\bigr].
$$

The policy improvement theorem shows that the new policy is at least as good as the old one.

### 9.3 Policy iteration and value iteration

Policy iteration alternates:

1. evaluate the current policy;
2. improve it greedily.

Value iteration compresses these two steps into the update

$$
V_{k+1}(s)
=
\max_a \sum_{s',r} p(s',r\mid s,a)\bigl[r+\gamma V_k(s')\bigr].
$$

This directly applies the Bellman optimality operator.

## 10. Monte Carlo and temporal-difference learning

Dynamic programming requires a model.
Model-free methods replace exact expectations with sampled experience.

### 10.1 Monte Carlo evaluation

Monte Carlo policy evaluation estimates $V^\pi(s)$ by averaging complete sampled returns from episodes that visit $s$.
Its target is unbiased but high variance:

$$
V(S_t) \leftarrow V(S_t) + \alpha\bigl(G_t - V(S_t)\bigr).
$$

### 10.2 Temporal-difference learning

TD learning bootstraps by using an estimate of the next value:

$$
V(S_t) \leftarrow V(S_t) + \alpha\delta_t,
$$

where the TD error is

$$
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t).
$$

This combines the low variance of bootstrapping with the model-free nature of sampling.

Key comparison:

- Monte Carlo waits until the episode ends and uses full return targets;
- TD updates immediately from one-step transitions.

## 11. Control with action values

For control, we often learn action values directly.

### 11.1 SARSA

SARSA is an on-policy TD control method.
Its update uses the action actually taken by the current behavior policy:

$$
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha\Bigl[
R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
\Bigr].
$$

Because the next action $A_{t+1}$ is sampled from the same policy being learned, SARSA reflects the value of the exploratory policy itself.

### 11.2 Q-learning

Q-learning is an off-policy TD control method.
Its target uses the greedy next action regardless of which action was actually sampled:

$$
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha\Bigl[
R_{t+1} + \gamma \max_{a'} Q(S_{t+1},a') - Q(S_t,A_t)
\Bigr].
$$

This separates:

- the behavior policy used for exploration; from
- the target policy being optimized.

Under suitable assumptions in the tabular case, Q-learning converges to $Q^*$.

### 11.3 SARSA versus Q-learning

The distinction matters.
SARSA evaluates the policy actually executed, including its exploratory moves.
Q-learning evaluates the greedy policy while potentially exploring with another policy.

As a result:

- SARSA can learn safer behavior in environments where exploratory actions are dangerous;
- Q-learning is often more aggressive because it learns toward a greedy target.

## 12. Function approximation and deep RL

Tabular methods assume each state-action pair can be stored explicitly.
This is impossible in large or continuous state spaces.

Function approximation replaces tables with parameterized estimators such as

$$
Q_\theta(s,a)
\quad \text{or} \quad
\pi_\theta(a\mid s).
$$

This enables generalization across states, but introduces instability because:

- updates are no longer local;
- bootstrapping targets depend on learned parameters; and
- data are correlated along trajectories.

Deep Q-networks, policy gradients, and actor-critic methods address these challenges with neural-network approximators and additional stabilization tricks.

## 13. Worked example: a tiny gridworld

Consider a deterministic gridworld with terminal goal reward $+1$, step cost $-0.04$, and discount $\gamma=0.95$.
If the agent is one step from the goal and can move directly into it, then under an optimal policy:

$$
Q^*(s,\text{right}) = 1
$$

if the episode ends immediately after entering the goal.
If another action leads to a nonterminal state $s'$ with value $0.7$, then

$$
Q^*(s,\text{up}) = -0.04 + 0.95(0.7) = 0.625.
$$

The Bellman optimality equation selects the larger quantity.
This is the central idea of value-based RL: rank actions by long-term consequences rather than immediate reward alone.

## 14. Structural perspective

From a compositional viewpoint, an MDP can be read as a system in which:

- states are the objects carrying information about decision context;
- policies map states to action distributions;
- transition kernels map state-action pairs to next-state distributions; and
- Bellman operators map value functions to updated value functions.

This perspective is helpful for organizing algorithms, but it does not replace the standard probabilistic and optimization-based derivations.

## 15. Common pitfalls

- Confusing reward with return: reward is one-step feedback, return aggregates future reward.
- Treating $Q(s,a)$ as immediate payoff only: it includes downstream consequences.
- Assuming off-policy methods are automatically better: they can be less stable with function approximation.
- Ignoring the role of exploration: without persistent exploration, value estimates can be systematically wrong.
- Forgetting discounting assumptions: some infinite-horizon statements require bounded rewards and $\gamma<1$.

## 16. Summary

Reinforcement learning studies sequential decision-making under uncertainty.
The MDP provides the formal framework, the Bellman equations express recursive structure, dynamic programming solves known-model problems, and temporal-difference methods learn from experience.
SARSA and Q-learning are the canonical value-based control algorithms that lead naturally into deep RL.

## 17. Next steps

The next note develops policy optimization methods.
Those methods replace value-only control with direct optimization of a parameterized policy, then combine policy and value learning in actor-critic algorithms.
