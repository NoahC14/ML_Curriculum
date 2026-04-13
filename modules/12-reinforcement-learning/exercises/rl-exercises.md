---
title: "Reinforcement Learning Exercises"
module: "12-reinforcement-learning"
lesson: "rl-exercises"
doc_type: "exercise"
topic: "rl-foundations-and-policy-gradients"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability"
  - "01-optimization/convexity-and-optimization"
  - "06-neural-networks/neural-networks-first-principles"
  - "12-reinforcement-learning/notes/rl-foundations"
  - "12-reinforcement-learning/notes/policy-gradients"
updated: "2026-04-13"
owner: "curriculum-team"
tags:
  - "reinforcement-learning"
  - "mdp"
  - "bellman-equation"
  - "td-learning"
  - "policy-gradient"
---

## Purpose

These exercises reinforce the mathematical and algorithmic foundations of reinforcement learning, from Bellman recursions to actor-critic methods.

## Exercise 1. MDP components

A mobile robot navigates a warehouse with noisy wheel motion.

1. Propose a state representation for the task.
2. Specify a reasonable action space.
3. Give one reward design that encourages fast task completion and one that encourages safety.
4. State one way in which the Markov property could fail for your state representation.

## Exercise 2. Return and discounting

Suppose an episode yields rewards $(2, -1, 0, 3)$ from times $t+1$ through $t+4$.

1. Compute $G_t$ when $\gamma=0$.
2. Compute $G_t$ when $\gamma=0.5$.
3. Compute $G_t$ when $\gamma=0.9$.
4. Explain in words how changing $\gamma$ changes the planning horizon.

## Exercise 3. Bellman expectation equation

Let $\pi$ be a policy on a finite MDP.

1. Starting from $V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]$, derive the Bellman expectation equation for $V^\pi$.
2. Derive the Bellman expectation equation for $Q^\pi$.
3. Explain why these equations are linear in $V^\pi$ and $Q^\pi$ once $\pi$ is fixed.

## Exercise 4. Bellman optimality

1. State the Bellman optimality equation for $V^*$.
2. State the Bellman optimality equation for $Q^*$.
3. Explain why the max operator makes the optimality equations harder to solve exactly than the expectation equations.

## Exercise 5. Policy evaluation on a two-state MDP

Consider an MDP with states $s_1,s_2$ and one action per state.
Rewards are $r(s_1)=1$ and $r(s_2)=2$.
Transitions are deterministic:

- from $s_1$ the next state is $s_2$;
- from $s_2$ the next state is $s_2$.

Let $\gamma=0.8$.

1. Write the Bellman equations for $V(s_1)$ and $V(s_2)$.
2. Solve for both values exactly.
3. Interpret why $V(s_2)$ exceeds the immediate reward $2$.

## Exercise 6. Exploration versus exploitation

Answer each in 2-4 sentences.

1. Why can a purely greedy agent fail even in a simple finite MDP?
2. Compare $\varepsilon$-greedy and Boltzmann exploration.
3. Give one setting where optimism in value initialization is useful and one where it may be misleading.

## Exercise 7. Model-based versus model-free

For each method below, classify it as model-based or model-free and justify briefly:

1. value iteration with known transitions;
2. SARSA from sampled episodes only;
3. Monte Carlo tree search with a learned simulator;
4. PPO with a neural-network policy and critic.

## Exercise 8. TD error

Suppose at time $t$ we observe:

$$
R_{t+1}=1.5,\qquad \gamma=0.9,\qquad V(S_t)=3.0,\qquad V(S_{t+1})=2.0.
$$

1. Compute the TD error $\delta_t$.
2. If $\alpha=0.1$, compute the updated value estimate at $S_t$.
3. Explain why TD learning is said to bootstrap.

## Exercise 9. SARSA and Q-learning

Suppose

$$
Q(S_t,A_t)=1.2,\qquad R_{t+1}=0.4,\qquad \gamma=0.95,\qquad \alpha=0.5.
$$

In the next state, assume:

- the behavior policy chooses $A_{t+1}=a_1$ with $Q(S_{t+1},a_1)=0.8$;
- the greedy action has value $\max_{a'}Q(S_{t+1},a')=1.4$.

1. Compute the SARSA target.
2. Compute the Q-learning target.
3. Compute the updated $Q(S_t,A_t)$ under each method.
4. Explain why the two updates differ.

## Exercise 10. Function approximation

Answer each in 2-4 sentences.

1. Why are tabular methods impractical in continuous state spaces?
2. Why does function approximation improve generalization?
3. Why can combining bootstrapping, off-policy learning, and nonlinear approximation create instability?

## Exercise 11. REINFORCE estimator

1. Write the episodic objective $J(\theta)$ for a stochastic policy.
2. Starting from the trajectory distribution, derive the REINFORCE estimator.
3. Explain why REINFORCE is unbiased.
4. Explain why REINFORCE often has high variance.

## Exercise 12. Baselines

Let $b(s)$ be a state-dependent baseline.

1. Show that
   $$
   \mathbb{E}_{A\sim\pi(\cdot\mid s)}[\nabla_\theta \log \pi_\theta(A\mid s)b(s)] = 0.
   $$
2. Explain why baselines can reduce variance without introducing bias.
3. Why is a state-value function a natural baseline?

## Exercise 13. Actor-critic interpretation

1. In one sentence, what does the actor learn?
2. In one sentence, what does the critic learn?
3. Why does a critic usually reduce variance relative to Monte Carlo returns?
4. What bias can enter when the critic is inaccurate?

## Exercise 14. PPO reasoning

Answer each in 2-4 sentences.

1. What is the probability ratio in PPO measuring?
2. Why is clipping used?
3. Why can PPO usually take multiple gradient epochs on one batch while plain REINFORCE cannot reuse the same data so freely?

## Exercise 15. Small design problem

Design a minimal RL experiment for a classroom notebook.

1. Choose either a tabular gridworld or CartPole.
2. State the algorithm you would teach first and why.
3. List the main hyperparameters.
4. Describe one plot you would use to help a student diagnose learning behavior.
