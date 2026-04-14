---
title: "Agency and Sequence Through Unity Theory"
module: "17-unity-theory-perspectives"
lesson: "agency-and-sequence"
doc_type: "notes"
topic: "unity-theory-sequence-memory-agency"
status: "draft"
prerequisites:
  - "17-unity-theory-perspectives/scope-memo"
  - "17-unity-theory-perspectives/glossary"
  - "09-sequence-models/sequence-modeling"
  - "12-reinforcement-learning/rl-foundations"
  - "12-reinforcement-learning/policy-gradients"
updated: "2026-04-13"
owner: "curriculum-team"
tags:
  - "unity-theory"
  - "sequence-models"
  - "reinforcement-learning"
  - "memory"
  - "agency"
  - "interpretive-note"
---

# Agency and Sequence Through Unity Theory

## Purpose

This note gives a bounded Unity Theory reading of sequence modeling and reinforcement learning.
Its aim is not to redefine either field.
Its aim is to connect standard objects from [Module 09 sequence modeling](../../09-sequence-models/notes/sequence-modeling.md) and [Module 12 reinforcement learning foundations](../../12-reinforcement-learning/notes/rl-foundations.md) to a small set of interpretive questions:

- what counts as identity across time;
- how memory supports that persistence under transformation;
- how policy can be read as coherent action under uncertainty; and
- how exploration and reward fit the Unity Theory terms `multiplicity` and `coherence`.

> [!NOTE]
> **Interpretive note.** The canonical explanations remain the RNN hidden-state recurrence, BPTT, LSTM gating, MDP formalism, Bellman equations, and policy-gradient methods. This note is a companion interpretation of those objects, not an alternative derivation.

## Boundary block

- `Standard claim`: Sequence models summarize past inputs into hidden states, and reinforcement learning formalizes sequential decision-making through MDPs, value functions, Bellman recursions, and policy optimization.
- `Interpretive claim`: Unity Theory offers a disciplined language for reading hidden-state persistence as a task-indexed form of identity across time, and policy learning as a constrained search for coherent action under uncertainty.
- `Scope note`: This note does not claim that sequence models literally solve metaphysical identity, that RL agents possess consciousness or intent, or that Unity Theory is part of standard ML doctrine.

## Canonical anchors used here

The note depends on the following canonical results and constructions.

From Module 09:

- hidden-state sequence summaries, where the note states that sequence models replace direct conditioning on the raw past with a learned state summary $h_t$ of $x_{1:t}$;
- recurrent updates of the form $h_t = \phi(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$;
- vanishing and exploding gradients as repeated Jacobian products;
- LSTM memory cells and gates as mechanisms for preserving and editing temporal information;
- beam search as maintenance of several plausible futures rather than a single greedy continuation.

From Module 12:

- the MDP formalism $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$;
- policies as action-selection rules, deterministic or stochastic;
- value functions $V^\pi$ and $Q^\pi$ as long-run evaluative summaries under a policy;
- Bellman equations as recursive statements of long-horizon consequence;
- exploration-exploitation tradeoffs;
- policy gradients and actor-critic methods as direct policy optimization coupled with learned value estimates.

## Correspondence table

| Unity Theory term | Canonical anchor in Modules 09 and 12 | Narrow use in this note |
| --- | --- | --- |
| identity | hidden state, memory cell, state estimate, value summary | what the model tries to preserve as task-relevant sameness across time |
| relation | recurrent update, transition kernel, Bellman recursion, actor-critic coupling | the structured links by which present state constrains future state and action |
| embodiment | finite-dimensional hidden state, LSTM cell state, tabular value table, neural policy network | the concrete realization of temporal or decision structure |
| multiplicity | many plausible continuations, stochastic policies, exploratory trajectories, beam candidates | constrained plurality of future paths consistent with current information |
| coherence | usable long-range memory, policy-value consistency, reward-aligned long-term action | stability relative to the task objective and dynamics |
| informational action | recurrent update, chosen action, policy-gradient step, TD update | converting available information into an admissible state or policy change |

## 1. Sequence modeling as identity across time

### Canonical anchor

Module 09 motivates hidden-state models by replacing direct dependence on the entire raw past with a learned state summary:

$$
h_t = \text{summary of } x_{1:t}.
$$

A plain recurrent update then takes the form

$$
h_t = \phi(W_{xh}x_t + W_{hh}h_{t-1} + b_h).
$$

This is the standard mechanism by which a sequence model carries information forward.
The model does not preserve the whole past literally.
It preserves only whatever aspects of the past remain useful for the downstream prediction task.

### Interpretive reading

The Unity Theory term `identity` is useful here if it is kept narrow.
The hidden state is not the identity of an object in any absolute sense.
It is the model's current embodied criterion for what should count as the same relevant situation as the sequence unfolds.

That criterion is task-indexed:

- in language modeling, it may preserve syntactic or semantic constraints needed for the next token;
- in sequence classification, it may preserve features needed for the final label;
- in encoder-decoder settings, it may preserve enough source information for later decoding.

On this reading, sequence modeling asks a disciplined version of the identity question:

> Which aspects of the past must remain stable through successive updates so that later predictions still refer to the same task-relevant situation?

This reading is justified only because Module 09 already makes state compression explicit.
The hidden state is a finite embodiment of that persistence problem.

### Scope note

This does not mean the hidden state stores a complete or faithful representation of reality.
It stores a lossy, learned, task-shaped summary.
Unity language is helpful only insofar as it sharpens the distinction between persistence of task identity and literal retention of every past detail.

## 2. Memory failure and memory repair

### Canonical anchor

Module 09 explains vanishing and exploding gradients through repeated Jacobian products during BPTT.
The result is that long-range temporal credit assignment can become unstable.
LSTMs and GRUs address this by introducing gates and, in the LSTM case, an explicit memory cell $c_t$ with an additive path.

### Interpretive reading

This gives a precise way to talk about `coherence` across time.
For sequence models, coherence is not a mystical harmony.
It is the practical stability of task-relevant information under repeated temporal transformation.

A plain RNN often loses coherence because the information needed to treat later states as continuous with earlier ones becomes numerically unusable.
LSTM gates can then be read as learned controls on temporal identity maintenance:

- the forget gate regulates which prior content should cease to count as relevant persistence;
- the input gate regulates what new content may enter the standing state;
- the output gate regulates which part of the maintained state is exposed for current computation.

In ordinary ML language, this is memory management under optimization constraints.
In the Unity Theory vocabulary, it is a disciplined example of preserving coherence of identity through transformation.

### Concrete cross-reference

Module 09 already states the right bounded interpretation in its Unity insertion point: sequence models can be viewed as systems that compress temporal experience into state under resource constraints.
This note simply develops that comment more explicitly.

## 3. Sequence decoding and multiplicity

### Canonical anchor

Module 09 contrasts greedy decoding with beam search.
Greedy decoding commits to the locally best next token.
Beam search keeps several high-scoring partial hypotheses alive and extends them in parallel.

### Interpretive reading

This is a useful place for the glossary term `multiplicity`.
At a given point in decoding, the future is not fixed by the present hidden state alone.
There may be several plausible continuations compatible with what the model currently knows.

Beam search is therefore an operational example of constrained multiplicity:

- many futures are available;
- not all are equally plausible;
- search keeps a structured subset alive rather than collapsing immediately to one path.

This does not make beam search philosophically deep.
It makes it a clean computational illustration of the more general idea that one task-relevant state can support several admissible forward realizations.

### Bridge to reinforcement learning

This matters for agency because the same structural issue appears in control.
A state rarely determines one inevitable action.
It defines a set of admissible possibilities whose quality must be evaluated under uncertainty and over time.

## 4. Reinforcement learning as coherent action under uncertainty

### Canonical anchor

Module 12 defines an MDP

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma),
$$

with policy, return, state-value, and action-value functions.
The core idea is that actions must be evaluated by their long-run consequences, not only by immediate reward.
Bellman equations express this recursively, and policy optimization methods seek action rules with high expected return.

### Interpretive reading

This is the narrow sense in which the word `agency` can be used productively in a companion note.
An RL agent is not assumed to have free will, consciousness, or humanlike deliberation.
It is an adaptive decision system whose policy converts state information into action under uncertainty and feedback.

The Unity Theory term `informational action` is appropriate here because a policy does exactly that:

- it receives a state or state estimate;
- it selects from an admissible action set;
- the action changes future observations and rewards;
- later updates revise the policy in light of those consequences.

Calling policy `coherent action` is warranted only when coherence is tied to the canonical RL objective.
A policy is coherent relative to an MDP when its local choices are organized by a stable long-run evaluative structure rather than by myopic one-step reward alone.

That long-run structure is encoded canonically by $V^\pi$ and $Q^\pi$.

## 5. Value functions as identity-across-time summaries

### Canonical anchor

Module 12 defines

$$
V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]
$$

and

$$
Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a].
$$

These are summaries of expected future return under a policy.
The Bellman equations recursively tie present evaluation to expected downstream consequences.

### Interpretive reading

For this note, value functions are a second example of `identity across time`.
A value function does not preserve sensory identity the way a hidden state tries to preserve predictive context.
Instead, it preserves evaluative identity:

- different trajectories that begin from the same state may vary in surface detail;
- the value function compresses them into a task-relevant summary of future consequence;
- the state is treated as the same decision point insofar as those downstream consequences are summarized by the same value estimate.

This is why the acceptance criterion for this card specifically mentions value functions.
They are one of the cleanest places where Unity Theory language can remain disciplined.
The persistence is not metaphysical sameness.
It is persistence of evaluative meaning across unfolding time.

### Scope note

This does not imply that a value function captures everything important about an agent or environment.
It captures what is relevant to expected return under a specified policy, transition law, reward function, and discount factor.

## 6. Exploration as multiplicity

### Canonical anchor

Module 12 formalizes the exploration-exploitation tradeoff and discusses strategies such as epsilon-greedy and softmax exploration.
It also distinguishes SARSA and Q-learning partly by how they relate to exploratory behavior and greedy targets.

### Interpretive reading

The glossary term `multiplicity` applies naturally here.
At a state $s$, the decision problem typically admits several feasible trajectories.
Exploration keeps that plurality open long enough to learn which possibilities are genuinely valuable.

This suggests a conservative interpretive claim:

> Exploration is multiplicity kept epistemically alive under uncertainty.

That is a useful reading because it explains why premature collapse to a single favored action can produce systematically wrong value estimates.
Without exploration, the agent acts as if the future had already been reduced to one path even when the evidence does not justify that reduction.

The SARSA versus Q-learning contrast is relevant here:

- SARSA evaluates the policy actually executed, including exploratory moves;
- Q-learning learns toward a greedy target while behavior may remain exploratory.

This is not just an algorithmic detail.
It shows that multiplicity can be represented differently depending on whether the learning target includes exploratory action as part of the policy being evaluated.

## 7. Reward as evaluative coherence

### Canonical anchor

Module 12 distinguishes one-step reward from return and emphasizes that value-based RL ranks actions by long-term consequences.
Policy-gradient methods then optimize expected return directly rather than relying only on greedy one-step selection.

### Interpretive reading

Reward can be read as a local evaluative signal, while return and value functions express a broader standard of `coherence` over time.
On this reading:

- immediate reward is not yet full coherence;
- coherence is the compatibility of successive actions with a stable long-run objective;
- Bellman recursions make that compatibility computationally explicit.

This reading is useful because it blocks a common philosophical overreach.
If one says "reward is value" without qualification, the statement becomes vague or false.
The tighter claim is:

> Within a specified MDP, reward is the local feedback signal from which long-run evaluative coherence is recursively constructed.

That keeps the interpretation tied to RL formalism instead of metaphor.

## 8. Policy gradients and actor-critic as coupled agency

### Canonical anchor

The policy-gradient note defines the objective $J(\theta)$, derives REINFORCE, introduces baselines and advantages, and presents actor-critic methods as coupled policy and value learning.
An actor updates action probabilities.
A critic estimates value information used to reduce variance and improve directionality.

### Interpretive reading

Actor-critic is a natural place to connect `informational action` and `coherence`.
The actor alone selects actions, but the critic supplies an evaluative summary that ties local policy changes to long-horizon consequences.

This creates a bounded interpretive picture:

- the actor is the current realization of action choice;
- the critic is the learned evaluative memory of how present choices relate to future return;
- the advantage estimate measures whether a sampled action improved on the policy's standing expectation at that state.

In Unity Theory terms, the pair can be read as a coordination between action and evaluative identity.
In canonical RL terms, it is simply coupled policy and value estimation.
Both descriptions point to the same formal machinery, but only the second one is standard doctrine.

## 9. Why the sequence-RL connection matters

The sequence and RL modules concern different canonical problems, but they share a structural question:

> How can a system preserve enough of the past to act or predict well in the future when only finite state can be carried forward at each step?

In Module 09, the answer appears as hidden states, memory cells, and decoding strategies.
In Module 12, it appears as state representations, value functions, and policies optimized for long-run return.

This is why the two modules belong together in a Unity companion note.
Both are about selective persistence under transformation:

- sequence models preserve predictive context across observation updates;
- RL preserves decision-relevant evaluation across action-dependent state transitions.

That shared structure is what makes `identity across time` a productive interpretive phrase here.

## 10. Non-claims and failure modes

To keep this note within scope, several stronger claims must be rejected explicitly.

- This note does not claim that recurrent states or value functions are full theories of personal identity.
- This note does not claim that RL agents possess moral or phenomenological agency.
- This note does not claim that reward provides an objective notion of value outside the specified task.
- This note does not claim that multiplicity is always desirable; unchecked branching can be computationally or statistically harmful.
- This note does not claim that Unity Theory improves the derivations of BPTT, Bellman equations, or the policy-gradient theorem.

If those stronger claims are desired, they belong in an `Exploratory note`, not in this interpretive companion.

## 11. Research-facing questions

The following questions stay within the allowed scope because they begin from canonical objects already taught.

1. Can hidden-state diagnostics be framed as measuring how well task identity persists across long temporal transformations?
2. Can exploration schedules be compared as different ways of maintaining useful multiplicity before value estimates become reliable?
3. Can actor-critic architectures be analyzed as different decompositions of action selection and evaluative memory?
4. Can representation-learning tools for sequence state tracking improve partially observed RL by making state identity more stable across time?

These are prompts for later work, not established claims.

## Summary

Sequence models and reinforcement learning already have their own standard mathematics.
What Unity Theory contributes here, if used carefully, is a compact cross-module language for talking about persistence, plurality, and evaluative action across time.

The disciplined correspondences are:

- hidden states and memory cells as finite embodiments of task-indexed identity persistence;
- beam search and exploration as operational forms of multiplicity;
- value functions as summaries of evaluative identity across possible futures; and
- policy learning as coherent informational action under uncertainty.

That is a legitimate companion reading precisely because it remains downstream of the canonical account rather than competing with it.
