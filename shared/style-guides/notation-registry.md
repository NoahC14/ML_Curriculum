---
title: "Notation Registry"
module: "shared"
doc_type: "reference"
topic: "notation-registry"
status: "draft"
updated: "2026-04-15"
owner: "curriculum-team"
tags:
  - "style-guide"
  - "notation"
  - "cross-module-consistency"
---

# Notation Registry

## Purpose

This registry records the repository-wide default meanings for high-reuse mathematical symbols.
It is the canonical reference for resolving notation collisions across modules.
Module-local notes may introduce additional symbols, but they should avoid reassigning the meanings below unless the local context is purely standard mathematics and the symbol is redefined explicitly.

## Resolution Rules

1. Reserve bold lowercase symbols such as $\mathbf{x}$ and $\mathbf{w}$ for vectors when coordinates matter.
2. Reserve uppercase Roman symbols such as $X$, $Y$, $Z$, and $H$ for spaces, random variables, or matrices only when the surrounding section makes the role explicit.
3. Reserve calligraphic symbols such as $\mathcal{D}$, $\mathcal{H}$, and $\mathcal{L}$ for datasets, hypothesis classes, and objective-like aggregates.
4. Use $I$ for a finite sample index set in categorical diagrams and structural notes.
5. Use $D$ for degree matrices in graph-learning material only.

## Core Registry

| Symbol | Standard meaning | First introduction | Usage notes |
| --- | --- | --- | --- |
| $\mathbf{x}$ | input vector or point in Euclidean space | `modules/00-math-toolkit/notes/01-linear-algebra.md` | Use bold when the object is explicitly a vector. |
| $x_i$ | $i$th input example or coordinate depending on context | `modules/00-math-toolkit/notes/02-multivariable-calculus.md` | Redefine locally when switching between coordinates and samples. |
| $X$ | input space, design matrix, or random variable | `modules/00-math-toolkit/notes/03-probability-statistics.md` | State the role explicitly in each note. Use `\mathbf{X}` only for matrix-emphasis contexts. |
| $y_i$ | target attached to example $x_i$ | `modules/00-math-toolkit/notes/03-probability-statistics.md` | In regression/classification notes, keep paired with $x_i$. |
| $Y$ | output space, label random variable, or response vector | `modules/00-math-toolkit/notes/03-probability-statistics.md` | In categorical notes, default to output or label space. |
| $\hat{y}$ | model prediction | `modules/02-statistical-learning/notes/model-evaluation.md` | Reserve hats for predicted quantities, not latent variables. |
| $Z$ | latent space or latent variable | `modules/00-math-toolkit/notes/04-information-theory.md` | In category-theory figures, use interchangeably with $H$ only if the surrounding prose fixes the role. |
| $H$ | representation or hidden-state space | `modules/00-math-toolkit/notes/06-ml-categorical-mapping.md` | Prefer $H$ for learned feature or hidden-state spaces. |
| $\mathcal{D}$ | dataset or empirical sample | `modules/00-math-toolkit/notes/06-ml-categorical-mapping.md` | Do not reuse for degree matrices or diagram index objects. |
| $I$ | finite sample index set | `shared/style-guides/diagram-notation.md` | Replaces legacy categorical use of $D$ for sample indices. |
| $D$ | degree matrix in graph learning | `modules/13-graph-learning/notes/graph-learning.md` | Avoid using plain $D$ for datasets outside graph contexts. |
| $\Theta$ | parameter space or parameter collection space | `modules/00-math-toolkit/notes/05-category-theory-primer.md` | Use for spaces; use $\theta$ for a specific parameter vector when practical. |
| $\theta$ | parameter vector, scalar parameter, or policy parameter | `modules/00-math-toolkit/notes/03-probability-statistics.md` | Allowed across probabilistic, linear, neural, and RL notes with the same broad meaning: learnable parameters. |
| $\phi$ | feature map, representation map, or encoder-like transformation | `modules/00-math-toolkit/notes/05-category-theory-primer.md` | Prefer for input-to-representation maps. |
| $\psi$ | output map, decoder, or readout nonlinearity | `modules/05-probabilistic-modeling/notes/graphical-models.md` | Use only after local definition because meanings vary more than for $\phi$. |
| $\eta$ | natural parameter, learning-rate-adjacent scalar, or natural transformation component | `modules/00-math-toolkit/notes/06-ml-categorical-mapping.md` | In category-theory notes, reserve $\eta_A,\eta_B$ for natural transformations. |
| $\ell$ | pointwise loss | `modules/00-math-toolkit/notes/06-ml-categorical-mapping.md` | Reserve $\mathcal{L}$ for aggregate or dataset-level objectives. |
| $\mathcal{L}$ | aggregate objective, negative log-likelihood, or training loss | `modules/00-math-toolkit/solutions/linear-algebra-solutions.md` | Distinguish from $\ell$ in notes that use both. |
| $R(\theta)$ | population risk | `modules/00-math-toolkit/notes/06-ml-categorical-mapping.md` | Keep unhatted $R$ for population or idealized quantities. |
| $\widehat{R}(\theta)$ | empirical risk | `modules/00-math-toolkit/notes/06-ml-categorical-mapping.md` | Reserve hats for empirical estimates. |
| $\lambda$ | regularization weight, eigenvalue, or dual multiplier depending on local section | `modules/00-math-toolkit/derivations/svd-derivation.md` | Redefine explicitly because the symbol has multiple standard mathematical uses. |
| $\mu$ | mean or strong-convexity constant | `modules/00-math-toolkit/notes/03-probability-statistics.md` | Reintroduce when switching from probability to optimization. |
| $\sigma$ | sigmoid, standard deviation, or activation/nonlinearity | `modules/00-math-toolkit/derivations/svd-derivation.md` | Must be redefined locally before use. |
| $A$ | adjacency matrix or generic object in category theory | `modules/00-math-toolkit/notes/01-linear-algebra.md` | In graph notes, capitalize as adjacency matrix and define immediately. |
| $L$ | Laplacian, loss surrogate, or linear map label depending on note | `modules/00-math-toolkit/derivations/chain-rule-matrices.md` | Avoid plain $L$ for losses when $\mathcal{L}$ is already in use. |
| $W^{[\ell]}$ | neural-network weight matrix at layer $\ell$ | `modules/06-neural-networks/derivations/backpropagation.md` | Keep bracketed layer indexing in neural-network material. |
| $b^{[\ell]}$ | neural-network bias at layer $\ell$ | `modules/06-neural-networks/derivations/backpropagation.md` | Pair with $W^{[\ell]}$. |
| $h_t$ | recurrent hidden state at time $t$ | `modules/09-sequence-models/notes/sequence-modeling.md` | Reserve time subscripts for sequential hidden-state updates. |
| $Q$, $K$, $V$ | query, key, and value maps or tensors | `modules/10-transformers-llms/derivations/self-attention.md` | Keep uppercase for transformer notation. |
| $\pi_\theta(a \mid s)$ | policy with parameters $\theta$ | `modules/12-reinforcement-learning/derivations/policy-gradient-theorem.md` | Reserve $\pi$ for policies or categorical projections only when context is clear. |

## Collisions Resolved in This Pass

| Legacy usage | Resolution |
| --- | --- |
| `$D$` as a dataset index object in shared categorical figures | Replaced with `$I$` and documented as deprecated. |
| Workspace-specific absolute markdown links | Replaced with repo-relative links so symbol references remain portable. |
| Optimization SVG palette and serif typography drifting from the shared figure library | Restyled to use the shared figure palette, line weight, and sans-serif font policy. |

## Scope Notes

- This registry is intentionally conservative: it standardizes the symbols most likely to create cross-module confusion.
- Less common derivation-local symbols should still be defined in each note's notation section.
- If a future module needs a durable new symbol family, add it here and cross-link the introducing file.
