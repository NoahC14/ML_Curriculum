---
title: "Category Theory Exercises"
module: "00-math-toolkit"
lesson: "category-theory-exercises"
doc_type: "exercise"
topic: "category-theory"
status: "draft"
prerequisites:
  - "00-math-toolkit/category-theory-primer"
  - "00-math-toolkit/linear-algebra"
updated: "2026-04-09"
owner: "curriculum-team"
tags:
  - "category-theory"
  - "exercises"
  - "composition"
  - "ml-foundations"
---

## Purpose

These exercises are organized to move from core definitions to ML-facing interpretation:

- Tier 1: definition fluency and standard examples;
- Tier 2: diagrams, universal properties, and coherence;
- Tier 3: ML synthesis.

## Exercise 1: Verify the category axioms in `Set`

> **Problem.** Let $A$, $B$, $C$, and $D$ be sets, and let
> $$
> f : A \to B, \qquad g : B \to C, \qquad h : C \to D
> $$
> be functions.
> Prove that function composition is associative and that identity functions satisfy the left and right identity laws.

**Hints**

- Evaluate both composites on an arbitrary element $a \in A$.
- Write out the identity function explicitly.

**Deliverables**

- A short proof.

## Exercise 2: Linear maps as morphisms in `Vect_\mathbb{R}`

> **Problem.** Let $T : \mathbb{R}^2 \to \mathbb{R}^3$ and $S : \mathbb{R}^3 \to \mathbb{R}^2$ be linear maps represented by matrices
> $$
> T(\mathbf{x}) = \mathbf{A}\mathbf{x},
> \qquad
> S(\mathbf{y}) = \mathbf{B}\mathbf{y}.
> $$
> Show that $S \circ T$ is also linear, and identify its matrix.

**Hints**

- Start from $(S \circ T)(\mathbf{x}) = S(T(\mathbf{x}))$.
- Use the definition of matrix multiplication.

**Deliverables**

- A derivation.
- One sentence relating this to stacked linear layers.

## Exercise 3: A non-example from groups

> **Problem.** Give an example of a function between groups that is not a group homomorphism. Explain precisely which structure-preservation condition fails.

**Hints**

- Compare $f(xy)$ with $f(x)f(y)$.

**Deliverables**

- A concrete example.
- A one-paragraph explanation.

## Exercise 4: Read a commutative diagram as an equation

> **Problem.** Consider the square
> $$
> \begin{CD}
> X @>f>> Y \\
> @VhVV   @VVgV \\
> Z @>k>> W
> \end{CD}
> $$
> Write the equation expressing that this diagram commutes. Then explain in plain language what that equation means.

**Hints**

- Compare the two paths from $X$ to $W$.

**Deliverables**

- The algebraic equation.
- A plain-language interpretation.

## Exercise 5: Product universal property in `Set`

> **Problem.** Let $X$, $A$, and $B$ be sets, with maps
> $$
> f : X \to A, \qquad g : X \to B.
> $$
> Define a map $\langle f, g \rangle : X \to A \times B$, and prove that it satisfies the product equations with the projections $\pi_A$ and $\pi_B$.

**Hints**

- Define $\langle f, g \rangle(x)$ componentwise.
- Then check the projection identities directly.

**Deliverables**

- The definition of $\langle f, g \rangle$.
- A short proof.

## Exercise 6: Coproduct case analysis

> **Problem.** Suppose a dataset consists of either an image example or a tabular example. Explain how a disjoint-union viewpoint $A \sqcup B$ models this situation. Then describe what data are required to define a map from $A \sqcup B$ into a common label space $Y$.

**Hints**

- Think branchwise: what should happen on the image branch, and what should happen on the tabular branch?

**Deliverables**

- A short prose explanation.
- A diagram or symbolic description.

## Exercise 7: Functor sanity check

> **Problem.** Let $F : \mathrm{Vect}_{\mathbb{R}} \to \mathrm{Set}$ be the forgetful functor sending each vector space to its underlying set and each linear map to its underlying function.
> Show that $F$ preserves identities and composition.

**Hints**

- The underlying function of an identity linear map is an identity function.
- The underlying function of a composite linear map is the composite of the underlying functions.

**Deliverables**

- A short proof.

## Exercise 8: Naturality square

> **Problem.** Let $F, G : \mathcal{C} \to \mathcal{D}$ be functors, and let $\eta : F \Rightarrow G$ be a natural transformation. Write the naturality condition for a morphism $f : A \to B$ and explain why it says the family $\eta_A$ is coherent rather than arbitrary.

**Hints**

- There are two paths from $F(A)$ to $G(B)$.

**Deliverables**

- The naturality equation.
- A short explanation in prose.

## Exercise 9: Feature pipeline as composition

> **Problem.** Let
> $$
> n : X \to X', \qquad \phi : X' \to H, \qquad c : H \to Y
> $$
> model normalization, feature extraction, and classification.
> Write the full predictor as one composite map. Then explain what associativity tells you about how to group these stages in code or analysis.

**Hints**

- There is only one valid order of application.
- Associativity is about parenthesization, not about reordering.

**Deliverables**

- The composite formula.
- A short interpretation.

## Exercise 10: Label-preserving augmentation

> **Problem.** Let $a : X \to X$ be an augmentation map and $y : X \to \mathcal{Y}$ be a label function. Write the commutative square expressing label preservation. Then explain why this matters for supervised learning.

**Hints**

- Compare $y(x)$ with $y(a(x))$.

**Deliverables**

- The commuting equation.
- A short explanation.

## Exercise 11: Encoder-decoder factorization

> **Problem.** An autoencoder factors a reconstruction map as
> $$
> X \xrightarrow{e} Z \xrightarrow{d} X.
> $$
> Explain what is gained conceptually by introducing the latent object $Z$ rather than treating the reconstruction as a single opaque end-to-end map.

**Hints**

- Think about compression, bottlenecks, and interpretability.

**Deliverables**

- A one-paragraph answer.

## Exercise 12: Transfer learning diagram

> **Problem.** Let
> $$
> f_{\mathrm{src}} : X_{\mathrm{src}} \to H,
> \qquad
> t : H \to H',
> \qquad
> c_{\mathrm{tgt}} : H' \to Y_{\mathrm{tgt}}.
> $$
> Draw or describe the composite map used for transfer learning. Then state one reason why the intermediate adaptation map $t$ may be necessary.

**Hints**

- Source and target tasks may not use exactly the same representation space.

**Deliverables**

- The composite formula.
- One concrete reason for including $t$.

## Exercise 13: Monoidal intuition in a two-branch model

> **Problem.** A multimodal model processes an image $x_{\mathrm{img}}$ and a text input $x_{\mathrm{text}}$ in parallel. Explain, at an intuitive level, how a product-like or tensor-like operation captures "parallel first, combine later."

**Hints**

- Distinguish sequential composition from parallel composition.

**Deliverables**

- A short prose explanation.

## Exercise 14: Scope and limits

> **Problem.** Give two examples of ML questions that categorical language helps clarify, and two examples of ML questions that still require other mathematical tools such as optimization, probability, or numerical analysis.

**Hints**

- One good split is "what composes?" versus "what is statistically or numerically valid?"

**Deliverables**

- Four short items, each with one sentence of explanation.
