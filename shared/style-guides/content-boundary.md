# Content Boundary

## Purpose
This guide defines how the course separates standard machine learning content from interpretive or speculative companion material. Its function is governance, not decoration: a reader or reviewer should be able to inspect any page and immediately tell which claims are canonical, which are structural commentary, and which are original to this curriculum.

## Core policy
- Canonical ML exposition comes first.
- Standard material must never be presented as if it depends on Unity Theory or a novel categorical reading.
- Category theory may clarify composition, abstraction, invariance, and transfer, but it does not replace standard mathematics or empirical argument.
- Unity Theory is companion material and must be labeled as interpretive, exploratory, or speculative when used.
- When in doubt, over-label the non-canonical material.

## Content classes

### 1. Canonical
Use for material that is standard, accepted, and teachable from conventional ML, statistics, optimization, linear algebra, probability, or computer science sources.

Typical examples:
- definitions of risk, loss, gradient, estimator, likelihood, and posterior
- standard derivations for linear regression, logistic regression, backpropagation, and attention
- accepted claims about model classes, optimization methods, and evaluation protocols
- ordinary mathematical facts used in the course

Required label:
- `Canonical`

Citation expectation:
- cite standard textbooks, lecture notes, papers, or other established sources when the claim is not elementary enough to stand without citation

### 2. Structural clarification
Use for category-theoretic or abstraction-oriented commentary that restates standard content without changing the underlying claim.

Typical examples:
- describing an ML pipeline as a composition of maps
- treating invariance as a preservation statement
- using objects, morphisms, or diagrams to clarify an already-standard construction

Required label:
- `Structural note`

Rule:
- the underlying mathematical result must still be stated in conventional ML or mathematical language nearby

Citation expectation:
- cite the canonical ML source for the underlying concept
- cite a category-theory source only if nontrivial categorical terminology is used

### 3. Interpretive correspondence
Use for crosswalks between canonical ML material and Unity Theory or a nonstandard categorical interpretation that is meant to illuminate, not assert equivalence.

Typical examples:
- relating representation stability to coherence
- reading identity and relation into a latent-space discussion
- proposing a disciplined conceptual mapping between an ML object and a Unity Theory term

Required label:
- `Interpretive note`

Rule:
- state the mapping explicitly and name both sides of it
- say that the passage is interpretive, companion, or heuristic
- do not imply that the mapping is part of standard ML literature unless it is actually sourced as such

Citation expectation:
- cite the canonical ML source for the underlying model, theorem, or experiment
- cite the internal Unity Theory or companion note separately if the interpretation is original to this course

### 4. Exploratory or speculative extension
Use for original hypotheses, research prompts, conjectural frameworks, or stronger claims that go beyond accepted literature.

Typical examples:
- proposing a new compositional learning framework
- suggesting a novel categorical equivalence not established in the literature
- extending Unity Theory language into a research claim about optimization, agency, or representation

Required label:
- `Exploratory note` or `Speculative research note`

Rule:
- place this material after the canonical exposition, usually in a boxed section, sidebar, appendix, or `unity/` note
- do not mix speculative claims into theorem statements, core derivations, or assessment prompts unless the assignment is explicitly a research exercise

Citation expectation:
- separate sourced facts from original conjecture
- do not attach a standard citation in a way that makes the original claim look established
- if no external support exists, say that the proposal is original, exploratory, or a research direction

## Labeling conventions
Use the following labels verbatim unless a module has a stricter local convention:

- `Canonical`
- `Structural note`
- `Interpretive note`
- `Exploratory note`
- `Speculative research note`

Recommended usage pattern:
- main lesson text, derivations, theorem statements, algorithm definitions, and implementation instructions default to `Canonical`
- category-theoretic commentary inside a standard lesson should usually be `Structural note`
- Unity Theory crosswalks inside a standard lesson should usually be `Interpretive note`
- stronger original proposals belong in `Exploratory note` or `Speculative research note`

## Visual and typographic cues
The repository is markdown-first, so boundary cues must work in plain text before any future site styling exists.

### Canonical material
- keep in the main body under ordinary section headings
- do not place canonical material in special callout boxes unless the entire lesson uses callouts consistently
- theorem-like statements, derivations, worked examples, and labs should normally remain visually plain

### Non-canonical companion material
Use a visibly separate block with an explicit label at the top. Any of the following are acceptable:

```md
> [!NOTE]
> **Interpretive note.** This is a companion reading, not part of the canonical derivation.
```

```md
> [!TIP]
> **Structural note.** Categorical language is being used here to clarify composition.
```

```md
> [!WARNING]
> **Speculative research note.** This proposal is exploratory and is not established ML doctrine.
```

Presentation rules:
- the label must appear in bold at the start of the block
- use a distinct block, not an inline aside buried in a paragraph
- do not place interpretive or speculative content under headings that sound canonical, such as `Theorem`, `Definition`, `Proof`, or `Algorithm`, unless the heading itself is explicitly marked
- for longer companion passages, prefer a subsection titled with the label, such as `## Interpretive note: identity and relation in latent spaces`

## Citation rules at the boundary

### When material is canonical
- cite standard sources in the normal way
- avoid citing internal Unity or companion notes as if they establish the canonical claim

### When a section crosses from canonical to interpretive
- first state the canonical result or mechanism
- provide the standard citation for that result
- then mark the transition with `Interpretive note` or `Structural note`
- if the interpretation is original to this course, cite the local companion artifact separately or state that the framing is original to this curriculum

### When a section is speculative
- identify exactly which statements are sourced and which are conjectural
- use phrases such as `we propose`, `one possible interpretation`, `exploratory hypothesis`, or `research direction`
- never write a speculative claim in the same citation sentence as an established result unless the distinction is explicit

### Citation anti-patterns
- do not cite a standard ML paper next to an original Unity Theory interpretation in a way that implies endorsement
- do not present analogy, metaphor, or vocabulary alignment as a theorem
- do not rely on category-theory terminology alone as evidence for an ML claim

## Placement rules
- Core notes, derivations, labs, and assessments should be overwhelmingly canonical.
- Companion material may appear in sidebars, appendices, comparison tables, `unity/` directories, or explicitly marked sections.
- Module 00 may use category theory in primer mode, but still must distinguish standard mathematics from structural commentary.
- Module 16 may use more formal category theory, but should still mark which claims are standard category theory and which are course-specific interpretations.
- Module 17 may contain more interpretive and speculative material, but it still must distinguish sourced claims from original synthesis.

## Reviewer checklist
Use this checklist when reviewing any page, notebook, lesson, derivation, or module README.

- Can a reader identify the content class of every nontrivial section within a few seconds?
- Is all standard ML content presented in ordinary canonical language before companion framing appears?
- Are Unity Theory passages labeled `Interpretive note`, `Exploratory note`, or `Speculative research note` as appropriate?
- If category theory is used, is it serving structural clarification rather than replacing the standard derivation?
- Are mappings between ML concepts and Unity Theory concepts stated explicitly on both sides?
- Are theorem statements, definitions, algorithms, and proofs free of unlabeled speculative language?
- Are citations split cleanly between established sources and original course interpretations?
- Could a skeptical subject-matter reviewer tell what is accepted literature and what is original to this course?

## Audit test for acceptance
A page passes this governance standard only if all of the following are true:

- standard, accepted material is clearly recognizable as standard
- Unity Theory and novel categorical content are always labeled as interpretive, exploratory, or speculative
- a reviewer can scan the page and immediately see where the canonical lesson ends and the companion layer begins

## Short author checklist
Before publishing a page, ask:

1. Have I stated the canonical concept in standard terms first?
2. Have I marked every non-canonical passage with one of the approved labels?
3. Have I separated standard citations from original interpretive claims?
4. Would this page still read as a credible ML lesson if the companion blocks were removed?
