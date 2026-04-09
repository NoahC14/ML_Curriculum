# Markdown Style Guide

## Purpose
Standardize markdown used for notes, derivations, exercises, solutions, references, and planning documents without over-specifying contributor workflow.

## Scope
Apply this guide to:
- module `README.md` files
- notes in `notes/`
- derivations in `derivations/`
- exercise and solution writeups
- syllabus and shared reference documents

If a file type needs stricter local rules, keep those additions narrow and document them in the nearest `README.md`.

## Minimal Rules
- Prefer short sections with explicit transitions.
- Define symbols before using them.
- Keep standard ML exposition canonical-first.
- Use category theory only where it clarifies structure.
- Label Unity Theory material as companion, interpretive, exploratory, or speculative.
- Prefer one idea per subsection over long omnibus sections.

## Front Matter
Use YAML front matter at the top of all substantial instructional markdown files. Short placeholder `README.md` files may omit it until they become real content.

### Required fields
- `title`
- `module`
- `doc_type`
- `status`

### Recommended fields
- `lesson`
- `topic`
- `prerequisites`
- `updated`
- `owner`
- `tags`

### Template
```yaml
---
title: "Gradient Descent as Iterated Optimization"
module: "01-optimization"
lesson: "gradient-descent-basics"
doc_type: "notes"
topic: "gradient-descent"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
  - "00-math-toolkit/multivariable-calculus"
updated: "2026-04-09"
owner: "curriculum-team"
tags:
  - "optimization"
  - "first-order-methods"
---
```

### Field conventions
- `module`: use the repository module slug such as `03-linear-models`.
- `doc_type`: use one of `readme`, `notes`, `derivation`, `exercise`, `solution`, `reference`, `planning`, or `unity-note`.
- `status`: prefer `stub`, `draft`, `reviewed`, or `final`.
- `prerequisites`: list repo-relative concept or lesson identifiers, not prose paragraphs.
- `updated`: use ISO date format `YYYY-MM-DD`.

See `shared/style-guides/examples/markdown-front-matter-example.md` for a complete example.

## Heading Hierarchy
Use headings as a semantic outline, not as visual decoration.

- `#` document title only when front matter is followed by a visible title section
- `##` primary sections
- `###` subsections
- `####` rare, only when a subsection genuinely needs internal structure

Avoid skipping heading levels. Most instructional documents should stop at `###`.

### Recommended section patterns
For notes:
- Motivation
- Assumptions and Notation
- Main Development
- Worked Example
- Computational Interpretation
- Limitations or Scope Notes

For derivations:
- Goal
- Assumptions and Notation
- Derivation
- Result
- ML Relevance
- Limitations

For exercises:
- Problem
- Hints
- Deliverables

## Theorem-Like Blocks
Use bold labels with a blockquote for theorem-like content so the style stays portable across markdown renderers.

### Standard forms
```md
> **Definition.** A function `f : R^d -> R` is convex if for all `x, y in R^d` and `t in [0, 1]`,
> `f(tx + (1 - t)y) <= tf(x) + (1 - t)f(y)`.
```

```md
> **Proposition.** If `f` is differentiable and convex, then
> `f(y) >= f(x) + \nabla f(x)^\top (y - x)` for all `x, y`.
```

```md
> **Proof Sketch.** Apply the definition of convexity to the line segment from `x` to `y`,
> differentiate with respect to the segment parameter, and evaluate at zero.
```

### Allowed labels
- `Definition`
- `Proposition`
- `Theorem`
- `Lemma`
- `Corollary`
- `Proof`
- `Proof Sketch`
- `Example`
- `Remark`
- `Exercise`
- `Warning`

### Style rules
- Start with the label in bold, followed by a period.
- Keep each block focused on one statement.
- Use `Proof Sketch` unless a full proof is required for learning goals.
- Follow a theorem-like block with a plain-language interpretation when the result is easy to misread.

## Math Conventions
Use standard LaTeX syntax that renders in common markdown math engines.

### Inline math
Use single dollar delimiters for short expressions inside prose.

```md
The gradient $\nabla f(x)$ points in the direction of steepest ascent.
```

Rules:
- Keep inline math short.
- Do not place full derivation steps inline.
- Introduce symbols in surrounding prose before reusing them repeatedly.

### Display math
Use double dollar delimiters for standalone equations.

```md
$$
\hat{y} = Xw + b\mathbf{1}
$$
```

Use display math for:
- objective functions
- update rules
- probability factorizations
- equations that need room for readable notation

### Aligned derivations
Use `aligned` inside display math when showing multi-step reasoning.

```md
$$
\begin{aligned}
L(w) &= \frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)^2 \\
     &= \frac{1}{n}\|Xw - y\|_2^2
\end{aligned}
$$
```

Rules:
- Align on `=`, `<=`, or other main operators.
- One logical transformation per line.
- Omit low-value algebra only when the intended audience can reasonably fill it in.

### Numbered equations
Use `equation` for equations you will reference later in the same document.

```md
\begin{equation}
\nabla_w L(w) = \frac{2}{n}X^\top(Xw - y)
\label{eq:linear-regression-gradient}
\end{equation}
```

Rules:
- Number only equations that are referenced later.
- Use stable labels with prefixes such as `eq:`, `def:`, or `thm:`.
- Refer to equations in prose as `Equation {eq:linear-regression-gradient}` only if the renderer supports label resolution; otherwise write "the gradient equation above".

### General notation rules
- Use `\mathbb{R}`, `\mathbb{E}`, `\mathrm{Var}`, and similar standard operators where appropriate.
- Make dimensions explicit when they matter, such as `X \in \mathbb{R}^{n \times d}`.
- Prefer `\top` over raw transpose apostrophes.
- Prefer semantic notation over visually clever notation.

## Code and Paths
- Wrap inline code, filenames, paths, identifiers, and environment variables in backticks.
- Keep code fences language-tagged when possible, such as `python`, `bash`, `yaml`, or `text`.
- Move reusable code out of prose docs and into `src/`, then link to it.

## Citations
Use a lightweight author-year style in prose with a matching reference entry in the local references section or bibliography file.

### In-text citation pattern
```md
(Bishop, 2006)
(Goodfellow, Bengio, and Courville, 2016)
```

### Reference entry pattern
```md
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Goodfellow, I., Bengio, Y., and Courville, A. (2016). *Deep Learning*. MIT Press.
```

Rules:
- Do not invent citations.
- Prefer primary sources for technical claims and standard textbooks for canonical background.
- Keep full bibliographic normalization for `shared/bibliography/` as a later tooling concern.

## Cross-References
Prefer repo-relative markdown links so files remain navigable in GitHub and local editors.

### File links
```md
See [Module 01 README](../../modules/01-optimization/README.md).
```

### Section links
```md
See [Assumptions and Notation](#assumptions-and-notation).
```

### Cross-reference rules
- Use descriptive link text, not raw paths as prose.
- Link to the closest stable artifact rather than repeating content.
- When referring across modules, mention both the module number and concept name in surrounding prose.

## Contributor Guidance
- Optimize for consistency, clarity, and mathematical readability.
- Do not add custom admonition syntaxes that depend on a specific site generator unless the repo adopts one explicitly.
- If a rule here adds friction without improving readability or reuse, keep the simpler option and document the exception locally.
