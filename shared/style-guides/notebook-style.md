# Notebook Style Guide

## Purpose
Standardize educational notebooks so they are readable, reproducible, and structurally consistent across modules.

## Scope
Apply this guide to notebooks in module `notebooks/`, project notebooks, and exploratory notebooks that are intended to remain in the repository.

## Notebook Naming
Use the filename pattern:

```text
NB-{module}-{topic}-{sequence}.ipynb
```

Examples:
- `NB-00-linear-algebra-vectors-01.ipynb`
- `NB-03-linear-regression-normal-equation-02.ipynb`
- `NB-10-transformer-attention-mechanics-01.ipynb`

### Naming rules
- `module`: two-digit module number such as `00`, `03`, or `10`
- `topic`: short kebab-case topic slug
- `sequence`: two-digit sequence number within that topic

Use a new sequence number when there are multiple notebooks for the same topic, not when making minor revisions.

## Required Notebook Flow
Every teaching notebook should run top-to-bottom and follow this cell order.

1. Title and metadata markdown cell
2. Learning goals markdown cell
3. Setup code cell
4. Optional imports and configuration explanation markdown cell
5. Alternating narrative and code cells for the main development
6. Interpretation or discussion markdown cell after any major result
7. Closing summary markdown cell
8. Optional exercises or next steps markdown cell

Do not begin a notebook with unexplained code.

## Opening Cell Template
The first markdown cell should include:
- notebook title
- module and topic
- estimated runtime
- prerequisites
- expected outputs

Example:

```md
# Gradient Descent on a Quadratic Objective

- Module: 01 Optimization
- Topic: first-order optimization
- Estimated runtime: 5 minutes
- Prerequisites: multivariable calculus, vectors and matrices, Python basics
- Expected outputs: loss curve, parameter trajectory, comparison of learning rates
```

## Narrative Cell Density
Keep the notebook explanatory without turning it into a wall of text.

### Rules of thumb
- Include a markdown cell before each major code block or conceptual shift.
- Keep most markdown cells to 3-8 short lines.
- Split long mathematical exposition into multiple cells.
- After plots, metrics, or tables, add a short interpretation cell unless the takeaway is obvious and already stated.

Use notebooks to support understanding, not to replace the more durable treatment in `notes/` or `derivations/`.

## Code Cell Style
Code cells should be small, purposeful, and runnable in order.

### Rules
- One conceptual action per cell where possible.
- Keep imports near the top.
- Avoid hidden state across distant cells.
- Prefer deterministic behavior by setting seeds when randomness matters.
- Move reusable helpers into module `src/` when they exceed notebook-local scope.

### Optional code cell headers
If a notebook is long or has multiple phases, start a code cell with a short comment:

```python
# Generate a synthetic regression dataset
```

Do not add decorative comments to every cell.

## Output Expectations
Outputs should support pedagogy and reproducibility.

### Keep outputs when they are:
- small numeric summaries
- key plots
- short tables
- brief sanity checks

### Clear or avoid outputs when they are:
- long debug dumps
- repeated package installation logs
- oversized data previews
- noisy progress bars that obscure the main lesson

### Plot expectations
- label axes
- include titles when the plot meaning is not already obvious
- keep legends readable
- ensure random examples are reproducible when possible

## Math in Notebooks
- Use markdown math conventions from `shared/style-guides/markdown-style.md`.
- Put derivation-heavy content in markdown cells, not code comments.
- If a derivation exceeds a few cells, move the full derivation to `derivations/` and link to it.

## Metadata and Kernel Contract
Repository notebooks should preserve valid `nbformat` metadata and declare the intended kernel.

### Required metadata expectations
- `nbformat` and `nbformat_minor` must be present
- `metadata.kernelspec.name` must be set
- `metadata.kernelspec.display_name` must be set
- `metadata.language_info.name` should be `python`

### Preferred kernel defaults
Use Python 3 unless a notebook explicitly requires another kernel.

Example metadata shape:

```json
{
  "kernelspec": {
    "display_name": "Python 3",
    "name": "python3"
  },
  "language_info": {
    "name": "python"
  }
}
```

Do not commit notebooks with missing kernelspec metadata.

## Reproducibility Rules
- A fresh kernel run should succeed top-to-bottom.
- Any required data source must be documented near the setup cells.
- Any package dependency outside the base environment must be stated explicitly.
- If execution time is long, note the expected runtime and lighter fallback path.

## When To Split a Notebook
Split a notebook when:
- it exceeds one coherent lesson goal
- setup and theory dominate the notebook more than the experiment itself
- output volume becomes hard to review
- reusable functions are carrying most of the logic

In those cases, move durable exposition to markdown docs and durable code to `src/`.

## Contributor Guidance
- Keep notebook rules minimal and practical.
- Favor clarity over cleverness.
- Treat notebooks as teaching artifacts first and exploratory scratchpads second once committed.
