---
title: "Figure Consistency Audit"
module: "shared"
doc_type: "reference"
topic: "figure-consistency-audit"
status: "draft"
updated: "2026-04-15"
owner: "curriculum-team"
tags:
  - "style-guide"
  - "figures"
  - "audit"
---

# Figure Consistency Audit

## Scope

This audit covers the shared figure assets under `shared/figures/` and the module markdown files that reference them.
It also records cross-reference defects discovered while checking figure citations because broken figure links and workspace-bound paths were the main consistency failures in the current repo state.

## Asset Sets Reviewed

| Asset set | Files reviewed | Result |
| --- | --- | --- |
| `shared/figures/category-theory/rendered/` | shared categorical SVG and PNG exports | Pass after notation normalization for the empirical-risk pipeline source. |
| `shared/figures/optimization/` | `convex-set-segment.svg`, `convex-function-chord.svg` | Updated to match shared palette, font policy, and line-weight conventions. |
| Markdown figure references | Module 00, 01, 13, 16, and 17 references found by repo-wide search | Corrected stale absolute paths and one missing reciprocal note link. |

## Findings and Resolutions

| ID | Finding | Status | Resolution |
| --- | --- | --- | --- |
| FIG-01 | Optimization SVGs used a different palette, heavier line weights, and serif typography than the shared category-theory library. | Fixed | Restyled both optimization figures to use the shared dark-blue/rust/gray-blue palette, default sans-serif font stack, and `1.8` stroke width. |
| FIG-02 | The empirical-risk shared diagram used `$D$` for a sample index object, colliding with graph-learning use of `$D$` for the degree matrix. | Fixed | Updated the source spec and notation guide to use `$I$` for finite sample indices and `$\mathcal{D}$` for datasets. |
| REF-01 | `modules/17-unity-theory-perspectives/notes/scope-memo.md` linked to a workspace-specific absolute path. | Fixed | Replaced with a repo-relative link to `shared/style-guides/content-boundary.md`. |
| REF-02 | `modules/17-unity-theory-perspectives/notes/glossary.md` contained repeated workspace-specific absolute links. | Fixed | Replaced all companion-note references with stable repo-relative links. |
| REF-03 | `modules/13-graph-learning/notes/graph-learning.md` linked to `spectral-graph-methods.md` via a stale absolute path. | Fixed | Replaced with `./spectral-graph-methods.md`. |
| REF-04 | The graph-learning pair had only a one-way explicit note link from `graph-learning.md` to `spectral-graph-methods.md`. | Fixed | Added the reciprocal link in `spectral-graph-methods.md` back to `graph-learning.md`. |

## Current Shared Figure Style

All reusable figures should now follow these style tokens:

- Main stroke and primary text: `#1F3A5F`
- Accent stroke/text: `#B35C44`
- Guide text: `#7A8796`
- Warm neutral object fill: `#F6F3EA`
- Default reusable stroke width: `1.8`
- Font policy: default exported sans-serif stack, not local serif styling

## Remaining Risks

- The category-theory rendered assets should be regenerated whenever `shared/figures/category-theory/sources/diagrams.json` changes so the PNG and SVG exports stay synchronized with the source spec.
- A few non-figure markdown docs still use custom admonition syntax; that is a markdown-style issue rather than a figure-style issue and was left outside this pass.

## Acceptance Check

- No known shared figure currently mixes a separate palette or font family.
- The main figure-reference defects found in this pass are corrected.
- The notation collision between categorical dataset indexing and graph degree matrices is resolved at the shared-guide level.
