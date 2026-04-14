---
title: "Diagram Notation Guide"
module: "shared"
doc_type: "reference"
topic: "figure-notation"
status: "draft"
updated: "2026-04-13"
owner: "curriculum-team"
tags:
  - "style-guide"
  - "category-theory"
  - "diagrams"
---

# Diagram Notation Guide

## Purpose

This guide defines the notation and visual conventions for reusable category-theory figures stored in `shared/figures/category-theory/`. Its role is narrow: keep categorical figures consistent across the Module 00 primer, Module 16 consolidation notes, and related ML structural notes.

## Source of Truth

- Diagram specs live in `shared/figures/category-theory/sources/diagrams.json`.
- Rendered assets live in `shared/figures/category-theory/rendered/`.
- The renderer is `tooling/scripts/render_category_theory_diagrams.py`.

Contributors should edit the JSON source first, then re-render all assets.

## Naming and File Rules

- Use lowercase kebab-case slugs such as `naturality-square` and `empirical-risk-pipeline`.
- Keep one rendered SVG and one rendered PNG per slug.
- Use short titles that name the mathematical pattern, not only the application.
- Prefer reusable labels so the same figure can be cited from more than one note.

## Symbol Conventions

### Core objects

- Use `$X$`, `$Y$`, `$Z$`, and `$H$` for input, output, latent, and representation spaces unless a note already fixes more specific notation.
- Use `$A$`, `$B$`, and `$C$` for generic category-theory examples.
- Use `$D$` for a finite dataset index object, not for the raw dataset as an unordered prose concept.
- Use `$\Theta$` for parameter space.

### Core morphisms

- Use `$f,g,h$` for generic morphisms.
- Use `$\phi$` for feature maps.
- Use `$c$` or `$h$` for downstream prediction heads when a figure is about supervised composition.
- Use `$e$` and `$d$` for encoder and decoder maps.
- Use `$a$` for augmentations or symmetry actions on inputs.
- Use `$\tilde{a}$` for the corresponding latent-space action.
- Use `$\eta_A,\eta_B$` for components of a natural transformation.

### Structural maps

- Use `$\pi_A,\pi_B$` for product projections.
- Use `$\iota_A,\iota_B$` for coproduct inclusions.
- Use `$\mathrm{id}_X$` or `$\mathrm{id}_Y$` for identity morphisms.
- Use `$\ell$` for pointwise loss and `$\mathrm{avg}$` for empirical averaging.

## Diagram Types

### Commutative diagrams

Use node-arrow diagrams when the figure states equality of composites, factorization, equivariance, or universal properties.

Required habits:

- Place objects in rounded rectangles.
- Use the accent color only to highlight the composite or universal map that the note is emphasizing.
- Keep diagrams sparse enough that each commuting condition can be read without a legend.

### String diagrams

Use string-style figures when sequential and parallel composition must both be visually salient, especially for monoidal intuition, residual pathways, or parallel ML branches.

Required habits:

- Wires denote objects or typed channels.
- Boxes denote morphisms or typed operations.
- Vertical or left-to-right order denotes sequential composition.
- Parallel wires denote monoidal or branchwise composition.

## Visual Conventions

- Node fill: warm neutral background to distinguish objects from page background.
- Main edge and text color: dark blue for ordinary maps and labels.
- Accent color: muted rust for the primary composite, residual path, or emphasized universal map.
- Guide text: gray-blue for captions and secondary connectors.
- Stroke weight: consistent across all figures in the rendered library.

Avoid:

- adding multiple accent colors in a single diagram;
- switching between `$H$` and `$Z$` for the same role inside one figure;
- mixing generic category notation and application-specific notation without a reason stated in the nearby prose.

## When to Use Which Figure

- Use `basic-composition` and `feature-pipeline` for introductory composition.
- Use `commutative-triangle`, `augmentation-label-square`, and `representation-equivariance` for commuting conditions.
- Use `product-universal-property` and `coproduct-universal-property` for universal property discussions.
- Use `functor-composition` and `naturality-square` for functorial and naturality claims.
- Use `empirical-risk-pipeline`, `attention-factorization`, `message-passing-square`, and `agent-pipeline` for ML pipeline illustrations.
- Use `monoidal-parallel` and `residual-block-string` when parallel composition or bypass structure matters more than objectwise commuting equalities.

## Contributor Workflow

1. Add or modify a diagram spec in `shared/figures/category-theory/sources/diagrams.json`.
2. Reuse the symbol conventions in this guide unless the destination note has already fixed different symbols.
3. Run `python tooling/scripts/render_category_theory_diagrams.py`.
4. Verify that both `svg` and `png` outputs are created in `shared/figures/category-theory/rendered/`.
5. Add a file-path reference in the note that uses the diagram.

## Current Library Coverage

The current shared library covers:

- basic categories and composition;
- commuting triangles and squares;
- products and coproducts;
- functors and natural transformations;
- monoidal and string-diagram intuition;
- supervised pipelines, empirical risk, and encoder-decoder factorizations;
- equivariance, attention, message passing, and agent workflow structure.
