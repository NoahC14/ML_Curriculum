# Category Theory Figures

This directory stores reusable categorical diagrams used in:

- `modules/00-math-toolkit/notes/05-category-theory-primer.md`
- `modules/00-math-toolkit/notes/06-ml-categorical-mapping.md`
- `modules/16-category-theory-for-ml/notes/categories-functors-nts.md`
- `modules/16-category-theory-for-ml/notes/diagrammatic-reasoning.md`

## Structure

- `sources/diagrams.json`: declarative source of truth for all diagram layouts and labels.
- `rendered/`: generated `svg` and `png` files.

## Rendering

Run:

```powershell
python tooling/scripts/render_category_theory_diagrams.py
```

The renderer reads `sources/diagrams.json` and writes both SVG and PNG files into `rendered/`.
