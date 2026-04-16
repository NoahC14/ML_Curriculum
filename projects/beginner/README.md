# Beginner Projects

This track implements Card 11.1 by packaging small, self-contained projects for learners finishing Parts I and II of the curriculum. Every project is feasible with Modules 00-05 only and is designed to connect mathematical formulation, implementation, and written interpretation.

## Track design goals
- keep datasets small, tabular, and low-friction;
- reinforce linear regression, logistic regression, regularization, and evaluation;
- move from highly guided work toward semi-open comparative analysis; and
- keep each project within a documented `4-8` hour window.

## Project sequence

| Project | Focus | Level | Estimated time | Primary modules |
|---|---|---|---|---|
| [`housing-price-prediction/`](./housing-price-prediction/README.md) | Linear regression, residual analysis, feature interpretation | Guided | `4-5` hours | `00-03` |
| [`binary-classification-evaluation/`](./binary-classification-evaluation/README.md) | Logistic regression, thresholds, precision/recall, ROC-AUC | Guided | `4-6` hours | `01-04` |
| [`regularization-comparison/`](./regularization-comparison/README.md) | Ridge vs lasso, optimization behavior, coefficient stability | Structured | `5-7` hours | `01-04` |
| [`discriminative-vs-generative/`](./discriminative-vs-generative/README.md) | Logistic regression, SVM, Naive Bayes, GMM comparison | Semi-open | `6-8` hours | `02-05` |

## Shared expectations
- Each project includes `README.md`, `template.ipynb`, and `rubric.md`.
- Rubrics emphasize correctness, analysis quality, and written interpretation.
- Students should embed concise written analysis directly in the notebook unless the spec requests a separate memo.
- Projects should use reproducible splits, fixed random seeds, and explicit metric reporting.

## Dataset policy
Beginner projects intentionally avoid heavy preprocessing, external data wrangling, or compute-heavy pipelines. Where possible, they use synthetic data or small `scikit-learn` datasets so the learning focus stays on modeling decisions rather than data cleaning overhead.
