# Rubric: Regularization Comparison Study

Total: 100 points

## 1. Experimental setup and correctness - 20 points
- `17-20`: The synthetic dataset, splits, baseline, and regularized models are all implemented correctly and reproducibly.
- `12-16`: The setup is mostly correct, with only modest gaps in documentation or completeness.
- `6-11`: The setup is partial or contains major issues in model comparison.
- `0-5`: The workflow is missing or substantially incorrect.

## 2. Regularization comparison quality - 25 points
- `22-25`: OLS, ridge, and lasso are compared fairly, with sensible hyperparameter choices and clear reporting.
- `16-21`: The comparison is mostly complete but one important family, setting, or fairness control is thin.
- `8-15`: The comparison is partial or weakly justified.
- `0-7`: The comparison is too incomplete to support conclusions.

## 3. Analysis quality - 25 points
- `22-25`: The notebook clearly explains generalization tradeoffs, coefficient shrinkage, sparsity, and signs of overfitting.
- `16-21`: The analysis is mostly correct but somewhat generic or uneven.
- `8-15`: The analysis is mostly descriptive, with limited theoretical connection.
- `0-7`: The analysis is missing or unsupported.

## 4. Written interpretation - 20 points
- `17-20`: The writeup makes a precise, evidence-based recommendation about when to use OLS, ridge, or lasso.
- `12-16`: The recommendation is reasonable but only partly justified.
- `6-11`: The interpretation is vague or weakly tied to evidence.
- `0-5`: The interpretation is missing or substantially unsupported.

## 5. Communication and reproducibility - 10 points
- `9-10`: The notebook is organized, readable, and uses fixed seeds with clear tables or plots.
- `6-8`: Mostly readable with minor clarity or reproducibility issues.
- `3-5`: Weak organization or multiple missing reproducibility details.
- `0-2`: Not meaningfully readable.

## Minimum completeness conditions
The submission cannot earn above `69/100` unless it:

- compares OLS, ridge, and lasso on the same dataset split;
- reports both predictive error and parameter-behavior evidence; and
- includes a written interpretation section.
