# Rubric: Housing Price Prediction

Total: 100 points

## 1. Problem framing and data understanding - 15 points
- `13-15`: The notebook clearly states the prediction task, interprets feature meanings, and summarizes the generated dataset accurately.
- `9-12`: The framing is mostly correct, with only minor gaps in feature interpretation or dataset summary.
- `5-8`: The task is only partially framed or the dataset summary is incomplete.
- `0-4`: The framing is missing or substantially incorrect.

## 2. Model implementation and correctness - 25 points
- `22-25`: The baseline and linear regression models are implemented correctly, use a coherent split, and produce valid outputs.
- `16-21`: The core models are present and mostly correct, with modest issues in setup or completeness.
- `8-15`: The workflow is only partly correct or omits an important required model.
- `0-7`: The implementation is missing or substantially incorrect.

## 3. Evaluation and analysis quality - 25 points
- `22-25`: Required regression metrics and plots are present, interpreted correctly, and used to support a clear diagnosis of model behavior.
- `16-21`: Evaluation is mostly complete, but one metric, plot, or interpretation step is thin.
- `8-15`: Evaluation is partial or mostly descriptive.
- `0-7`: Evaluation is too incomplete to support conclusions.

## 4. Written interpretation - 25 points
- `22-25`: The writeup explains coefficients, residual behavior, limitations, and next steps in precise course-aligned language.
- `16-21`: Interpretation is sensible but somewhat generic or weakly tied to evidence.
- `8-15`: Interpretation is shallow or only loosely connected to the results.
- `0-7`: Interpretation is missing or unsupported.

## 5. Communication and reproducibility - 10 points
- `9-10`: The notebook is organized, readable, uses fixed seeds, and documents the workflow clearly.
- `6-8`: The notebook is generally readable but has minor clarity or reproducibility issues.
- `3-5`: Organization is weak or several reproducibility details are missing.
- `0-2`: The submission is difficult to follow or not reproducible.

## Minimum completeness conditions
The submission cannot earn above `69/100` unless it:

- compares a constant baseline with at least one learned linear model;
- reports all required regression metrics; and
- includes a written interpretation section.
