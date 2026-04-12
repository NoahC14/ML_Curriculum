# Rubric: Discriminative vs Generative Models

Total: 100 points

## 1. Experimental setup and reproducibility - 15 points
- `13-15`: Notebook runs in a coherent order, uses fixed random seeds, documents the dataset split, and keeps preprocessing consistent across comparisons.
- `9-12`: Core setup is correct but one reproducibility detail or preprocessing choice is unclear.
- `5-8`: Setup is partially correct, but the split, scaling, or evaluation pipeline is inconsistent.
- `0-4`: Experimental setup is incomplete or not reproducible.

## 2. Discriminative models - 20 points
- `17-20`: Logistic regression and SVM are both trained correctly, hyperparameters are stated, and the writeup explains why these are discriminative models.
- `12-16`: Both models are present and mostly correct, but explanation or implementation details are thin.
- `6-11`: Only one discriminative model is implemented correctly, or both are implemented with major gaps.
- `0-5`: Discriminative modeling work is missing or substantially incorrect.

## 3. Generative models - 20 points
- `17-20`: Gaussian Naive Bayes and class-conditional GMM are both trained correctly, the density-modeling assumptions are explained, and the notebook shows how class predictions are produced.
- `12-16`: Both models are present and mostly correct, but the GMM classifier logic or interpretation is underdeveloped.
- `6-11`: Only one generative model is implemented correctly, or both contain major conceptual errors.
- `0-5`: Generative modeling work is missing or substantially incorrect.

## 4. Evaluation and comparison quality - 20 points
- `17-20`: All four models are compared on the same held-out test set with multiple metrics, confusion matrices, and a clear metric table.
- `12-16`: Comparison is mostly complete but one important metric, visualization, or fairness control is missing.
- `6-11`: Evaluation is partial or mixes validation and test results in a confusing way.
- `0-5`: Comparison is too incomplete to support conclusions.

## 5. Written analysis - 20 points
- `17-20`: Analysis connects empirical outcomes to modeling assumptions, training procedures, and likely failure modes. Claims are specific and evidence-based.
- `12-16`: Analysis addresses the required prompts but remains somewhat generic or under-argued.
- `6-11`: Analysis is mostly descriptive, with limited connection to theory or evidence.
- `0-5`: Analysis is missing, minimal, or unsupported.

## 6. Communication and notebook quality - 5 points
- `5`: Clear headings, readable tables/plots, concise prose, and consistent notation.
- `3-4`: Generally readable, with minor clarity or organization issues.
- `1-2`: Hard to follow because of weak organization or poor presentation.
- `0`: Submission is not meaningfully readable.

## Minimum completeness conditions
The submission cannot earn above `69/100` unless it:

- includes at least two discriminative and two generative models;
- evaluates them on the same dataset split; and
- includes a written comparison section.

## Instructor notes
Use this project as a reusable assessment artifact for Modules 03-05. The key grading question is not only whether the code runs, but whether the student can relate performance differences back to assumptions about \( p(y \mid x) \), \( p(x \mid y) \), independence, and mixture structure.
