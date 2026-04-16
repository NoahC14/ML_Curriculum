# Binary Classification with Evaluation

## Project metadata
- Level: Guided
- Estimated time: `4-6` hours
- Primary modules: `01-04`
- Main topics: logistic regression, classification thresholds, confusion matrices, ROC-AUC

## Project overview
This project asks students to train a binary logistic regression classifier and evaluate it with more than a single accuracy number. The emphasis is on connecting the probabilistic output of logistic regression to threshold-dependent evaluation.

Students will use the Wisconsin breast cancer dataset from `scikit-learn`. The dataset is small enough for rapid iteration and clean enough that the main work stays focused on modeling and evaluation.

## Why this dataset
This dataset is appropriate because it:

- is binary, so the logistic-regression story stays central;
- is widely used, small, and already packaged in `scikit-learn`;
- supports precision/recall tradeoffs in a meaningful way; and
- avoids feature-engineering complexity that would distract from evaluation concepts.

## Prerequisites
Students should be comfortable with:

- sigmoid probabilities and cross-entropy;
- train/validation/test splits;
- confusion matrices, precision, recall, F1, and ROC-AUC; and
- basic feature standardization.

## Learning objectives
By the end of the project, students should be able to:

1. fit a reproducible binary logistic regression pipeline;
2. explain the difference between a score, a probability, and a hard class prediction;
3. compare threshold choices instead of accepting the default threshold uncritically;
4. interpret confusion-matrix tradeoffs in application terms; and
5. justify a model-selection decision with evidence from multiple metrics.

## Deliverables
Submit:

- a completed notebook based on [`template.ipynb`](./template.ipynb);
- concise written answers embedded in the notebook; and
- any brief appendix notes justifying threshold or regularization choices.

Instructors can grade the submission with [`rubric.md`](./rubric.md).

## Required tasks

### 1. Frame the classification problem
Explain:

- what the positive class represents;
- why logistic regression is a suitable first classifier; and
- why accuracy alone may be insufficient.

### 2. Prepare and inspect the data
Create a reproducible workflow that:

- loads the dataset;
- creates train, validation, and test splits;
- standardizes features; and
- reports class balance.

### 3. Train logistic regression models
Train:

- one baseline logistic regression model; and
- one tuned variant with a justified change in regularization strength or classification threshold.

### 4. Evaluate with multiple metrics
On held-out data, report:

- accuracy;
- precision;
- recall;
- F1 score;
- ROC-AUC; and
- a confusion matrix.

Also include one threshold sweep or threshold comparison table.

### 5. Write the analysis
Answer:

1. Which threshold would you choose and why?
2. What kinds of mistakes does the model make?
3. How does changing the threshold affect precision and recall?
4. In what setting would you tolerate more false positives or more false negatives?

## Suggested workflow
1. Fit a default model first.
2. Use the validation split to reason about threshold choice.
3. Freeze the final threshold before the test-set comparison.
4. Use the final markdown cells to connect empirical tradeoffs back to the evaluation toolkit.

## Expected submission quality
A strong submission:

- keeps validation and test usage clearly separated;
- uses at least one threshold comparison beyond the default `0.5`;
- interprets the confusion matrix in application language; and
- supports conclusions with metrics instead of impressions.
