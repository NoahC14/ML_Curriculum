# Regularization Comparison Study

## Project metadata
- Level: Structured
- Estimated time: `5-7` hours
- Primary modules: `01-04`
- Main topics: ridge regression, lasso, coefficient shrinkage, bias-variance reasoning

## Project overview
This project asks students to compare how unregularized linear regression, ridge regression, and lasso behave on the same noisy prediction task. The project turns the regularization discussion from Modules 01-03 into an empirical study about stability, overfitting, and coefficient shrinkage.

Students will use a synthetic regression dataset with correlated features and polynomial expansion. This setup makes overfitting visible while keeping the mathematics familiar.

## Why this dataset
This dataset is appropriate because it:

- makes the need for regularization visible without requiring domain-specific preprocessing;
- allows direct comparison of coefficient magnitudes and sparsity;
- keeps training fast enough to repeat several experiments; and
- connects naturally to bias-variance and optimization discussions.

## Prerequisites
Students should be comfortable with:

- least squares and mean squared error;
- gradient-descent or optimization-based fitting intuition;
- the idea of overfitting and validation-based model selection; and
- the definitions of ridge and lasso penalties.

## Learning objectives
By the end of the project, students should be able to:

1. compare unregularized, ridge, and lasso regression on a shared task;
2. explain how penalty strength changes coefficient behavior and generalization;
3. connect empirical results to bias-variance tradeoffs;
4. interpret why correlated features can destabilize unregularized regression; and
5. write a short recommendation about when each regularization method is appropriate.

## Deliverables
Submit:

- a completed notebook based on [`template.ipynb`](./template.ipynb);
- concise analysis embedded in the notebook; and
- any brief appendix notes needed to justify hyperparameter choices.

Instructors can grade the submission with [`rubric.md`](./rubric.md).

## Required tasks

### 1. Describe the modeling setting
Explain:

- why correlated or high-variance features can cause trouble for ordinary least squares; and
- what ridge and lasso are expected to change.

### 2. Build the dataset and baseline
Use the provided synthetic-data scaffold to:

- generate reproducible train, validation, and test splits;
- expand features or use the provided polynomial representation; and
- establish an unregularized linear-regression baseline.

### 3. Compare regularized models
Train and compare:

- ordinary least squares;
- at least two ridge settings; and
- at least two lasso settings.

Report the chosen regularization strengths and justify the final settings briefly.

### 4. Evaluate prediction and coefficient behavior
On held-out data, report:

- MSE or RMSE across models;
- a validation comparison table;
- coefficient magnitude summaries; and
- one visualization showing how regularization changes fit quality or parameter values.

### 5. Write the analysis
Answer:

1. Which model generalized best, and why?
2. How did ridge and lasso differ in coefficient behavior?
3. Did lasso produce meaningful sparsity?
4. What evidence suggests overfitting or instability in the unregularized model?
5. When would you prefer ridge over lasso, or lasso over ridge?

## Suggested workflow
1. Fit the unregularized baseline first.
2. Use the validation split to tune regularization strength.
3. Compare both predictive error and coefficient behavior.
4. Use the final section to connect the evidence back to regularization theory.

## Expected submission quality
A strong submission:

- compares all three model families on the same split;
- uses both predictive and parameter-level evidence;
- explains shrinkage and sparsity without overclaiming; and
- clearly distinguishes validation choices from final test reporting.
