# Housing Price Prediction

## Project metadata
- Level: Guided
- Estimated time: `4-5` hours
- Primary modules: `00-03`
- Main topics: linear regression, mean squared error, residual analysis, feature interpretation

## Project overview
This project asks students to build a first end-to-end regression workflow on a small synthetic housing dataset. The emphasis is not on data wrangling. It is on translating the linear regression material from Module 03 into a careful prediction and interpretation workflow.

Students will work with a generated dataset whose features imitate common housing variables such as square footage, bedroom count, home age, school score, and commute score. The notebook provides the data-generation cell so all students start from the same reproducible setup.

## Why this dataset
This dataset is appropriate for the beginner track because it:

- avoids external downloads and licensing friction;
- keeps feature meanings interpretable;
- supports train/validation/test splitting without preprocessing detours; and
- makes residual plots and coefficient interpretation easy to discuss.

## Prerequisites
Students should be comfortable with:

- vectors, matrices, and basic descriptive statistics;
- train/test splits and mean squared error;
- gradient descent or normal-equation intuition; and
- the interpretation of coefficients in linear models.

## Learning objectives
By the end of the project, students should be able to:

1. fit and evaluate a linear regression model on a tabular dataset;
2. compare a constant predictor baseline against a learned model;
3. interpret coefficients and residual plots in domain language;
4. diagnose underfitting or feature mismatch from model errors; and
5. write a short conclusion about where a linear model is helpful and where it is limited.

## Deliverables
Submit:

- a completed notebook based on [`template.ipynb`](./template.ipynb);
- short written responses inside the notebook; and
- any brief appendix notes used to justify modeling choices.

Instructors can grade the submission with [`rubric.md`](./rubric.md).

## Required tasks

### 1. Frame the prediction problem
Explain:

- what the target variable represents;
- which features you expect to increase or decrease price; and
- why linear regression is a reasonable first model.

### 2. Inspect the dataset
Use the generated dataset to report:

- sample count and feature names;
- basic summary statistics;
- pairwise patterns or plots for at least two features; and
- a brief note about any visible feature scaling differences.

### 3. Build baselines and linear models
Train and compare:

- a constant baseline that predicts the training-set mean;
- one linear regression model fit on the full feature set; and
- one variant with a small feature-engineering or feature-selection decision that you justify.

### 4. Evaluate the model
On held-out data, report:

- mean squared error;
- root mean squared error;
- mean absolute error; and
- an \(R^2\) score.

Also include one residual plot and one predicted-vs-actual plot.

### 5. Interpret the result
Write a short analysis addressing:

1. Which features appear most influential and why?
2. Where does the model make its largest errors?
3. What do the residuals suggest about linearity and noise?
4. If you had to improve the model without leaving Modules 00-03, what would you try next?

## Suggested workflow
1. Run the notebook once with the provided synthetic dataset.
2. Establish the constant baseline before fitting a learned model.
3. Keep one primary linear model and one justified variant.
4. Use the closing section to connect the results back to least squares and model assumptions.

## Expected submission quality
A strong submission:

- keeps the workflow reproducible;
- reports all required regression metrics;
- uses plots to support interpretation rather than decoration; and
- distinguishes observed evidence from speculation.
