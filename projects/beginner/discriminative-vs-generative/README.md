# Discriminative vs Generative Models

## Project overview
This beginner project asks students to compare how discriminative and generative classifiers behave on the same supervised learning task. The core contrast is:

- discriminative models learn a decision boundary or conditional rule for \( p(y \mid x) \);
- generative models encode assumptions about how features are produced within each class and use Bayes-style reasoning to classify.

Students train and compare four models on the Wisconsin breast cancer dataset:

- discriminative: logistic regression and support vector machine;
- generative: Gaussian Naive Bayes and a class-conditional Gaussian mixture model.

The finished artifact should read like a compact empirical study, not just a set of code cells.

## Why this dataset
The Wisconsin breast cancer dataset is appropriate for Modules 02-05 because it is:

- tabular and small enough for fast iteration;
- binary classification, which keeps evaluation straightforward;
- continuous-valued, which makes Gaussian modeling assumptions explicit; and
- rich enough for logistic regression, SVM, Naive Bayes, and GMMs to behave differently.

The project uses only `numpy`, `matplotlib`, and `scikit-learn`.

## Prerequisites
Students should be comfortable with:

- train/validation/test splits and basic evaluation metrics;
- logistic regression from Module 03;
- kernel methods and SVMs from Module 04;
- Gaussian Naive Bayes, latent-variable modeling, and EM from Module 05; and
- confusion matrices, F1, ROC-AUC, and error analysis from the evaluation toolkit.

## Learning objectives
By the end of the project, students should be able to:

1. train at least two discriminative and two generative classifiers on a shared dataset;
2. explain the modeling assumptions behind each approach in plain mathematical language;
3. compare optimization and training procedures, including convex optimization versus EM-style fitting;
4. evaluate models with multiple metrics instead of accuracy alone; and
5. write a short evidence-based argument about when a generative or discriminative approach is preferable.

## Deliverables
Submit:

- a completed notebook based on [`template.ipynb`](./template.ipynb);
- a short written comparison embedded in the notebook; and
- any brief appendix notes needed to justify tuning choices or failure cases.

Instructors can grade the submission with [`rubric.md`](./rubric.md).

## Required tasks
Complete the following in the notebook.

### 1. Frame the modeling question
State, in your own words:

- what makes logistic regression and SVM discriminative;
- what makes Gaussian Naive Bayes and a Gaussian mixture model generative; and
- which assumptions you expect to help or hurt on this dataset.

### 2. Prepare the data
Build a reproducible pipeline that:

- loads the dataset;
- creates train, validation, and test splits;
- standardizes features where appropriate; and
- records class balance and feature dimensions.

### 3. Train two discriminative models
Train and evaluate:

- logistic regression;
- one SVM model.

At minimum, report the hyperparameters you used and justify them briefly.

### 4. Train two generative models
Train and evaluate:

- Gaussian Naive Bayes;
- a class-conditional Gaussian mixture model with one fitted mixture per class.

You should explain how the GMM classifier converts per-class density estimates into class predictions.

### 5. Compare metrics and behavior
On the same held-out test set, compare:

- accuracy;
- precision;
- recall;
- F1 score;
- ROC-AUC; and
- confusion matrices.

Also compare:

- training procedure;
- sensitivity to preprocessing and hyperparameters;
- interpretability of assumptions; and
- where each model appears to fail.

### 6. Write a short analysis
Answer the following prompts:

1. Which model performed best on the chosen metrics, and why?
2. Which model made the strongest assumptions about feature generation?
3. Did the stronger generative assumptions help or hurt performance?
4. How does the optimization story differ between logistic regression, SVM, Naive Bayes, and GMM?
5. In what kind of real problem would you choose a generative model over a discriminative one?

## Suggested workflow
1. Run the notebook once with the provided baseline settings.
2. Tune only a small number of hyperparameters on the validation split.
3. Freeze the final model settings before the test-set comparison.
4. Use the final section to connect empirical behavior back to the theory from Modules 03-05.

## Expected submission quality
A strong submission:

- keeps the experiment reproducible;
- compares all four required models on the same split;
- distinguishes assumptions from observed performance;
- interprets metrics instead of listing them; and
- states limitations clearly, especially for Gaussian assumptions and mixture complexity.

## Optional extensions
If time permits, choose one extension:

- add a missing-feature experiment and discuss robustness;
- compare linear and RBF SVMs;
- vary the number of GMM components and discuss bias-variance tradeoffs; or
- examine calibration quality in addition to classification accuracy.
