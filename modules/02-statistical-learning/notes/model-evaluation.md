---
title: "Model Evaluation Toolkit"
module: "02-statistical-learning"
lesson: "model-evaluation"
doc_type: "notes"
topic: "evaluation-metrics-and-validation"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability"
  - "00-math-toolkit/linear-algebra"
  - "01-optimization/convexity-and-optimization"
  - "02-statistical-learning/statistical-learning-foundations"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "statistical-learning"
  - "model-evaluation"
  - "classification-metrics"
  - "regression-metrics"
  - "roc"
  - "precision-recall"
  - "calibration"
---

## Purpose

These notes standardize how the curriculum evaluates supervised models.
The goal is to make train, validation, and test methodology explicit, define the most common regression and classification metrics, and introduce the reusable toolkit in `shared/src/evaluation.py`.

The companion notebook [SL-02-evaluation-toolkit.ipynb](../notebooks/SL-02-evaluation-toolkit.ipynb) demonstrates the workflow on one regression model and one classification model.

## Learning objectives

After working through this note, you should be able to:

- distinguish training, validation, and test roles in a disciplined evaluation pipeline;
- explain when to use MAE, MSE, RMSE, and $R^2$ for regression;
- compute and interpret accuracy, precision, recall, F1, ROC-AUC, and PR-AUC for binary classification;
- read a confusion matrix and relate it to thresholded decisions;
- explain how ROC and precision-recall curves arise from sweeping a decision threshold;
- interpret calibration plots and the Brier score for probabilistic classifiers; and
- use `shared/src/evaluation.py` to keep evaluation logic consistent across later modules.

## 1. Evaluation is part of the learning problem

Training minimizes an objective on observed data.
Evaluation asks a different question: how well does the trained model support the decision or prediction task on new data?

That distinction matters because the same model can look strong under one metric and weak under another.
It can also look strong on the training sample while generalizing poorly.

For this reason, evaluation should be designed before model comparison begins.
At minimum, that means:

- fixing the prediction target;
- fixing the metric or family of metrics;
- fixing the train, validation, and test protocol; and
- keeping the test set untouched until final reporting.

## 2. Train, validation, and test splits

Suppose a dataset is

$$
S = \{(x_i, y_i)\}_{i=1}^n.
$$

We often partition it into three disjoint subsets:

- a **training set**, used to fit model parameters;
- a **validation set**, used to choose hyperparameters or thresholds; and
- a **test set**, used once at the end for final evaluation.

The core discipline is role separation.
If the test set influences model choice, it stops being a valid estimate of post-selection performance.

### Why a validation set is needed

A model family usually comes with choices such as:

- regularization strength;
- polynomial degree;
- feature subset;
- class probability threshold; or
- architecture size.

Choosing among these using the test set leaks information from evaluation into training.
The validation set exists to absorb that selection pressure instead.

### Typical workflow

1. Split the data into train, validation, and test partitions.
2. Fit candidate models on the training partition.
3. Choose hyperparameters using validation performance.
4. Refit if needed using the approved training procedure.
5. Report final performance once on the held-out test partition.

The helper `train_valid_test_split` in `shared/src/evaluation.py` implements this pattern for reusable labs.

### Cross-validation

When data are limited, one validation split may be unstable.
In $K$-fold cross-validation, the training portion is divided into $K$ folds, each fold plays the role of validation once, and the scores are averaged.

Cross-validation is usually used for model selection, not as a replacement for a final untouched test set.

### Split cautions

- Stratify classification splits when class imbalance matters.
- Keep all preprocessing that learns from data inside the training fold.
- Use time-aware splits for temporal data.
- Use grouped splits when observations from the same entity should not appear across train and test.

## 3. Regression metrics

For regression, the target is numeric and the prediction is usually a real-valued estimate $\hat{y}$.
Let true targets be $y_1, \dots, y_m$ and predictions be $\hat{y}_1, \dots, \hat{y}_m$ on an evaluation set of size $m$.

### Mean absolute error

The **mean absolute error** is

$$
\operatorname{MAE} = \frac{1}{m}\sum_{i=1}^m |y_i - \hat{y}_i|.
$$

MAE measures average absolute deviation in the original target units.
It is easy to interpret and less sensitive to large outliers than squared-error metrics.

### Mean squared error

The **mean squared error** is

$$
\operatorname{MSE} = \frac{1}{m}\sum_{i=1}^m (y_i - \hat{y}_i)^2.
$$

Because the residual is squared, large errors count disproportionately.
MSE is often convenient analytically and aligns with Gaussian-noise modeling assumptions.

### Root mean squared error

The **root mean squared error** is

$$
\operatorname{RMSE} = \sqrt{\operatorname{MSE}}.
$$

RMSE keeps the same outlier sensitivity as MSE but returns to the target units.
That often makes it easier to compare against domain tolerances.

### Coefficient of determination

The **coefficient of determination** is

$$
R^2 = 1 - \frac{\sum_{i=1}^m (y_i - \hat{y}_i)^2}{\sum_{i=1}^m (y_i - \bar{y})^2},
$$

where

$$
\bar{y} = \frac{1}{m}\sum_{i=1}^m y_i.
$$

This compares the fitted model against the baseline predictor that always outputs the sample mean.
Values near $1$ indicate a strong fit relative to that baseline.
Values near $0$ indicate little improvement over the mean predictor.
Negative values mean the model is worse than the mean baseline on the evaluation set.

### Practical regression guidance

- Use MAE when absolute error in native units is the clearest operational quantity.
- Use MSE or RMSE when large deviations should be penalized heavily.
- Use $R^2$ as a relative fit summary, not as the sole reporting metric.

The function `regression_metrics` in `shared/src/evaluation.py` returns this standard summary for later labs.

## 4. Classification metrics begin with the confusion matrix

For binary classification, each example belongs to either the negative class or the positive class.
A thresholded classifier outputs either positive or negative.

The resulting counts form the **confusion matrix**:

$$
\begin{array}{c|cc}
 & \text{Predicted negative} & \text{Predicted positive} \\
\hline
\text{Actual negative} & TN & FP \\
\text{Actual positive} & FN & TP
\end{array}
$$

The entries mean:

- $TP$: true positives;
- $TN$: true negatives;
- $FP$: false positives; and
- $FN$: false negatives.

Every threshold-based classification metric is a function of these counts.

### Accuracy

The **accuracy** is

$$
\operatorname{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}.
$$

Accuracy is intuitive, but it can be misleading under class imbalance.
If only $1\%$ of examples are positive, always predicting the negative class gives $99\%$ accuracy while failing completely on the positive class.

### Precision

The **precision** is

$$
\operatorname{Precision} = \frac{TP}{TP + FP}.
$$

Precision answers:
among the examples predicted positive, how many were actually positive?

High precision matters when false positives are costly.

### Recall

The **recall** or **true positive rate** is

$$
\operatorname{Recall} = \frac{TP}{TP + FN}.
$$

Recall answers:
among the actually positive examples, how many did the classifier recover?

High recall matters when false negatives are costly.

### F1 score

The **F1 score** is the harmonic mean of precision and recall:

$$
F_1 = 2 \cdot \frac{\operatorname{Precision}\cdot\operatorname{Recall}}{\operatorname{Precision} + \operatorname{Recall}}.
$$

Because it is a harmonic mean, F1 is low whenever either precision or recall is low.
It is useful when both false positives and false negatives matter and a single thresholded summary is needed.

### Worked confusion-matrix example

Suppose a classifier produces:

$$
TP = 18,\quad FP = 6,\quad FN = 2,\quad TN = 74.
$$

Then:

$$
\operatorname{Accuracy} = \frac{18 + 74}{100} = 0.92,
$$

$$
\operatorname{Precision} = \frac{18}{18 + 6} = 0.75,
$$

$$
\operatorname{Recall} = \frac{18}{18 + 2} = 0.90,
$$

and

$$
F_1 = 2 \cdot \frac{0.75 \cdot 0.90}{0.75 + 0.90} \approx 0.818.
$$

This is a useful example because accuracy looks excellent, but precision is materially lower than recall.
That gap may or may not be acceptable, depending on the application.

The helper `confusion_matrix_report` returns the matrix and label order used in plotting and downstream reporting.

## 5. ROC curves

Many classifiers output a score or probability $s(x)$ rather than only a class label.
To turn that score into a label, choose a threshold $t$ and predict positive when $s(x) \geq t$.

As $t$ changes, the confusion matrix changes.
This generates the **receiver operating characteristic** or **ROC** curve.

### Coordinates

The ROC curve plots:

- **false positive rate**

$$
\operatorname{FPR} = \frac{FP}{FP + TN},
$$

against

- **true positive rate**

$$
\operatorname{TPR} = \frac{TP}{TP + FN}.
$$

For each threshold $t$, we get one point $(\operatorname{FPR}, \operatorname{TPR})$.

### Worked threshold example

Consider six examples with labels

$$
y = [1, 1, 0, 0, 1, 0]
$$

and scores

$$
s = [0.95, 0.80, 0.70, 0.55, 0.40, 0.20].
$$

At threshold $t = 0.80$, predicted positives are the first two examples.
Then:

- $TP = 2$;
- $FP = 0$;
- $FN = 1$; and
- $TN = 3$.

So

$$
\operatorname{TPR} = \frac{2}{3}, \qquad \operatorname{FPR} = 0.
$$

At threshold $t = 0.55$, predicted positives are the first four examples.
Then:

- $TP = 2$;
- $FP = 2$;
- $FN = 1$; and
- $TN = 1$,

so

$$
\operatorname{TPR} = \frac{2}{3}, \qquad \operatorname{FPR} = \frac{2}{3}.
$$

Lowering the threshold tends to increase both true positives and false positives.
The ROC curve records that tradeoff across all thresholds.

### AUC-ROC

The **area under the ROC curve** is denoted ROC-AUC.
A value of $1$ indicates perfect ranking of positives above negatives.
A value of $0.5$ corresponds to random ranking in the binary case.

ROC-AUC is threshold-free, but it can look optimistic in highly imbalanced problems because the false positive rate normalizes by the large number of negatives.

The helper `roc_curve_data` returns ROC coordinates and area, while `plot_roc_curve` renders the result.

## 6. Precision-recall curves

The **precision-recall** or **PR** curve uses the same threshold sweep but plots:

- recall on the horizontal axis; and
- precision on the vertical axis.

This curve is especially useful when the positive class is rare or operationally important.

### Why PR curves matter under imbalance

If positives are rare, a small false positive rate can still correspond to many false positives in absolute count.
ROC may then hide practical failure.
Precision reveals that failure immediately because it falls when false positives accumulate.

### Worked threshold example

Using the same toy example:

- at threshold $t = 0.80$, precision is $1.0$ and recall is $2/3$;
- at threshold $t = 0.55$, precision is $2/4 = 0.5$ and recall remains $2/3$;
- at threshold $t = 0.40$, recall rises to $1.0$, but precision falls because more negatives are predicted positive.

So lowering the threshold usually pushes the PR curve toward higher recall and often lower precision.

### AUC-PR and average precision

Two common summaries are:

- the trapezoidal area under the PR curve; and
- **average precision**, a ranking-based summary used widely in information retrieval and detection.

PR summaries should be interpreted relative to the positive-class prevalence.
A PR-AUC of $0.40$ may be strong if positives occur only $5\%$ of the time, but weak if the dataset is balanced.

The helpers `precision_recall_curve_data` and `plot_precision_recall_curve` support this analysis in the shared toolkit.

## 7. Calibration and reliability diagrams

A classifier can rank examples well and still produce poor probabilities.
For decisions involving downstream risk, uncertainty, or resource allocation, probability calibration matters.

### Calibration

A probabilistic classifier is **well calibrated** if among examples assigned probability near $p$, approximately a fraction $p$ actually belong to the positive class.

For example, among all examples receiving predicted probability about $0.70$, roughly $70\%$ should be positive.

### Reliability diagram

A **reliability diagram** or **calibration plot** groups predictions into bins and compares:

- the mean predicted probability in each bin; against
- the observed fraction of positives in that bin.

Perfect calibration lies on the diagonal line

$$
y = x.
$$

If plotted points fall below the diagonal, the model is overconfident in those bins.
If they lie above the diagonal, the model is underconfident.

### Brier score

For binary outcomes, the **Brier score** is

$$
\operatorname{Brier} = \frac{1}{m}\sum_{i=1}^m (p_i - y_i)^2,
$$

where $p_i$ is the predicted probability of the positive class and $y_i \in \{0,1\}$ is the true label.

This is a proper scoring rule for probabilistic forecasts.
Lower values are better.

### Practical note

Calibration plots can become noisy in small samples because each bin may contain few examples.
That is not a defect of the concept.
It means the empirical estimate of event frequency is itself uncertain.

The helper `calibration_curve_data` returns reliability-diagram inputs and the Brier score, and `plot_calibration_curve` renders the comparison.

## 8. Choosing metrics by task

No metric is universally best.
Metric choice should follow the decision problem.

### Regression

- Use MAE when typical absolute miss size is the main operational concern.
- Use RMSE when large misses should be penalized more sharply.
- Use $R^2$ as a comparative summary, not a substitute for error units.

### Classification

- Use accuracy only when class balance and error costs make it meaningful.
- Use precision when false positives are costly.
- Use recall when false negatives are costly.
- Use F1 when both precision and recall matter and one thresholded summary is needed.
- Use ROC-AUC for general ranking discrimination.
- Use PR-AUC when the positive class is rare or the retrieval task is asymmetric.
- Use calibration diagnostics when predicted probabilities will be consumed directly.

## 9. Practical evaluation workflow

The reusable workflow for this repository is:

1. create the split with `train_valid_test_split`;
2. fit on the training partition only;
3. choose hyperparameters and decision thresholds using validation metrics;
4. compute final summaries on the test partition with `regression_metrics` or `classification_metrics`;
5. inspect diagnostic structures such as confusion matrices, ROC curves, PR curves, and calibration plots; and
6. report metrics together with the task context that makes them meaningful.

This pattern is demonstrated in [SL-02-evaluation-toolkit.ipynb](../notebooks/SL-02-evaluation-toolkit.ipynb) and reused downstream.
For example, later linear-model material can import the same helpers from `shared/src/evaluation.py` instead of reimplementing evaluation logic locally.

## 10. Category theory insertion point

The canonical content here is statistical and operational.
If category theory is used, it should remain supplementary.
One reasonable view is that an evaluation pipeline is a composition

$$
\text{data split} \to \text{fit} \to \text{predict} \to \text{metric map},
$$

where different metrics expose different structure preserved or lost under the prediction map.
That viewpoint can clarify compositional design, but it does not replace the standard definitions.

## 11. Unity Theory insertion point

Any Unity Theory language here should remain interpretive.
A cautious framing is that evaluation tests whether learned representations preserve enough relational structure for the target task under distributional variation.
The core machinery, however, remains standard model assessment.

## Summary

Model evaluation is not a cosmetic final step.
It determines what a model is being asked to optimize in practice and what claims can be justified from the results.

The central lessons are:

- separate training, validation, and test roles cleanly;
- choose metrics that match the decision problem rather than defaulting to accuracy;
- use confusion matrices to understand thresholded classification behavior;
- use ROC and PR curves to study threshold tradeoffs;
- use calibration plots when probabilities matter; and
- reuse `shared/src/evaluation.py` so later modules inherit a consistent evaluation standard.

## References

- Trevor Hastie, Robert Tibshirani, and Jerome Friedman, *The Elements of Statistical Learning*, 2nd ed.
- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*.
- Shai Shalev-Shwartz and Shai Ben-David, *Understanding Machine Learning: From Theory to Algorithms*.
