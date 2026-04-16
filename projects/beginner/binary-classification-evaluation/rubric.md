# Rubric: Binary Classification with Evaluation

Total: 100 points

## 1. Data pipeline and correctness - 20 points
- `17-20`: The dataset split, scaling, and logistic-regression workflow are correct and reproducible.
- `12-16`: The pipeline mostly works, with only minor setup or documentation issues.
- `6-11`: The pipeline is partly correct but contains major gaps in split handling or preprocessing.
- `0-5`: The workflow is incomplete or substantially incorrect.

## 2. Evaluation discipline - 25 points
- `22-25`: The notebook reports all required metrics, includes a threshold comparison, and keeps validation and test evidence clearly separated.
- `16-21`: Evaluation is mostly complete but misses one important metric, comparison, or control.
- `8-15`: Evaluation is partial or somewhat confused.
- `0-7`: Evaluation is too incomplete to support conclusions.

## 3. Analysis quality - 25 points
- `22-25`: The analysis correctly explains precision/recall tradeoffs, threshold effects, and likely error patterns.
- `16-21`: The analysis is mostly correct but somewhat generic or underdeveloped.
- `8-15`: The analysis is mostly descriptive and weakly connected to the metrics.
- `0-7`: The analysis is missing or substantially incorrect.

## 4. Written interpretation - 20 points
- `17-20`: The writeup communicates clearly which operating point is preferred and why, with application-aware reasoning.
- `12-16`: The interpretation is understandable but only partly justified.
- `6-11`: The interpretation is vague or weakly tied to evidence.
- `0-5`: The interpretation is missing or unsupported.

## 5. Communication and notebook quality - 10 points
- `9-10`: Clear headings, readable plots/tables, concise prose, and consistent notation.
- `6-8`: Generally readable with modest clarity issues.
- `3-5`: Hard to follow because of weak organization or presentation.
- `0-2`: Not meaningfully readable.

## Minimum completeness conditions
The submission cannot earn above `69/100` unless it:

- fits at least one logistic regression model correctly;
- reports the full required metric set;
- compares at least two threshold choices; and
- includes a written interpretation section.
