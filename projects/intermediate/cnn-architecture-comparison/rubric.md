# Rubric: CNN Architecture Comparison

Total: `100 points`

## 1. Problem framing and design rationale - 20 points

- `17-20`: The comparison question is clear, the chosen architectures are meaningfully different, and design expectations are stated before results are discussed.
- `12-16`: The comparison is sensible but the rationale or expected tradeoffs are only partly developed.
- `6-11`: The setup is vague, weakly motivated, or compares models without a clear design question.
- `0-5`: The project lacks a coherent framing.

## 2. Implementation and code quality - 20 points

- `17-20`: Training code is organized, reproducible, and correct; data handling and model definitions are readable and well structured.
- `12-16`: The implementation works with minor reproducibility or organization gaps.
- `6-11`: The code runs only partially or includes major quality problems that weaken the comparison.
- `0-5`: The implementation is substantially incomplete or unreliable.

## 3. Experimental discipline and diagnostics - 20 points

- `17-20`: The evaluation protocol is controlled, hyperparameters are documented, and at least two diagnostics are used to interpret training behavior.
- `12-16`: Core experiments are present, but diagnostics or controls are somewhat thin.
- `6-11`: Evaluation is partial, inconsistent, or weakly documented.
- `0-5`: Experimental discipline is insufficient to support conclusions.

## 4. Results and comparative analysis - 25 points

- `22-25`: Metrics, class-level behavior, and ablations are interpreted carefully, with specific links to architecture and optimization choices.
- `16-21`: Results are mostly sound, but some claims remain generic or under-supported.
- `8-15`: Results are reported descriptively with limited technical interpretation.
- `0-7`: Results are too incomplete or unsupported to justify the conclusions.

## 5. Communication and written analysis - 15 points

- `13-15`: The report is clear, concise, and technically precise, with figures and tables integrated into the argument.
- `9-12`: The analysis is readable but somewhat uneven or repetitive.
- `4-8`: The writeup is hard to follow or weakly connected to evidence.
- `0-3`: Written analysis is missing or not meaningful.

## Minimum completeness conditions

The submission cannot earn above `69/100` unless it:

- compares two CNN-based models on the same data split;
- includes at least one diagnostic plot beyond final accuracy;
- documents the main training choices; and
- includes a written analysis section.
