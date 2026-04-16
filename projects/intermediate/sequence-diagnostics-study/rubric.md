# Rubric: Sequence Prediction with Diagnostics Analysis

Total: `100 points`

## 1. Task framing and theoretical motivation - 20 points

- `17-20`: The sequence task is clearly defined, the comparison question is well motivated, and the expected pathologies are identified in advance.
- `12-16`: The task is clear, but the theoretical motivation or comparison logic is underdeveloped.
- `6-11`: The project has a weakly specified goal or little connection to sequence-model theory.
- `0-5`: Framing is missing or incoherent.

## 2. Implementation and code quality - 20 points

- `17-20`: The training pipeline is correct, readable, and reproducible, with clean handling of sequence batching and evaluation.
- `12-16`: The implementation mostly works but has minor quality or reproducibility gaps.
- `6-11`: There are major implementation issues, unclear code structure, or weak reproducibility.
- `0-5`: The implementation is substantially incomplete.

## 3. Diagnostics and experimental rigor - 25 points

- `22-25`: Gradient and training diagnostics are collected systematically and used to support conclusions about stability and learning dynamics.
- `16-21`: Diagnostics are present and useful, but the analysis does not fully exploit them.
- `8-15`: Diagnostics are partial, weakly interpreted, or inconsistently tied to the experiments.
- `0-7`: Diagnostic evidence is too limited to support the discussion.

## 4. Results and error analysis - 20 points

- `17-20`: Metrics and qualitative failures are analyzed carefully, with clear links to context length, architecture, or stabilization choices.
- `12-16`: Evaluation is solid but some conclusions remain generic.
- `6-11`: Results are reported with limited error analysis or weak linkage to design choices.
- `0-5`: Results are too incomplete or unsupported.

## 5. Written analysis and communication - 15 points

- `13-15`: The report is precise, well organized, and evidence-based.
- `9-12`: The report is readable but somewhat thin or uneven.
- `4-8`: The explanation is difficult to follow or mostly descriptive.
- `0-3`: Written analysis is missing or not meaningful.

## Minimum completeness conditions

The submission cannot earn above `69/100` unless it:

- compares at least two sequence-model variants;
- includes gradient or stability diagnostics;
- evaluates on a held-out split or clearly separated evaluation set; and
- includes a written analysis section.
