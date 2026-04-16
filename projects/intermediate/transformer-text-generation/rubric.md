# Rubric: Transformer Text Generation

Total: `100 points`

## 1. Setup and design rationale - 20 points

- `17-20`: The corpus, tokenization, model size, and comparison question are clearly justified and well scoped to the available compute.
- `12-16`: The setup is mostly sensible but some design choices are weakly justified.
- `6-11`: The setup is underspecified, poorly scoped, or not clearly aligned with the stated question.
- `0-5`: The project lacks a coherent setup.

## 2. Implementation and code quality - 20 points

- `17-20`: The transformer pipeline is correct, organized, and reproducible, with clear handling of batching, masking, and generation.
- `12-16`: The implementation mostly works but has minor correctness or clarity issues.
- `6-11`: Major implementation gaps or quality issues weaken the results.
- `0-5`: The pipeline is substantially incomplete or unreliable.

## 3. Experimental rigor and diagnostics - 20 points

- `17-20`: Training controls are documented, diagnostics are informative, and the comparison is run under fair conditions.
- `12-16`: Experiments are mostly solid, but diagnostics or controls are only partly developed.
- `6-11`: Experimental discipline is uneven or weakly documented.
- `0-5`: The evidence base is too limited to support conclusions.

## 4. Evaluation and generative analysis - 25 points

- `22-25`: Held-out metrics, sample generations, and decoding comparisons are interpreted carefully, with attention to both quality and failure modes.
- `16-21`: Evaluation is useful but some conclusions remain generic or under-supported.
- `8-15`: Results are presented descriptively with limited technical analysis.
- `0-7`: Evaluation is too incomplete or unsupported.

## 5. Written analysis and communication - 15 points

- `13-15`: The report is concise, technically precise, and explicit about limitations and compute constraints.
- `9-12`: The report is readable but somewhat thin or uneven.
- `4-8`: The report is hard to follow or weakly argued.
- `0-3`: Written analysis is missing or not meaningful.

## Minimum completeness conditions

The submission cannot earn above `69/100` unless it:

- trains at least one transformer language model;
- includes one justified design comparison;
- reports both held-out metrics and generated-text evidence; and
- includes a written analysis section.
