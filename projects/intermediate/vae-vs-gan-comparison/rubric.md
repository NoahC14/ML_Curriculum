# Rubric: Generative Model Comparison, VAE vs GAN

Total: `100 points`

## 1. Framing and objective-level understanding - 20 points

- `17-20`: The project clearly explains what the VAE and GAN objectives optimize and frames a focused comparison question.
- `12-16`: The framing is mostly sound but objective-level distinctions are not fully developed.
- `6-11`: The comparison is weakly motivated or conceptually muddled.
- `0-5`: The project lacks a coherent framing.

## 2. Implementation and code quality - 20 points

- `17-20`: Both models are implemented or adapted correctly, code is readable, and the pipeline is reproducible.
- `12-16`: Both models are present with minor correctness or code-quality gaps.
- `6-11`: One model is substantially weaker, or implementation quality limits trust in the results.
- `0-5`: The implementation is substantially incomplete.

## 3. Experimental rigor and diagnostics - 20 points

- `17-20`: Hyperparameters, stabilization choices, and training behavior are documented clearly, including evidence about GAN instability or VAE tradeoffs.
- `12-16`: Core experiments are present, but diagnostics or controls are somewhat thin.
- `6-11`: Experimental discipline is uneven or weakly explained.
- `0-5`: The evidence base is too limited for credible conclusions.

## 4. Comparative evaluation and analysis - 25 points

- `22-25`: The submission compares reconstructions, samples, diversity, and limitations carefully, without overclaiming from weak metrics.
- `16-21`: The analysis is useful but somewhat generic or under-supported.
- `8-15`: Results are presented descriptively with limited technical comparison.
- `0-7`: Comparative analysis is too incomplete or unsupported.

## 5. Written communication - 15 points

- `13-15`: The report is organized, precise, and explicit about tradeoffs and limits.
- `9-12`: The report is readable but uneven or thin in places.
- `4-8`: The writeup is hard to follow or mostly descriptive.
- `0-3`: Written analysis is missing or not meaningful.

## Minimum completeness conditions

The submission cannot earn above `69/100` unless it:

- includes both a VAE and a GAN;
- presents generated outputs from both models;
- discusses stability or objective tradeoffs explicitly; and
- includes a written analysis section.
