# shared/src

Reusable Python utilities shared across curriculum modules live here.
The goal is to keep evaluation and training-analysis logic consistent across labs instead of reimplementing small helpers inside each notebook.

## Current modules

- `evaluation.py`: train/validation/test splitting, regression and classification summaries, and standard evaluation plots introduced in Module 02.
- `training_diagnostics.py`: loss curves, gradient norm tracking, activation summaries and histograms, learning-rate visualization, and confusion-matrix rendering for neural-network labs.

## Typical usage

```python
from shared.src.training_diagnostics import (
    activation_summary,
    append_gradient_snapshot,
    plot_activation_histograms,
    plot_gradient_norms,
    plot_learning_rates,
    plot_loss_curves,
)
```

Use the evaluation toolkit for prediction metrics and the diagnostics toolkit for training dynamics.
The diagnostics helpers are written to be array-first and framework-agnostic where practical:

- pass NumPy arrays directly;
- pass PyTorch tensors or parameters and the helpers will convert them through `.detach().cpu().numpy()` when available;
- keep long-lived plotting and tracking code in `shared/src/` rather than notebook-local utility cells.

## Cross-module expectations

- Module 07 uses `training_diagnostics.py` in `DL-04-diagnostics-demo.ipynb` to visualize vanishing and exploding gradients on a live training run.
- Module 08 should reuse the same helpers for CNN activation maps, gradient flow, and confusion-matrix checks on vision classifiers.
- Module 10 should reuse the same helpers for transformer learning-rate schedules, attention-block activation diagnostics, and classifier-head confusion matrices.
