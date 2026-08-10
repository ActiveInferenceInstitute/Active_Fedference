"""Torch-free configuration defaults for the PyTorch point-mass MLP complement.

Single source of truth for the PyTorch complement's configuration defaults,
consumed by BOTH:

* :mod:`fedference.bnn_baseline_torch` (which imports torch at module level and
  therefore cannot be imported by the manuscript-variable generator), and
* :mod:`manuscript_vars.loaders` (the ``BNN_*`` token fallbacks when the
  executed ``bnn_torch.json`` report is absent — PyTorch not installed).

Keeping the values here (importable without torch) means the token fallbacks
are the *actual* experiment defaults, never re-typed string literals.
"""

from __future__ import annotations

#: Hidden-layer width of the point-mass MLP.
BNN_HIDDEN_DIM_DEFAULT: int = 16
#: Per-client training steps.
BNN_N_STEPS_DEFAULT: int = 200
#: Beta-loss robustness parameter (``beta -> 0`` recovers cross-entropy).
BNN_BETA_DEFAULT: float = 0.5
#: Server-side robust-aggregation strength.
BNN_ROBUSTNESS_DEFAULT: float = 0.5
#: Number of federated clients in the experiment.
BNN_N_CLIENTS_DEFAULT: int = 5

__all__ = [
    "BNN_HIDDEN_DIM_DEFAULT",
    "BNN_N_STEPS_DEFAULT",
    "BNN_BETA_DEFAULT",
    "BNN_ROBUSTNESS_DEFAULT",
    "BNN_N_CLIENTS_DEFAULT",
]
