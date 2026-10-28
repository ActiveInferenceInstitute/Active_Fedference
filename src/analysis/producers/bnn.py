"""Existing bnn report production; no persistence or workflow imports.

Friston et al. (2024) and Mildner et al. (2025) study meanings and configured
seed/budget controls are retained from the canonical analysis workflow.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

from experiment_config import ExperimentConfig
from fedference.experiments import run_bnn_robustness_report


def _bnn_report(config: ExperimentConfig, *, smoke: bool = False) -> dict:
    """Exploratory point-estimate logistic accuracy under label contamination."""
    if smoke:
        return run_bnn_robustness_report(
            config.seeds[0],
            n_seeds=min(config.bnn_n_seeds, 4),
            n_per=min(config.bnn_n_per, 20),
        )
    return run_bnn_robustness_report(
        config.seeds[0],
        n_seeds=config.bnn_n_seeds,
        n_per=config.bnn_n_per,
    )


class BnnTorchOptions(TypedDict, total=False):
    """Validated optional keyword arguments for the Torch complement."""

    n_clients: int
    n_per: int
    hidden_dim: int
    n_steps: int
    robustness: float
    beta: float
    contamination_levels: tuple[float, ...]


def _bnn_torch_report(
    config: ExperimentConfig,
    *,
    options: BnnTorchOptions,
    run_experiment: Callable[..., dict],
) -> dict:
    """Produce the Mildner et al. (2025) Torch complement from explicit inputs.

    Orchestration owns optional-backend admission and configuration I/O; the
    selected domain operation receives the unchanged seed and validated options.
    """
    return dict(run_experiment(seed=config.seeds[0], **options))
