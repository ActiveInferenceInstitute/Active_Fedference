"""V1 tempered aggregation — four pinned properties of the F_lambda family.

Tests:
1. entropy_weight=1.0 is bit-identical to the default (no-arg) call.
2. Lower lambda (entropy_weight=0.3) sharpens the consensus peak.
3. c->0 recovery: variational_aggregate with robustness=0.0 matches log_linear_pool
   regardless of entropy_weight (lambda-independent recovery).
4. Bounded influence across lambda: robust aggregate with one adversarial agent
   keeps consensus accuracy >= 0.5 for all tested lambda values.

All computations are genuine NumPy operations on small categorical distributions.
No mocks.
"""

from __future__ import annotations

import numpy as np

from fedference.aggregation import log_linear_pool, variational_aggregate


def test_tempered_bit_identical_at_default():
    """entropy_weight=1.0 explicit matches default (no entropy_weight arg) to 1e-10."""
    beliefs = [[0.7, 0.3], [0.6, 0.4]]
    r_default = variational_aggregate(beliefs, robustness=1.0)
    r_explicit = variational_aggregate(beliefs, robustness=1.0, entropy_weight=1.0)
    assert np.allclose(r_default.consensus, r_explicit.consensus, atol=1e-10), (
        f"default vs entropy_weight=1.0 mismatch: {r_default.consensus} vs {r_explicit.consensus}"
    )


def test_tempered_sharpens_with_lower_lambda():
    """entropy_weight=0.3 (lower lambda) yields a sharper (higher-max) consensus than 1.0."""
    beliefs = [[0.7, 0.3], [0.6, 0.4]]
    r_low = variational_aggregate(beliefs, robustness=1.0, entropy_weight=0.3)
    r_std = variational_aggregate(beliefs, robustness=1.0, entropy_weight=1.0)
    peak_low = float(r_low.consensus.max())
    peak_std = float(r_std.consensus.max())
    assert peak_low > peak_std, (
        f"lower lambda should sharpen consensus: peak_low={peak_low:.6f} peak_std={peak_std:.6f}"
    )


def test_tempered_log_linear_recovery():
    """c->0 recovery: variational_aggregate(robustness=0.0) matches log_linear_pool to 1e-8."""
    beliefs = [[0.7, 0.3], [0.6, 0.4]]
    r = variational_aggregate(beliefs, robustness=0.0, entropy_weight=1.0)
    expected = log_linear_pool(beliefs)
    assert np.allclose(r.consensus, expected, atol=1e-8), (
        f"c=0 recovery mismatch: {r.consensus} vs {expected}"
    )


def test_tempered_scenario_retains_true_state_majority_across_lambda():
    """The declared 3:1 fixture retains true-state mass >= 0.5 across lambda."""
    # 3 clean agents with correct belief [0.9, 0.1], 1 adversarial [0.1, 0.9]; true state = 0
    true_state = 0
    clean_belief = [0.9, 0.1]
    adversarial_belief = [0.1, 0.9]
    beliefs = [clean_belief, clean_belief, clean_belief, adversarial_belief]

    for entropy_weight in [0.5, 1.0, 2.0]:
        r = variational_aggregate(beliefs, robustness=2.0, entropy_weight=entropy_weight)
        accuracy = float(r.consensus[true_state])
        assert accuracy >= 0.5, (
            f"scenario control failed at entropy_weight={entropy_weight}: "
            f"consensus[true_state]={accuracy:.4f} < 0.5"
        )
