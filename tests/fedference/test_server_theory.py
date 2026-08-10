"""Executable scope guard for the MAJ-1 objective-orientation question."""

from __future__ import annotations

import numpy as np
import pytest

from fedference.server_theory import (
    construct_normalized_weight_no_go_witness,
    construct_orientation_witness,
    construct_raw_log_pool_no_go_witness,
    heuristic_weight_block,
    objective_weight_block,
)


def test_asymmetric_kl_witness_separates_objective_and_heuristic_updates() -> None:
    consensus = np.asarray([0.50, 0.30, 0.20])
    beliefs = np.asarray([[0.90, 0.05, 0.05], [0.45, 0.35, 0.20], [0.05, 0.15, 0.80]])
    witness = construct_orientation_witness(
        consensus,
        beliefs,
        robustness=2.0,
    )
    assert witness.max_absolute_gap > 1e-3
    assert not np.allclose(witness.objective_weights, witness.heuristic_weights)
    assert "not a universal no-go theorem" in witness.result_scope


def test_identical_consensus_and_beliefs_remove_orientation_gap() -> None:
    belief = np.asarray([0.6, 0.3, 0.1])
    witness = construct_orientation_witness(
        belief,
        [belief, belief],
        robustness=1.0,
    )
    assert witness.max_absolute_gap == pytest.approx(0.0, abs=1e-15)
    assert np.allclose(witness.objective_weights, [1.0, 1.0])


def test_orientation_updates_require_positive_robustness_and_matching_dimensions() -> None:
    with pytest.raises(ValueError, match="positive"):
        objective_weight_block([0.5, 0.5], [[0.5, 0.5]], robustness=0.0)
    with pytest.raises(ValueError, match="positive"):
        heuristic_weight_block([0.5, 0.5], [[0.5, 0.5]], robustness=-1.0)
    with pytest.raises(ValueError, match="dimension"):
        construct_orientation_witness(
            [0.5, 0.5],
            [[0.3, 0.3, 0.4]],
            robustness=1.0,
        )


def test_raw_log_pool_no_go_witness_has_two_exact_blocks_and_nonzero_contradiction() -> None:
    """The declared C1 separable class cannot realize both raw q blocks."""
    witness = construct_raw_log_pool_no_go_witness()
    assert witness.max_q_block_error < 1e-14
    assert witness.tangential_contradiction_norm > 1e-3
    assert witness.scales == (1.0, 2.0)
    for scale, source in zip(witness.scales, witness.sources):
        log_scores = scale * np.log(source)
        expected = np.exp(log_scores - np.max(log_scores))
        expected /= expected.sum()
        assert np.allclose(expected, witness.consensus, atol=1e-14)
    assert "not a universal" in witness.result_scope


def test_raw_log_pool_no_go_witness_requires_a_nonuniform_multistate_consensus() -> None:
    with pytest.raises(ValueError, match="non-uniform"):
        construct_raw_log_pool_no_go_witness(np.asarray([0.5, 0.5]))
    with pytest.raises(ValueError, match="at least two"):
        construct_raw_log_pool_no_go_witness(np.asarray([1.0]))


def test_normalized_weight_companion_has_equal_reverse_weights_but_forward_gap() -> None:
    """Normalising a cannot repair the stated forward-KL objective class."""
    witness = construct_normalized_weight_no_go_witness(robustness=1.7)
    assert witness.normalized_weight_max_absolute_gap < 1e-14
    assert witness.forward_difference_gap > 1e-3
    assert witness.reverse_kl_difference_a == pytest.approx(-0.4 * np.log(2.0))
    assert witness.reverse_kl_difference_b == pytest.approx(-0.4 * np.log(2.0))
    assert witness.forward_kl_difference_a == pytest.approx(-0.2 * np.log(3.0))
    assert witness.forward_kl_difference_b == pytest.approx(-0.1 * np.log(3.0))


def test_normalized_weight_companion_rejects_nonpositive_robustness() -> None:
    with pytest.raises(ValueError, match="positive"):
        construct_normalized_weight_no_go_witness(robustness=0.0)
