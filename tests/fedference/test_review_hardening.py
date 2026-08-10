"""Focused regression tests for the red-team API and review-grid contract."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.report_schemas import ReportSchemaError, validate_report
from fedference.aggregation import (
    AggregationConfig,
    aggregate,
    aggregate_result,
    aggregation_free_energy,
    log_linear_pool,
    robust_aggregate,
    variational_aggregate,
)
from fedference.belief_sharing import share_round
from fedference.experiments import run_review_grid
from fedference.generalized_bayes import (
    cavity,
    generalized_posterior,
    softmax,
    update_factor,
)
from fedference.statistics import (
    cohens_d_from_rank_biserial,
    d_equivalent_from_rank_biserial,
)

BELIEFS = np.asarray(
    [[0.80, 0.15, 0.05], [0.70, 0.20, 0.10], [0.10, 0.20, 0.70]],
    dtype=float,
)


def test_canonical_aggregation_keywords_and_deprecated_alias_parity() -> None:
    canonical = log_linear_pool(
        local_posteriors=BELIEFS,
        base_weights=[1.0, 2.0, 1.0],
    )
    with pytest.warns(DeprecationWarning, match="beliefs"):
        with pytest.warns(DeprecationWarning, match="weights"):
            legacy = log_linear_pool(
                beliefs=BELIEFS,
                weights=[1.0, 2.0, 1.0],
            )
    assert np.array_equal(canonical, legacy)


def test_raw_and_normalized_effective_weights_have_distinct_contracts() -> None:
    robust = robust_aggregate(
        local_posteriors=BELIEFS,
        base_weights=[1.0, 2.0, 1.0],
        robustness=1.5,
    )
    assert np.all(robust.raw_effective_weights <= np.asarray([1.0, 2.0, 1.0]) + 1e-12)
    assert np.isclose(robust.normalized_effective_weights.sum(), 1.0)
    with pytest.warns(DeprecationWarning, match="agent_weights"):
        assert np.array_equal(robust.agent_weights, robust.normalized_effective_weights)

    variational = variational_aggregate(
        local_posteriors=BELIEFS,
        base_weights=[1.0, 2.0, 1.0],
        robustness=1.0,
        max_iter=8,
        multistart=False,
    )
    assert np.all(variational.raw_effective_weights <= np.asarray([1.0, 2.0, 1.0]) + 1e-12)


def test_cavity_recombines_with_site_factor_and_alias_warns() -> None:
    global_posterior = np.asarray([0.55, 0.30, 0.15])
    site_factor = np.asarray([0.50, 0.25, 0.25])
    result = cavity(
        global_posterior=global_posterior,
        site_factor=site_factor,
    )
    recombined = result * site_factor
    recombined /= recombined.sum()
    assert np.allclose(recombined, global_posterior)
    with pytest.warns(DeprecationWarning, match="posterior"):
        with pytest.warns(DeprecationWarning, match="factor"):
            legacy = cavity(posterior=global_posterior, factor=site_factor)
    assert np.allclose(legacy, result)


def test_generalized_bayes_loss_alias_preserves_result() -> None:
    prior = np.log(np.asarray([0.4, 0.35, 0.25]))
    loss = np.asarray([0.1, 0.3, 0.8])
    canonical = generalized_posterior(prior, loss_by_state=loss, tau=0.7)
    with pytest.warns(DeprecationWarning, match="loss_vec"):
        with pytest.warns(DeprecationWarning, match="learning_rate"):
            legacy = generalized_posterior(prior, loss_vec=loss, learning_rate=0.7)
    assert np.allclose(canonical, legacy)


def test_canonical_tau_alias_has_the_same_semantics_as_learning_rate() -> None:
    prior = np.log(np.asarray([0.4, 0.35, 0.25]))
    loss = np.asarray([0.1, 0.3, 0.8])
    canonical = generalized_posterior(prior, loss_by_state=loss, tau=0.7)
    with pytest.warns(DeprecationWarning, match="learning_rate"):
        legacy = generalized_posterior(prior, loss_by_state=loss, learning_rate=0.7)
    assert np.allclose(canonical, legacy)


def test_conditional_world_contrast_is_robust_minus_naive() -> None:
    from fedference.experiments.conditional_world import (
        ConditionalScenario,
        _trial,
    )

    scenario = ConditionalScenario(
        scenario_id="test",
        true_state=0,
        target_state=1,
        observability=0.7,
        attack="confident_wrong",
        adversary_weight=0.5,
        n_contaminated=2,
    )
    result = _trial(np.random.default_rng(4), scenario, n_agents=7, robustness=1.5)
    assert result["contrast"] == pytest.approx(result["robust"][0] - result["naive"][0])


def test_d_equivalent_is_canonical_and_old_name_is_only_an_adapter() -> None:
    assert d_equivalent_from_rank_biserial(0.5) == pytest.approx(1.1547005383792515)
    with pytest.warns(DeprecationWarning, match="d_equivalent"):
        assert cohens_d_from_rank_biserial(0.5) == pytest.approx(d_equivalent_from_rank_biserial(0.5))
    assert np.isinf(d_equivalent_from_rank_biserial(1.0))


def test_review_grid_is_deterministic_nested_and_selection_free() -> None:
    first = run_review_grid(
        seed=17,
        n_seeds=2,
        n_trials=2,
        n_agents=3,
        rates=(0.0, 0.5),
        divergences=("KLD", "RKL"),
    )
    second = run_review_grid(
        seed=17,
        n_seeds=2,
        n_trials=2,
        n_agents=3,
        rates=(0.0, 0.5),
        divergences=("KLD", "RKL"),
    )
    assert first == second
    assert first["analysis_profile"] == "diagnostic_review_grid"
    assert first["selection_status"].startswith("selection-free")
    assert first["independent_unit"].startswith("seed")
    assert first["controls"]["clean_control_present"]
    assert first["precision_plan"]["target_status"] == "not_evaluated"
    assert first["precision_plan"]["target_met"] is None
    assert first["statistics"]["fdr_alpha"] == 0.05
    assert first["statistics"]["power_alpha"] == 0.05
    assert first["statistics"]["planning_alternative"] == "greater"
    for row in first["conditional_world"]["by_scenario"].values():
        assert len(row["contrast_by_seed"]) == 2
        assert row["n_trials"] == 2
    for mechanism in first["statistics"]["by_mechanism"].values():
        assert mechanism["bh_family_ownership"].startswith("one BH family")
        for rate in mechanism["by_rate"].values():
            assert len(rate["methods"]["RKL"]["contrast_by_seed"]) == 2
    validate_report("robustness_review_grid", first)


def test_versioned_report_reader_rejects_unsupported_schema_version() -> None:
    payload = {
        "schema_version": "1.0",
        "accuracy_by_method_and_rate": {},
        "naive_degrades_with_rate": False,
        "robust_above_threshold_at_worst_rate": False,
        "accuracy_threshold": 0.5,
        "verdict": {},
        "accuracy_at_verdict_rate": {},
        "per_rate_summary": {},
        "paired_tests_by_rate": {},
        "any_robust_wins": False,
        "worst_rate": 0.9,
        "verdict_rate": 0.8,
        "kind": "clean",
        "n_agents": 3,
        "n_contaminated": 1,
        "n_trials": 2,
        "naive_verdict_rate_mean": 0.5,
        "n": 2,
        "fdr_alpha": 0.05,
        "power_alpha": 0.05,
        "power_alternative": "greater",
        "target_power": 0.8,
        "headline_power": 0.5,
        "headline_n_for_target_power": 10,
        "prospective_n_for_target_power": 10,
        "headline_method": "RKL",
        "headline_selection_rule": "declared",
        "headline_tie_set": ["RKL"],
        "headline_tie_break": "declared",
        "headline_is_display_selection": True,
        "largest_mean_difference_method": "RKL",
        "worst_rate_best_method": "RKL",
        "analysis_unit": "seed",
        "trial_structure": "nested",
        "paired_test_alternative": "two-sided",
        "fdr_family_ownership": "declared",
        "d_equivalent_status": "declared",
        "true_state": 0,
        "attack_target_state": 1,
        "rates": [0.0, 0.9],
        "divergences": ["KLD", "RKL"],
        "server_robustness_by_label": {"KLD": 0.0, "RKL": 1.0},
        "seed": 0,
    }
    with pytest.raises(ReportSchemaError, match="unsupported schema_version"):
        validate_report("robustness_sweep", payload)


def test_aggregation_free_energy_rejects_negative_raw_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        aggregation_free_energy(
            consensus_posterior=BELIEFS[0],
            raw_effective_weights=[1.0, -0.1, 1.0],
            local_posteriors=BELIEFS,
            robustness=1.0,
        )


def test_share_round_canonical_fields_and_legacy_alias() -> None:
    canonical = share_round(local_posteriors=BELIEFS, exclude_self=False)
    assert canonical.shared_posteriors.shape == BELIEFS.shape
    with pytest.warns(DeprecationWarning, match="agent_beliefs"):
        legacy = share_round(agent_beliefs=BELIEFS, exclude_self=False)
    assert np.allclose(canonical.shared_posteriors, legacy.shared_posteriors)


@pytest.mark.parametrize(
    "true_state",
    [True, np.nan, "not-an-index", 9],
)
def test_share_round_rejects_malformed_true_state(true_state: object) -> None:
    with pytest.raises(ValueError, match="true_state"):
        share_round(local_posteriors=BELIEFS, true_state=true_state)


def test_share_round_alias_conflicts_and_fail_closed_inputs() -> None:
    with pytest.raises(TypeError, match="both"):
        share_round(local_posteriors=BELIEFS, agent_beliefs=BELIEFS)
    with pytest.raises(TypeError, match="both"):
        share_round(local_posteriors=BELIEFS, base_weights=[1.0, 1.0, 1.0], weights=[1.0, 1.0, 1.0])
    with pytest.raises(TypeError, match="unexpected"):
        share_round(local_posteriors=BELIEFS, typo=1)
    with pytest.raises(TypeError, match="required"):
        share_round()
    with pytest.raises(ValueError, match="AggregationConfig"):
        share_round(local_posteriors=BELIEFS, config="not-a-config")
    with pytest.raises(ValueError, match="mutually exclusive"):
        share_round(
            local_posteriors=BELIEFS,
            config=AggregationConfig(),
            method="naive",
        )


def test_generalized_bayes_validation_and_alpha_branches() -> None:
    prior = np.log(np.asarray([0.5, 0.3, 0.2]))
    loss = np.asarray([0.1, 0.2, 0.7])
    with pytest.raises(ValueError, match="at least one"):
        softmax(np.asarray([]))
    with pytest.raises(ValueError, match="finite"):
        softmax(np.asarray([np.nan]))
    with pytest.raises(TypeError, match="both"):
        generalized_posterior(prior, loss, loss_vec=loss)
    with pytest.raises(TypeError, match="both"):
        generalized_posterior(prior, loss, tau=0.5, learning_rate=1.0)
    with pytest.raises(TypeError, match="unexpected"):
        generalized_posterior(prior, loss, unsupported=1.0)
    with pytest.raises(TypeError, match="required"):
        generalized_posterior(prior)
    with pytest.raises(ValueError, match="same shape"):
        generalized_posterior(prior, loss[:2])
    with pytest.raises(ValueError, match="non-empty"):
        generalized_posterior(np.asarray([]), np.asarray([]))
    with pytest.raises(ValueError, match="finite"):
        generalized_posterior(np.asarray([np.nan, 0.0]), loss[:2])
    with pytest.raises(ValueError, match="tau"):
        generalized_posterior(prior, loss, tau=-1.0)
    with pytest.raises(ValueError, match="unknown"):
        generalized_posterior(prior, loss, divergence="not-a-divergence")

    # Exercise both finite Alpha-Renyi normalization branches: alpha below one
    # uses the full support, while equal losses force the alpha>1 all-state
    # prefix rather than a face solution.
    below_one = generalized_posterior(prior, loss, divergence="AR", alpha=0.5)
    above_one = generalized_posterior(
        prior,
        np.asarray([0.1, 0.1, 0.1]),
        divergence="AR",
        alpha=2.0,
    )
    near_kl = generalized_posterior(prior, loss, divergence="AR", alpha=1.0 + 1e-8)
    assert np.isclose(below_one.sum(), 1.0)
    assert np.isclose(above_one.sum(), 1.0)
    assert np.allclose(near_kl, generalized_posterior(prior, loss))


def test_cavity_and_update_factor_fail_closed_compatibility_paths() -> None:
    posterior = np.asarray([0.55, 0.30, 0.15])
    factor = np.asarray([0.50, 0.25, 0.25])
    with pytest.raises(TypeError, match="both"):
        cavity(global_posterior=posterior, posterior=posterior, site_factor=factor)
    with pytest.raises(TypeError, match="both"):
        cavity(global_posterior=posterior, site_factor=factor, factor=factor)
    with pytest.raises(TypeError, match="unexpected"):
        cavity(global_posterior=posterior, site_factor=factor, typo=1)
    with pytest.raises(TypeError, match="required"):
        cavity()

    canonical = update_factor(
        old_site_factor=factor,
        old_global_posterior=posterior,
        new_global_posterior=np.asarray([0.7, 0.2, 0.1]),
    )
    with pytest.warns(DeprecationWarning):
        legacy = update_factor(
            old_factor=factor,
            old_posterior=posterior,
            new_posterior=np.asarray([0.7, 0.2, 0.1]),
        )
    assert np.allclose(canonical, legacy)
    with pytest.raises(TypeError, match="both"):
        update_factor(
            old_site_factor=factor,
            old_factor=factor,
            old_global_posterior=posterior,
            new_global_posterior=posterior,
        )
    with pytest.raises(TypeError, match="unexpected"):
        update_factor(
            old_site_factor=factor,
            old_global_posterior=posterior,
            new_global_posterior=posterior,
            typo=1,
        )
    with pytest.raises(TypeError, match="required"):
        update_factor()


def test_aggregation_validation_alias_and_dispatch_paths() -> None:
    with pytest.raises(TypeError, match="both"):
        log_linear_pool(local_posteriors=BELIEFS, beliefs=BELIEFS)
    with pytest.raises(TypeError, match="unexpected"):
        log_linear_pool(BELIEFS, typo=1)
    with pytest.raises(TypeError, match="required"):
        log_linear_pool()
    with pytest.raises(TypeError, match="required"):
        robust_aggregate()
    with pytest.raises(TypeError, match="required"):
        aggregate_result()

    with pytest.raises(ValueError, match="length"):
        aggregation_free_energy(
            consensus_posterior=np.asarray([0.5]),
            raw_effective_weights=[1.0, 1.0, 1.0],
            local_posteriors=BELIEFS,
            robustness=1.0,
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        aggregation_free_energy(
            consensus_posterior=np.asarray([0.5, np.nan, 0.5]),
            raw_effective_weights=[1.0, 1.0, 1.0],
            local_posteriors=BELIEFS,
            robustness=1.0,
        )
    with pytest.raises(ValueError, match="positive finite sum"):
        aggregation_free_energy(
            consensus_posterior=np.asarray([0.0, 0.0, 0.0]),
            raw_effective_weights=[1.0, 1.0, 1.0],
            local_posteriors=BELIEFS,
            robustness=1.0,
        )
    with pytest.raises(ValueError, match="finite values"):
        aggregation_free_energy(
            consensus_posterior=BELIEFS[0],
            raw_effective_weights=[1.0, np.nan, 1.0],
            local_posteriors=BELIEFS,
            robustness=1.0,
        )

    with pytest.warns(DeprecationWarning, match="beliefs"):
        aggregate(beliefs=BELIEFS)
    with pytest.raises(TypeError, match="required"):
        aggregate()
    with pytest.raises(TypeError, match="both"):
        aggregate(BELIEFS, base_weights=[1.0, 1.0, 1.0], weights=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="unknown naive"):
        aggregate(BELIEFS, method="naive", robustness=1.0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        aggregate(BELIEFS, config=AggregationConfig(), method="naive")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_seeds": 1}, "n_seeds"),
        ({"n_trials": 1}, "n_trials"),
        ({"rates": ()}, "rates"),
        ({"rates": (0.5, 0.2)}, "sorted"),
        ({"rates": (0.2, 0.2)}, "duplicates"),
        ({"rates": (-0.1, 0.2)}, r"\[0, 1\]"),
        ({"divergences": ("RKL",)}, "KLD"),
        ({"divergences": ("KLD", "RKL", "RKL")}, "duplicates"),
        ({"fdr_alpha": 0.0}, "fdr_alpha"),
        ({"power_alpha": 1.0}, "power_alpha"),
        ({"planning_alternative": "bogus"}, "planning_alternative"),
    ],
)
def test_review_grid_rejects_invalid_controls(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        run_review_grid(**kwargs)


def test_review_grid_consumes_configured_statistical_levels() -> None:
    report = run_review_grid(
        seed=11,
        n_seeds=2,
        n_trials=2,
        n_agents=3,
        rates=(0.0, 0.5),
        divergences=("KLD", "RKL"),
        target_max_mcse=1.0,
        fdr_alpha=0.10,
        power_alpha=0.01,
        planning_alternative="two-sided",
    )
    assert report["statistics"]["fdr_alpha"] == 0.10
    assert report["statistics"]["power_alpha"] == 0.01
    assert report["statistics"]["planning_alternative"] == "two-sided"
    for mechanism in report["statistics"]["by_mechanism"].values():
        for rate_row in mechanism["by_rate"].values():
            for method_row in rate_row["methods"].values():
                assert method_row["planning_power"] >= 0.0
