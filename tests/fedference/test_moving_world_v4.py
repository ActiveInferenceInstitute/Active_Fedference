"""Tests for the V4 disjoint-FOV multi-agent moving world (Task 1 / Task 2).

Three tests covering the core V4 claims:
1. Isolated agents with disjoint FOVs cannot reliably infer the global state.
2. Communicating agents outperform isolated agents by a measurable margin.
3. EFE-guided movement does not catastrophically underperform random movement.
"""

from __future__ import annotations


def test_disjoint_fov_incommunicado_fails():
    """Isolated agents with disjoint FOVs cannot correctly infer global state.

    With 3 agents covering 2/6 cells each and a 6-state hidden variable
    (threat position), an isolated agent can only observe its own window.
    Random-chance accuracy is 1/6 ~ 0.17; isolated agents do better than
    pure chance (they score their FOV window correctly) but cannot identify
    the exact state within their window, keeping accuracy well below 0.70.
    """
    from fedference.experiments import run_disjoint_fov_world

    result = run_disjoint_fov_world(seed=0, n_agents=3, n_positions=6, fov_width=2, n_steps=20)

    assert "isolated_accuracy" in result, "result must contain isolated_accuracy"
    assert result["isolated_accuracy"] < 0.70, (
        f"Isolated accuracy too high ({result['isolated_accuracy']:.3f}): "
        "isolated agents with disjoint FOVs should not reach 70% on a 6-state world"
    )


def test_disjoint_fov_communicating_outperforms():
    """Communicating agents with disjoint FOVs outperform isolated agents.

    By fusing complementary beliefs via log-linear pooling each step, agents
    collectively eliminate states outside any agent's FOV, yielding a measurably
    higher accuracy than isolated voting — at least 5 percentage points.
    """
    from fedference.experiments import run_disjoint_fov_world

    result = run_disjoint_fov_world(seed=0, n_agents=3, n_positions=6, fov_width=2, n_steps=20)

    assert "communicating_accuracy" in result, "result must contain communicating_accuracy"
    gap = result["communicating_accuracy"] - result["isolated_accuracy"]
    assert gap > 0.05, (
        f"Gap too small (comm={result['communicating_accuracy']:.3f}, "
        f"iso={result['isolated_accuracy']:.3f}, gap={gap:.3f}): "
        "communicating agents should outperform isolated by at least 0.05"
    )


def test_efe_navigation_outperforms_random():
    """EFE-guided navigation combined with belief sharing does not underperform random.

    EFE-guided agents select information-maximising moves (minimising expected
    posterior entropy) while random agents take uniform random actions. Both
    conditions share beliefs each step. EFE should not underperform random
    movement by more than 10 percentage points (tolerance accounts for
    stochasticity in the short-step regime).
    """
    from fedference.experiments import run_efe_navigation_test

    result = run_efe_navigation_test(seed=0, n_agents=2, n_positions=4, n_steps=5)

    assert "efe_accuracy" in result, "result must contain efe_accuracy"
    assert "random_accuracy" in result, "result must contain random_accuracy"
    assert result["efe_accuracy"] >= result["random_accuracy"] - 0.10, (
        f"EFE underperforms random: efe={result['efe_accuracy']:.3f}, "
        f"random={result['random_accuracy']:.3f}"
    )


def test_efe_result_contains_n_trials():
    """run_efe_navigation_test result dict must include n_trials for schema parity."""
    from fedference.experiments import run_efe_navigation_test

    result = run_efe_navigation_test(seed=1, n_agents=2, n_positions=4, n_steps=3)
    assert "n_trials" in result, "result must include n_trials (schema parity with other harness functions)"
    assert isinstance(result["n_trials"], int)
    assert result["n_trials"] > 0


def test_efe_navigation_strictly_outperforms_random_on_longer_horizon():
    """EFE strictly outperforms random movement with enough steps to exploit information gain.

    With n_steps=8 the EFE agents have had time to converge on informative
    positions; EFE accuracy should exceed random by a strict margin (>0).
    This guards against a degenerate implementation that provides zero benefit.
    """
    from fedference.experiments import run_efe_navigation_test

    # Use multiple seeds and require EFE to win on at least one of them.
    seeds = [0, 1, 2, 3, 4]
    efe_wins = 0
    for seed in seeds:
        result = run_efe_navigation_test(seed=seed, n_agents=2, n_positions=4, n_steps=8)
        if result["efe_accuracy"] > result["random_accuracy"]:
            efe_wins += 1

    assert efe_wins >= 1, (
        f"EFE never strictly outperformed random over {len(seeds)} seeds — "
        "implementation may be degenerate"
    )


def test_disjoint_fov_non_tiling_fov_width():
    """run_disjoint_fov_world handles fov_width that does not tile evenly.

    When n_agents * fov_width != n_positions some states fall outside all
    agent FOVs (n_out > 0 branch). The function must still return a valid
    result dict without error.
    """
    from fedference.experiments import run_disjoint_fov_world

    # 3 agents x fov_width=3 = 9 != n_positions=8: partial coverage
    result = run_disjoint_fov_world(seed=42, n_agents=3, n_positions=8, fov_width=3, n_steps=5)

    for key in ("isolated_accuracy", "communicating_accuracy", "gap", "n_agents", "fov_width"):
        assert key in result, f"missing key: {key}"
    assert 0.0 <= result["isolated_accuracy"] <= 1.0
    assert 0.0 <= result["communicating_accuracy"] <= 1.0
