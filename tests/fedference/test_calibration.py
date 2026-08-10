"""Held-out calibration and evaluation separation for MAJ-8."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fedference.aggregation import AggregationConfig
from fedference.calibration import (
    CalibrationEpisode,
    CalibrationResult,
    CandidateScore,
    calibrate_aggregation,
    evaluate_locked_aggregation,
)


def _episodes(
    prefix: str,
    *,
    shift: float = 0.0,
) -> tuple[CalibrationEpisode, ...]:
    return (
        CalibrationEpisode(
            episode_id=f"{prefix}-clean",
            world_family="clean",
            local_posteriors=np.asarray(
                [
                    [0.8 - shift, 0.2 + shift],
                    [0.7 - shift, 0.3 + shift],
                    [0.6 - shift, 0.4 + shift],
                ]
            ),
            true_state=0,
        ),
        CalibrationEpisode(
            episode_id=f"{prefix}-outlier",
            world_family="severe",
            local_posteriors=np.asarray(
                [
                    [0.8 - shift, 0.2 + shift],
                    [0.75 - shift, 0.25 + shift],
                    [0.01 + shift, 0.99 - shift],
                ]
            ),
            true_state=0,
        ),
    )


def test_calibration_is_deterministic_and_freezes_a_declared_candidate() -> None:
    candidates = (
        AggregationConfig(method="naive"),
        AggregationConfig(method="robust", robustness=1.0),
        AggregationConfig(
            method="robust",
            robustness=3.0,
            max_iter=512,
            tol=1e-6,
        ),
    )
    first = calibrate_aggregation(_episodes("cal"), candidates)
    second = calibrate_aggregation(reversed(_episodes("cal")), reversed(candidates))
    assert first.selected.fingerprint == second.selected.fingerprint
    assert first.calibration_sha256 == second.calibration_sha256
    assert first.calibration_episode_sha256s == second.calibration_episode_sha256s
    assert first.calibration_episode_world_families == ("clean", "severe")
    assert first.candidates == second.candidates
    assert first.selected in candidates
    assert all(np.isfinite(score.mean_log_score) for score in first.candidates)


def test_calibration_ties_use_the_canonical_configuration_fingerprint() -> None:
    candidates = (
        AggregationConfig(method="naive", robustness=1.0),
        AggregationConfig(method="naive", robustness=2.0),
    )
    first = calibrate_aggregation(_episodes("cal"), candidates)
    second = calibrate_aggregation(_episodes("cal"), reversed(candidates))
    expected = min(candidate.fingerprint for candidate in candidates)
    assert first.selected.fingerprint == expected
    assert second.selected.fingerprint == expected


def test_locked_evaluation_rejects_calibration_episode_reuse() -> None:
    calibration = calibrate_aggregation(
        _episodes("cal"),
        (
            AggregationConfig(method="naive"),
            AggregationConfig(method="variational", robustness=1.0),
        ),
    )
    evaluation = evaluate_locked_aggregation(
        _episodes("eval", shift=0.01),
        calibration,
    )
    assert np.isfinite(evaluation.mean_log_score)
    with pytest.raises(ValueError, match="overlap calibration"):
        evaluate_locked_aggregation(_episodes("cal"), calibration)
    relabelled = replace(
        _episodes("cal")[0],
        episode_id="renamed-evaluation-world",
        world_family="renamed-family",
    )
    with pytest.raises(ValueError, match="contents overlap calibration"):
        evaluate_locked_aggregation((relabelled,), calibration)


def test_calibration_rejects_duplicate_ids_and_candidates() -> None:
    episode = _episodes("same")[0]
    config = AggregationConfig(method="naive")
    with pytest.raises(ValueError, match="episode ids"):
        calibrate_aggregation((episode, episode), (config,))
    with pytest.raises(ValueError, match="candidate configurations"):
        calibrate_aggregation((episode,), (config, config))
    with pytest.raises(ValueError, match="episode contents"):
        calibrate_aggregation(
            (episode, replace(episode, episode_id="relabeled")),
            (config,),
        )


def test_calibration_episode_normalizes_and_rejects_invalid_simplex_rows() -> None:
    episode = CalibrationEpisode(
        episode_id="scaled",
        world_family="clean",
        local_posteriors=np.asarray([[8.0, 2.0], [3.0, 7.0]]),
        true_state=0,
    )
    assert np.allclose(episode.local_posteriors.sum(axis=1), 1.0)
    with pytest.raises(ValueError, match="finite"):
        CalibrationEpisode(
            episode_id="nan",
            world_family="clean",
            local_posteriors=np.asarray([[np.nan, 1.0], [0.5, 0.5]]),
            true_state=0,
        )


def test_calibration_contract_rejects_forged_results_and_duplicate_evaluation() -> None:
    episodes = _episodes("cal")
    result = calibrate_aggregation(
        episodes,
        (
            AggregationConfig(method="naive"),
            AggregationConfig(method="robust", robustness=1.0),
        ),
    )
    with pytest.raises(ValueError, match="one of the candidates"):
        replace(
            result,
            selected=AggregationConfig(method="robust", robustness=99.0),
        )
    with pytest.raises(ValueError, match="SHA-256"):
        replace(result, calibration_sha256="bad")
    with pytest.raises(ValueError, match="canonical calibration payload"):
        replace(result, calibration_sha256="f" * 64)
    losing_config = next(
        score.config for score in result.candidates if score.config != result.selected
    )
    with pytest.raises(ValueError, match="deterministic highest-score candidate"):
        replace(result, selected=losing_config)
    with pytest.raises(ValueError, match="align"):
        replace(result, calibration_episode_sha256s=())
    with pytest.raises(ValueError, match="per-episode mean"):
        CandidateScore(
            config=AggregationConfig(method="naive"),
            mean_log_score=0.0,
            per_episode_log_scores=(-1.0,),
        )
    with pytest.raises(ValueError, match="evaluation episode ids"):
        evaluate_locked_aggregation(
            (_episodes("eval")[0], _episodes("eval")[0]),
            result,
        )


def test_calibration_hash_binds_the_complete_candidate_declaration() -> None:
    result = calibrate_aggregation(
        _episodes("cal"),
        (AggregationConfig(method="naive"),),
    )
    score = result.candidates[0]
    tampered_config = replace(result.selected, tol=result.selected.tol / 2.0)
    with pytest.raises(ValueError, match="canonical calibration payload"):
        replace(
            result,
            selected=tampered_config,
            candidates=(replace(score, config=tampered_config),),
        )
    changed_scores = tuple(
        value + 0.01 for value in score.per_episode_log_scores
    )
    with pytest.raises(ValueError, match="canonical calibration payload"):
        replace(
            result,
            candidates=(
                replace(
                    score,
                    mean_log_score=float(np.mean(changed_scores)),
                    per_episode_log_scores=changed_scores,
                ),
            ),
        )
    with pytest.raises(ValueError, match="canonical calibration payload"):
        replace(
            result,
            calibration_episode_world_families=("retagged", "severe"),
        )


def test_calibration_rejects_nonconverged_and_fallback_candidates() -> None:
    episode = _episodes("cal")[0]
    with pytest.raises(ValueError, match="did not converge during calibration"):
        calibrate_aggregation(
            (episode,),
            (
                AggregationConfig(
                    method="variational",
                    robustness=1.0,
                    max_iter=1,
                    multistart=False,
                ),
            ),
        )
    fallback_episode = CalibrationEpisode(
        episode_id="fallback",
        world_family="extreme",
        local_posteriors=np.asarray([[0.99, 0.01], [0.01, 0.99]]),
        true_state=0,
    )
    with pytest.raises(ValueError, match="numerical fallback during calibration"):
        calibrate_aggregation(
            (fallback_episode,),
            (AggregationConfig(method="robust", robustness=1_000_000.0),),
        )


def test_calibration_episode_owns_immutable_belief_bytes() -> None:
    source = np.asarray([[0.8, 0.2], [0.7, 0.3]])
    episode = CalibrationEpisode(
        episode_id="immutable",
        world_family="clean",
        local_posteriors=source,
        true_state=0,
    )
    source[0] = [0.1, 0.9]
    assert np.array_equal(episode.local_posteriors[0], np.asarray([0.8, 0.2]))
    with pytest.raises(ValueError, match="read-only"):
        episode.local_posteriors[0, 0] = 0.5


def test_calibration_episode_legacy_beliefs_alias_warns() -> None:
    with pytest.warns(DeprecationWarning, match="beliefs is deprecated"):
        episode = CalibrationEpisode(
            episode_id="legacy",
            world_family="clean",
            beliefs=np.asarray([[0.8, 0.2], [0.7, 0.3]]),
            true_state=0,
        )
    with pytest.warns(DeprecationWarning, match="CalibrationEpisode.beliefs"):
        assert np.array_equal(episode.beliefs, episode.local_posteriors)


def test_calibration_scores_reject_coercion_and_misaligned_episode_counts() -> None:
    config = AggregationConfig(method="naive")
    with pytest.raises(ValueError, match="finite scalars"):
        CandidateScore(
            config=config,
            mean_log_score=0.0,
            per_episode_log_scores=("0.0",),
        )
    with pytest.raises(ValueError, match="mean_log_score"):
        CandidateScore(
            config=config,
            mean_log_score=True,
            per_episode_log_scores=(1.0,),
        )
    score = CandidateScore(
        config=config,
        mean_log_score=0.0,
        per_episode_log_scores=(0.0,),
    )
    with pytest.raises(ValueError, match="align with calibration episodes"):
        CalibrationResult(
            selected=config,
            candidates=(score,),
            calibration_episode_ids=("cal-1", "cal-2"),
            calibration_episode_sha256s=("a" * 64, "b" * 64),
            calibration_episode_world_families=("clean", "severe"),
            calibration_sha256="c" * 64,
        )
