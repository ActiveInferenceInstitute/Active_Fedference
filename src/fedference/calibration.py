"""Held-out calibration for server robustness controls (MAJ-8).

Calibration episodes are explicitly separated from confirmatory episodes. The
selector uses mean log score—a proper scoring rule—and returns a frozen
:class:`~fedference.aggregation.AggregationConfig` plus content hashes. It does
not inspect or optimize any later evaluation episode.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ._validation import as_pmf_matrix
from .aggregation import AggregationConfig, aggregate_result
from .evidence import canonical_sha256

ArrayF = np.ndarray
_EPS = 1e-12


@dataclass(frozen=True, init=False)
class CalibrationEpisode:
    """One independent world used only for hyperparameter calibration."""

    episode_id: str
    world_family: str
    local_posteriors: ArrayF
    true_state: int

    def __init__(
        self,
        episode_id: str,
        world_family: str,
        local_posteriors: ArrayF | None = None,
        true_state: int | None = None,
        **legacy: object,
    ) -> None:
        """Construct one episode using canonical or warned legacy names."""
        if "beliefs" in legacy:
            if local_posteriors is not None:
                raise TypeError(
                    "local_posteriors and deprecated beliefs cannot both be supplied"
                )
            local_posteriors = legacy.pop("beliefs")  # type: ignore[assignment]
            warnings.warn(
                "beliefs is deprecated; use local_posteriors",
                DeprecationWarning,
                stacklevel=2,
            )
        if legacy:
            names = ", ".join(sorted(legacy))
            raise TypeError(f"unexpected keyword argument(s): {names}")
        if local_posteriors is None:
            raise TypeError("local_posteriors is required")
        if true_state is None:
            raise TypeError("true_state is required")
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "world_family", world_family)
        object.__setattr__(self, "local_posteriors", local_posteriors)
        object.__setattr__(self, "true_state", true_state)
        self.__post_init__()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.episode_id, str)
            or not self.episode_id.strip()
            or not isinstance(self.world_family, str)
            or not self.world_family.strip()
        ):
            raise ValueError("episode_id and world_family must be non-empty")
        matrix = as_pmf_matrix(self.local_posteriors, name="local_posteriors")
        if matrix.shape[1] < 2:
            raise ValueError("local_posteriors must contain at least two states")
        if (
            isinstance(self.true_state, bool)
            or not isinstance(self.true_state, (int, np.integer))
            or not 0 <= int(self.true_state) < matrix.shape[1]
        ):
            raise ValueError("true_state must index the shared state space")
        matrix = matrix.copy()
        matrix.setflags(write=False)
        object.__setattr__(self, "local_posteriors", matrix)

    @property
    def beliefs(self) -> ArrayF:
        """Deprecated alias for :attr:`local_posteriors`."""
        warnings.warn(
            "CalibrationEpisode.beliefs is deprecated; use local_posteriors",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.local_posteriors


@dataclass(frozen=True)
class CandidateScore:
    """Calibration score for one candidate configuration."""

    config: AggregationConfig
    mean_log_score: float
    per_episode_log_scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.config, AggregationConfig):
            raise ValueError("config must be an AggregationConfig")
        if not isinstance(self.per_episode_log_scores, (tuple, list)):
            raise ValueError("per_episode_log_scores must be a sequence")
        object.__setattr__(
            self,
            "per_episode_log_scores",
            tuple(self.per_episode_log_scores),
        )
        if not self.per_episode_log_scores:
            raise ValueError("per_episode_log_scores must be non-empty")
        if any(
            isinstance(score, (bool, np.bool_))
            or not isinstance(
                score,
                (int, float, np.integer, np.floating),
            )
            for score in self.per_episode_log_scores
        ):
            raise ValueError("per_episode_log_scores must be finite scalars")
        scores = np.asarray(self.per_episode_log_scores, dtype=np.float64)
        if scores.ndim != 1 or not np.all(np.isfinite(scores)):
            raise ValueError("per_episode_log_scores must be finite scalars")
        if (
            isinstance(self.mean_log_score, (bool, np.bool_))
            or not isinstance(
                self.mean_log_score,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(self.mean_log_score)
            or not np.isclose(
                float(self.mean_log_score),
                float(np.mean(scores)),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError("mean_log_score must equal the per-episode mean")
        object.__setattr__(self, "mean_log_score", float(self.mean_log_score))


def _candidate_payload(score: CandidateScore) -> dict[str, object]:
    """Return the canonical, JSON-compatible declaration for one candidate."""
    return {
        "config": score.config.as_dict(),
        "fingerprint": score.config.fingerprint,
        "mean_log_score": score.mean_log_score,
        "per_episode_log_scores": list(score.per_episode_log_scores),
    }


def _calibration_payload(
    *,
    selected: AggregationConfig,
    candidates: tuple[CandidateScore, ...],
    calibration_episode_ids: tuple[str, ...],
    calibration_episode_sha256s: tuple[str, ...],
    calibration_episode_world_families: tuple[str, ...],
    primary_estimand: str,
) -> dict[str, object]:
    """Return the complete declared payload bound by ``calibration_sha256``."""
    return {
        "calibration_episode_ids": list(calibration_episode_ids),
        "calibration_episode_sha256s": list(calibration_episode_sha256s),
        "calibration_episode_world_families": list(
            calibration_episode_world_families
        ),
        "candidates": [_candidate_payload(score) for score in candidates],
        "primary_estimand": primary_estimand,
        "selected_config_fingerprint": selected.fingerprint,
    }


@dataclass(frozen=True)
class CalibrationResult:
    """Frozen selected configuration and self-verifying calibration provenance."""

    selected: AggregationConfig
    candidates: tuple[CandidateScore, ...]
    calibration_episode_ids: tuple[str, ...]
    calibration_episode_sha256s: tuple[str, ...]
    calibration_episode_world_families: tuple[str, ...]
    calibration_sha256: str
    primary_estimand: str = "mean held-out log score over calibration worlds"

    def __post_init__(self) -> None:
        if not isinstance(self.selected, AggregationConfig):
            raise ValueError("selected must be an AggregationConfig")
        for name in (
            "candidates",
            "calibration_episode_ids",
            "calibration_episode_sha256s",
            "calibration_episode_world_families",
        ):
            if not isinstance(getattr(self, name), (tuple, list)):
                raise ValueError(f"{name} must be a sequence")
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.candidates or any(
            not isinstance(score, CandidateScore) for score in self.candidates
        ):
            raise ValueError("candidates must contain CandidateScore values")
        object.__setattr__(
            self,
            "candidates",
            tuple(sorted(self.candidates, key=lambda score: score.config.fingerprint)),
        )
        fingerprints = tuple(score.config.fingerprint for score in self.candidates)
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("candidate configurations must be unique")
        if self.selected.fingerprint not in fingerprints:
            raise ValueError("selected configuration must be one of the candidates")
        if (
            not self.calibration_episode_ids
            or any(
                not isinstance(episode_id, str) or not episode_id.strip()
                for episode_id in self.calibration_episode_ids
            )
            or len(set(self.calibration_episode_ids)) != len(self.calibration_episode_ids)
        ):
            raise ValueError("calibration_episode_ids must be non-empty and unique")
        if len(self.calibration_episode_sha256s) != len(self.calibration_episode_ids):
            raise ValueError("calibration episode hashes must align with episode ids")
        if len(self.calibration_episode_world_families) != len(
            self.calibration_episode_ids
        ) or any(
            not isinstance(world_family, str) or not world_family.strip()
            for world_family in self.calibration_episode_world_families
        ):
            raise ValueError("calibration world families must align with episode ids")
        if any(
            len(score.per_episode_log_scores) != len(self.calibration_episode_ids)
            for score in self.candidates
        ):
            raise ValueError("candidate scores must align with calibration episodes")
        if len(set(self.calibration_episode_sha256s)) != len(self.calibration_episode_sha256s):
            raise ValueError("calibration episode contents must be unique")
        if any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
            for digest in self.calibration_episode_sha256s
        ):
            raise ValueError("calibration episode hashes must be SHA-256 digests")
        if (
            not isinstance(self.calibration_sha256, str)
            or len(self.calibration_sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in self.calibration_sha256)
        ):
            raise ValueError("calibration_sha256 must be a SHA-256 digest")
        if not isinstance(self.primary_estimand, str) or not self.primary_estimand.strip():
            raise ValueError("primary_estimand must be non-empty")
        selected_score = min(
            self.candidates,
            key=lambda candidate: (
                -candidate.mean_log_score,
                candidate.config.fingerprint,
            ),
        )
        if self.selected.fingerprint != selected_score.config.fingerprint:
            raise ValueError(
                "selected configuration must be the deterministic highest-score candidate"
            )
        expected_sha256 = canonical_sha256(
            _calibration_payload(
                selected=self.selected,
                candidates=self.candidates,
                calibration_episode_ids=self.calibration_episode_ids,
                calibration_episode_sha256s=self.calibration_episode_sha256s,
                calibration_episode_world_families=self.calibration_episode_world_families,
                primary_estimand=self.primary_estimand,
            )
        )
        if self.calibration_sha256 != expected_sha256:
            raise ValueError(
                "calibration_sha256 does not match the canonical calibration payload"
            )


def _episode_content_payload(episode: CalibrationEpisode) -> dict[str, object]:
    matrix = np.asarray(episode.local_posteriors, dtype=np.float64)
    return {
        "local_posteriors": matrix.tolist(),
        "true_state": int(episode.true_state),
    }


def _score(
    episodes: tuple[CalibrationEpisode, ...],
    config: AggregationConfig,
) -> CandidateScore:
    """Score a candidate only when every aggregation execution is trustworthy."""
    scores: list[float] = []
    for episode in episodes:
        aggregation = aggregate_result(
            episode.local_posteriors,
            config=config,
        )
        if aggregation.fallback_events:
            raise ValueError(
                "candidate configuration used a numerical fallback during calibration "
                f"on episode {episode.episode_id!r}: {aggregation.fallback_events}"
            )
        if not aggregation.converged:
            raise ValueError(
                "candidate configuration did not converge during calibration "
                f"on episode {episode.episode_id!r}"
            )
        scores.append(
            float(
                np.log(
                    max(
                        aggregation.consensus[int(episode.true_state)],
                        _EPS,
                    )
                )
            )
        )
    return CandidateScore(
        config=config,
        mean_log_score=float(np.mean(scores)),
        per_episode_log_scores=tuple(scores),
    )


def calibrate_aggregation(
    episodes: Iterable[CalibrationEpisode],
    candidates: Iterable[AggregationConfig],
) -> CalibrationResult:
    """Select and freeze the highest-log-score candidate on calibration data.

    Ties are resolved by the canonical configuration fingerprint, making the
    decision independent of caller iteration order.
    """
    episode_tuple = tuple(episodes)
    candidate_tuple = tuple(candidates)
    if not episode_tuple:
        raise ValueError("calibration episodes must be non-empty")
    if not candidate_tuple:
        raise ValueError("candidate configurations must be non-empty")
    if any(not isinstance(episode, CalibrationEpisode) for episode in episode_tuple):
        raise ValueError("calibration episodes must be CalibrationEpisode values")
    if any(not isinstance(config, AggregationConfig) for config in candidate_tuple):
        raise ValueError("candidates must be AggregationConfig values")
    episode_ids = tuple(episode.episode_id for episode in episode_tuple)
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("calibration episode ids must be unique")
    fingerprints = tuple(config.fingerprint for config in candidate_tuple)
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("candidate configurations must be unique")

    episode_tuple = tuple(sorted(episode_tuple, key=lambda episode: episode.episode_id))
    candidate_tuple = tuple(sorted(candidate_tuple, key=lambda config: config.fingerprint))
    episode_sha256s = tuple(
        canonical_sha256(_episode_content_payload(episode))
        for episode in episode_tuple
    )
    if len(set(episode_sha256s)) != len(episode_sha256s):
        raise ValueError("calibration episode contents must be unique")
    scored = tuple(_score(episode_tuple, config) for config in candidate_tuple)
    selected_score = min(
        scored,
        key=lambda candidate: (
            -candidate.mean_log_score,
            candidate.config.fingerprint,
        ),
    )
    calibration_hash = canonical_sha256(
        _calibration_payload(
            selected=selected_score.config,
            candidates=scored,
            calibration_episode_ids=tuple(
                episode.episode_id for episode in episode_tuple
            ),
            calibration_episode_sha256s=episode_sha256s,
            calibration_episode_world_families=tuple(
                episode.world_family for episode in episode_tuple
            ),
            primary_estimand="mean held-out log score over calibration worlds",
        )
    )
    return CalibrationResult(
        selected=selected_score.config,
        candidates=scored,
        calibration_episode_ids=tuple(episode.episode_id for episode in episode_tuple),
        calibration_episode_sha256s=episode_sha256s,
        calibration_episode_world_families=tuple(
            episode.world_family for episode in episode_tuple
        ),
        calibration_sha256=calibration_hash,
    )


def evaluate_locked_aggregation(
    episodes: Iterable[CalibrationEpisode],
    calibration: CalibrationResult,
) -> CandidateScore:
    """Evaluate a frozen configuration and reject calibration/evaluation overlap."""
    if not isinstance(calibration, CalibrationResult):
        raise ValueError("calibration must be a CalibrationResult")
    evaluation = tuple(episodes)
    if not evaluation:
        raise ValueError("evaluation episodes must be non-empty")
    if any(not isinstance(episode, CalibrationEpisode) for episode in evaluation):
        raise ValueError("evaluation episodes must be CalibrationEpisode values")
    evaluation_ids = tuple(episode.episode_id for episode in evaluation)
    if len(set(evaluation_ids)) != len(evaluation_ids):
        raise ValueError("evaluation episode ids must be unique")
    overlap = set(calibration.calibration_episode_ids).intersection(
        episode.episode_id for episode in evaluation
    )
    if overlap:
        raise ValueError(f"evaluation episodes overlap calibration data: {sorted(overlap)}")
    evaluation = tuple(sorted(evaluation, key=lambda episode: episode.episode_id))
    content_overlap = set(calibration.calibration_episode_sha256s).intersection(
        canonical_sha256(_episode_content_payload(episode))
        for episode in evaluation
    )
    if content_overlap:
        raise ValueError("evaluation episode contents overlap calibration data under different ids")
    return _score(evaluation, calibration.selected)


__all__ = [
    "CalibrationEpisode",
    "CalibrationResult",
    "CandidateScore",
    "calibrate_aggregation",
    "evaluate_locked_aggregation",
]
