"""Evidence-bound single-machine pilots for the open research lanes.

The functions in this module are intentionally small smoke/pilot runners. They
bind existing mathematical primitives to explicit estimands, controls, and
deterministic seeds; they do not promote a pilot into confirmatory evidence.
Optional PyTorch is imported only inside the BNN runner.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .aggregation import AggregationConfig
from .bnn_fedgvi import DiagonalGaussian, FedGVIServerState, load_server_checkpoint, save_server_checkpoint
from .calibration import CalibrationEpisode, calibrate_aggregation, evaluate_locked_aggregation
from .evidence import canonical_sha256
from .protocol_parity import fedgvi_bnn_parity_matrix


def _validate_profile(profile: str) -> None:
    if profile not in ("smoke", "pilot"):
        raise ValueError("single-machine pilots accept only 'smoke' or 'pilot'")


def run_calibration_pilot(*, seed: int = 0, profile: str = "pilot") -> dict[str, object]:
    """Run leakage-free calibration and a disjoint locked evaluation."""
    _validate_profile(profile)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    n_calibration = 4 if profile == "smoke" else 12
    n_evaluation = 3 if profile == "smoke" else 8
    rng = np.random.default_rng(seed)

    def make_episode(index: int, *, evaluation: bool) -> CalibrationEpisode:
        truth = int(rng.integers(0, 3))
        rows = rng.dirichlet(np.full(3, 0.8), size=5)
        rows[:, truth] += 1.5
        rows /= rows.sum(axis=1, keepdims=True)
        prefix = "evaluation" if evaluation else "calibration"
        return CalibrationEpisode(f"{prefix}-{index:03d}", "synthetic-three-state", rows, truth)

    calibration_episodes = tuple(make_episode(index, evaluation=False) for index in range(n_calibration))
    evaluation_episodes = tuple(make_episode(index, evaluation=True) for index in range(n_evaluation))
    candidates = tuple(
        AggregationConfig(method="robust", robustness=value, max_iter=128, tol=1e-9)
        for value in (0.0, 0.1, 0.25)
    )
    result = calibrate_aggregation(calibration_episodes, candidates)
    locked = evaluate_locked_aggregation(evaluation_episodes, result)
    try:
        evaluate_locked_aggregation((calibration_episodes[0],), result)
    except ValueError as exc:
        overlap_control: dict[str, object] = {
            "status": "rejected",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    else:  # pragma: no cover - overlap rejection is a release invariant
        overlap_control = {"status": "accepted_unexpectedly"}
    return {
        "status": "pilot",
        "profile": profile,
        "seed": seed,
        "n_calibration_episodes": n_calibration,
        "n_evaluation_episodes": n_evaluation,
        "primary_estimand": "mean held-out log score over independent calibration worlds",
        "independent_unit": "calibration world; agents and states are nested",
        "selected_config": result.selected.as_dict(),
        "selected_config_fingerprint": result.selected.fingerprint,
        "calibration_sha256": result.calibration_sha256,
        "candidate_scores": [
            {
                "config": score.config.as_dict(),
                "fingerprint": score.config.fingerprint,
                "mean_log_score": score.mean_log_score,
                "per_episode_log_scores": list(score.per_episode_log_scores),
            }
            for score in result.candidates
        ],
        "locked_evaluation": {
            "config_fingerprint": result.selected.fingerprint,
            "mean_log_score": locked.mean_log_score,
            "per_episode_log_scores": list(locked.per_episode_log_scores),
            "episode_ids": [episode.episode_id for episode in evaluation_episodes],
        },
        "overlap_negative_control": overlap_control,
        "no_claim": (
            "calibration separates tuning from evaluation but does not make the "
            "robust server heuristic objective-backed or universally robust"
        ),
    }


def _synthetic_bnn_data(
    seed: int,
    *,
    n_clients: int,
    n_per_class: int,
    contamination: float,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """Create matched binary shards and an independent held-out set."""
    if n_clients < 1 or n_per_class < 1 or not 0.0 <= contamination <= 1.0:
        raise ValueError("invalid synthetic BNN data controls")
    rng = np.random.default_rng(seed)
    clients: list[tuple[np.ndarray, np.ndarray]] = []
    for client_id in range(n_clients):
        negative = rng.normal(loc=(-1.0, 0.0), scale=0.65, size=(n_per_class, 2))
        positive = rng.normal(loc=(1.0, 0.0), scale=0.65, size=(n_per_class, 2))
        x = np.vstack((negative, positive)).astype(np.float32)
        labels = np.concatenate((np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)))
        if contamination and client_id == n_clients - 1:
            n_flip = max(1, int(round(2 * n_per_class * contamination)))
            labels[:n_flip] = 1 - labels[:n_flip]
        y = np.eye(2, dtype=np.float32)[labels]
        clients.append((x, y))
    test_negative = rng.normal(loc=(-1.0, 0.0), scale=0.65, size=(max(8, n_per_class), 2))
    test_positive = rng.normal(loc=(1.0, 0.0), scale=0.65, size=(max(8, n_per_class), 2))
    test_x = np.vstack((test_negative, test_positive)).astype(np.float32)
    test_y = np.concatenate(
        (np.zeros(test_negative.shape[0], dtype=int), np.ones(test_positive.shape[0], dtype=int))
    )
    return clients, test_x, test_y


def _run_bnn_state(
    *,
    clients: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    device: Any,
    beta: float,
    rounds: int,
    local_steps: int,
    hidden_dim: int,
    start_state: FedGVIServerState | None = None,
    start_round: int = 0,
) -> tuple[FedGVIServerState, tuple[str, ...]]:
    """Advance the real BNN site/cavity protocol for a bounded number of rounds."""
    import torch

    from .bnn_variational_torch import VariationalMLP

    template = VariationalMLP(2, hidden_dim=hidden_dim, output_dim=2, seed=seed, beta=beta).to(device)
    dimension = template.to_diagonal_gaussian().mean.size
    prior = DiagonalGaussian(np.zeros(dimension), np.ones(dimension))
    state = start_state or FedGVIServerState.initialize(prior, n_clients=len(clients))
    checkpoint_fingerprints: list[str] = []
    for round_index in range(start_round, rounds):
        posteriors: dict[int, DiagonalGaussian] = {}
        for client_id, (features, labels) in enumerate(clients):
            model = VariationalMLP(
                2,
                hidden_dim=hidden_dim,
                output_dim=2,
                seed=seed + 1000 * round_index + client_id,
                beta=beta,
            ).to(device)
            model.fit_from_cavity(
                state.cavity(client_id),
                torch.as_tensor(features),
                torch.as_tensor(labels),
                n_steps=local_steps,
                lr=0.03,
                n_mc=1,
                kl_weight=1.0,
                beta=beta,
            )
            posteriors[client_id] = model.to_diagonal_gaussian()
        state = state.advance_round(posteriors, schedule="parallel")
        checkpoint_fingerprints.append(state.fingerprint)
    return state, tuple(checkpoint_fingerprints)


def _bnn_log_score(
    state: FedGVIServerState,
    *,
    seed: int,
    device: Any,
    beta: float,
    hidden_dim: int,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> float:
    import torch

    from .bnn_variational_torch import VariationalMLP

    model = VariationalMLP(2, hidden_dim=hidden_dim, output_dim=2, seed=seed, beta=beta).to(device)
    model.load_diagonal_gaussian(state.posterior())
    probabilities = model.predict_proba(torch.as_tensor(test_x), deterministic=True).cpu().numpy()
    return float(np.mean(np.log(np.maximum(probabilities[np.arange(test_y.size), test_y], 1e-12))))


def run_fedgvi_bnn_pilot(
    *,
    seed: int = 0,
    profile: str = "pilot",
    requested_device: str = "cpu",
) -> dict[str, object]:
    """Run a portable CPU/MPS FedGVI-vs-PVI proper-score pilot.

    The source-scale CUDA profile remains declarative.  This pilot uses a small
    synthetic binary task so the protocol, checkpoint/resume equivalence, and
    device receipt can be exercised on one workstation without pretending to
    reproduce FashionMNIST source-scale numbers.
    """
    _validate_profile(profile)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    try:
        from .torch_bnn import configure_torch_determinism, resolve_torch_device
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("PyTorch is required for the portable FedGVI BNN pilot") from exc
    configure_torch_determinism(seed)
    device, device_receipt = resolve_torch_device(requested_device, allow_cpu_fallback=True)
    rounds = 1 if profile == "smoke" else 2
    local_steps = 1 if profile == "smoke" else 3
    n_per_class = 4 if profile == "smoke" else 8
    contamination_levels = (0.0, 0.6)
    rows: list[dict[str, object]] = []
    checkpoints: list[str] = []
    for contamination in contamination_levels:
        clients, test_x, test_y = _synthetic_bnn_data(
            seed + int(contamination * 100),
            n_clients=3,
            n_per_class=n_per_class,
            contamination=contamination,
        )
        method_states: dict[str, FedGVIServerState] = {}
        method_scores: dict[str, float] = {}
        for method, beta in (("fedgvi", 0.5), ("pvi-nll", 0.0)):
            state, state_checkpoints = _run_bnn_state(
                clients=clients,
                seed=seed,
                device=device,
                beta=beta,
                rounds=rounds,
                local_steps=local_steps,
                hidden_dim=8,
            )
            method_states[method] = state
            checkpoints.extend(state_checkpoints)
            method_scores[method] = _bnn_log_score(
                state,
                seed=seed,
                device=device,
                beta=beta,
                hidden_dim=8,
                test_x=test_x,
                test_y=test_y,
            )
        # Round-level interruption/resume gate: save at the first round and
        # replay the remaining deterministic update from that exact state.
        if rounds > 1:
            with tempfile.TemporaryDirectory(prefix="active-fedference-bnn-") as scratch:
                first, _ = _run_bnn_state(
                    clients=clients,
                    seed=seed,
                    device=device,
                    beta=0.5,
                    rounds=1,
                    local_steps=local_steps,
                    hidden_dim=8,
                )
                checkpoint = save_server_checkpoint(Path(scratch) / "round-1.json", first)
                resumed = load_server_checkpoint(checkpoint)
                resumed_final, _ = _run_bnn_state(
                    clients=clients,
                    seed=seed,
                    device=device,
                    beta=0.5,
                    rounds=rounds,
                    local_steps=local_steps,
                    hidden_dim=8,
                    start_state=resumed,
                    start_round=1,
                )
                resume_equivalent = resumed_final.fingerprint == method_states["fedgvi"].fingerprint
        else:
            resume_equivalent = True
        rows.append(
            {
                "contamination": contamination,
                "fedgvi_log_score": method_scores["fedgvi"],
                "pvi_nll_log_score": method_scores["pvi-nll"],
                "fedgvi_minus_pvi_log_score": method_scores["fedgvi"] - method_scores["pvi-nll"],
                "checkpoint_resume_equivalent": resume_equivalent,
                "n_clients": len(clients),
                "rounds": rounds,
                "local_steps": local_steps,
            }
        )
    return {
        "status": "pilot",
        "profile": profile,
        "seed": seed,
        "device": device_receipt.__dict__,
        "backend": "torch-mean-field-diagonal-gaussian",
        "protocol_parity": fedgvi_bnn_parity_matrix().as_dict(),
        "rows": rows,
        "checkpoint_fingerprints": list(checkpoints),
        "primary_estimand": (
            "paired held-out log-score difference between cavity-conditioned FedGVI and PVI/NLL"
        ),
        "independent_unit": "independently seeded end-to-end synthetic BNN run",
        "negative_controls": {
            "pvi_nll_baseline": "matched beta=0 cavity/site-factor protocol",
            "checkpoint_resume": all(bool(row["checkpoint_resume_equivalent"]) for row in rows),
        },
        "source_scale_boundary": (
            "FashionMNIST/MNIST/KMNIST and CUDA source-scale execution remain open "
            "external or later portable lanes"
        ),
        "no_claim": (
            "a synthetic CPU/MPS pilot does not establish the source-scale BNN "
            "effect or a server robustness theorem"
        ),
        "report_fingerprint": canonical_sha256(rows),
    }


__all__ = ["run_calibration_pilot", "run_fedgvi_bnn_pilot"]
