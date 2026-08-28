"""Exercise direct, one-machine process, and loopback-socket boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from fedference import AggregationConfig, aggregate_result, share_round
from fedference.federation import (
    PROTOCOL_VERSION,
    load_socket_replay,
    run_multiprocess_round,
    run_socket_round,
    validate_socket_replay,
)


def _rounded(values: np.ndarray) -> list[float]:
    """Return stable six-decimal output without changing stored evidence."""
    return [round(float(value), 6) for value in values]


def _prepare_empty_directory(path: str | Path) -> Path:
    """Create a caller-owned directory, refusing to mix with existing files."""
    output_dir = Path(path).expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _write_json(path: Path, payload: object) -> None:
    """Write deterministic, finite JSON for the replay CLI boundary."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new or empty caller-owned directory for beliefs, consensus, and replay JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run equivalent aggregation through each supported local boundary."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        output_dir = _prepare_empty_directory(args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))

    local_posteriors = np.asarray(
        [
            [0.70, 0.20, 0.10],
            [0.60, 0.30, 0.10],
            [0.10, 0.10, 0.80],
        ],
        dtype=np.float64,
    )
    config = AggregationConfig(
        method="robust",
        robustness=1.5,
        max_iter=64,
        tol=1e-9,
    )

    direct = share_round(
        local_posteriors,
        config=config,
        exclude_self=True,
        true_state=0,
    )
    reference_consensus = aggregate_result(local_posteriors, config=config).consensus
    process_consensus = run_multiprocess_round(local_posteriors, config=config)
    replay_path = output_dir / "replay.json"
    socket = run_socket_round(
        local_posteriors,
        config=config,
        round_id="example-round",
        auth_key="local-example-key-not-a-production-secret",
        replay_path=replay_path,
    )
    socket_consensus = np.asarray(socket["consensus"], dtype=np.float64)

    beliefs_path = output_dir / "beliefs.json"
    consensus_path = output_dir / "consensus.json"
    _write_json(beliefs_path, local_posteriors.tolist())
    _write_json(consensus_path, reference_consensus.tolist())
    persisted_replay = load_socket_replay(replay_path)
    persisted_replay_valid = validate_socket_replay(
        persisted_replay,
        local_posteriors,
        reference_consensus,
        config=config,
    )

    payload = {
        "artifacts": sorted(path.name for path in output_dir.iterdir()),
        "config_fingerprint": config.fingerprint,
        "direct_sharing": {
            "exclude_self": True,
            "global_consensus": _rounded(direct.consensus),
            "mean_accuracy": round(direct.mean_accuracy, 6),
            "mean_surprise": round(direct.mean_surprise, 6),
            "shared_posteriors": [_rounded(row) for row in direct.shared_posteriors],
        },
        "example": "federation_boundaries",
        "parity": {
            "direct_bit_identical_to_reference": bool(
                np.array_equal(direct.consensus, reference_consensus)
            ),
            "process_bit_identical_to_reference": bool(
                np.array_equal(reference_consensus, process_consensus)
            ),
            "socket_bit_identical_to_reference": bool(
                np.array_equal(reference_consensus, socket_consensus)
            ),
            "socket_reference_bit_identical": bool(socket["bit_identical"]),
        },
        "replay": {
            "persisted_replay_valid": persisted_replay_valid,
            "socket_replay_valid": bool(socket["replay_valid"]),
        },
        "socket": {
            "authenticated": bool(socket["authenticated"]),
            "n_workers": int(socket["n_workers"]),
            "protocol_version": int(socket["protocol_version"]),
            "round_id": str(socket["round_id"]),
        },
        "transport_scope": "one-machine process and loopback socket only",
    }
    assert int(socket["protocol_version"]) == PROTOCOL_VERSION
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
