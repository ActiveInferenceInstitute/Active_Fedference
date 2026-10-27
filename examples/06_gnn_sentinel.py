"""Export the canonical sentinel world as a GNN model without a GNN dependency.

The matrices describe the categorical specialization inspired by Friston et al.
(2024), Figures 1/4. They do not encode the full belief-sharing protocol or
FedGVI client updates. Run with ``uv run --locked python
examples/06_gnn_sentinel.py --format gnn`` to emit the exchange document.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from fedference.pomdp import build_sentinel_world


def main(argv: Sequence[str] | None = None) -> int:
    """Serialize source-owned matrices, retaining their native axis order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "gnn"), default="json")
    parser.add_argument(
        "--verify-extracted",
        type=Path,
        help="compare upstream extraction JSON to the native model",
    )
    args = parser.parse_args(argv)
    world = build_sentinel_world()
    matrices = {
        name: world[name][0].reshape(-1).tolist() if name in ("C", "D") else world[name][0].tolist()
        for name in ("A", "B", "C", "D")
    }
    n_states, n_obs, n_controls = (world[key] for key in ("n_states", "n_obs", "n_controls"))
    if args.verify_extracted is not None:
        try:
            extracted = json.loads(args.verify_extracted.read_text(encoding="utf-8"))
            if (
                not isinstance(extracted, dict)
                or "status" in extracted
                or extracted.get("extraction_schema_version") != "1.0.0"
            ):
                raise ValueError("expected a successful GNN extraction schema 1.0.0 object")
            dimensions = (("num_states", n_states), ("num_observations", n_obs), ("num_actions", n_controls))
            for key, expected in dimensions:
                if extracted.get(key) != expected:
                    raise ValueError(f"extracted {key} differs from the native model")
            for name, key in (("A", "A_matrix"), ("B", "B_matrix"), ("C", "C_vector"), ("D", "D_vector")):
                if not np.array_equal(np.asarray(extracted.get(key)), np.asarray(matrices[name])):
                    raise ValueError(f"extracted {name} shape or values differ from the native model")
        except (OSError, UnicodeError, ValueError) as exc:
            parser.error(str(exc))
    model = "\n".join(
        [
            "# Active Fedference sentinel exchange model",
            "",
            "## GNNSection",
            "ActInfPOMDP",
            "",
            "## GNNVersionAndFlags",
            "GNN v1.0",
            "",
            "## ModelName",
            "Active Fedference canonical sentinel",
            "",
            "## ModelAnnotation",
            "Source: fedference.pomdp.build_sentinel_world() with canonical defaults.",
            "Categorical Friston et al. (2024) Figs. 1/4 specialization; no protocol replication claim.",
            "C contains log-preference potentials; D is an initial probability vector.",
            "Control order: " + ", ".join(world["control_labels"]) + ".",
            "",
            "## StateSpaceBlock",
            f"A[{n_obs},{n_states},type=float]",
            "# B[next_state,previous_state,action]",
            f"B[{n_states},{n_states},{n_controls},type=float]",
            f"C[{n_obs},type=float]",
            f"D[{n_states},type=float]",
            f"s[{n_states},type=float]",
            f"o[{n_obs},type=int]",
            "u[1,type=int]",
            "",
            "## Connections",
            "D>s",
            "s>A",
            "A>o",
            "s>B",
            "u>B",
            "",
            "## InitialParameterization",
            *(f"{name}=" + json.dumps(value, allow_nan=False) for name, value in matrices.items()),
            "",
            "## ModelParameters",
            f"num_states: {n_states}",
            f"num_observations: {n_obs}",
            f"num_actions: {n_controls}",
            "",
        ]
    )
    if args.format == "gnn":
        print(model, end="")
    else:
        print(
            json.dumps(
                {
                    "example": "gnn_sentinel",
                    "gnn_model": model,
                    "control_labels": list(world["control_labels"]),
                    "b_axis_order": ["next_state", "previous_state", "action"],
                    "proves_protocol_replication": False,
                    "extraction_verified": args.verify_extracted is not None,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
