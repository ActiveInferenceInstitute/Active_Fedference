# src/ — Active Fedference project logic

Core implementations, deterministic analysis pipeline, figure factories, and
manuscript-variable generation for the active-inference federation study.

## Layout

```text
src/
├── fedference/             # Domain logic + explicit I/O/optional-Torch boundaries
├── fedference_cli/         # Installed list/run/benchmark/verify/replay adapter
├── analysis/               # workflow.py + schemas + canonical artifact contract
├── figures/                # Figure generators used by reproducible scripts
├── manuscript_vars/         # Deterministic manuscript token generator package
├── publication/            # Publication packaging, freshness, and surface/accessibility validation
├── experiment_config.py     # Single source of truth for experiment parameters
├── manuscript_variables.py  # Re-export shim over manuscript_vars/
├── invariants.py           # Contract checks for algebraic limits
├── documentation.py        # API doc helpers
├── project_paths.py        # Project-root path helpers
└── _runtime.py             # Shared utility helpers
```

`project_paths.resolve_script_project_root()` is the shared boundary for
script entry points that need to target a checkout other than the one
containing the script. It gives an explicit `--project-root` value precedence
over the validated `ACTIVE_FEDFERENCE_PROJECT_ROOT` review override, so
individual scripts do not grow incompatible path-selection rules.

## Entry points and usage

```python
from experiment_config import load_experiment_config
from fedference import AggregationConfig, aggregate_result, generalized_posterior
from fedference.experiments import run_belief_sharing

cfg = load_experiment_config()
print(cfg.seed)
aggregation = AggregationConfig(method="robust", robustness=1.5)
print(aggregate_result([[0.3, 0.7], [0.4, 0.6]], config=aggregation).consensus)
print(run_belief_sharing(cfg.seed).keys())
```

## Principles

- Keep mathematical primitives in `src/fedference/`.
- Keep scripts thin: this package provides testable functions, scripts invoke
  them and provide stable subprocess/CI boundaries rather than reimplementing
  domain logic.
- Keep outputs deterministic by default (`seed` in `manuscript/config.yaml`).
- Keep API exports synchronized with `__init__.py` and `__all__`.
- Keep mathematical primitives side-effect free. Restrict filesystem/network
  effects to the named evidence, data, checkpoint, and transport adapters, all
  with explicit caller-owned paths.
- Keep Torch behind the optional `bnn` extra; `fedference.torch_bnn` must never
  become a default import of the NumPy/SciPy core.
- Registry state declares intended evidence. Only a verified `RunReceipt` plus
  the relevant scientific gates can support an executed-result statement.
- The HTML validator enforces a deterministic accessibility subset; passing it
  must never be presented as WCAG or PDF/UA conformance.

Prefer composing the typed boundaries rather than importing implementation
details: `AggregationConfig` + `aggregate_result` for new pooling calls,
`share_round`/federation adapters for transport, `evidence`/receipt classes for
provenance, and `src/figures/_metadata.py` plus report schemas for publication
visuals. Legacy public names remain supported by the API-stability contract.

## Analysis and figures

`src/analysis/workflow.py` is the orchestration layer for study execution and
artifacts; every JSON report and the figure registry are validated against
`src/analysis/report_schemas.py` at the single write boundary before anything
lands on disk. `src/analysis/artifacts.py` separately declares the complete
expected Stage-02 hand-off for audits, so validators never promote arbitrary
files discovered under `output/` (see [`analysis/README.md`](analysis/README.md)).
`src/figures/*.py` are figure generators; each file maps to one or more files
in `output/figures/` and has a corresponding test in `tests/figures/`.

## Commands

From project root:

```bash
uv run --locked --extra dev pytest tests/
uv run --locked fedference list --json
uv run --locked python scripts/02_run_analysis.py
uv run --locked python scripts/z_generate_manuscript_variables.py
```

## Further reading

- [`AGENTS.md`](AGENTS.md)
- [`STYLE.md`](STYLE.md)
- [`../docs/core/architecture.md`](../docs/core/architecture.md)
