# src/ — Active Fedference source tree

Business logic for robust federated active inference. Deep architecture:
[`../docs/core/architecture.md`](../docs/core/architecture.md).

## Layout

```
src/
├── fedference/          # Domain logic + explicit boundary adapters (NO infrastructure imports)
├── fedference_cli/      # Installed list/run/benchmark/verify/replay adapter
├── analysis/            # workflow.py orchestrator + report_schemas.py write-boundary validator
├── figures/             # manuscript figure generators + _common.py
├── manuscript_vars/     # Manuscript token generator package (loaders/tokens/render/generate)
├── publication/         # Publication packaging, identifiers, Zenodo, freshness, and surface validation
├── experiment_config.py # Loads manuscript/config.yaml
├── manuscript_variables.py  # Re-export shim over manuscript_vars/
├── invariants.py
├── documentation.py     # API doc helpers
├── project_paths.py
└── _runtime.py
```

## `fedference/` package (domain layer)

Mathematical modules use NumPy/SciPy and remain free of filesystem/network
effects. Named boundary modules may perform explicit caller-authorized I/O:
`evidence.py`, `external_data.py`, the benchmark resource loader, checkpoint
writers, and socket replay persistence. Optional Torch modules remain behind
the `bnn` extra. See the module table in
[`../docs/core/architecture.md`](../docs/core/architecture.md).

**Layer contract:** no `import infrastructure` anywhere under `fedference/`.

```bash
! grep -rn "import infrastructure" src/fedference/
```

Tests: `tests/fedference/test_*.py` (mirror module names).

## Orchestration modules

| Module | Role |
| --- | --- |
| `analysis/workflow.py` | `run_analysis_pipeline()` — JSON reports + figures |
| `analysis/report_schemas.py` | Typed report/figure-registry schemas + write-boundary validator |
| `figures/*.py` | One generator per PNG; shared palette in `_common.py` |
| `manuscript_variables.py` | `generate_variables()` → `{{TOKEN}}` map |
| `experiment_config.py` | `ExperimentConfig` from `manuscript/config.yaml` |
| `publication/pipeline_freshness.py` | Content-hashed analysis → hydration → render receipts |
| `publication/clean_checkout.py` | Clean Git/tracking/import release-evidence probe |
| `publication/identifiers.py` | Shared DOI normalization and resolver URL contract |
| `publication/zenodo.py` | Typed standard-library Zenodo draft/release boundary; no implicit publication |
| `publication/web_package.py` | Web assets/references plus deterministic HTML accessibility invariants |
| `publication/surface_validation.py` | Rendered PDF/slide/web surface invariants; no PDF/UA claim |
| `fedference_cli/__init__.py` | Installed CLI; explicit output paths and registry/evidence dispatch |

These modules also stay infrastructure-free in this project.

## Style

[`STYLE.md`](STYLE.md) · [`../docs/development/style_guide.md`](../docs/development/style_guide.md)

## Adding code

1. Domain math → `fedference/<module>.py`
2. Tests → `tests/fedference/test_<module>.py`
3. ISC row → [`../ISA.md`](../ISA.md)
4. Coverage gate → [`../docs/reference/verification-commands.md`](../docs/reference/verification-commands.md)

## See also

- [`README.md`](README.md)
- [`../scripts/AGENTS.md`](../scripts/AGENTS.md)
