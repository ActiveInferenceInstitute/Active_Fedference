# Script conventions — Active Fedference

Orchestration rules for standalone repo-root commands under `scripts/`.

## Thin orchestrator rules

Scripts **coordinate** — they never **compute**:

```python
# CORRECT
from analysis.workflow import run_analysis_pipeline

paths = run_analysis_pipeline()
for label, path in paths.items():
    print(f"{label}: {path}")
```

```python
# WRONG — math belongs in src/fedference/
consensus = np.exp(np.sum(np.log(beliefs), axis=0))
```

## Import patterns

Project scripts run with `src/` on `PYTHONPATH` (via pytest `conftest` or
`uv run --locked` from project directory):

```python
from analysis.workflow import run_analysis_pipeline
from manuscript_variables import generate_variables
```

Do not import `infrastructure.*` from project scripts unless wrapping a
repository-root pipeline script pattern — this project's scripts call `src/`
only.

## Output layout

Write only through `src/analysis/workflow.py` and figure generators — do not
invent new output paths without updating
[`../docs/operations/output-layout.md`](../docs/operations/output-layout.md).

Standard tree:

```
output/
├── reports/     # JSON experiment reports (schema-validated at the write boundary)
├── figures/     # PNG figures + sibling PDFs from the generators, plus figure_registry.json
├── data/        # manuscript_variables.json
├── manuscript/  # Token-resolved markdown copies
├── release/     # Release bundle + provenance manifest (build_release.py)
└── pdf/         # Rendered PDF via the sibling template render script
```

## Architecture reference

[`../docs/core/architecture.md`](../docs/core/architecture.md)

## See also

- [`AGENTS.md`](AGENTS.md)
- [`../docs/development/style_guide.md`](../docs/development/style_guide.md)
