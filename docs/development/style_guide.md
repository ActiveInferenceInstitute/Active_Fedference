# Style guide

Seven rules for modifying Active Fedference source, tests, and manuscript prose.

## Rule 1: Zero mocks

See [`testing_philosophy.md`](testing_philosophy.md). No exceptions.

## Rule 2: Infrastructure delegation

`src/fedference/` must not import `infrastructure.*`. Orchestration modules in
this project (`analysis/`, `figures/`, `manuscript_variables.py`) also stay
infrastructure-free by design.

Local scripts under `scripts/` call into `src/` only. Template-side render
scripts under the template repo's `scripts/` directory may use
`infrastructure.*`.

## Rule 3: Thin orchestrator

| Layer | Allowed | Forbidden |
| --- | --- | --- |
| `src/fedference/` mathematical modules | Math, types, deterministic RNG | File I/O, plotting, infrastructure |
| Named `fedference` boundary adapters | Explicit evidence/data/checkpoint/replay I/O to caller-owned paths; optional Torch behind extras | Hidden writes, default Torch imports, `infrastructure.*` |
| `src/analysis/workflow.py` | Serialize JSON via `_write_json` (schema-validated), call experiments + figures | New formulas; bypassing `_write_json` |
| `src/figures/` | Matplotlib I/O, styling via `_common.py` | Domain algorithms |
| `scripts/` | `main(argv=None)`, argparse, `--project-root`, subprocess ordering, output/status mapping, format-specific validation | Domain formulas, experiment loops, ad-hoc artifact discovery, silent fallback to another checkout |

## Rule 4: Show, not tell

Manuscript and doc references must name files and symbols:

```markdown
`fedference.aggregation.robust_aggregate` at `robustness=0.0` equals the
project `log_linear_pool`, tested in
`tests/fedference/test_core_identities.py`. Describe its relation to Friston
Eq. 7 only as the documented categorical posterior-log-potential
specialization, never as a reconstruction of the full source protocol.
```

## Rule 5: Explicit paths

Use project-qualified paths in cross-repo docs:

- `src/fedference/aggregation.py`
- `manuscript/19_results_robustness.md`

Inside the project tree, relative paths from the file you edit are fine.

## Rule 6: Type hints

- `from __future__ import annotations` at the top of every module
- Full annotations on all public functions and methods
- Module docstring cites the relevant Friston equation/figure or FedGVI mechanism

## Rule 7: Error messages

Raise `ValueError` (or a narrow domain exception) with actionable context:

```python
# GOOD
if robustness < 0:
    raise ValueError(f"robustness must be non-negative, got {robustness}")
```

Avoid bare `assert` in public APIs (use asserts only in tests or internal invariants).

## Manuscript-specific

- Numbers in prose → `{{TOKEN}}` only ([`../manuscript/tokens-and-labels.md`](../manuscript/tokens-and-labels.md))
- Figure references → `[@fig:label]` matching `{#fig:label}` in [`../../manuscript/SYNTAX.md`](../../manuscript/SYNTAX.md)
- **Three robustness axes** labeled honestly in results/discussion sections
  (axis 1 = per-client result inherited only under the source theorem's matching
  assumptions; axis 2 = sharp server heuristic, recovery-limit guarantee only;
  axis 3 = server objective-backed effective-weight bound, conservative — not
  estimator-level B-robustness)

## See also

- [`agent_instructions.md`](agent_instructions.md)
- [`../core/architecture.md`](../core/architecture.md)
- [`../../src/STYLE.md`](../../src/STYLE.md)
