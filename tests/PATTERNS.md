# Test patterns — Active Fedference

Zero-mock conventions for `tests/`.

## Forbidden

Anywhere under `tests/`:

- `unittest.mock`, `MagicMock`, `@patch`, `create_autospec`, `pytest-mock`

## Patterns

### Seeded NumPy

```python
def test_log_linear_pool_uniform_weights():
    rng = np.random.default_rng(0)
    beliefs = rng.dirichlet(np.ones(4), size=3)
    from fedference.aggregation import log_linear_pool
    consensus = log_linear_pool(beliefs)
    assert np.isclose(consensus.sum(), 1.0)
```

### Exact identity pins

Recovery limits use tight tolerances and explicit arrays — see
`tests/fedference/test_core_identities.py`.

### Figure tests

Call generators with fixture data; assert PNG path exists and is non-empty
(`tests/figures/`).

### Workflow integration

`tests/analysis/test_workflow.py` runs `run_analysis_pipeline` against temp or
project output dirs with real JSON writes.

### Report-schema accept/reject

`tests/analysis/test_report_schemas.py` drives real payloads through the real
validator (`analysis.report_schemas.validate_report` /
`check_figure_contract`) — no mocks. Valid payloads are derived mechanically
from the `_REPORT_SCHEMAS` table, so every declared schema gets accept coverage
automatically; reject cases drop or mistype one required field at a time and
assert `ReportSchemaError` names the report and field. `int` is deliberately
probed with a `bool` and `number` with a numeric string, so the checker must
reject `True` as an int and must not coerce string numerics. Add a bespoke shape
test only for variant payloads such as `bnn_torch` (its `ok` vs `skipped: ...`
branches).

### Stage-contract tests

The stage-contract tests in the same module read the real committed artifacts
under `output/reports/*.json` and `output/figures/figure_registry.json` and
re-run them through `validate_report` and `check_figure_contract`, so a drifted
on-disk artifact fails against its declared schema and every figure's declared
input fields are proven present.

## conftest.py requirements

- `os.environ.setdefault("MPLBACKEND", "Agg")`
- `sys.path` includes project `src/`

## Coverage command

```bash
uv run --locked --extra dev pytest tests/ --cov=src --cov-fail-under=90
```

## See also

- [`../docs/development/testing_philosophy.md`](../docs/development/testing_philosophy.md)
- [`AGENTS.md`](AGENTS.md)
