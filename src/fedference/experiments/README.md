# `src/fedference/experiments/` — experiment harness

Seeded study families behind the registered research profiles. Producers return
report payloads; orchestration, schema validation and persistence stay in
[`../../analysis/workflow.py`](../../analysis/workflow.py).

## Modules

| Area | Modules |
| --- | --- |
| Study families | `belief_sharing.py`, `complexity.py`, `conditional_world.py`, `cross_study.py`, `navigation.py`, `parameter_recovery.py`, `robustness.py`, `sensitivity.py`, `worlds.py` |
| Review and diagnostics | `review_grid.py`, `gallery.py`, `heuristic_characterization.py`, `diagnostics.py` |
| Assembly | `report_bundle.py`, `_common.py` |

## Contract

- Every family consumes an explicit `ExperimentConfig`; seeds come from the
  configured budget, never from a module-level default.
- Families return payloads. They do not write files, choose output paths or
  validate against report schemas; the workflow does that at one write boundary.
- Claim labels stay honest: `heuristic_characterization.py` characterizes the
  server heuristic, while `robustness.py` and `sensitivity.py` report seeded
  study behavior. Neither establishes a universal robustness theorem.
- Run through a registered profile; a verified receipt is what supports an
  executed-result statement.

## Check it

```bash
uv run --locked --extra dev pytest tests/fedference -q
```

The producer and write-ownership gates are in
[`../../../tests/analysis/test_producer_modularity.py`](../../../tests/analysis/test_producer_modularity.py).
Parent contract: [../AGENTS.md](../AGENTS.md).
