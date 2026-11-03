# `tests/integration/` — required upstream GNN runtime

Two modules exercise the real upstream parser and the real PyMDP backend:

- [test_gnn_model_cli.py](test_gnn_model_cli.py) validates native and extension
  models through the installed CLI with `--gnn-root`.
- [test_gnn_pymdp.py](test_gnn_pymdp.py) runs the PyMDP agent, verifies beliefs,
  policies and actions, and checks replay parity and receipt verification.

Both belong to the `gnn` validation profile. The default `native` profile does
not collect them. The `gnn` and `full` profiles require a real runtime:

```bash
export GNN_ROOT=/path/to/pinned/upstream
export GNN_PARSER_DEPENDENCIES=/path/to/parser/dependencies
uv run --locked --extra dev pytest tests/integration --validation-profile=gnn -q
```

`GNN_PYTHON` selects the interpreter independently of the source tree, and
`GNN_BACKEND_LOCK` can override the backend lock. Missing or invalid runtime
configuration refuses with `GNN_PROFILE_RUNTIME_REQUIRED` before fixtures run.
A required profile cannot report success with skipped tests.

Shard assignment is defined in
[`../../scripts/validation_profiles.py`](../../scripts/validation_profiles.py)
and verified by [`../test_gnn_ci_profiles.py`](../test_gnn_ci_profiles.py).

Parent contract: [../AGENTS.md](../AGENTS.md).
