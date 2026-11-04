# `src/analysis/producers/` — report producers

Report producers grouped by existing study family. Each module consumes an
explicit `ExperimentConfig` and returns a payload; canonical orchestration,
schema validation and persistence stay in
[`../workflow.py`](../workflow.py).

## Modules

| Module | Family |
| --- | --- |
| `belief.py` | Belief-sharing and recovery reports. |
| `bnn.py` | BNN baseline and robustness reports. |
| `diagnostics.py` | Diagnostics and evidence reports. |
| `robustness.py` | Robustness, sensitivity and contamination reports. |
| `worlds.py` | World, navigation and environment reports. |
| `__init__.py` | Package marker; no orchestration. |

## Contract

- Producers are pure with respect to the filesystem: no writes, no output
  directories, no registry mutation.
- Seeds and study sizes come from the configured budget. Do not hardcode them.
- `workflow.py` imports producer functions directly and owns the single
  `_write_json` boundary. Producer modularity and write ownership are gated by
  [`../../../tests/analysis/test_producer_modularity.py`](../../../tests/analysis/test_producer_modularity.py).
- A producer returns the payload shape declared in
  [`../schemas/`](../schemas/README.md); it does not validate that shape itself.

Parent contract: [../AGENTS.md](../AGENTS.md).
