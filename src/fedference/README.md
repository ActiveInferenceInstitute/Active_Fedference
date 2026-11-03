# `src/fedference/` — domain package

The mathematical core: discrete-categorical FedGVI, the belief-sharing bridge,
and the explicit adapters that may perform caller-authorized I/O.

`__init__.py` is the public surface. It exports the FedGVI core (divergences,
losses, generalized Bayes, aggregation), the application path
(`LabeledAggregationRequest`, `aggregate_labeled`), belief sharing, evidence
receipts, statistics and provenance. Active-inference machinery (POMDP,
Dirichlet learning, expected free energy, Bayesian model reduction) is imported
lazily from submodules so the core import stays light.

## Layout

| Area | Modules |
| --- | --- |
| Aggregation and application | `aggregation.py`, `application.py`, `aggregation_comparators.py` |
| Divergences and generalized Bayes | `divergences.py`, `losses.py`, `generalized_bayes.py` |
| Belief sharing and inference | `belief_sharing.py`, `belief_updating.py`, `expected_free_energy.py`, `dirichlet_learning.py`, `pomdp.py` |
| Agents and runs | `agents.py`, `colonies.py`, `single_machine.py`, `trials.py`, `scoring.py` |
| Recovery and calibration | `bayesian_model_reduction.py`, `continuous_recovery.py`, `calibration.py`, `complexity.py`, `hierarchy_tasks.py` |
| Evidence and provenance | `evidence.py`, `provenance.py`, `research_registry.py`, `protocol_parity.py`, `external_data.py` |
| Statistics and baselines | `statistics.py`, `benchmark.py`, `bnn_baseline.py`, `bnn_fedgvi.py`, `server_theory.py`, `contamination.py` |
| Hybrid and optional Torch | `hybrid.py`, `hybrid_tracking.py`, `bnn_baseline_torch.py`, `bnn_variational_torch.py`, `torch_bnn.py`, `bnn_defaults.py` |
| Internal validation | `_validation.py` |
| Subpackages | `config/`, `data/`, `evidence_contracts/`, `experiments/`, `federation/`, `gnn/`, `models/` |

## Layer contract

Mathematical primitives are side-effect free. Filesystem and network effects
belong only to the named boundary adapters (`evidence.py`, `external_data.py`,
the benchmark loader, checkpoint writers, and socket replay persistence), all
with caller-owned paths. No module under this package may import
`infrastructure`.

Torch stays behind the optional `bnn` extra and must never enter the default
import graph. The cross-layer rationale is in
[`../../docs/core/architecture.md`](../../docs/core/architecture.md).

## Check it

```bash
uv run --locked --extra dev pytest tests/fedference -q
uv run --locked ruff check src/ tests/ scripts/ examples/ _fedference_build_backend.py
uv run --locked mypy src/
```

Tests mirror module names under [tests/fedference/](../../tests/fedference/).
The package contract is [AGENTS.md](AGENTS.md).
