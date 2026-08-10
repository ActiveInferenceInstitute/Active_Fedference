# tests/ — Active Fedference test suite

Zero-mock, seeded NumPy tests with **≥90%** coverage on `../src/`.
Philosophy: [`../docs/development/testing_philosophy.md`](../docs/development/testing_philosophy.md).

## Layout

```
tests/
├── conftest.py              # MPLBACKEND=Agg; sys.path → src/
├── fedference/              # Mirrors src/fedference/
├── figures/                 # Mirrors src/figures/
├── analysis/test_workflow.py
├── test_manuscript_variables.py
├── test_experiment_config.py
├── test_invariants.py
├── test_documentation.py
├── test_scripts_smoke.py
├── test_docs_contract.py
├── test_token_provenance.py
├── test_xref_integrity.py
├── test_docstrings.py
├── test_caption_completeness.py
├── test_web_publication_contract.py
├── test_token_tables.py
└── PATTERNS.md
```

## `tests/fedference/`

| Module | Focus |
| --- | --- |
| `test_core_identities.py` | Central recovery limits (ISC spine) |
| `test_core_edges.py` | Numerical edge cases |
| `test_divergences.py` | Divergence dispatch |
| `test_belief_updating.py` | VFE / infer_states |
| `test_belief_sharing.py` | share_round |
| `test_dirichlet_learning.py` | Language acquisition |
| `test_expected_free_energy.py` | EFE identity |
| `test_bayesian_model_reduction.py` | BMR |
| `test_pomdp.py` | Sentinel world |
| `test_agents.py` | SentinelEnsemble |
| `test_contamination.py` | Saboteurs |
| `test_statistics.py` | Wilcoxon + BH-FDR |
| `test_experiments.py` | Four seeded experiment contracts |
| `test_bnn_baseline.py` | FedGVI logreg |
| `test_aggregation_config.py` / `test_aggregation_comparators.py` | Rich aggregation API, compatibility, comparator controls |
| `test_server_theory.py` / `test_calibration.py` | Theory witnesses and calibration/evaluation separation |
| `test_evidence.py` / `test_research_registry.py` | Versioned specs, receipts, live-source verification, registry invariants |
| `test_external_data.py` / `test_benchmark.py` | Hash-bound archives, deterministic splits, packaged compatibility data |
| `test_bnn_fedgvi.py` / `test_torch_bnn.py` | Site/cavity checkpoints and explicit CPU/MPS device receipts |
| `test_transport_envelope.py` / `test_socket_transport.py` | Versioned envelopes, authentication, enforced loopback binding, persistent replay guards, event ordering |
| `test_hybrid.py` / `test_hybrid_tracking.py` | Hybrid recovery limits and honestly named on-policy tracking diagnostics |

## `tests/figures/`

One file per figure generator (`test_free_energy_comparison.py`, …).

Publication, CLI, documentation, metadata, freshness, clean-checkout, and
rendered-surface contracts—including deterministic HTML accessibility failures—
live in the top-level `tests/test_*.py` files. Smoke
or pilot execution proves only the declared implementation path; it is not
confirmatory scientific evidence.

## Authoritative command

```bash
uv run --locked --extra dev pytest tests/ \
  --cov=src \
  --cov-fail-under=90
```

## See also

- [`PATTERNS.md`](PATTERNS.md)
- [`../docs/development/agent_instructions.md`](../docs/development/agent_instructions.md)
- [`README.md`](README.md)
