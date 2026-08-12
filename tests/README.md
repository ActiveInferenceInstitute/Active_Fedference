# tests/ — Quick reference

Active Fedference test suite. Full contract: [`AGENTS.md`](AGENTS.md).

```bash
uv run --locked --extra dev pytest tests/ --cov=src --cov-fail-under=90
```

Fast feedback and explicit profiles:

```bash
uv run --locked pytest tests/ -m "not slow" -q
uv run --locked pytest tests/ -m integration -q
uv run --locked pytest tests/ -m publication -q
```

The profiles select real seeded tests; they do not replace the full coverage
gate. Temporary PDF-review and local test artifacts under `.tmp/` are ignored
and never belong in a release snapshot.

| Suite slice | Covers |
| --- | --- |
| `fedference/` | `src/fedference/*` domain modules |
| `test_fedference_cli.py` | Installed CLI parsing, output isolation, receipts, verification, and unimplemented-run failure |
| `figures/` | `src/figures/*` generators |
| `analysis/` | `src/analysis/workflow.py`; `test_report_schemas.py` covers the write-boundary validator and stage contracts |
| `test_manuscript_variables.py` | Token hydration |
| `test_experiment_config.py` | `manuscript/config.yaml` |
| `test_scripts_smoke.py` | Auxiliary script subprocess smoke |
| `test_invariants.py` | Numerical invariants (PMF, recovery, EFE, KL) |
| `test_build_backend.py` | PEP 517 backend pinning, archive normalization, metadata, and source-manifest contract |
| `test_release_manifest.py` | Release bundle manifest and provenance fingerprint |
| `test_release_preflight.py` | Release metadata, rendered-surface, and provenance prerequisites |
| `test_validation_receipt.py` | Full-suite coverage receipt, input hashes, environment, and freshness checks |
| `test_pipeline_freshness.py` | Analysis → hydration → render provenance ordering and content digests |
| `test_clean_checkout.py` | Clean-clone tracking, required release inputs, and isolated import probes |
| `test_publication_metadata.py` | Generated metadata surface consistency |
| `test_docs_contract.py` | Every source-owned `AGENTS.md`/`README.md`, docs consistency, Mermaid structure, retired-name gate, SYNTAX registry, and stale-language gate |
| `test_documentation.py` | Documentation contract oracle |
| `test_manuscript_claim_audit.py` | Manuscript claim-strength / three-axis audit |
| `test_token_provenance.py` | No hardcoded decimals/percentages in prose |
| `test_token_sourcing.py` | Token source provenance |
| `test_xref_integrity.py` | Dangling cross-reference detection |
| `test_docstrings.py` | Docstring coverage and style |
| `test_caption_completeness.py` | Figure/table caption requirements, estimands, uncertainty, and replication units |
| `test_web_publication_contract.py` | Web publication metadata contract |
| `test_surface_validation.py` | PDF structure, extracted text, and rendered-surface checks |
| `test_token_tables.py` | Token table structure |
| `test_runtime_surface.py` | No retired runtime markers or test-double APIs |

Roadmap-foundation tests under `tests/fedference/` cover the public aggregation
configuration and legacy parity, experimental comparators and theory witness,
calibration/evaluation separation, evidence schemas, research registry, pinned
external-data parsing, FedGVI site/cavity checkpoints, optional CPU/MPS device
receipts, protocol-parity labels, honestly named on-policy hybrid tracking
diagnostics, versioned transport envelopes, in-process and SQLite-backed local
replay guards, enforced loopback binding, packaged default data, independently
recomputable config hashes, and dirty-tree receipt disposition. Strict receipt
tests additionally bind the live full Git commit,
clean tree, and `uv.lock` digest. These are implementation gates; they do not
convert a smoke run into scientific evidence.

Philosophy: [`../docs/development/testing_philosophy.md`](../docs/development/testing_philosophy.md)
