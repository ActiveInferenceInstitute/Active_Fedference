# Active Fedference — Technical Reference

**Standalone research project** (separate git repository at
``docxology/active_fedference``). Reimplements **FedGVI**
(Mildner et al., 2025) in the discrete-categorical setting and connects it to
Friston et al. (2024) federated **belief-sharing**.

**Acceptance contract:** [`ISA.md`](ISA.md) · **Documentation hub:** [`docs/README.md`](docs/README.md)

This checkout is intentionally a standalone development and review repository;
the v1.0.4 public snapshot is published at the target below; v1.0.3, v1.0.2,
v1.0.1, and v0.1.0 remain prior Zenodo versions.
The configured interim remote is
[`docxology/active_fedference`](https://github.com/docxology/active_fedference)
(`origin`); the public destination is
[`ActiveInferenceInstitute/Active_Fedference`](https://github.com/ActiveInferenceInstitute/Active_Fedference).
Work on a `codex/*` branch. Commit, push, and Zenodo publication are separate
release actions and require explicit authorization; do not force-push or reset
a dirty worktree.

## What this project is

Robust federated active inference. **Central project-local identity (tested):**
`robust_aggregate(robustness=0)` $\equiv$ `log_linear_pool`. The comparison
with Friston et al. Eq. 7 is a categorical posterior-log-potential
specialization under documented shared-support, admitted-potential, and
fixed-weight assumptions; it is not a reconstruction of the complete
message-passing protocol. The KLD/NLL/$\beta=0$ client recovery is a separate
project-local limit.

Science narrative and three robustness axes:
[`docs/core/conceptual-foundations.md`](docs/core/conceptual-foundations.md).

## Layer contract

`src/fedference/` must not import `infrastructure.*` (ISC-21). Mathematical
primitives use the default NumPy/SciPy dependency set. Explicit boundary
adapters (`evidence`, external-data/benchmark loading, checkpoints, and socket
replays) may perform caller-authorized I/O; optional `*_torch` and `torch_bnn`
modules require an extra and must never enter the default import graph. Scripts
remain thin orchestrators. Three aggregation methods in
``src/fedference/aggregation.py``:
``log_linear_pool`` (categorical Eq. 7 bridge), ``robust_aggregate`` (server heuristic),
and ``variational_aggregate`` (objective-backed conservative server rule).

Full module map, experiments, and artifacts:
[`docs/core/architecture.md`](docs/core/architecture.md) ·
[`docs/core/experiments-and-artifacts.md`](docs/core/experiments-and-artifacts.md).

## Composable public boundaries

- Pure domain operations are importable from `fedference`; use
  `AggregationConfig` + `aggregate_result` for new integrations and retain
  `aggregate`, `log_linear_pool`, and `robust_aggregate` for compatibility.
- Direct belief sharing, one-machine process federation, and loopback sockets
  consume the same validated configuration and transport envelopes; transport
  adapters must not duplicate aggregation math.
- Evidence, checkpoint, external-data, replay, DOI, and report-writing code is
  explicit boundary code. Reports cross the typed `_write_json` boundary only;
  optional Torch/BNN modules stay outside the default import graph.
- The installed `fedference_cli` is a composable adapter, not a second domain
  implementation: `__init__.py` is the compatibility facade, `_parser.py`
  owns process grammar, `_commands.py` owns registry dispatch, and `_support.py`
  owns output isolation and receipt construction. Its package contract is
  [`src/fedference_cli/README.md`](src/fedference_cli/README.md).
- Figures are generated through `src/figures/`, whose metadata registry,
  manuscript captions, report payloads, and output filenames must agree.

## Conventions

- `from __future__ import annotations`; full type hints; Friston/FedGVI citations in docstrings
- `np.random.default_rng(seed)` everywhere; never global `np.random`
- No mocks; $\ge 90\%$ coverage on `src/`
- Manuscript numbers are `{{TOKEN}}` only — see [`docs/manuscript/tokens-and-labels.md`](docs/manuscript/tokens-and-labels.md)
- Generated `output/` is a committed reviewer snapshot regenerated from source;
  `.tmp/` is local review scratch space and must never be released.
- README/docs diagrams use fenced GitHub-compatible Mermaid with quoted labels
  whenever punctuation could be parsed as syntax; validate statically and with
  the renderer before release.
- Every figure has source relation, estimand, unit, uncertainty disposition,
  replication unit, and concise `alt_text` in `_metadata.py`, with a
  self-contained caption and final PDF/HTML visual QA. HTML remains the
  accessibility-enhanced canonical surface. The source-current combined PDF
  requests tagged structure and is released only when `pdfinfo` reports
  `Tagged: yes`, qpdf exposes a non-empty `/Lang` and `StructTreeRoot`, and
  the source-bound language check passes; some Poppler builds omit the
  language line even when `/Lang` is present. Tagged structure is not a PDF/UA
  conformance claim.

## New module checklist

1. `src/fedference/<name>.py` + `tests/fedference/test_<name>.py`
2. Add ISC row to [`ISA.md`](ISA.md) (never re-number existing ISCs)
3. Run coverage gate (below)

## Validation commands

```bash
uv run --locked --extra dev pytest tests/ \
  --cov=src --cov-fail-under=90

# Fast feedback and explicit integration/publication profiles
uv run --locked pytest tests/ -m "not slow" -q
uv run --locked pytest tests/ -m integration -q
uv run --locked pytest tests/ -m publication -q

rg -n "import infrastructure" src/fedference/ && \
  { echo "Layer leak"; exit 1; } || echo "Clean"

# Check every Mermaid block in README.md and docs/ before rendering
uv run --locked python scripts/validate_mermaid.py
uv run --locked python scripts/validate_mermaid.py --render --renderer npx \
  --output-dir .tmp/mermaid-render

# Figure/caption and reader-surface contracts
uv run --locked pytest tests/test_caption_completeness.py tests/figures/ -q

# Verify the release bundle is not stale relative to current sources
uv run --locked python scripts/build_release.py --verify

# PEP 517 distributions are reproducible only under an explicit source epoch.
export SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)"
uv build --out-dir .tmp/dist-a
uv build --out-dir .tmp/dist-b
cmp .tmp/dist-a/*.whl .tmp/dist-b/*.whl
cmp .tmp/dist-a/*.tar.gz .tmp/dist-b/*.tar.gz
```

Analysis report payloads are validated against typed schemas at the write
boundary (`src/analysis/report_schemas.py`: `TypedDict` shapes plus per-figure
dependency contracts), so a malformed field fails when written rather than when a
figure consumes it. The release manifest records a provenance fingerprint
(SHA-256 over the declared source/config inputs) and `--verify` rejects a bundle
that no longer matches those inputs.

Full probe list: [`docs/reference/verification-commands.md`](docs/reference/verification-commands.md).

The cross-layer extension and orchestration rules are in
[`docs/development/modularity.md`](docs/development/modularity.md). Read that
guide before adding a domain operation, research lane, CLI command, report,
figure, script, or durable documentation page.

Scientific claim boundaries and the MAJ-1 server-rule evidence ladder are
maintained in [`docs/research/manuscript-claim-audit.md`](docs/research/manuscript-claim-audit.md),
[`docs/research/extended-statistical-audit-2026-07-14.md`](docs/research/extended-statistical-audit-2026-07-14.md),
and [`docs/todo/scholarship-and-phase-plan.md`](docs/todo/scholarship-and-phase-plan.md).
Federation trust boundaries and publication-surface accessibility are
maintained separately in
[`docs/security/active_fedference-threat-model.md`](docs/security/active_fedference-threat-model.md)
and [`docs/manuscript/accessibility.md`](docs/manuscript/accessibility.md).

## See also

- [`README.md`](README.md) — pitch and quick run
- [`STANDALONE.md`](STANDALONE.md) — confidentiality and standalone core
- [`manuscript/AGENTS.md`](manuscript/AGENTS.md) — manuscript editing
- Scope and related work: [`manuscript/22_discussion_related_work.md`](manuscript/22_discussion_related_work.md)
