# Active Fedference — Forward TODO

> This is a forward roadmap, not a snapshot of completed work. Test, coverage,
> report, figure, render, and release counts are generated into `output/` and
> refreshed by the validation pipeline; they are intentionally not duplicated
> here because hard-coded counts become stale immediately after a source change.

This file is the top-level signpost for future work. Each open item has a scoped
page under [`docs/todo/`](docs/todo/README.md) with the implementable details —
the scoped page is the single owner of an item's scope; this index never
duplicates it. The phase plan and source map are in
[`docs/todo/scholarship-and-phase-plan.md`](docs/todo/scholarship-and-phase-plan.md)
and [`docs/research/literature-audit.md`](docs/research/literature-audit.md).
Completed work belongs in [`ISA.md`](ISA.md), generated validation reports,
release notes, or git history — never in this file.

---

## Baseline Contract

Preserve the project's current scientific framing while taking any item below:

1. **Client-side FedGVI** is the source-theorem-backed robustness axis: under
   the cited FedGVI loss, divergence, and regularity assumptions, bounded
   β/rcce losses in the generalized posterior inherit the corresponding
   bounded-influence result. This repository tests recovery and finite
   empirical cases; it does not independently prove general accuracy or
   robustness outside those assumptions.
2. **`robust_aggregate`** is a sharp server heuristic with the recovery-limit
   guarantee and a scoped no-go proposition: the declared continuously
   differentiable separable block-objective class cannot produce its raw
   log-pool update. This is not a universal no-objective theorem and does not
   make the heuristic objective-backed.
3. **`variational_aggregate`** is the objective-backed server rule: it minimizes
   a stated free energy and carries a proven raw effective-weight bound with
   empirical redescending behavior, but this is *not* an estimator-level
   bounded-influence (B-robustness) proof for the normalized consensus — and it
   is conservative in the declared comparison (where its more diffuse
   consensus does not win peak accuracy), not universally dominated.
4. No figure, table, README claim, or manuscript sentence grants axis 2 a guarantee
   it does not have, claims axis 3 has a proven bounded-influence/B-robustness
   guarantee it does not have, claims axis 3 wins on accuracy unless the
   statistics show it, or implies true multi-machine federation before that
   transport exists.

## Evidence Boundary

The current evidence supports only: conditional robustness under declared
contamination and attack geometry, communication benefit under complementary
views, proper-score controls without a demonstrated robust belief-quality
advantage, the bounded source-bound review-grid slice with preserved negative
controls, and the formal recovery identities. The mechanism gallery contains
both positive and negative robust contrasts; the hierarchy runs do not improve
location; and the BNN complement reverses at the terminal contamination level.
The hybrid API is a tested representation slice, not a completed continuous
active-inference task. Any new scientific phase therefore begins with a
preregistered primary estimand, independent unit, falsifier, and no-claim
boundary from the
[scholarship-indexed phase plan](docs/todo/scholarship-and-phase-plan.md).
No future phase may promote a larger model, a secure channel, a deeper diagram,
or a positive single attack result into a theorem about `robust_aggregate`.

---

## Gates For Any Item

The authoritative bounded wrapper is `uv run --locked python scripts/validate_all.py
full`; its embedded `source` profile runs Ruff, mypy, invariants, the domain
layer check, and exact-set release build/verification before the full coverage
suite. The test suite no longer touches the committed `output/` snapshot: the
subprocess smoke tests redirect all pipeline writes into a temporary scaffold
via `ACTIVE_FEDFERENCE_PROJECT_ROOT` (see `src/project_paths.py`). As general
hygiene, still regenerate the publication outputs after source changes that
alter results, so the committed snapshot stays derived from the declared
configuration — `tests/test_report_scale_guard.py` verifies its scale. The
release-grade end-to-end procedure (fresh clone, full ladder, rendered-surface
count invariants, raster reads, fingerprint verify, cross-vendor verdict) is
scoped in
[`docs/todo/release-and-verification-ladder.md`](docs/todo/release-and-verification-ladder.md).

```
# Full source gate
uv run --locked --extra dev pytest tests/ --cov=src --cov-fail-under=90

# Central identity gate
uv run --locked python -c "
from fedference.aggregation import robust_aggregate, variational_aggregate, log_linear_pool
import numpy as np
b = [[.7,.3],[.6,.4]]
assert np.allclose(robust_aggregate(b, robustness=0).consensus, log_linear_pool(b))
assert np.allclose(variational_aggregate(b, robustness=0).consensus, log_linear_pool(b))
"

# Manuscript provenance gate
uv run --locked pytest tests/test_xref_integrity.py tests/test_caption_completeness.py \
  tests/test_token_provenance.py tests/test_manuscript_variables.py -q

# Layer-boundary gate
! grep -rn "import infrastructure" src/fedference/

# Publication package gate
uv run --locked python scripts/validate_all.py package

# TODO hygiene gate
uv run --locked pytest tests/test_docs_contract.py -q

# Ruff lint gate
uv run --locked ruff check src/ tests/
# Expected: 0 E501, 0 F401, 0 F811, 0 F841 (CI-clean)
# line-length 110 in pyproject.toml — E501 spans >=111 trigger
```

---

## Major — Scientific Upgrades & Deep Extensions

Research projects, not engineering tasks; each requires a mathematical design
pass before implementation, and each begins from the preregistration
requirements in the [Evidence Boundary](#evidence-boundary). Ordered by impact
on the project's scientific standing. Residual scope lives in each scoped page.

| # | Priority | TODO | Effort | Dependencies |
|---|----------|------|--------|-------------|
| MAJ-8 | 🔴 Critical | [Calibrate robustness without evaluation leakage](docs/todo/adaptive-robustness-calibration.md) — bounded pilot now exercises disjoint calibration/evaluation and overlap rejection; freeze `robustness` and `entropy_weight` before confirmatory evaluation | 1–3 months | Scoped MAJ-1 result; may develop in parallel |
| MAJ-2A | 🔴 Critical | [Portable protocol-faithful FedGVI BNN](docs/todo/faithful-fedgvi-bnn-lane.md) — synthetic CPU/MPS cavity/site-factor/factor-replacement pilot is executable; extend to source datasets, locked M4 budget, checkpoints, and proper-score inference | 2–6 months | Public foundation |
| MAJ-2B | 🟢 External | [Exact source-scale CUDA replication](docs/todo/faithful-fedgvi-bnn-lane.md) — preserve the source configuration declaratively and execute only when external CUDA resources exist | External lane | MAJ-2A and external CUDA |
| MAJ-6 | 🟡 High | [Three-dataset external benchmark pack](docs/todo/external-benchmark-domain-pilot.md) — dataset-level summary and receipt controls are implemented; execute the pinned UCI pack confirmatorily with negative controls and source-bound manuscript artifacts | 2–6 months | Public foundation; calibration policy |
| MAJ-7 | 🟡 High | [Friston source-protocol reconstruction](docs/todo/faithful-friston-protocol-replication.md) — parity audit and analogue negative control are executable; resolve the Eq. 2 and Figures 5, 7, and 9 matrices before any exact-replication claim | 5–9 months | Source-protocol extraction |
| MAJ-3 | 🟡 High | [Continuous/hybrid state spaces](docs/todo/beyond-discrete-categorical-state-spaces.md) — matched hybrid controls and singular-covariance negative control are piloted; freeze a confirmatory benchmark with discrete, continuous, and oracle controls | 5–9 months | Hybrid recovery gates; scoped server-theory boundary |
| MAJ-5 | 🟡 High | [Hierarchy and richer tasks](docs/todo/deeper-hierarchy-task-family.md) — deterministic Four Rooms and Key-Door controls are piloted; preregister task units before a general hierarchy claim | 5–9 months | MAJ-3 recovery gates |
| MAJ-4A | 🟡 High | [Authenticated local multi-node emulator](docs/todo/true-multi-machine-federation.md) — Docker, mTLS by default, HMAC compatibility, checkpoint/restart, and deterministic network-fault controls | 7–12 months | Stable transport envelope |
| MAJ-4B | 🟢 External | [Physical multi-host validation](docs/todo/true-multi-machine-federation.md) — require receipts from distinct hosts; never infer this claim from local containers | External v1.x lane | MAJ-4A and external hosts |

## Staged release waves

The release names are evidence gates, not deadlines or claims that the open
research has already succeeded.

| Wave | Target | Required outcome |
| --- | --- | --- |
| Public foundation | v0.1 | Stable aggregation configuration, registry, receipts, installed CLI, wheel/sdist smoke, complete local release ladder, two isolated fresh-clone reproductions, and external confidentiality/license/author approval |
| Server theory | v0.2 | MAJ-1 scoped no-go evidence recorded in ISA/audits, MAJ-8 locked calibration, comparator evidence, negative controls, and a source-bound theory pack |
| Portable FedGVI | v0.3 | MAJ-2A protocol parity and M4 evidence plus the MAJ-6 three-dataset pack; MAJ-2B remains declarative and external |
| Protocol and tasks | v0.4 | MAJ-7 reconstruction evidence, a controlled hybrid benchmark, and both hierarchy tasks with recovery and falsifier controls |
| Stable platform | v1.0 | Stable schemas and deprecation policy, three independently verifiable evidence packs, both documentation paths, MAJ-4A emulator, clean-clone reproduction, and release approval |
| Research expansion | v1.x | Streaming/nonstationary sharing first, then multimodal missingness; privacy and secure aggregation only after a threat model and leakage protocol |

## Minor — Maintenance and Release Integrity

| # | Priority | TODO | Effort | Dependencies |
|---|----------|------|--------|-------------|
| MIN-2 | 🔴 Critical | [Clone-correct release and full verification ladder](docs/todo/release-and-verification-ladder.md) — provision safe headroom, verify wheel/sdist installation, run the complete ladder twice from isolated fresh clones, obtain a structured independent verdict, and retain external release authority | 1–3 days after headroom | None (blocks any push/release) |

## Roadmap control

The [scholarship-indexed phase plan](docs/todo/scholarship-and-phase-plan.md) is
an active governance document for the major phases, not an additional major
item. It remains open because it must be updated whenever a phase changes its
estimand, falsifier, source bundle, or claim boundary.

## Parked Tracks

On the long-term watchlist, not actively scoped. Their minimum estimand,
falsifier, source bridge, and claim boundary are recorded in the
[parked-track plan](docs/todo/scholarship-and-phase-plan.md#parked-tracks), so
"parked" means "not yet authorized for implementation," not "scientifically
undefined." Reconsider on community contribution or a relevant upstream FedGVI
extension.

- **Multi-modal beliefs** (beyond the sentinel world's location×proximity×pose×gaze;
  prioritize explicit missingness after streaming)
- **Privacy-preserving federation** (differential privacy, secure aggregation)
- **Online / streaming active inference** (first v1.x priority; continuous
  belief updating without episode boundaries)
- **Natural language output generation** (LLM-based belief summaries, not
  Friston's mechanical language emergence)

---

## Removal Rule

When an item is finished, move only the acceptance evidence to `ISA.md`,
generated reports, or release notes, then delete its row here and its scoped page
under `docs/todo/`. Do not leave finished TODO entries behind as status history.
