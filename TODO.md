# Active Fedference — Active Forward Backlog

> This file contains only open work. Closed implementation and release evidence
> is retained in [`ISA.md`](ISA.md), source-bound reports, release records, and
> Git history. Generated counts and hashes are deliberately not copied here.

The authoritative public integration repository is
[`ActiveInferenceInstitute/Active_Fedference`](https://github.com/ActiveInferenceInstitute/Active_Fedference)
(`origin` in the public checkout). The separate
[`docxology/active_fedference`](https://github.com/docxology/active_fedference)
repository remains the private/interim evidence mirror; aliases such as
`public` and `origin` depend on the checkout and must be inspected before use.
“Push to main” therefore means: fix an
exact candidate SHA, pass the applicable confidentiality and verification
gates, push a `codex/*` branch to the required remotes, obtain the applicable
review checks, and merge into public `main` with a normal merge commit. Hosted
Actions is billing-locked org-wide, so those checks mean the same gates
reproduced locally unless a run actually executed. It never means
a direct, forced, squashed, rebased, stale-head, or check-bypassing main push.

Every open item has one owner page under
[`docs/todo/`](docs/todo/README.md). Tiers describe the size and coordination
surface of the remaining work, not whether its gate is optional: a Minor item
can still block a release. The source-indexed dependency map is
[`docs/todo/scholarship-and-phase-plan.md`](docs/todo/scholarship-and-phase-plan.md).

The [2026-10-30 reconciliation](docs/research/repository-reconciliation-2026-10-30.md)
owns the latest dated sync observations, interrupted-work dispositions, and
engineering/validation sequence. The
[2026-10-27 repository review](docs/research/repository-review-2026-10-27.md)
remains historical evidence. Neither closes a release or scientific gate.

The v1.1.0 GitHub release (published 2026-09-20) and its linked Zenodo version
(record 22149133, DOI `10.5281/zenodo.22149133`, publication date 2026-09-19)
already exist; [`docs/research/release-state-20261027.md`](docs/research/release-state-20261027.md)
records the tag, asset, and PDF digests. Retire release creation and Zenodo
publication as future actions. MED-5 remains open only for the retrospective
exact-SHA certification of the published commit, an attested record of the
historical publication approval, MIN-3 closure, and the one-time private
synchronization. The sequence below describes those obligations and must not be
used to recreate or move the existing tag, re-upload assets, or republish
Zenodo bytes.

Adopted, not open: the GNN–FEP cutover pins `v5.1.0` at
`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf` in `.github/gnn-pin.json`, executes
the estate's shared reader instead of the deleted interim pin reader, and merged
as `c9d1c90` (#23); [`.github/GNN.md`](.github/GNN.md) carries the evidence.

## Non-negotiable scientific boundary

1. Client-side FedGVI is the source-theorem-backed robustness axis only under
   the cited loss, divergence, model, and contamination assumptions.
2. `robust_aggregate` is a sharp server heuristic with exact recovery at zero
   robustness and the scoped MAJ-1 no-go result for its declared separable
   objective class. It has neither a general variational objective nor an
   estimator-level bounded-influence guarantee.
3. `variational_aggregate` is objective-backed and has the declared raw
   effective-weight bound. It does not thereby acquire an estimator-level
   B-robustness guarantee or a universal accuracy advantage.
4. Transport integrity, application provenance, calibration, scientific
   validity, deployment suitability, and downstream decisions are separate
   claim classes. Passing one class cannot certify another.

## Execution order

The active post-publication path is:

1. certify the already-published v1.1.0 release commit
   `c75954181184545acc31cda17c1013d4dfb34d31` through MIN-2's sequential
   two-clone campaign and obtain the selected identified owner-author verdict;
   the release merge, tag, GitHub release, and Zenodo version predate this
   campaign (Zenodo publication date 2026-09-19; GitHub release 2026-09-20), so
   the campaign is retrospective and authorizes no new tag or publication;
2. record the historical Zenodo publication approval from an external record,
   or retain an explicit evidence gap for it; the published version cannot be
   approved retroactively;
3. complete the separate accessible Template renderer review in a clean
   checkout, use its exact merged commit, and finish the source-only visual-
   scholarship candidate as post-v1.1.0 documentation work;
4. run the candidate-specific portion of MIN-2, push the exact visual branch to
   both remotes, and merge it normally into public `main`; hosted Actions is
   billing-locked org-wide, so the applicable gate must be reproduced locally;
5. complete and normally merge MIN-3's public documentation and repository-
   discovery handoff, including removal of its closed forward-backlog entry;
6. finish MED-5 by synchronizing that final public `main` into private
   `origin/main` exactly once with an equal tree;
7. execute MAJ-8 preregistration and calibration freeze before MAJ-6 may inspect
   sealed confirmatory rows; and
8. select any later scientific or product version only after the confirmatory
   result is known and reviewed, and cut it through a fresh release-only PR
   rather than retrofitting v1.1.0.

ISC-89 and ISC-242 remain open. The exact-SHA owner-author verdict required for
v1.1 release acceptance does not close the broader independent/cross-vendor
reproduction lane. No fresh explicit approval remains for v1.1.0: the Zenodo
version is already public, so the historical approval must be attested from an
external record or retained as an explicit gap.

## Minor — bounded maintenance and evidence checks

| ID | Open item | Prerequisite | Acceptance evidence |
| --- | --- | --- | --- |
| MIN-2 | [Clone-correct integration and release verification](docs/todo/release-and-verification-ladder.md) | Exact candidate or final-public-main SHA; clean pinned renderer; safe external namespace and headroom where the clone phase applies | Candidate history audit; green source/render/package gates; two isolated final-SHA clone receipts; eligible structured verdict; ISC-242 remains open until its complete evidence set exists |
| MIN-3 | [Release documentation and repository-discovery handoff](docs/todo/release-documentation-handoff.md) | Verified v1.1.0 tag, GitHub release, and published Zenodo record 22149133; stable final public `main` | Link and installed-artifact probes show the public repository, version, DOI, checksums, install path, application guide, and accessibility boundaries consistently; no stale development identity remains on current-release surfaces |

## Medium — cross-surface integration and publication

| ID | Open item | Prerequisite | Acceptance evidence |
| --- | --- | --- | --- |
| MED-4 | [Visual scholarship and accessible-slide integration](docs/todo/visual-scholarship-integration.md) | Public `main` passing its applicable gate; exact merged Template renderer commit in a separate clean checkout; source-bound reports; approved sanitized history with additive corrective commits based on refreshed public `main`; no use of the user-owned dirty Template checkout | Reviewed Template merge SHA; exact audited Fedference visual branch; all figure pairs, long descriptions, exact-value fallbacks, HTML/reveal/Beamer/PDF QA, confidentiality audit, public PR and the locally reproduced post-merge gate; the three frontmatter plates rejoin the complex-figure inventory with rendered `output/web/` pages |
| MED-5 | [v1.1.0 release, GitHub/Zenodo publication, and private synchronization](docs/todo/v1-1-release-publication.md) | Published v1.1.0 identity (`c759541`, tag `v1.1.0`, GitHub release 2026-09-20, Zenodo 22149133); MIN-2 retrospective certification of that exact commit and eligible `pass`; historical publication approval attested or recorded as a gap | MIN-2 two-clone receipts and eligible verdict for `c759541`; attested historical publication approval or explicit gap record; downloaded-and-reverified GitHub assets and published Zenodo record; one tree-identical private-main merge; exact SHAs, URLs, and hashes recorded |
| MED-6 | [Independent and cross-vendor reproduction](docs/todo/independent-reproduction.md) | Fixed final public artifact set and complete execution packet | Both an identified independent-human/Advisor review and a genuinely different-vendor execution record, with commands, environments, findings, and dispositions; ISC-89 closes only when the conjunctive final-bundle criterion is satisfied |

## Major — scientific upgrades and external execution

| ID | Open item | Prerequisite | Acceptance evidence |
| --- | --- | --- | --- |
| MAJ-8 | [Freeze robustness calibration without evaluation leakage](docs/todo/adaptive-robustness-calibration.md) | Published v1.1 application base; verified one-time private synchronization; completed MIN-3 documentation handoff; preregistered three-dataset policy; immutable source/data/split seals | Separate robust and variational selections; complete candidate and failure tables; stable hierarchical-bootstrap selection; frozen design digest and pilot-derived common confirmatory `N`; no final-test access |
| MAJ-6 | [Run the three-dataset external confirmation](docs/todo/external-benchmark-domain-pilot.md) | Exact valid MAJ-8 frozen-design digest; unchanged source/lock/archive/member/partition seals; fresh write-once namespace | Twelve-cell paired primary family, separate clean noninferiority family, MCSE and Holm dispositions, solver/provenance matrix, exact-value tables, source-bound manuscript artifacts, and integrity-valid receipt regardless of outcome direction |
| MAJ-2A | [Run portable protocol-faithful FedGVI BNN evidence](docs/todo/faithful-fedgvi-bnn-lane.md) | Source-revision parity matrix; required Torch lane; pilot-frozen local budget | Cavity/site-factor protocol on the declared source datasets; checkpoint/resume evidence; locked paired proper-score inference; explicit CPU/MPS fallback and parity receipts |
| MAJ-2B | [Run exact source-scale CUDA replication](docs/todo/faithful-fedgvi-bnn-lane.md) | MAJ-2A plus suitable external CUDA resources | Source-configuration execution receipt and source-defined estimand/seed comparison; declarative configuration alone never counts as execution |
| MAJ-7 | [Reconstruct the Friston source protocols](docs/todo/faithful-friston-protocol-replication.md) | Resolved Eq. 2 and Figures 5, 7, and 9 source matrices | Parameter, estimand, native-unit, and independent-unit parity plus a protocol negative control; unresolved rows retain the paper-constrained label |
| MAJ-3 | [Confirm continuous/hybrid state-space behavior](docs/todo/beyond-discrete-categorical-state-spaces.md) | Recovery and covariance gates; MAJ-8-style frozen calibration/budget boundary | Matched confirmatory tracking worlds, complete discrete/continuous/oracle controls, proper-score inference, retained failures, and bounded representation claims |
| MAJ-5 | [Confirm deeper hierarchy across richer tasks](docs/todo/deeper-hierarchy-task-family.md) | MAJ-3 recovery gates; preregistered task-family units and compute budget | Locked Four Rooms and Key-Door comparisons across task units, corrected secondary family, negative controls, source-bound figures/tables, and task-specific fallback language when replication fails |
| MAJ-4A | [Build an authenticated local multi-node emulator](docs/todo/true-multi-machine-federation.md) | Stable protocol-v1 envelope and reviewed threat model | mTLS-default Docker execution, identity/certificate controls, restart-durable replay, declared network-fault matrix, and in-process consensus equivalence |
| MAJ-4B | [Validate physical multi-host federation](docs/todo/true-multi-machine-federation.md) | MAJ-4A plus approved distinct hosts and key-management boundary | Receipts from physically distinct hosts with cross-host restart and fault evidence; local containers cannot satisfy this unit |

## Parked, not authorized

The following tracks are defined in the
[phase plan](docs/todo/scholarship-and-phase-plan.md#parked-tracks) but are not
active implementation items: a sharper objective-backed replacement for the
server heuristic, streaming/nonstationary inference, multimodal missingness,
privacy-preserving federation, and language-summary generation.
Moving one into the active table first requires its source bundle, threat or
evaluation protocol, primary estimand, independent unit, falsifier, and claim
boundary to be reviewed. No parked track has a version, ordering priority, or
implementation authority until a separate governance review promotes it.

## Shared verification floor

Each item runs the smallest relevant tests during implementation and the full
applicable gate before integration. The command of record remains:

```bash
uv run --locked pytest tests/ -m "not slow" -q
uv run --locked pytest tests/ -m integration -q
uv run --locked pytest tests/ -m publication -q
uv run --locked pytest tests/test_examples.py -q
uv run --locked --extra dev pytest tests/ --cov=src --cov-fail-under=90
uv run --locked ruff check src/ tests/ scripts/ examples/ _fedference_build_backend.py
uv run --locked mypy src/
uv run --locked python scripts/validate_mermaid.py
uv run --locked python scripts/validate_mermaid.py --render --renderer npx --output-dir .tmp/mermaid-render
uv run --locked python scripts/validate_pipeline_freshness.py
uv run --locked python scripts/build_release.py --verify
uv run --locked python scripts/validate_all.py full
```

Publication-facing work additionally requires the real renderer, browser,
figure, PDF, slide, reproducible-build, installed-wheel/sdist, and
candidate-history gates named by its scoped page. A green suite establishes
only the boundary it tests.

## Removal and audit rule

When an item closes, write its exact acceptance evidence to `ISA.md`, a
source-bound receipt, release notes, or the relevant hosted record, then remove
its row and scoped page from this active backlog in the same reviewed change.
Do not keep closed checkboxes, retrospective narratives, stale pass counts, or
old candidate hashes here. Git history and the durable evidence surfaces—not
the forward queue—preserve the audit trail.
