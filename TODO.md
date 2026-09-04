# Active Fedference — Active Forward Backlog

> This file contains only open work. Closed implementation and release evidence
> is retained in [`ISA.md`](ISA.md), source-bound reports, release records, and
> Git history. Generated counts and hashes are deliberately not copied here.

The authoritative public integration repository is
[`ActiveInferenceInstitute/Active_Fedference`](https://github.com/ActiveInferenceInstitute/Active_Fedference)
(`public`). The `origin` remote, `docxology/active_fedference`, remains the
private/interim evidence repository. “Push to main” therefore means: fix an
exact candidate SHA, pass the applicable confidentiality and verification
gates, push a `codex/*` branch to the required remotes, obtain green review
checks, and merge into public `main` with a normal merge commit. It never means
a direct, forced, squashed, rebased, stale-head, or check-bypassing main push.

Every open item has one owner page under
[`docs/todo/`](docs/todo/README.md). Tiers describe the size and coordination
surface of the remaining work, not whether its gate is optional: a Minor item
can still block a release. The source-indexed dependency map is
[`docs/todo/scholarship-and-phase-plan.md`](docs/todo/scholarship-and-phase-plan.md).

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

The active critical path is:

1. use the exact reviewed and merged accessible Template renderer commit from
   a separate clean checkout and complete the source-only visual-scholarship
   candidate;
2. run the candidate-specific portion of MIN-2, push the exact visual branch to
   both remotes, merge it normally into public `main`, and require green
   post-merge CI;
3. run MED-5A: create and inspect the linked Zenodo draft, bind the reserved DOI
   to the release identity, merge the release-only PR normally, and require
   green post-merge public-main CI;
4. certify that exact final public-main merge commit through MIN-2's sequential
   two-clone campaign and obtain the selected identified owner-author verdict;
5. run MED-5B's publication phase: after an exact-SHA `pass`, create the
   immutable tag and verified GitHub release, update and verify the Zenodo draft,
   pause for fresh approval, then publish and independently verify Zenodo;
6. complete and normally merge MIN-3's public documentation and repository-
   discovery handoff, including removal of its closed forward-backlog entry;
7. finish MED-5B by synchronizing that final public `main` into private
   `origin/main` exactly once with an equal tree;
8. execute MAJ-8 preregistration and calibration freeze before MAJ-6 may inspect
   sealed confirmatory rows; and
9. select any later scientific version only after the confirmatory result is
   known and reviewed.

ISC-89 and ISC-242 remain open. The exact-SHA owner-author verdict required for
v1.1 tagging does not close the broader independent/cross-vendor reproduction
lane, and Zenodo publication still requires a fresh explicit approval after
the draft and GitHub release have been verified.

## Minor — bounded maintenance and evidence checks

| ID | Open item | Prerequisite | Acceptance evidence |
| --- | --- | --- | --- |
| MIN-2 | [Clone-correct integration and release verification](docs/todo/release-and-verification-ladder.md) | Exact candidate or final-public-main SHA; clean pinned renderer; safe external namespace and headroom where the clone phase applies | Candidate history audit; green source/render/package gates; two isolated final-SHA clone receipts; eligible structured verdict; ISC-242 remains open until its complete evidence set exists |
| MIN-3 | [Release documentation and repository-discovery handoff](docs/todo/release-documentation-handoff.md) | Verified public tag, GitHub release, and published Zenodo version | Link and installed-artifact probes show the public repository, version, DOI, checksums, install path, application guide, and accessibility boundaries consistently; no stale development identity remains on current-release surfaces |

## Medium — cross-surface integration and publication

| ID | Open item | Prerequisite | Acceptance evidence |
| --- | --- | --- | --- |
| MED-4 | [Visual scholarship and accessible-slide integration](docs/todo/visual-scholarship-integration.md) | Green public `main`; exact merged Template renderer commit in a separate clean checkout; source-bound reports; approved four-commit sanitized replay based on refreshed public `main`; no use of the user-owned dirty Template checkout | Reviewed Template merge SHA; exact four-commit Fedference visual branch; all figure pairs, long descriptions, exact-value fallbacks, HTML/reveal/Beamer/PDF QA, confidentiality audit, green PR and post-merge public-main workflows |
| MED-5 | [v1.1.0 release, GitHub/Zenodo publication, and private synchronization](docs/todo/v1-1-release-publication.md) | MED-4 merged and green; current metadata approvals for MED-5A; MIN-2 exact-final-SHA certification and `pass` only after the release PR merges and before MED-5B | Release-only normal-merge PR; immutable annotated tag; downloaded-and-reverified GitHub assets; verified linked Zenodo version after fresh publish approval; one tree-identical private-main merge; exact SHAs, URLs, and hashes recorded |
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
active implementation items: streaming/nonstationary inference, multimodal
missingness, privacy-preserving federation, and language-summary generation.
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
