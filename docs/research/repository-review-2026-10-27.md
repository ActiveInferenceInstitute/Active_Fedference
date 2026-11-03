# Repository review and improvement scopes — 2026-10-27

## Scope and disposition

This review covers repository identity and synchronization, dependency and CI
contracts, domain/API boundaries, GNN interoperability, evidence/transport
boundaries, documentation, and existing research/publication plans. It combines
repository-wide static checks with targeted source inspection and runtime
probes. It is not a line-by-line audit of every file, a renewed literature
review, a security certification, a full visual review, or an independent
review of the changes authored during this session.

The existing scientific claim boundaries remain in the
[claim audit](manuscript-claim-audit.md). The active scientific order remains
the [phase plan](../todo/scholarship-and-phase-plan.md); proposed engineering
work below does not change that order or authorize publication. In particular,
ISC-89 and ISC-242 remain open in the checked-in acceptance contract.

## Synchronization evidence

Live `git ls-remote origin HEAD refs/heads/main`, local HEAD and tracking-ref
inspection, and `git -c core.fsmonitor=false status --short` established:

| Repository | Local HEAD and live remote main | Initial worktree | Disposition |
| --- | --- | --- | --- |
| Active Fedference | `c75954181184545acc31cda17c1013d4dfb34d31` | Clean, on `main` | Already synchronized; no pull or merge needed |
| Generalized Notation Notation | `f0dfb68368280f3a43ff91ac28db058c968002d8` | Clean, on `main` | Already the latest remote-main source observed during this review |

Both `origin` URLs point to the corresponding ActiveInferenceInstitute public
repositories. Fedference's local `origin/main` also matches the first SHA.
`gh-axi release view v1.1.0` identifies the same Fedference release commit;
`gh-axi run list --limit 3` reports successful completed workflows, including
public-main run `35488624905`. These observations are dated network snapshots,
not continuing synchronization guarantees. No evidence-mirror synchronization,
push, merge, tag, release mutation, or Zenodo publication occurred.

The initial ordinary status invocation emitted `error: could not read IPC
response`; the explicit command-local `core.fsmonitor=false` probe then
succeeded. No persistent Git configuration was changed. Implementation is on
`codex/repo-review-gnn-20261027`, with uncommitted changes for review.

## Coverage and findings

| Area | Evidence inspected or exercised | Finding / boundary |
| --- | --- | --- |
| Repository/distribution | Remotes, live heads, release record, `pyproject.toml`, `uv.lock`, `MANIFEST.in`, build backend and packaging tests | Source is 1.1.0; several contributor surfaces still described an interim origin and a 1.0.4/1.1.0.dev0 development state |
| Public domain/API | Aggregation recovery/solver contracts, `_validation.py`, `pomdp.py`, `belief_updating.py`, public examples, application/CLI separation | Preserve numerical methods and legacy interfaces; three reproduced invalid-input cases below need focused fixes |
| GNN | Current upstream `gnn.extract` entry point, extraction schema, orientation code, and real subprocess probes | No previous GNN references or dependency found in source, scripts, tests, docs, or project metadata; bounded exchange is now demonstrated |
| Evidence/federation | Typed evidence/registry surfaces, transport encoding, output isolation, source/validation fingerprint declarations | Native `.npy/.npz` encoding and protocol-v1 compatibility remain separate from external model notation and physical deployment |
| Analysis/figures | Typed write boundary, report/figure registry, artifact and visual-contract inventories, tests | Existing schemas are extensive; registry and producer growth justify staged modularization, not wholesale rewrites |
| Publication | Metadata emitter, freshness verifier, release verifier, release plans | Starting-snapshot verification passed; authored source/docs changes require fresh source-bound release evidence before a later release |
| Tests/CI | Workflow, timed module inventory, test profiles, runtime/test-double scans, Ruff and mypy | Broad tests include substantial seeded computations even outside the slow profile; measure before changing profiles |
| Research roadmap | Current minor/medium/major pages, scholarship/phase policy, ISC-89/242 | Hosted releases and successful CI do not close retained reproduction/certification criteria or prove private synchronization |

### Reproduced normalization edge

The public `fedference.pomdp.normalise_columns` (`src/fedference/pomdp.py:62`)
accepts a non-finite matrix
and returns NaNs rather than rejecting it. The following runtime probe returned
`[[nan, 0.5], [nan, 0.5]]`:

```bash
uv run --locked python -c '
import numpy as np
from fedference.pomdp import normalise_columns
print(normalise_columns(np.array([[float("nan"), 1.0], [1.0, 1.0]])))
'
```

This is a bounded input-validation finding, not evidence that canonical seeded
worlds or published results are invalid. A follow-up should reject non-finite
inputs before clipping/division, cover NaN and both infinities plus finite
overflow of column sums, preserve valid finite results, inspect all callers,
and run the source coverage gate. A separate contract decision should resolve
the documented non-negative input versus the current tiny-negative tolerance;
do not silently change that numerical policy in the finite-value fix.

### Reproduced hierarchy and likelihood input holes

Two additional public-input probes demonstrate that the validation concerns
extend beyond the column normalizer:

```bash
uv run --locked python -c '
import numpy as np
from fedference.pomdp import LayerSpec, build_nlevel_world
from fedference.belief_updating import infer_states
world = build_nlevel_world([
    LayerSpec(2, default_prior=np.array([1.1, -0.1])), LayerSpec(9),
])
print(world["level_priors"][0])
print(infer_states(np.array([[-1.0, 0.5], [2.0, 0.5]]), 0, np.log([0.5, 0.5])))
'
```

The hierarchy builder accepts `[1.1, -0.1]` because its check
(`src/fedference/pomdp.py:635`, with conditioned priors at line 658) tests shape and
unit sum without non-negativity. State inference accepts the negative
likelihood and produces approximately `[2e-12, 1]`, because its likelihood
message (`src/fedference/belief_updating.py:95`) clips the selected row before
taking logs. The input has unit-sum
columns, so a sum-only check would not address this case.

Treat these as correctness issues at caller-data boundaries, with priority
before expanding model import. They do not demonstrate failures for the
canonical valid fixtures. Repair the hierarchy default and conditioned-prior
checks together; repair likelihood admission before clipping, with NaN,
infinity, negative, empty and multimodal cases. Preserve legitimate zero
likelihood handling and valid seeded recovery results. Inspect the corresponding
inference and hierarchy callers, including compatibility world constructors
with analogous prior checks, and run the full source coverage gate for the
fix; this maintenance patch does not change those scientific primitives.

### Retained GNN negative-control failure

GNN 5.0.1's headless extractor preserved every canonical A/B/C/D value and the
explicit `B[next_state, previous_state, action]` order. However, replacing A
with `[[1.0]]` produced a warning about `(1, 1)` versus `(9, 9)` and a successful
extraction even with `strict_validation=True` and `on_error="raise"`. The first
probe therefore failed its expectation that upstream would reject the model.
The raw first run is retained locally as
`.tmp/review-20261027/gnn-first.log`; it is deliberately not release evidence.

The example's `--verify-extracted` boundary now compares the extraction schema,
dimensions, and all native matrices explicitly. The corrected negative control
requires this local boundary to reject the corrupted result, whether a future
upstream returns an error envelope or still only warns. It does not assert
that upstream must retain the warning-only behavior. No upstream code was
patched, vendored, or implicitly installed.

## Bounded changes delivered

1. Correct the canonical public checkout/remote guidance in root and
   contributor instructions and the forward-backlog introduction. Keep
   historical release records and pending gate dispositions intact.
2. Add the [sentinel exchange example](../../examples/06_gnn_sentinel.py), using
   the native world builder and JSON numeric literals rather than a second
   implementation of model equations. Default JSON and direct GNN Markdown
   output are both available; file writes remain caller-controlled redirection.
3. Add the optional extraction-parity verifier and real subprocess
   [GNN regression tests](../../tests/test_gnn_interop.py). An explicit invalid
   `GNN_ROOT` fails. Absence skips only the two external integration cases;
   export, exact parity, corruption, schema, missing-value, and B-transposition
   checks remain available without GNN.
4. Document current-upstream discovery, the tested revision, compatible Python
   range, the extraction warning limitation, and the standalone/core boundary
   in the [example ladder](../../examples/README.md).
5. Document temporary-directory, Git/pytest discovery, and child build
   interpreter isolation in the
   [testing guide](../development/testing_philosophy.md).

The GNN lane covers one categorical state factor and one observation modality.
It is numeric exchange evidence, not GNN-backend execution, a general importer,
FedGVI protocol interchange, or Friston protocol replication.

## Validation evidence and retained limits

The final focused command used Python 3.12.15, the locked dev environment, an
explicit current `GNN_ROOT`, and GitVault temporary storage outside the source
checkout. Both `sys.prefix` and `sys.base_prefix` resolve on GitVault.

```bash
uv run --locked --extra dev pytest \
  tests/test_gnn_interop.py tests/test_examples.py tests/test_docs_contract.py \
  tests/test_runtime_surface.py tests/test_build_backend.py \
  tests/fedference/test_provenance.py -q
```

Result: **104 passed**. This includes real upstream extraction, the retained
corruption negative control, B-axis transposition, missing/non-finite data,
incompatible schemas/error envelopes, public examples, source-archive checks,
and source/runtime provenance. Without `GNN_ROOT`, the separate GNN run yielded
**10 passed, 2 skipped**. The two skips are the explicitly external cases.
Ruff passed across source/tests/scripts/examples/build backend; mypy passed
all **142** checked source/example files; all seven Mermaid blocks and the
three emitted metadata surfaces passed their respective checks.
The explicit missing-`GNN_ROOT` negative probe returned exit 1 at the selected
checkout assertion, as required; it did not skip or discover an alternate tree.

The separate exploratory repository run used Python 3.14.9 and
`-m "not slow and not publication"`. It started before the edits, so its
in-memory documentation assertion predates the alias correction. Its retained
result was:

```text
FAILED tests/fedference/test_provenance.py::test_non_git_source_never_invents_a_revision
FAILED tests/test_ci_shards.py::test_real_pytest_plugin_selects_only_its_module_and_records_execution
FAILED tests/test_cli_surface_failure_matrix.py::test_support_helpers_fail_before_writes_and_bind_real_git_state
FAILED tests/test_cli_surface_failure_matrix.py::test_cli_source_requirements_fail_before_application_destination_creation
FAILED tests/test_cli_surface_failure_matrix.py::test_verify_distinguishes_unrequested_source_and_invalid_checkout
FAILED tests/test_docs_contract.py::test_canonical_public_git_contract_is_consistent_across_active_release_pages
FAILED tests/test_fedference_cli.py::test_cli_refuses_committed_reviewer_output
7 failed, 2523 passed, 330 deselected, 894 warnings in 2279.02s
```

The six fixture/discovery failures are covered by the corrected 70-test rerun;
the documentation failure is covered by the corrected 60-test documentation
module and combined 104-test run. The original red run remains red. These
targeted reruns do not constitute a fresh complete coverage/publication gate
on an immutable final commit.

The author validation retained these failures and corrections:

- The initial upstream-only shape-rejection assumption failed, as detailed
  above; explicit local parity verification addresses that admission gap.
- The old documentation assertion failed after correcting the alias guidance.
  Its replacement checks the actual evidence-mirror URL, with the remaining
  public/review/integration policy assertions preserved. All 60 documentation
  tests passed afterward.
- A documentation scan run concurrently with packaging discovered a temporary
  expanded sdist directory and reported its absent historical PDF. Repeating
  the scan after the build completed passed; do not interpret this as a missing
  source-repository PDF. Keep source discovery and packaging appropriately
  isolated in ENG-5.
- An initial Python 3.12 run produced **97 passed, 3 failed**: explicit selection
  of the base interpreter made no-isolation child builds miss `setuptools`.
  Binding `UV_PYTHON` to the populated external project-environment interpreter
  resolved it; all 15 packaging tests and the combined final 104-test run passed.
- Temporary fixtures inside this source checkout made the supposedly non-Git
  provenance fixture inherit the enclosing repository. The isolated failing
  reproduction is retained. All four provenance tests passed with a task-owned
  GitVault temporary directory outside this checkout; no provenance behavior
  was relaxed to make a test pass.
- A broader rerun outside the checkout still produced **38 passed, 3 failed**:
  GitVault's ancestor repository/configuration also affected Git and pytest
  discovery. Explicit `GIT_CEILING_DIRECTORIES` and `PYTEST_ADDOPTS=--rootdir=.`
  boundaries then yielded **70 passed** across the shard-planner, CLI failure
  matrix, installed CLI, and provenance modules. The testing guide records this
  setup. Both failed environments remain in the local logs.

The installed-wheel example also passed outside the source working directory;
its native package imported from `site-packages` without GNN or Torch entering
the core import graph. Scratch wheel/sdist builds were byte-identical in two
fixed-epoch builds, and the source archive included the new example, tests and
review. These are local packaging checks, not publication assets or an
independent release verdict.

Starting-snapshot metadata, analysis/hydration/render freshness, and release
verification passed. After the edits, release verification correctly returned
exit **2** with:

```text
release bundle failed: validation receipt preflight failed: validation receipt input hashes are stale; validation receipt input digest is stale
```

That failure is retained. The published/generated snapshot was not rewritten
to erase the freshness boundary. A complete fresh coverage/publication/render/
two-clone campaign, visual review, external reproduction, private-mirror sync,
and live Zenodo asset verification were not performed by this maintenance work.

## Medium engineering proposals

These scopes are proposals for separate reviewed changes. Sizes are relative;
no performance, release, or scientific outcome is promised.

### ENG-1 — current-release and backlog reconciliation (first)

- **Surface:** root guidance, `STANDALONE.md`, release guide, active TODO pages,
  ISA summary versus historical iterations, repository discovery metadata.
- **Prerequisite:** read-only verification of the actual public GitHub and
  Zenodo records, exact release assets, eligible verdicts, two-clone receipts,
  and evidence-mirror state. This session checked GitHub identity, not Zenodo
  asset bytes or the private mirror.
- **Deliverable:** one dated release-state ledger that distinguishes observed
  publication from satisfied project acceptance criteria; retire backlog rows
  only against their complete evidence. Derive current identity prose from
  canonical config where practical; preserve historical evidence verbatim.
- **Acceptance / falsifier:** all current-release links and aliases resolve;
  a mismatched DOI, missing receipt, changed asset, or stale branch must prevent
  closure. Existing open ISC-89/242 dispositions cannot disappear merely
  because a tag is hosted.

### ENG-2 — reusable categorical GNN adapter

- **Surface:** a new explicit domain/boundary adapter, public exports, CLI
  adapter, model fixtures, ISA row, sdist inventory, source receipts.
- **Prerequisite:** the demonstrated sentinel exchange and documented upstream
  extraction schema; define state/action labels and vector/axis conventions.
- **Deliverable:** typed single-factor categorical import/export for admitted
  models, with explicit unsupported-feature errors and a content-bound receipt
  containing upstream revision/tree state, parser/extractor source hashes,
  schema version, model digest, and native-result digest.
- **Acceptance / falsifier:** exact values and labels survive supported round
  trips; reject missing/duplicate parameters, non-finite values, contradictory
  dimensions, axis swaps, unsupported factors, and changed upstream closure.
  Default `import fedference` remains GNN/Torch-free. Test supported GNN Python
  versions separately from Fedference's broader Python 3.10 floor.
- **Limit:** no backend or scientific equivalence claim. Do not serialize site
  factors/cavities as ordinary categorical posteriors without a new contract.

### ENG-3 — upstream conformance and compatibility lane

- **Surface:** CI and explicit optional integration profile.
- **Deliverable:** pin one approved GNN revision for reproducible required
  checks, and separately report a current-main compatibility probe. Cache and
  import only the declared source closure in an isolated interpreter.
- **Acceptance / falsifier:** execute real extraction and the retained shape
  negative control; a bad root, unreviewed parser change, schema change, or
  wrong matrix must fail the appropriate lane. Do not silently advance the pin
  or make network currentness part of offline default package imports.

### ENG-4 — report and publication module boundaries

- **Surface:** `analysis/report_schemas.py`, `analysis/workflow.py`,
  `publication/web_package.py`, `publication/zenodo.py`, `fedference/evidence.py`.
- **Deliverable:** split by existing schema family, producer operation, HTML
  resource policy, remote-operation phase, and receipt verification concern;
  retain compatibility facades and a single validation/write boundary.
- **Acceptance / falsifier:** unchanged serialized schemas, finding codes,
  public imports, byte-stable deterministic outputs, and representative invalid
  inputs rejected before writes. A changed receipt meaning or weakened
  fail-closed behavior blocks the split. Refactor one concern per change.

### ENG-5 — test cost, environment, and isolation

- **Surface:** pytest profiles, `.github/ci/module-times.json`, shard planner,
  `tests/conftest.py`, contributor setup.
- **Deliverable:** profile actual expensive modules, separate fast mechanics
  from retained seeded integration/scientific tests, and remove obsolete
  ancestor-checkout `sys.path` assumptions after outside-checkout probes.
  Make the chosen external Python explicit: an unpinned local `uv run` during
  this review selected Python 3.14.9 after the initial explicit 3.12.15 setup.
- **Acceptance / falsifier:** identical complete test inventory across shards,
  no dropped numerical tests, Python 3.10/3.12/3.13 compatibility retained,
  concurrent scratch namespaces, source coverage floor retained, and commands
  independent of unrelated ancestor packages. Record runtime improvement only
  after comparable measurements; do not invent a target from old timings.
  Preserve the demonstrated Git/pytest discovery boundaries when task-owned
  temporary roots sit under a separately configured workspace repository.

### ENG-6 — boundary-value and write-custody hardening

- **Surface:** POMDP normalization first, then evidence/checkpoint/replay and
  CLI output boundaries selected by concrete reproductions.
- **Deliverable:** fix the reproduced normalization, hierarchy-prior, and
  likelihood-admission holes as isolated numerical-boundary changes; audit
  analogous numeric boundaries and concurrent/aborted writes without changing
  admitted scientific methods. Keep exclusive creation, atomic publication,
  symlink policy, and failure cleanup explicit where required.
- **Acceptance / falsifier:** real malformed numeric inputs and real competing
  subprocess writers cannot yield accepted corrupt or overwritten evidence;
  valid current fixtures preserve their numerical results. New findings need
  separate reproductions; this review is not a completed security audit.

## Major scientific and deployment scopes

The standing medium lanes also remain governed by their existing pages:

| Existing lane | Required next evidence |
| --- | --- |
| [MED-4 visual scholarship](../todo/visual-scholarship-integration.md) | Resolve the current hosted/reviewer state against exact renderer/source commits, complete figure/slide/HTML/PDF evidence, and the public integration checks; this review performs no fresh visual acceptance |
| [MED-5 publication](../todo/v1-1-release-publication.md) | Reconcile already hosted release identity with the exact certification, immutable asset, Zenodo, and private-sync receipts before retiring the active page; a published tag cannot replace missing evidence |
| [MED-6 reproduction](../todo/independent-reproduction.md) | Package one fixed final evidence set for the independent human/Advisor and genuinely different-vendor execution lanes; same-family local validation cannot close ISC-89 |

Existing scoped pages remain the authority. Engineering GNN exchange can aid
configuration custody, but cannot discharge any scientific prerequisite.

| Lane | Deliverable and implementation boundary | Required evidence / decisive failure |
| --- | --- | --- |
| [MAJ-8 calibration](../todo/adaptive-robustness-calibration.md) | Separate robust/variational tuning under sealed train/calibration/test partitions; freeze source, data, budgets, and selection policy | Complete failures/candidate table, hierarchical-bootstrap stability and frozen design digest; any final-test leakage invalidates the freeze |
| [MAJ-6 external confirmation](../todo/external-benchmark-domain-pilot.md) | Execute the registered three-dataset paired comparison only after MAJ-8 | Twelve-cell primary family, separate clean noninferiority family, MCSE/Holm disposition, solver/provenance matrix; retain valid negative/null results |
| [MAJ-2A/B FedGVI](../todo/faithful-fedgvi-bnn-lane.md) | Portable cavity/site-factor source-protocol lane, then separately resourced source-scale CUDA execution | Source-revision/parameter/estimand parity, checkpoint/resume and device receipts; a point-estimate baseline or declared CUDA config cannot substitute |
| [MAJ-7 Friston protocol](../todo/faithful-friston-protocol-replication.md) | Reconstruct the source factor graph, message schedule and source figures before claiming full protocol replication | Resolved Eq. 2 and Figures 5/7/9 parity, native/independent units and protocol negative control; categorical A/B/C/D exchange is insufficient |
| [MAJ-3 continuous/hybrid](../todo/beyond-discrete-categorical-state-spaces.md) | Recovery-gated continuous/hybrid inference with calibrated matched controls | Covariance/recovery gates, proper-score confirmation and full discrete/continuous/oracle comparators; unstable covariance or failed recovery blocks promotion |
| [MAJ-5 richer hierarchy](../todo/deeper-hierarchy-task-family.md) | Preregistered task-family comparisons using existing generic LayerSpec construction | Independent task units, matched controls, correction and task-specific fallback; increased depth alone cannot establish benefit |
| [MAJ-4A/B federation](../todo/true-multi-machine-federation.md) | Local mTLS/certificate/restart emulator, then physical distinct-host deployment | Fault/replay/identity matrices and per-host receipts; containers on one host do not satisfy physical federation, privacy, or Byzantine resilience |

A **further GNN representation extension** would be major if it changes the
admitted model family: multifactor/hierarchical models, continuous/hybrid
blocks, or distributed factor graphs. First define semantics, supported
operators, units, schedules, and precision policy; then require recovery to
the existing native model, an independently derived oracle, rejection of
unsupported semantics, and backend execution on matched seeded fixtures.
Do not promote this proposal into the active roadmap until the existing phase
governance accepts its source bundle, primary estimand, independent unit,
falsifier, budget, and claim boundary.

## Suggested execution order

1. Review the bounded maintenance/example diff and reconcile ENG-1 evidence.
2. Fix the reproduced numerical boundaries as isolated changes; measure
   ENG-5 test costs while keeping the complete source gate intact.
3. Implement ENG-2, then ENG-3; negotiate upstream warning/error semantics
   without weakening local matrix/label checks.
4. Split one ENG-4 module concern at a time and execute only reproduced ENG-6
   hardening tasks, retaining every compatibility/failure fixture.
5. Close the prerequisite publication/documentation/synchronization gates
   using their actual evidence, then follow MAJ-8 → MAJ-6. Other major lanes
   retain their own source, resource, and independence prerequisites.

Every release remains a separate authorized action. The checked-in reviewer
snapshot and immutable v1.1.0 release bytes were not regenerated by this work;
source-bound receipt/fingerprint verification after these changes is expected
to require the normal producer sequence before a future release.
