# Active Fedference — Robust Federated Active Inference

[![Public GitHub repository](https://img.shields.io/badge/GitHub-ActiveInferenceInstitute%2FActive_Fedference-181717?logo=github)](https://github.com/ActiveInferenceInstitute/Active_Fedference)

Active Fedference is a research project that reimplements **FedGVI** (Federated
Generalized Variational Inference; Mildner, Hamelijnck, Giampouras & Damoulas,
2025, PMLR 267; arXiv:2502.00846) in the discrete-categorical setting and connects it to
the federated **belief-sharing** scenario of Friston et al.
(2024), *Federated inference and belief sharing* (Neurosci. Biobehav. Rev.
156:105500).

## Start here

The complete path for applying the software to your own categorical beliefs is
the [`application guide`](docs/application-guide.md). It covers installation,
the shared-state input contract, method selection, result diagnostics,
recipient-specific sharing, local process/socket boundaries, CLI receipts, and
troubleshooting.

The default runtime requires Python 3.10 or newer and
[`uv`](https://docs.astral.sh/uv/getting-started/installation/); it does not
require Torch or a GPU. These commands use a POSIX shell. To validate the
included labeled request, aggregate it, and verify its application receipt from
the root of the source revision containing this guide:

```bash
uv sync --locked
FEDFERENCE_APP_ROOT="$(mktemp -d /tmp/active-fedference-app.XXXXXX)"
uv run --locked fedference aggregate \
  --input examples/data/labeled_aggregation_request.json \
  --output-dir "$FEDFERENCE_APP_ROOT/run" \
  --project-root .
uv run --locked fedference verify \
  "$FEDFERENCE_APP_ROOT/run/receipt.json" \
  --require-nominal-solver
```

This source carries the `1.1.0` release identity and reserved version DOI
[`10.5281/zenodo.22149133`](https://doi.org/10.5281/zenodo.22149133).
The intended UTC publication date is 2026-09-19. Check the
[v1.1.0 release record](https://github.com/ActiveInferenceInstitute/Active_Fedference/releases/tag/v1.1.0)
for publication and the five verified assets; a source version or reserved DOI
alone does not establish publication. See the
[release installation and checksum instructions](docs/application-guide.md#install-the-v110-release).
The v1.0.4 release predates this application interface; its DOI applies only
to that earlier release and PDF.

The same labeled envelope is available as a pure Python operation:

```python
from fedference import LabeledAggregationRequest, aggregate_labeled
from pathlib import Path

request = LabeledAggregationRequest.from_json(
    Path("examples/data/labeled_aggregation_request.json").read_text()
)
result = aggregate_labeled(request)
print(result.state_labels, result.aggregation.consensus)
print(result.aggregation.solver_status)
```

Every posterior must be a one-dimensional, finite, non-negative categorical
mass vector over the same ordered state labels. Caller masses are preserved in
the canonical request and exposed separately as normalized rows in the result.
For iterative rules, accept a result only under your application's explicit
policy for `solver_status` and `fallback_events`; see the
[`input and result contract`](docs/application-guide.md#the-labeled-request).

Choose the narrowest interface that matches the job:

| Need | Use | Boundary |
| --- | --- | --- |
| Fuse labeled caller data and retain evidence | `fedference aggregate` | Strict JSON plus request/result/application receipt |
| Fuse labeled caller data in Python | `aggregate_labeled` | Pure typed operation; no file I/O |
| Use the smallest numeric operation | `aggregate_result` | Labels/provenance remain caller-owned |
| Give each agent a shared belief | `share_round_result` | In-process; optionally excludes the recipient's own belief |
| Exercise process isolation | `fedference.federation.run_multiprocess_round_result` | Spawned worker processes on one machine |
| Exercise transport and replay | `fedference.federation.run_socket_round_result` | Loopback TCP, optional HMAC, digest-checked replay |
| Run a registered research profile | `fedference run` / `verify` | Isolated outputs plus a content-bound receipt |
| Use the BNN complement | install the `bnn` extra | Optional Torch path, outside the default import graph |

The complete five-example ladder is in [`examples/README.md`](examples/README.md),
and the source-owned application guide explains how to adapt each boundary.
It compares all three aggregation rules, distinguishes aggregation from belief
sharing and federation, and demonstrates application/research receipts and
replay. An application receipt proves declared input/output/provenance
integrity, not scientific validity or a downstream decision. These are
usage examples, not a complete Friston protocol reconstruction, a universal
robustness result, a production multi-host network, or confirmatory evidence.

## Release records

The prior source-bound v1.0.4 reviewer snapshot is published and its release metadata
has been checked against the live Zenodo record. The public source, release tag,
and immutable research artifact now cross-reference the same version.

- **Published DOI:** [`10.5281/zenodo.21972644`](https://doi.org/10.5281/zenodo.21972644)
  · [Zenodo record](https://zenodo.org/records/21972644)
- **Prior public release:** [v1.0.4 GitHub release](https://github.com/ActiveInferenceInstitute/Active_Fedference/releases/tag/v1.0.4)
- **Public source repository:**
  [`ActiveInferenceInstitute/Active_Fedference`](https://github.com/ActiveInferenceInstitute/Active_Fedference)
- **Released manuscript PDF:**
  [`Active_Fedference_Research_Manuscript_v1.0.4_Zenodo_10.5281-zenodo.21972644.pdf`](Active_Fedference_Research_Manuscript_v1.0.4_Zenodo_10.5281-zenodo.21972644.pdf)

The released PDF embeds the published DOI and public repository URL as release
metadata; both cross-references were rechecked after publication. The v1.0.3,
v1.0.2, v1.0.1, and v0.1.0 records remain available as public prior versions.
This checkout is
the standalone development/review source; its configured `origin` is the
interim [`docxology/active_fedference`](https://github.com/docxology/active_fedference)
remote.

The headline result is a **project-local** identity. On the same categorical
inputs, the zero-robustness server path is bit-identical to the project's
log-linear pool. In code:

```python
import numpy as np

from fedference.aggregation import log_linear_pool, robust_aggregate

local_posteriors = np.array(
    [[0.70, 0.20, 0.10], [0.60, 0.30, 0.10], [0.10, 0.10, 0.80]],
    dtype=np.float64,
)

# Exact project identity: zero-robustness robust pooling == log-linear pooling.
assert np.array_equal(
    robust_aggregate(local_posteriors, robustness=0.0).consensus,
    log_linear_pool(local_posteriors),
)
```

The comparison with Friston et al. (2024) Eq. 7 is a **categorical
posterior-log-potential specialization**, not a reconstruction of its complete
message-passing protocol. It assumes a shared categorical support,
`q_n = softmax(m_n)` for the admitted local message potential `m_n`, and a
fixed project weight mapping. Then `softmax(sum_n w_n log q_n)` equals
`softmax(sum_n w_n m_n)` because the local normalizers are state-independent.
It does not reproduce the source factor graph, cavity/message schedule, or all
source protocol choices.

Robustness is therefore a tested recovery-limit extension of the *project's*
categorical pool, not a competitor to or full reconstruction of Friston's
protocol. The source-conditional robust client loss has its own
bounded-influence result; the objective-backed variational server rule instead
bounds its raw effective-weight update and shows redescending behavior only on
declared paths. The client KLD/NLL/beta=0 limit and the server
zero-robustness identity are separate recovery statements. The sharper
server-side `robust_aggregate` rule is reported separately as a heuristic whose
proven property is the project-local recovery limit.

## The problem this solves

The project log-linear pool is a product-of-experts: every agent holds a
multiplicative veto. Under the categorical bridge assumptions, it gives a
scoped view of the belief-sharing mechanism related to Friston. In the fixed
categorical world and attack geometry studied here, a miscalibrated or
adversarial sentinel can drag the colony off the truth; the source mechanism
itself is not a contamination experiment. FedGVI provides robust
federated-learning machinery, and Active Fedference evaluates this scoped
categorical bridge rather than claiming a universal literature gap.

## Architecture

Thin orchestrators provide stable subprocess/CI boundaries for root selection,
validation, and release sequencing; reusable mathematics and publication
logic remain importable in `src/`, with named boundary adapters for evidence,
data, checkpoints, and transport.

The installed CLI follows the same modular contract: `fedference_cli/__init__.py`
is a compatibility facade, `_parser.py` owns the process grammar,
`_commands.py` owns registry-backed dispatch, and `_support.py` owns atomic
writes, output isolation, validation, and receipts. The package map and
extension recipe are documented in
[`src/fedference_cli/README.md`](src/fedference_cli/README.md) and
[`docs/development/modularity.md`](docs/development/modularity.md).

```mermaid
flowchart TD
    accTitle: Reader-level Active Fedference architecture
    accDescr: Callers enter through typed library or CLI boundaries, the mathematical core owns aggregation, explicit adapters own effects, and source-bound reports feed accessible publication surfaces.
    callers["Library callers, applications, and tests"] --> boundary["Typed Python API and installed CLI"]
    boundary --> core["NumPy/SciPy mathematical and active-inference core"]
    core --> federation["In-process, spawned-process, and loopback federation adapters"]
    core --> analysis["Typed analysis reports and evidence receipts"]
    adapters["Explicit data, checkpoint, replay, and optional-Torch boundaries"] --> core
    config["Source configuration, registries, and dependency lock"] --> core
    config --> analysis
    analysis --> publication["Figures, hydrated manuscript, HTML, PDF, and slides"]
    publication --> gates["Freshness, accessibility, package, and release gates"]
```

**Text equivalent.**

| From | To | Meaning |
| --- | --- | --- |
| Library callers, applications, and tests | Typed Python API and installed CLI | Callers select the narrowest stable boundary instead of reimplementing domain logic. |
| Typed boundary | NumPy/SciPy core | Validated requests delegate to reusable aggregation and active-inference operations. |
| Core | Federation adapters | In-process, process, and loopback routes reuse the same aggregation rules. |
| Explicit adapters | Core | Data, checkpoint, replay, and optional-Torch effects enter only through named boundaries. |
| Configuration and registries | Core and typed analysis | Source-owned controls determine execution and report interpretation. |
| Typed analysis | Reader surfaces | Reports feed figures and hydrated HTML, PDF, and slide outputs. |
| Reader surfaces | Publication gates | Freshness, accessibility, packaging, and release checks remain separate from scientific claims and publication authority. |

The architecture diagram is source documentation, not an import-graph claim:
the executable layer gate remains `src/fedference/`-only and the report-schema
validator remains the single JSON write boundary. All README/docs Mermaid
blocks use GitHub-compatible fenced syntax. Check them without a browser with:

```bash
uv run --locked python scripts/validate_mermaid.py
uv run --locked python scripts/validate_mermaid.py --render --renderer npx \
  --output-dir .tmp/mermaid-render
```

The second command invokes Mermaid CLI and writes only review scratch files;
the `.mmd` sources remain in Markdown so GitHub renders the same diagrams.
The detailed module inventory below is supplemented by the responsibility and
module tables in [`docs/core/architecture.md`](docs/core/architecture.md).

- **`src/fedference/` core** — `divergences`, `losses`, `generalized_bayes`,
  `aggregation`, `belief_sharing`, and the implementation-derived `complexity`
  catalog. The recovery limits live here and are pinned
  by tests (the $\beta\to0$, $\alpha\to1$, robustness$\to0$ corners).
- **Active-inference machinery** — `pomdp` (sentinel world), `belief_updating`
  (variational state inference + free energy), `dirichlet_learning` (language
  acquisition), `expected_free_energy`, `bayesian_model_reduction`.
- **Ensemble** — `agents.SentinelEnsemble` (shared vs private factors),
  `colonies` (seeded colony builders), `contamination` (confident-wrong /
  label-noise saboteurs).
- **Experiments** — the `experiments/` subpackage wires the locked primitives into nine
  JSON-serializable core studies (three categorical source-mechanism analogues plus the
  robustness sweep, moving-world and hierarchical extensions, sensitivity,
  parameter recovery, and a hierarchical structure-learning study) plus extension studies (contamination gallery, robustness
  onset, descent comparison, etc.); `statistics.py` supplies the paired Wilcoxon
  test and BH-FDR that qualify the declared server contrast; `bnn_baseline.py`
  provides a separate exploratory generalized-Bayes point-estimate
  logistic-regression baseline comparing joint NLL/L2 and RCCE/L2
  configurations (its legacy `AR` argument is only an L2-coefficient selector,
  so the comparison does not isolate an RCCE-only effect), and the optional
  `bnn_baseline_torch.py` and `bnn_variational_torch.py` modules provide
  composable point-mass and mean-field MLP complements when torch is installed.
  The completed synthetic-pilot cavity wiring is retained as implementation
  evidence; source-data/CUDA parity future work remains open and cannot be
  inferred from that pilot.
- **Complexity diagnostic** — `run_complexity_scaling` calculates the dense
  implementation orders, measures seeded aggregation/sharing/inference scaling
  on the configured machine, and records timing variability without promoting
  machine-specific slopes to general performance claims.
- **Research/evidence boundary** — `research_registry` declares source bundles,
  datasets, profiles, estimands, falsifiers, and no-claim outcomes;
  `external_data` owns hash-checked caller-cached acquisition; `evidence`
  writes versioned content-bound receipts; and the installed CLI exposes those
  contracts without writing into the committed reviewer snapshot.
- **Federation transport** — ``federation/`` subpackage is a real queue-based
  single-machine multiprocess transport (multiprocessing queues + worker OS
  processes: `process.py`, `server.py`, `worker.py`, `transport.py`), verified
  bit-identical to in-process aggregation end-to-end. The same protocol also
  has a tested loopback-TCP adapter with versioned round/worker/configuration
  envelopes, optional HMAC framing, and persisted digest-verified replay
  (`socket_transport.py`). A caller-shared `ReplayGuard` rejects round-id reuse
  within one process, while `PersistentReplayGuard` makes that claim durable
  across local process restarts using caller-owned SQLite state. Docker/mTLS
  emulation, multi-host replay-domain design, and physical cross-host
  deployment remain open, separate lanes; see the
  [threat model](docs/security/active_fedference-threat-model.md).

The central **project-local** identity is
`robust_aggregate(robustness=0) == log_linear_pool`; with its default
`entropy_weight=1`, `variational_aggregate` has the same zero-robustness
recovery. Under the documented categorical posterior-log-potential assumptions,
the pool specializes the source Eq. 7; it is not a reconstruction of the full
source protocol. Separately, `generalized_posterior(KLD, NLL)` equals
closed-form prior×likelihood Bayes.

The stable rich-result entry point is `aggregate_result`. A single validated
configuration travels unchanged through direct sharing, queue/process
federation, and loopback sockets:

```python
from fedference import AggregationConfig, aggregate_result

local_posteriors = [
    [0.70, 0.20, 0.10],
    [0.60, 0.30, 0.10],
    [0.10, 0.10, 0.80],
]
config = AggregationConfig(
    method="variational",
    robustness=1.5,
    entropy_weight=0.8,
    max_iter=64,
    tol=1e-9,
)
result = aggregate_result(local_posteriors, config=config)
print(
    result.consensus,
    result.solver_status,
    result.fallback_events,
    config.fingerprint,
)
```

The existing top-level aggregation functions and `aggregate(...)` retain their
array-returning compatibility behavior. Passing both a configuration and legacy
tuning arguments fails explicitly instead of silently choosing one. A nominal
rich result has an empty `fallback_events` tuple; any numerical base-weight
substitution is explicit, and a trajectory returned from that substituted state
is not promoted to solver convergence. A multi-start operation can still report
a fallback from a discarded start while returning a separately converged,
fallback-free start.

### Composable integration surfaces

The package is deliberately assembled from small, typed boundaries:

- `fedference.aggregation` provides pure categorical pooling and the rich
  `aggregate_result(..., config=AggregationConfig(...))` contract.
- `fedference.application` adds ordered state labels, stable agent identities,
  canonical semantic hashing, and `aggregate_labeled` without duplicating the
  aggregation mathematics.
- `fedference.belief_sharing` reuses the same aggregation protocol for direct
  rounds and exposes `share_round_result`; `fedference.federation.process` and
  `fedference.federation.socket_transport` provide one-machine process and
  loopback rich-result adapters without duplicating the math. Existing
  consensus/diagnostic compatibility returns remain unchanged.
- `fedference.evidence`, `external_data`, checkpoint helpers, and replay
  guards own explicit caller-authorized I/O and produce verifiable application
  and research receipts.
- `fedference_cli` is a thin aggregate/registry/run/benchmark/verify/replay boundary;
  optional Torch/BNN modules are imported only when the corresponding extra is
  selected.
- `src/figures` consumes validated reports, while `_metadata.py`, manuscript
  captions, and `output/figures/figure_registry.json` keep visual meaning and
  provenance synchronized.

Public legacy names remain available. New code should compose the typed
configuration/result path and treat transport, evidence, rendering, and
publication as adapters around the same core operation.

## How to run

From the project root:

```bash
# Install the pure NumPy/SciPy core with the committed lockfile.
uv sync --locked

# Add the reproducibility, test, lint, type-check, and CPU/Torch tooling.
uv sync --locked --extra dev

# Optional mean-field BNN and CPU/MPS support
uv sync --locked --extra bnn

# Run the full project test suite with the 90% coverage gate on src/
uv run --locked --extra dev pytest tests/ \
  --cov=src --cov-fail-under=90

# Spot-check four of the nine tracked studies (deterministic under the config seed)
uv run --locked python -c "
from fedference import experiments as e
seed = 20240601
print('belief sharing :', e.run_belief_sharing(seed))
print('language       :', e.run_language_acquisition(seed)['final_kl'])
print('emergence      :', e.run_emergence(seed)['convergence'])
print('robustness     :', e.run_robustness_sweep(seed)['any_robust_wins'])
"

# Inspect the source-bound complexity catalog and measured scaling report after analysis
uv run --locked python -c '
import json
from pathlib import Path
r = json.loads(Path("output/reports/complexity_scaling.json").read_text())
print([(row["method"], row["axis"], row["observed_log_log_slope"]) for row in r["measurements"]])
'
```

### Installable artifacts

The source checkout is the reproducibility workspace. For a packaged install,
build the wheel or source distribution with the pinned lockfile and install
the resulting artifact into an isolated environment:

```bash
export SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)"
FEDFERENCE_PACKAGE_ROOT="$(mktemp -d /tmp/active-fedference-package.XXXXXX)"
uv build --out-dir "$FEDFERENCE_PACKAGE_ROOT/dist"
uv venv "$FEDFERENCE_PACKAGE_ROOT/env"
uv pip install --python "$FEDFERENCE_PACKAGE_ROOT/env/bin/python" \
  "$FEDFERENCE_PACKAGE_ROOT"/dist/*.whl
cd "$FEDFERENCE_PACKAGE_ROOT"
"$FEDFERENCE_PACKAGE_ROOT/env/bin/python" -c \
  "import sys, fedference; assert 'torch' not in sys.modules; print(fedference.__version__)"
"$FEDFERENCE_PACKAGE_ROOT/env/bin/fedference" aggregate \
  --input /path/to/labeled_aggregation_request.json \
  --output-dir "$FEDFERENCE_PACKAGE_ROOT/wheel-run"
"$FEDFERENCE_PACKAGE_ROOT/env/bin/fedference" verify \
  "$FEDFERENCE_PACKAGE_ROOT/wheel-run/receipt.json" --require-nominal-solver

# Exercise the supported source-distribution surface independently.
uv venv "$FEDFERENCE_PACKAGE_ROOT/sdist-env"
uv pip install --python "$FEDFERENCE_PACKAGE_ROOT/sdist-env/bin/python" \
  "$FEDFERENCE_PACKAGE_ROOT"/dist/*.tar.gz
"$FEDFERENCE_PACKAGE_ROOT/sdist-env/bin/python" -c \
  "import sys, fedference; assert 'torch' not in sys.modules; print(fedference.__version__)"
"$FEDFERENCE_PACKAGE_ROOT/sdist-env/bin/fedference" aggregate \
  --input /path/to/labeled_aggregation_request.json \
  --output-dir "$FEDFERENCE_PACKAGE_ROOT/sdist-run"
"$FEDFERENCE_PACKAGE_ROOT/sdist-env/bin/fedference" verify \
  "$FEDFERENCE_PACKAGE_ROOT/sdist-run/receipt.json" --require-nominal-solver
```

The wheel contains the typed importable runtime, `fedference/py.typed`, CLI,
and packaged compatibility inputs;
the source distribution additionally carries the modular `docs/`, `manuscript/`,
`scripts/`, `tests/`, and runnable `examples/` trees, including the application
guide, all numbered examples, example data, and manuscript configuration
template, for archival and source-level reproduction.
The exact wheel/source-distribution installation and byte-reproducibility
probes are maintained in
[`docs/reference/verification-commands.md`](docs/reference/verification-commands.md).
The runtime package is MIT-licensed; see [`LICENSE`](LICENSE).

Experiment parameters (seed, per-study knobs, the robustness-sweep grid,
divergence labels, statistical $\alpha$) all live in
[`manuscript/config.yaml`](manuscript/config.yaml) → `experiment:`, mirroring the
keyword arguments of the `fedference.experiments` study functions exactly.

## Installed CLI and evidence contracts

The package installs `fedference` with six commands:

```bash
# Give every write-producing example a fresh caller-owned root.
FEDFERENCE_RUN_ROOT="$(mktemp -d /tmp/active-fedference-runs.XXXXXX)"
FEDFERENCE_TRANSPORT_ROOT="$(mktemp -d /tmp/active-fedference-transport.XXXXXX)"

# Primary own-data path: configuration comes only from the labeled request.
uv run --locked fedference aggregate \
  --input examples/data/labeled_aggregation_request.json \
  --output-dir "$FEDFERENCE_RUN_ROOT/application" \
  --project-root .
uv run --locked fedference verify \
  "$FEDFERENCE_RUN_ROOT/application/receipt.json" \
  --require-nominal-solver

# Inspect source-bound experiments, profiles, datasets, and source revisions.
# Add --json for the complete machine-readable registry.
uv run --locked fedference list

# Correctness-only registered run; output must be explicit and outside output/
uv run --locked fedference run server-theory \
  --profile smoke --seed 0 \
  --output-dir "$FEDFERENCE_RUN_ROOT/server-theory" \
  --project-root .

# Verify config/report bytes, configuration hash, and completion status
uv run --locked fedference verify \
  "$FEDFERENCE_RUN_ROOT/server-theory/receipt.json"

# Hash-checked UCI benchmark smoke
uv run --locked fedference benchmark \
  --dataset-id uci-banknote --profile smoke --seed 42 \
  --cache-dir "$FEDFERENCE_RUN_ROOT/uci-cache" \
  --output-dir "$FEDFERENCE_RUN_ROOT/banknote"

# Produce a loopback replay plus the caller-retained numeric inputs.
uv run --locked python examples/03_federation_boundaries.py \
  --output-dir "$FEDFERENCE_TRANSPORT_ROOT/round"

# Replay uses the recorded configuration by default and reports finding codes.
uv run --locked fedference replay \
  --replay "$FEDFERENCE_TRANSPORT_ROOT/round/replay.json" \
  --beliefs "$FEDFERENCE_TRANSPORT_ROOT/round/beliefs.json" \
  --consensus "$FEDFERENCE_TRANSPORT_ROOT/round/consensus.json" \
  --require-nominal-solver

# Publication mode also matches the live commit, tree, and uv.lock.
# It passes only when the receipt was created from this same clean checkout.
uv run --locked fedference verify \
  "$FEDFERENCE_RUN_ROOT/server-theory/receipt.json" \
  --require-clean-git --project-root .
```

`AgentPosterior`, `LabeledAggregationRequest`, `LabeledAggregationResult`, and
`ApplicationReceipt` are versioned application contracts. An application
receipt binds exactly canonical `request.json` and `result.json`, runtime/source
provenance, configuration and request hashes, and complete solver health.
`status="completed"` means execution completed, not that solver health is
nominal or a scientific/downstream decision is valid.

`ExperimentSpec`, `DatasetSpec`, and `RunReceipt` are separately versioned
research contracts. The registry records source bundles, primary estimands, independent
units, falsifiers, no-claim outcomes, smallest effects, MCSE targets, budgets,
comparison families, and execution profiles. A schema-1.2 receipt records the
full Git commit plus clean/dirty/unavailable tree state, environment lock,
runtime provenance, configuration and dataset hashes, seeds, backend/device, fallbacks,
checkpoints, outputs, and status. `config.json` is itself receipt-bound so the
configuration hash can be recomputed. Registry entries declare intended
evidence; they do not imply that an open experiment has succeeded. Exact
schema-1.1 research receipts remain readable with unavailable runtime
provenance rather than invented metadata.
Tabular benchmark rows additionally record per-method fallback and
non-convergence counts at held-out-prediction grain plus the maximum iteration
count. The receipt summarizes affected dataset/seed/method cells; those counts
are diagnostics, not independent scientific units.

Application verification is artifact-only unless source equivalence is
explicitly requested with `--project-root /path/to/checkout`; human output says
whether source equivalence was verified, not requested, or unavailable.
Research publication verification compares the full commit, clean tree state,
and `uv.lock` digest when strict Git mode is requested.

Write-producing commands require `--output-dir`, reject non-empty targets, and
refuse the committed `output/` reviewer snapshot. Smoke and pilot results are
never manuscript evidence. An application receipt proves integrity, not
scientific validity or a downstream decision. Command grammar, output files, exit behavior, and
copy-pasteable workflows are documented in
[`src/fedference_cli/README.md`](src/fedference_cli/README.md) and
[`examples/README.md`](examples/README.md).

## Manuscript

The manuscript under [`manuscript/`](manuscript/) reports all nine studies. Every
number in the prose is a `{{TOKEN}}` hydrated from analysis outputs — never typed
by hand (ISC-30/35/36). The "robust beats naive" verdict in particular is emitted
by the statistics module after a paired Wilcoxon test deflated with
Benjamini–Hochberg FDR, then injected into the robustness results sections
(e.g. [`19_results_robustness.md`](manuscript/19_results_robustness.md)).

## Cover image

The manuscript cover is configured in `manuscript/config.yaml` and stored at
`manuscript/cover_image.png` (regenerated by `src/figures/graphical_abstract.py`).

## Reproducing outputs

Generated outputs under `output/` are committed as deterministic reviewer
snapshots, not hand-maintained source. After any manuscript, experiment, or
figure-producing change, regenerate the reports, variables, PDF, web package,
and release manifest with the documented scripts before committing the refreshed
artifacts. Analysis report payloads are validated against typed schemas at the
write boundary (`src/analysis/report_schemas.py` — `TypedDict` shapes plus
per-figure dependency contracts), so a malformed or renamed field fails when it
is written, not when a figure later consumes it. The schema-4 release bundle
carries an acyclic `publication-payload-v1` manifest and a provenance
fingerprint (a SHA-256 over the declared source, manuscript, documentation,
producer-script, dependency-lock, and claim-audit inputs). Template-owned
artifact, evidence, statistics, validation, rendered-provenance, and snapshot
controls are finalized and verified after the payload rather than hashed back
into it.
The manifest also records the pipeline profile and generator version;
`uv run --locked python scripts/build_release.py --verify` recomputes the fingerprint
and names changed inputs when a bundle is stale. The guarded CLI also requires
fresh publication-profile analysis, validation, hydration, and render receipts;
it establishes a local reviewer bundle, not clean-clone reproduction or
external release authorization. Unreleased builds omit
`generated_at` so rebuilding the same evidence tree is byte-identical. An
approved release may add canonical UTC metadata with `--timestamp` or
`SOURCE_DATE_EPOCH`. Pipeline-stage receipts follow the same rule: their
content hashes are reproducible by default, and time metadata is explicit.
Wheel and source-distribution builds use an exactly pinned setuptools backend
and a small PEP 517 wrapper that normalizes archive metadata whenever
`SOURCE_DATE_EPOCH` is supplied. The release ladder builds both formats twice
and rejects backend-version, member-order, owner, or checkout-mtime drift.

Local PDF page renders and other review scratch files belong under `.tmp/` and
are ignored. They are useful for visual QA but are not publication evidence or
release inputs.

The validated HTML manuscript is the canonical accessibility-enhanced reader
surface. The source-current combined manuscript PDF is generated through the
tagged LuaLaTeX/tagpdf path and is released only after `pdfinfo` reports
`Tagged: yes`, qpdf exposes a non-empty `/Lang` and `StructTreeRoot`, and the
source-bound language check passes. Some Poppler builds omit the language line
from `pdfinfo` even when `/Lang` is present. Slide PDFs remain a separate
Beamer surface. Tagged structure does not establish PDF/UA conformance; the
exact automated and manual boundary is in
[`docs/manuscript/accessibility.md`](docs/manuscript/accessibility.md).

## Evidence status and active research

Every load-bearing claim is graded on four evidence levels: **formal/executable
identities** (an algebraic statement with a corresponding invariant or
negative-control test), **source-conditional results** (inherited from a cited
source only under that source's assumptions), **conditional empirical findings**
(true of the declared seeded simulation and its estimand, not generalized beyond
it), and **scoped implementation facts** (true of the executed code path, not a
universal property of the method family). The three robustness axes keep their
distinct guarantees: the client-side FedGVI update ($\beta$-loss / rcce) carries
the cited bounded-influence result under the source theorem's matching
assumptions; `robust_aggregate` is a sharp server heuristic whose positive
formal property is the exact zero-robustness recovery limit, with a scoped
no-go proposition excluding a declared separable objective class but not every
broader construction; and
`variational_aggregate` carries an objective-backed raw effective-weight bound,
not estimator-level B-robustness. The MAJ-1 characterization report covers
influence, finite-breakdown, attack-mechanism, state-space, agent-count,
robustness, and weight-imbalance diagnostics, with explicit negative controls and
no-claim metadata. See [`docs/research/manuscript-claim-audit.md`](docs/research/manuscript-claim-audit.md)
and [`docs/todo/scholarship-and-phase-plan.md`](docs/todo/scholarship-and-phase-plan.md).

## Conventions

- Pure NumPy/SciPy core (the optional `bnn` extra adds the PyTorch point-mass,
  mean-field, and CPU/MPS runtime modules without changing default
  dependencies); typed;
  `from __future__ import annotations`; deterministic
  via `np.random.default_rng(seed)` (never global `np.random`).
- No `infrastructure.*` imports inside `src/fedference/` (layer contract).
- No mocks anywhere — tests are real seeded computations with explicit numeric
  expectations; the no-mocks policy is part of this repository's acceptance
  contract.
- $\ge 90\%$ combined line and branch coverage on `src/`; branch measurement
  is enabled in the coverage configuration, and the release-facing achieved
  coverage record is `output/data/test_coverage_receipt.json`. `coverage_project.json`
  is only an ignored local convenience export and never reviewer evidence.
- New modules: `src/fedference/<name>.py` with a sibling
  `tests/fedference/test_<name>.py`; module docstring cites the relevant Friston (2024)
  equation/figure or FedGVI mechanism.

## Documentation

| Entry | Purpose |
| --- | --- |
| [`docs/application-guide.md`](docs/application-guide.md) | Install, adapt, diagnose, and retain categorical aggregation results |
| [`docs/README.md`](docs/README.md) | Modular documentation hub (architecture, testing, pipeline, ops) |
| [`examples/README.md`](examples/README.md) | Runnable API, method, federation, replay, and CLI ladder |
| [`docs/development/modularity.md`](docs/development/modularity.md) | Software, orchestration, report, figure, and documentation extension contract |
| [`AGENTS.md`](AGENTS.md) | Slim technical reference and validation commands |
| [`ISA.md`](ISA.md) | Live acceptance-criteria contract |
| [`STANDALONE.md`](STANDALONE.md) | Confidentiality and standalone/fork notes |
| [`docs/reference/api-stability.md`](docs/reference/api-stability.md) | Public API/schema compatibility and deprecation policy |
| [`docs/research/README.md`](docs/research/README.md) | Research audits, evidence ladders, and claim boundaries |
| [`docs/security/active_fedference-threat-model.md`](docs/security/active_fedference-threat-model.md) | Federation trust boundaries, abuse paths, and security no-claim rules |
| [`docs/manuscript/accessibility.md`](docs/manuscript/accessibility.md) | HTML accessibility contract and tagged-PDF boundary |

Start with the [`application guide`](docs/application-guide.md), continue with
[`examples/README.md`](examples/README.md) for every supported boundary, then
[`docs/development/quickstart.md`](docs/development/quickstart.md) for the full
source-to-publication validation sequence.
