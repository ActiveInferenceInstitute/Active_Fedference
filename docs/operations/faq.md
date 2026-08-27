# FAQ

## What is Active Fedference?

A discrete-categorical reimplementation of FedGVI connected to Friston et al.
(2024) federated belief-sharing. The central **project-local** identity is
`robust_aggregate(robustness=0) == log_linear_pool`. Under the documented
categorical posterior-log-potential assumptions, that pool specializes Friston
Eq. 7; it is not a reconstruction of the source's complete message-passing
protocol.

See [`../core/conceptual-foundations.md`](../core/conceptual-foundations.md).

## Can I use my own data?

Yes, when each participating client already produces a categorical posterior
over one shared ordered finite state space. The labeled API/CLI accepts one
finite, non-negative vector per identified agent, preserves the explicit state
order, and returns a categorical consensus plus solver diagnostics. It does not
train a raw-data model or turn logits, class labels, samples, or arbitrary
scores into calibrated posteriors.

Follow the [`application guide`](../application-guide.md) for the exact input
contract, a labeled-state recipe, base-weight semantics, and a result-acceptance
check.

## Which interface and aggregation method should I choose?

Start with `fedference aggregate` when you need labeled JSON and an application
receipt, or `aggregate_labeled` for pure Python. Use `aggregate_result` only when
your program already owns labels/evidence; use rich sharing, process, or socket
results only for those specific boundaries. The
[`interface table`](../application-guide.md#choose-the-narrowest-interface)
shows the imports and evidence returned by each choice.

`naive`, `robust`, and `variational` are distinct modeling choices, not a
quality ranking. The [`method table`](../application-guide.md#choose-an-aggregation-rule)
states their supported properties and no-claim boundaries. Construct one
explicit `AggregationConfig` across all compared boundaries; compatibility
defaults are not interchangeable configuration records.

## Where is the acceptance contract?

[`../../ISA.md`](../../ISA.md) — see its Criteria section for the current ISC range and checked/open counts (this drifts every iteration, so it is not snapshotted here); probes for verification.

## What is the current release status?

The public source and reviewer snapshot are maintained at
[`ActiveInferenceInstitute/Active_Fedference`](https://github.com/ActiveInferenceInstitute/Active_Fedference).
The current public release is
[v1.0.4](https://github.com/ActiveInferenceInstitute/Active_Fedference/releases/tag/v1.0.4)
with [Zenodo record 21972644](https://zenodo.org/records/21972644) and DOI
[`10.5281/zenodo.21972644`](https://doi.org/10.5281/zenodo.21972644). The
v1.0.3 and older records remain available as prior versions. This checkout is
the unreleased `1.1.0.dev0` application line; it has no version DOI/date, and
the v1.0.4 DOI must not be associated with its post-release code.

## Can I commit this project to the public template repo?

**No.** It is a standalone repository, not a subdirectory of the public
template repository. The v1.0.4 release is published from the standalone
target repository; future changes to this checkout require the explicit
target-repository review/push and Zenodo versioning workflow. Do not push it into the public template
repository. See
[`../../STANDALONE.md`](../../STANDALONE.md).

## Where does the math live?

Mathematical modules under `src/fedference/` use NumPy/SciPy and have no
`infrastructure.*` imports. Named evidence/data/checkpoint/replay adapters own
explicit I/O, and optional Torch modules remain behind extras.
[`../core/architecture.md`](../core/architecture.md).

## Where do manuscript numbers come from?

`{{TOKEN}}` hydration via `src/manuscript_variables.py` reading
`output/reports/*.json`. Those report payloads are schema-validated at the
write boundary (`src/analysis/report_schemas.py`), so a token can only cite a
field the producing study actually declared. Never type numbers in prose.
[`../manuscript/tokens-and-labels.md`](../manuscript/tokens-and-labels.md).

## What are the three robustness axes?

1. **Client-side (source-conditional):** `generalized_posterior` with bounded
   losses (`beta_loss`, `rcce`) — the FedGVI axis; the source theorem's
   bounded-influence result applies only under its matching assumptions.
2. **Server-side heuristic (sharp):** `robust_aggregate` reverse-KL
   reweighting — only the robustness→0 recovery limit is proven; everything
   sharper is a characterized implementation fact, not a theorem.
3. **Server-side objective-backed (conservative):** `variational_aggregate`
   forward-CE reweighting — exact block updates on `aggregation_free_energy`;
   proven raw effective-weight bound and empirical redescending response. The
   bound is an objective-level guarantee, **not** estimator-level
   B-robustness. Max-entropy consensus keeps it conservative. V1 adds an
   `entropy_weight` λ (default 1.0 bit-identical; smaller λ sharpens).

Do not conflate them in results prose: never grant one axis a guarantee only
another has. See
[`../core/conceptual-foundations.md`](../core/conceptual-foundations.md).

## Is true multi-machine federation in scope?

Not in the current evidence. The federation transport is single-host: queues,
OS processes, and loopback TCP with versioned configuration-bound envelopes,
optional HMAC, and digest-verified replay. The roadmap separates a future local
Docker/mTLS multi-node emulator from later receipts on physically distinct
hosts. A caller-shared `ReplayGuard` rejects round-id reuse within one process,
while `PersistentReplayGuard` extends that control across local process
restarts with caller-owned SQLite state. A shared multi-host replay domain and
both future deployment lanes remain open, and transport integrity would still
not establish privacy or Byzantine robustness. See the
[threat model](../security/active_fedference-threat-model.md).

## How do I inspect or run registered research?

```bash
FEDFERENCE_FAQ_ROOT="$(mktemp -d /tmp/active-fedference-faq.XXXXXX)"
uv run --locked fedference list
uv run --locked fedference run server-theory \
  --profile smoke --seed 0 \
  --output-dir "$FEDFERENCE_FAQ_ROOT/server-theory" \
  --project-root .
uv run --locked fedference verify \
  "$FEDFERENCE_FAQ_ROOT/server-theory/receipt.json"
```

Write-producing commands require an explicit empty directory outside committed
`output/`. Confirmatory profiles remain blocked until their pilot freezes the
effect, MCSE target, budget, comparison family, and configuration.

## How do I use the aggregation API?

Start with the copy-pasteable labeled-state recipe in
[`../application-guide.md`](../application-guide.md#labeled-python-api),
then run the numbered [`examples`](../../examples/README.md).

## How do I run all analysis studies?

```bash
uv run --locked python scripts/02_run_analysis.py
```

This produces validated reports and figures. It does not run tests, hydrate or
render the manuscript, validate web/PDF surfaces, or establish publication
readiness.

## How do I reproduce the source-current reviewer snapshot?

Use the exact producer order in the
[`two-pass rendering sequence`](../manuscript/rendering_pipeline.md). It binds
analysis, the full-suite validation receipt, final hydration, rendering, web
preparation, and freshness checks. The
[`contributor quickstart`](../development/quickstart.md) provides context and
stopping points.

## How do I run all acceptance gates?

Use [`../reference/verification-commands.md`](../reference/verification-commands.md).
Tests, source-only checks, generated-artifact freshness, rendered-surface
validation, package reproducibility, release-bundle verification, push, and
external publication are separate gates; no two-command shortcut establishes
all of them.

## How do I extend the hierarchy to 4 or more levels?

Define a `list[LayerSpec]` with the desired number of levels (topmost first,
L1 location last) and pass it to `build_nlevel_world`:

```python
from fedference.pomdp import LayerSpec, build_nlevel_world, nlevel_infer
import numpy as np

specs = [
    LayerSpec(n_states=2, labels=["low", "high"],
              default_prior=[0.5, 0.5],
              conditioned_priors=[[0.5, 0.5], [0.3, 0.7]]),
    LayerSpec(n_states=2, labels=["low_threat", "high_threat"],
              default_prior=[0.5, 0.5],
              conditioned_priors=[[0.5, 0.5], [0.2, 0.8]]),
    LayerSpec(n_states=2, labels=["quiet", "alert"],
              default_prior=[0.5, 0.5]),  # conditioned_priors=None -> uniform child prior
    LayerSpec(n_states=9, labels=list("012345678"), default_prior=[1/9]*9),
]
world = build_nlevel_world(specs)
A = np.asarray(world["L1"]["A"][0])
result = nlevel_infer(A, 4, world)
# result["q_levels"] is a list of per-level posteriors, all valid PMFs
```

`nlevel_infer` handles any depth automatically. No new aggregation logic is needed
— federation at each level reuses `log_linear_pool`.

## Why are there two documentation layers (root + `docs/`)?

- **Root** (`README.md`, `AGENTS.md`, `ISA.md`) — identity, contract, quick commands.
- **`docs/`** — modular operational rulebook (architecture, testing, pipeline, ops).

Start at [`../README.md`](../README.md).

## Forking or standalone use?

[`../../STANDALONE.md`](../../STANDALONE.md) — confidentiality, standalone core,
pipeline boundaries.

## See also

- [`../application-guide.md`](../application-guide.md)
- [`troubleshooting.md`](troubleshooting.md)
- [`../../TODO.md`](../../TODO.md) — forward backlog only
