# Runnable example ladder

These examples are the shortest source-backed route from categorical
aggregation to the repository's local federation and evidence boundaries. Run
them from a checkout after installing the locked environment:

```bash
uv sync --locked
```

The default environment is sufficient for all five examples and the installed
CLI. It does not install the optional Torch path. For adapting the example
beliefs, choosing a method, and interpreting solver diagnostics, start with the
[`application guide`](../docs/application-guide.md).

Every program writes one deterministic JSON document to standard output. The
three programs that create files require an explicit, new or empty
`--output-dir`; they never choose or clean a destination on the caller's behalf.
The ignored `.tmp/` tree is suitable for local scratch runs.

## 1. Canonical aggregation API

```bash
uv run --locked python examples/01_minimal_aggregation.py
```

[`01_minimal_aggregation.py`](01_minimal_aggregation.py) uses the stable
`AggregationConfig` plus `aggregate_result` path, prints the rich diagnostics,
and demonstrates the fail-closed negative-probability boundary. The supplied
`base_weights` are raw coefficients in the log pool. They are not normalized
before aggregation: scaling every raw coefficient changes consensus sharpness.
`normalized_effective_weights` are the corresponding normalized diagnostics.

## 2. Compare the declared server rules

```bash
uv run --locked python examples/02_compare_aggregation_methods.py
```

[`02_compare_aggregation_methods.py`](02_compare_aggregation_methods.py) runs
all three methods over the same fixed beliefs and sets every solver control
explicitly. It also checks the central project-local recovery: both the robust
path and the variational path with `entropy_weight=1` are bit-identical to the
naive log-linear pool when `robustness=0`. Interpret the labels narrowly:

- `naive` is the project's categorical posterior-log-potential specialization
  under the documented shared-support, admitted-potential, and fixed-weight
  assumptions. It is not a reconstruction of the full Friston et al. protocol.
- `robust` is the project server heuristic. Seeded contamination experiments
  contain both wins and reversals; this example does not establish a universal
  robustness theorem.
- `variational` is the objective-backed conservative server rule. Its documented
  raw effective-weight bound is not an estimator-level bounded-influence claim.

The expected six-decimal consensus vectors are:

| Method | Consensus | Normalized effective weights |
| --- | --- | --- |
| `naive` | `[0.750000, 0.107143, 0.142857]` | `[0.333333, 0.333333, 0.333333]` |
| `robust` | `[0.818356, 0.147523, 0.034121]` | `[0.518309, 0.463017, 0.018674]` |
| `variational` | `[0.416799, 0.309848, 0.273353]` | `[0.384618, 0.421713, 0.193669]` |

## 3. Cross the supported local federation boundaries

```bash
FEDFERENCE_ROUND_ROOT="$(mktemp -d /tmp/active-fedference-round.XXXXXX)"
uv run --locked python examples/03_federation_boundaries.py \
  --output-dir "$FEDFERENCE_ROUND_ROOT/federation"
```

[`03_federation_boundaries.py`](03_federation_boundaries.py) sends one explicit
configuration through direct `share_round`, a one-machine process round, and a
loopback-socket round. All three global results are bit-identical to the
canonical `aggregate_result` reference. Direct sharing additionally reports
per-recipient beliefs that omit the recipient's own broadcast; these differ
from its global consensus by design. The socket frames use a fixed non-secret
example HMAC key to exercise authenticated framing; this is a mechanics check,
not credential-management guidance.

The example writes `beliefs.json`, `consensus.json`, and the digest-verified
`replay.json`. Validate the persisted transcript through the installed CLI:

```bash
uv run --locked fedference replay \
  --replay "$FEDFERENCE_ROUND_ROOT/federation/replay.json" \
  --beliefs "$FEDFERENCE_ROUND_ROOT/federation/beliefs.json" \
  --consensus "$FEDFERENCE_ROUND_ROOT/federation/consensus.json" \
  --json \
  --require-nominal-solver
```

The recorded configuration is used by default. The JSON result should report
`"integrity_valid": true`, `"nominal_solver": true`, and an empty findings
array. Explicit method/tuning options are assertions and must equal the
recorded configuration. Human failures name stable finding codes rather than
printing a bare `FAIL`.

These adapters provide genuine process and framed TCP transport evidence on one
machine. The socket helper is loopback-only. This is not cross-host deployment,
mTLS evidence, or a reason to infer that a smoke result is confirmatory science.
The process and socket convenience helpers currently expose uniform base
weights; weighted direct sharing remains a separate public-API path.

## 4. CLI receipt and output-isolation workflow

```bash
FEDFERENCE_RECEIPT_ROOT="$(mktemp -d /tmp/active-fedference-receipt.XXXXXX)"
uv run --locked python examples/04_cli_receipt_workflow.py \
  --output-dir "$FEDFERENCE_RECEIPT_ROOT/workflow" \
  --project-root .
```

[`04_cli_receipt_workflow.py`](04_cli_receipt_workflow.py) launches the public
CLI in real child processes. It runs the registered `server-theory` smoke
profile, verifies the content-bound receipt, confirms that a second run cannot
reuse the non-empty evidence directory, and confirms that the CLI refuses to
write beneath the committed `output/` reviewer tree. Its printed summary omits
the intentionally variable timestamps, run identifier, absolute paths, and Git
tree state while the receipt itself retains them.

The resulting caller-owned directory contains:

```text
server-theory-smoke/
├── config.json
├── receipt.json
└── report.json
```

The smoke run demonstrates an executable, receipt-bound implementation path.
It does not promote the registered pilot to confirmatory evidence. See
[`../docs/core/conceptual-foundations.md`](../docs/core/conceptual-foundations.md)
for the scientific boundaries and
[`../docs/security/active_fedference-threat-model.md`](../docs/security/active_fedference-threat-model.md)
for the federation trust model.

## 5. Apply labeled caller data and verify its receipt

```bash
FEDFERENCE_APPLICATION_ROOT="$(mktemp -d /tmp/active-fedference-application.XXXXXX)"
uv run --locked python examples/05_labeled_application.py \
  --input examples/data/labeled_aggregation_request.json \
  --output-dir "$FEDFERENCE_APPLICATION_ROOT/run" \
  --project-root .
```

[`05_labeled_application.py`](05_labeled_application.py) is the primary
own-data example. It parses the strict schema-1.0 labeled request, runs
`aggregate_labeled`, invokes `fedference aggregate` over the same input, checks
that both rich results are bit-identical, and verifies the application receipt
with nominal solver health required. The example request deliberately includes
one non-normalized mass row and one omitted base weight: canonical
`request.json` preserves the masses and expands the omitted weight, while
`result.json` exposes normalized rows separately.

The resulting directory contains exactly:

```text
run/
├── receipt.json
├── request.json
└── result.json
```

Expected six-decimal consensus: `[0.818512, 0.147846, 0.033642]`; expected
solver status: `nominal`. The application receipt binds the two artifact bytes,
runtime/source provenance, request/configuration hashes, and solver health. It
proves integrity at those declared levels, not calibration, scientific
validity, a winning state, or approval of a downstream decision.

## Regression tests

The ladder is executed as real subprocesses with no mocks:

```bash
uv run --locked --extra dev pytest tests/test_examples.py -q
```

The federation/replay check is marked `integration`, so it can also be selected
explicitly:

```bash
uv run --locked --extra dev pytest tests/test_examples.py -m integration -q
```

Use a fresh scratch root on each run. Refusing a non-empty destination is an
evidence-preservation feature; the examples never delete or overwrite a prior
run. Next, see the [`application guide`](../docs/application-guide.md) for
caller-owned data and [`../src/fedference_cli/README.md`](../src/fedference_cli/README.md)
for the complete receipt and replay command contract.
