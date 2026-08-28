# `fedference_cli` — installed orchestration boundary

`fedference_cli` is the installed `fedference` command-line boundary for
labeled application aggregation and source-bound research runs. It is
intentionally small at the package surface:
the public `main()` entry point remains stable while the implementation is
split by responsibility. The package does not own research mathematics, report
schemas, or publication rendering.

Application users should begin with the project-wide
[`application guide`](../../docs/application-guide.md); this page is the
detailed command, receipt, and extension contract.

## Module map

| Module | Responsibility | Must not become responsible for |
| --- | --- | --- |
| `__init__.py` | Stable compatibility facade exporting `main` and the historical `_report_fallbacks` helper | Argument parsing, experiment dispatch, filesystem writes, or domain imports beyond the facade imports |
| `_parser.py` | `argparse` grammar, command selection, process-facing error mapping, and the timestamp supplied to a run | Experiment algorithms, receipt construction, or direct file writes |
| `_commands.py` | Dispatch for labeled `aggregate`, registry `run`/`benchmark`, receipt `verify`, and socket `replay` | Parser definition, duplicated aggregation math, or publication-snapshot writes |
| `_support.py` | Atomic JSON writes, project-root and output isolation, seed/control validation, registry summaries, and receipt construction | Selecting a study, choosing scientific parameters, or parsing command-line arguments |
| `__main__.py` | `python -m fedference_cli` process shim | Alternative command semantics |

The direction is deliberately one-way:

```text
fedference_cli.__main__
        -> fedference_cli.__init__
        -> fedference_cli._parser
             -> fedference_cli._commands
                  -> fedference domain/evidence/registry boundaries
             -> fedference_cli._support
```

The installed command is therefore useful for operators and CI while the
underlying functions remain importable for tests and library callers. Scripts
may invoke the CLI as a subprocess, but scripts must not reimplement these
checks or import private CLI helpers to perform research work.

## Stable public surface

```python
from fedference_cli import main

status = main(["list", "--json"])
```

The legacy import below remains available for compatibility with existing
tests and review tooling:

```python
from fedference_cli import _report_fallbacks
```

New integrations should prefer the typed APIs in `fedference` and its evidence
and registry modules. The CLI is an adapter for explicit process boundaries,
not the package's domain API.

## Which command to use

| Command | Use it for | Writes |
| --- | --- | --- |
| `fedference aggregate` | Validate and aggregate a strict schema-1.0 labeled request. Configuration comes only from the request JSON. | Canonical `request.json`, `result.json`, and application `receipt.json` in a new or empty `--output-dir` |
| `fedference list [--json]` | Inspect source, dataset, experiment, profile, runner, and no-claim declarations. Text output distinguishes executable and declared-only experiments. | Nothing |
| `fedference run EXPERIMENT` | Run one registry entry that has an executable adapter. Use `--cache-dir` only for `external-tabular`; `--device` only affects `fedgvi-bnn`. | `config.json`, `report.json`, and `receipt.json` in a new or empty `--output-dir` |
| `fedference benchmark` | Run the hash-checked registered UCI pack with explicit client and contamination controls. This boundary needs a caller-owned cache and may acquire external data. | The same three-file evidence directory |
| `fedference verify RECEIPT` | Auto-detect application/research receipts and verify their artifacts. `--project-root` requests source equivalence; `--require-nominal-solver` applies only to application receipts. | Nothing |
| `fedference replay` | Recompute a digest-only loopback-socket replay from caller-supplied beliefs/consensus and its recorded configuration; optionally emit structured JSON or require nominal health. | Nothing |

The registry can describe work that is not executable. `list` preserves those
declarations, while `run --help` advertises only entries with a current runner.
An `active`, `planned`, or executable label is not evidence that a study
succeeded.

## Labeled application receipt

The own-data path validates request, provenance requirements, and destination
safety before creating output:

```bash
FEDFERENCE_APPLICATION_ROOT="$(mktemp -d /tmp/active-fedference-app.XXXXXX)"
uv run --locked fedference aggregate \
  --input examples/data/labeled_aggregation_request.json \
  --output-dir "$FEDFERENCE_APPLICATION_ROOT/run" \
  --project-root .
uv run --locked fedference verify \
  "$FEDFERENCE_APPLICATION_ROOT/run/receipt.json" \
  --require-nominal-solver
```

Nominal aggregation exits 0. Non-nominal solver health exits 1 after retaining
all three files. Schema, usage, required-provenance, or unsafe-path failure exits
2 without creating the destination. The sorted success object has this shape:

```json
{"receipt": "/absolute/run/receipt.json", "request": "/absolute/run/request.json", "result": "/absolute/run/result.json", "solver_status": "nominal"}
```

Application receipt `status="completed"` records execution completion only.
The receipt verifies declared input/output/provenance integrity; it does not
prove nominal health unless that is separately required, and never proves
scientific validity or authorizes a downstream decision.

## Local receipt smoke

Prerequisite: install the locked default environment from the repository root
with `uv sync --locked`. Then run a deterministic, local server-theory smoke:

```bash
FEDFERENCE_CLI_ROOT="$(mktemp -d /tmp/active-fedference-cli.XXXXXX)"

uv run --locked fedference run server-theory \
  --profile smoke \
  --seed 0 \
  --output-dir "$FEDFERENCE_CLI_ROOT/server-theory" \
  --project-root .

uv run --locked fedference verify \
  "$FEDFERENCE_CLI_ROOT/server-theory/receipt.json"
```

The run prints machine-readable paths whose absolute prefix varies:

```text
{"report": ".../report.json", "receipt": ".../receipt.json"}
PASS: server-theory-smoke-<run-id> (git_tree_state=<clean-or-dirty>)
```

`config.json` records the selected experiment, profile, seeds, runner, and
registry fingerprint. `report.json` contains the validated lane-specific
result and the complete experiment declaration. `receipt.json` binds the
config/report bytes, configuration hash, source and tree state, lock digest,
device/backend, fallbacks, checkpoints, seeds, and timestamps. The receipt does
not hash itself. Its run identifier and timestamps intentionally vary between
executions, while the seeded report remains deterministic for a fixed source
and configuration.

Ordinary verification accepts an honestly recorded dirty development tree and
checks the bound artifacts. Strict verification is a separate source-equivalence
gate:

```bash
uv run --locked fedference verify \
  "$FEDFERENCE_CLI_ROOT/server-theory/receipt.json" \
  --require-clean-git \
  --project-root .
```

Smoke and pilot receipts demonstrate an implementation path; they are not
confirmatory evidence or publication authority.

## Socket replay verification

`run_socket_round(..., replay_path=...)` is the Python producer for the replay
consumed by this command. The replay intentionally stores digests and protocol
metadata rather than raw beliefs or consensus, so those numeric arrays must be
provided separately. The recorded configuration is read and validated by
default. Any explicit method/tuning flag is an assertion and must match the
recorded value:

```bash
uv run --locked fedference replay \
  --replay /path/to/replay.json \
  --beliefs /path/to/beliefs.json \
  --consensus /path/to/consensus.json \
  --json \
  --require-nominal-solver
```

Structured output includes `integrity_valid`, `solver_status`,
`nominal_solver`, recorded configuration/fingerprint, and stable categorized
findings. Integrity-valid deterministic nonconvergence may still carry a solver
finding. Exit 0 means integrity-valid and, when requested, nominal health. Exit
1 means an integrity mismatch or requested nominal-health failure. Malformed
command/input exits 2. Human failures print finding codes/messages instead of a
bare `FAIL`. The digest-only replay file is not a signature over its own bytes
or independent proof of an ephemeral port. Bind it into a higher-level receipt
when whole-file provenance is required.

## Output safety and common failures

- `aggregate`, `run`, and `benchmark` validate before creating their destination, reject a
  non-empty target, and reject every path that resolves beneath committed
  `output/`.
- Seeds must be unique non-negative integers. Repeat `--seed` only for runners
  that accept multiple seeds.
- Confirmatory profiles fail closed until their registry contract is frozen.
- `external-tabular` requires `--cache-dir`; the dedicated `benchmark` command
  exposes its dataset and contamination controls.
- A caller uses the exact configuration recorded by a socket replay. Supplying
  a numerically equivalent but differently fingerprinted solver configuration
  is a verification failure.
- `verify --require-nominal-solver` on a research receipt is misuse and exits 2;
  application artifact integrity and source equivalence are reported as
  separate levels.

## Adding a CLI workflow

1. Add or update the registry declaration in
   [`fedference/research_registry.py`](../fedference/research_registry.py),
   including the estimand, independent unit, falsifier, budget, MCSE target,
   comparison family, profile, and no-claim boundary.
2. Implement reusable execution in a domain or experiment module under
   [`fedference/`](../fedference/); keep it deterministic and independently
   testable.
3. Add only the dispatch branch and configuration serialization needed in
   [`_commands.py`](_commands.py). Keep validation and receipt mechanics in
   [`_support.py`](_support.py).
4. Add parser arguments in [`_parser.py`](_parser.py) only when the workflow
   needs a new caller-controlled value. Validate before creating the output
   directory.
5. Extend the zero-mock CLI tests in
   [`tests/test_fedference_cli.py`](../../tests/test_fedference_cli.py) and the
   relevant domain tests. A smoke run is implementation evidence, not
   confirmatory scientific evidence.
6. Update the user-facing command table in
   [`scripts/README.md`](../../scripts/README.md) only if a script changes;
   update the research and verification docs when the contract changes.

Every completed run must pass the typed report validator and write a
configuration-bound receipt. Run output belongs in an explicit, empty,
caller-owned directory outside the committed `output/` reviewer snapshot.

## Optional dependencies and safety

The default import path remains NumPy/SciPy-only. Torch/BNN support is selected
by the `bnn` extra and reports any device fallback explicitly. A CLI command
must not make optional dependencies part of the default import graph, silently
replace a failed algorithmic path, or claim CUDA, multi-host federation, or
confirmatory evidence from a CPU smoke run.

See the project-wide [application guide](../../docs/application-guide.md),
[modularity guide](../../docs/development/modularity.md),
the [API stability policy](../../docs/reference/api-stability.md), and the
[CLI contract tests](../../tests/test_fedference_cli.py).
