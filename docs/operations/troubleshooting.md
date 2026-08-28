# Troubleshooting

Symptom-driven fixes for Active Fedference.

## `import fedference` fails in application code

**Cause:** The source-layout package is not installed in the interpreter that
launched the application. Pytest's path fixture is a test-only convenience and
is not an application installation method.

**Fix from a source checkout:**

```bash
uv sync --locked
uv run --locked python -c "from fedference import aggregate_result; print(aggregate_result([[.7, .3], [.6, .4]]).consensus)"
```

For a non-editable consumer environment, follow the isolated wheel recipe in
the [`application guide`](../application-guide.md#install-a-wheel-or-source-distribution). Do not
silently repair this by setting a global `PYTHONPATH`.

If `uv` says an active `VIRTUAL_ENV` does not match the project `.venv`, it is
warning that it will use the locked project environment. Deactivate the
unrelated environment if that was not intentional; do not add `--active`
without deliberately choosing to mutate that other environment.

## A posterior matrix or base-weight vector is rejected

**Cause:** At least one posterior is empty, scalar, multidimensional,
non-finite, negative, all-zero, or a different length; or `base_weights` is not
one-dimensional, does not match the number of agents, contains invalid values,
or has no positive entry. Version 1.1 no longer flattens row/column matrices or
higher-rank tensors supplied where one categorical vector is required.

**Fix:** Confirm that every agent emits a genuinely one-dimensional, finite non-negative vector over the
same ordered state labels. Rows may be unnormalized positive masses because the
boundary normalizes them. Exact zeros are supported and floored internally for
log-domain operations; do not pre-emptively clip negative or non-finite data.
Validate the upstream producer instead.

A shared row length does not prove shared semantics. If two agents assign the
same columns to different label orders, the library cannot detect the mismatch
and the numeric result is meaningless. Keep the ordered labels beside the
matrix as shown in the
[`canonical application recipe`](../application-guide.md#labeled-python-api).

## CLI or example says the output directory must be empty

**Cause:** Write-producing commands preserve existing evidence and never clear
or mix a prior run on the caller's behalf.

**Fix:** Choose a new directory for each run:

```bash
FEDFERENCE_RUN_ROOT="$(mktemp -d /tmp/active-fedference-run.XXXXXX)"
uv run --locked fedference run server-theory \
  --profile smoke --seed 0 \
  --output-dir "$FEDFERENCE_RUN_ROOT/server-theory" \
  --project-root .
```

Archive or inspect the old directory separately. Do not delete retained
evidence merely to make a rerun pass.

## CLI refuses a path beneath `output/`

**Cause:** Project-local `output/` is the committed, producer-owned reviewer
snapshot. Ad hoc CLI writes there would mix caller runs with source-bound
artifacts.

**Fix:** Use a new caller-owned directory under `.tmp/`, `/tmp`, or another
application data root. Passing a deeper path beneath `output/` does not bypass
the guard.

## Aggregation configuration conflicts with method or tuning arguments

**Cause:** A typed `AggregationConfig` and compatibility arguments such as
`method` or `robustness` were supplied to the same adapter. The API refuses to
guess precedence.

**Fix:** Construct one explicit configuration and pass only `config=...` through
direct aggregation, sharing, process, and socket calls. Preserve its complete
dictionary or fingerprint for replay. See
[`Choose an aggregation rule`](../application-guide.md#choose-an-aggregation-rule).

## Spawned federation recurses, hangs, or fails during bootstrap

**Cause:** `run_multiprocess_round` always uses spawned processes. An unguarded
module body, `python -c`, notebook cell, or other non-importable entry point
cannot be safely re-imported by each child.

**Fix:** Put the call in a real Python file and protect it with:

```python
def main() -> None:
    # Construct beliefs/configuration and call run_multiprocess_round here.
    ...


if __name__ == "__main__":
    main()
```

Use [`examples/03_federation_boundaries.py`](../../examples/03_federation_boundaries.py)
as the executable reference. Increase `startup_timeout` only when measured
process-import latency exceeds the default; do not use an unbounded timeout to
hide a bootstrap error.

## Socket federation rejects the host or round identifier

**Cause:** The convenience transport accepts loopback hosts only, requires a
non-empty round identifier, and can reject a reused identifier when a replay
guard owns the replay domain.

**Fix:** Use `127.0.0.1`, `localhost`, or another validated loopback form and a
unique application round ID. `ReplayGuard` remembers IDs within one process;
`PersistentReplayGuard` uses caller-owned SQLite state across local restarts.
Neither is a shared multi-host replay domain.

## `fedference replay` reports an integrity or solver finding

**Cause:** The replay relationship is invalid: a belief, consensus, digest,
worker order, protocol field, loopback host, or aggregation configuration does
not match; or integrity is valid but recomputed solver health is non-nominal.
A well-formed mismatch exits 1 with a stable finding code/message. Malformed
JSON or an invalid array is a command error and exits 2 with a parser diagnostic.

**Fix:** Use the caller-retained beliefs and consensus from the same round, then
inspect the recorded aggregation event:

```bash
FEDFERENCE_REPLAY_PATH=/path/to/replay.json
uv run --locked python - "$FEDFERENCE_REPLAY_PATH" <<'PY'
import json
import sys
from pathlib import Path

events = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
event = next(item for item in events if item.get("event") == "aggregate")
print(json.dumps(event["aggregation_config"], indent=2, sort_keys=True))
PY
```

The command reads that recorded configuration by default. Any explicit
`method`, `robustness`, `entropy_weight`, `max_iter`, `tol`, or multistart flag
is an equality assertion, not a replacement. Add `--json` for categorized
machine-readable findings and `--require-nominal-solver` when integrity-valid
nonconvergence/fallback should also fail the command. A numerically equivalent
consensus does not compensate for a different configuration fingerprint.

## Receipt verification fails after files move or outside the checkout

**Cause:** Ordinary verification cannot find a bound artifact relative to the
receipt, or strict verification cannot resolve the source Git state and
`uv.lock` from the current directory.

**Fix:** Keep all artifacts bound by the receipt together. An application run
contains `request.json`, `result.json`, and `receipt.json`; a research run
contains `config.json`, `report.json`, and `receipt.json`. The receipt's own
directory is the default artifact root. If the receipt is stored separately
from an otherwise intact artifact directory, select that directory explicitly:

```bash
fedference verify /path/to/receipt.json --root /path/to/run
```

For strict source-equivalence verification from another directory, identify
the source checkout separately:

```bash
uv run --project /path/to/Active_Fedference --locked \
  fedference verify /path/to/run/receipt.json \
  --root /path/to/run \
  --require-clean-git \
  --project-root /path/to/Active_Fedference
```

`--root` resolves hash-bound artifacts; `--project-root` resolves source
provenance. Neither substitutes for the other. An ordinary receipt may
honestly record a dirty development tree. Strict verification requires the
exact recorded commit, clean tree state, and lock digest; it is a separate
source-equivalence gate.

## Literal `{{TOKEN}}` in the rendered PDF

**Cause:** Variable hydration not run, or token missing from `generate_variables()`.

**Fix:**

```bash
uv run --locked python scripts/02_run_analysis.py
# This proves token resolution only; it is not the final manuscript pass.
uv run --locked python scripts/z_generate_manuscript_variables.py --provisional-validation
cat output/data/manuscript_variables.json | head
```

For a source-current rendered surface, continue with the
[two-pass rendering sequence](../manuscript/rendering_pipeline.md): it runs
the provisional template pass, obtains the full-suite receipt, hydrates final
tokens, and performs a separate final template pass with explicit hydration
skipped.

Add missing keys in the `src/manuscript_vars/` package (`generate.py`,
`tokens.py`, or `loaders.py`); `src/manuscript_variables.py` is only the
re-export shim.

## Missing JSON reports during variable hydration

**Cause:** Analysis stage skipped; `output/reports/*.json` absent.

**Fix:**

```bash
uv run --locked python scripts/02_run_analysis.py
uv run --locked python scripts/z_generate_manuscript_variables.py --provisional-validation
```

Note: `HIER_*` and `NLEVEL3_*` tokens are strict loads from
`output/reports/hierarchical_world.json` and `output/reports/nlevel3_world.json`
respectively (see [`../manuscript/rendering_pipeline.md`](../manuscript/rendering_pipeline.md)).
A `FileNotFoundError` from `_hierarchical_variables()` or `_nlevel3_variables()`
means one of those two report files is missing — fix by re-running
`scripts/02_run_analysis.py`. Final non-draft hydration additionally
requires the fresh coverage receipt in the linked two-pass sequence.

## Config changed but figures/PDF unchanged

**Cause:** Render ran without re-analysis.

**Fix:**

Follow the [source-current two-pass rendering sequence](../manuscript/rendering_pipeline.md)
from analysis onward. It deliberately regenerates provisional inputs, runs the
full-suite receipt, then regenerates and renders receipt-backed final inputs;
do not try to reuse a prior final hydration after config changes.

## Pipeline freshness validation fails

**Cause:** A content-hashed stage receipt is missing, an upstream report changed
after hydration, or a rendered input/output no longer matches the recorded
boundary. Receipt validation is intentionally fail-closed; file modification
times are not treated as sufficient provenance.

**Fix:** Re-run the [source-current two-pass rendering sequence](../manuscript/rendering_pipeline.md)
in its exact order: analysis → provisional hydration → first template
stages 03–05 with `--skip-manuscript-hydration` → coverage receipt → final
hydration → second template stages 03–05 with the same skip flag → web
preparation and validation → render receipt → freshness validation. The receipt
must follow web preparation because that producer writes `output/web/`.

Do not hand-edit `output/data/pipeline_provenance.json`; it is generated by
`publication.pipeline_freshness` and the receipt does not make an external
template renderer itself content-addressable. Schema 3 replaces an older
receipt only when analysis is recorded, after which hydration and render must
be regenerated in order.

## Clean-checkout probe fails

**Cause:** The probe found an uncommitted change, an untracked required file, a
missing required tracked path, or an import failure. A development worktree is
allowed to be dirty, but it cannot be described as fresh-checkout evidence.

**Fix:** Inspect the listed paths in a separate clean clone and rerun:

```bash
uv run --locked python scripts/validate_clean_checkout.py
```

## Unresolved figure or cross-reference marker

**Cause:** Label mismatch between prose and figure anchor.

**Fix:**

1. Check label in [`../../manuscript/SYNTAX.md`](../../manuscript/SYNTAX.md).
2. Ensure `![Caption](../output/figures/name.png){#fig:label}` matches `[@fig:label]`.
3. Re-run analysis + variables + render.

## `ReportSchemaError` during analysis

**Cause:** A study payload drifted from its declared shape, or a figure
generator consumed a field its dependency contract does not declare. Reports
are validated at the single write boundary in
`src/analysis/report_schemas.py`.

**Fix:** Update the producer and the schema **together** in
`src/analysis/report_schemas.py` (and the figure dependency contract if a
generator's consumed fields changed). Never hand-edit `output/reports/*.json`
to satisfy the validator.

## `build_release.py --verify` fails

**Cause:** Either a byte-digest mismatch (an artifact was changed, added, or
removed after the bundle was built — including hand edits) or a provenance
fingerprint mismatch (a declared source, manuscript, documentation,
producer-script, dependency-lock, or claim-audit input changed since the
bundle was built, so the bundle is stale even though every listed digest
matches), or the required publication-profile analysis/test/hydration/render
receipt chain is stale. The diagnostic names changed inputs when available.

**Fix:** Regenerate outputs through the pipeline, then rebuild the bundle:

```bash
uv run --locked python scripts/build_release.py
uv run --locked python scripts/build_release.py --verify
```

If a no-op rebuild changes only release time metadata, the producer is stale:
schema 3 omits `generated_at` in unreleased builds. Do not hand-edit the
manifest. Re-run with the current producer and reserve `--timestamp` or
`SOURCE_DATE_EPOCH` for an approved release.

## Slide decks show "??" for cross-references

**Cause:** Cross-deck `\ref{...}` numbers are substituted from the combined
manuscript's aux map (`output/pdf/_combined_manuscript.aux`), which is a
transient producer-workspace artifact of the most recent combined build. It is
used by the dependent slide pass, then excluded from the public Git tree and
release bundle. On a first render with no aux yet, numeric lookup is
unavailable, but section references are rendered as visible `sec:*` labels
rather than unresolved "??" markers.

**Fix:** Re-run the template render after the combined build if numeric
cross-deck labels are required; the next slide pass resolves them to the
combined PDF's printed numbers. See
[`../manuscript/rendering_pipeline.md`](../manuscript/rendering_pipeline.md).

## PDF rendering fails: `mmdc` / Could not find Chrome

**Fix:**

```bash
npx --yes puppeteer browsers install chrome-headless-shell
uv run --locked python scripts/00_preflight.py
```

## LaTeX / citation errors

**Fix:**

```bash
grep -nE "Citation|undefined|Error" \
  output/pdf/*.log
```

The current renderer leaves `_combined_manuscript.log` and
`_latex_stdout.log` under `output/pdf/`. The release-facing rendered-surface
validator checks every local `*.log`; an obsolete log from an older renderer
(for example `_xelatex_stdout.log`) must therefore be removed by the rendering
producer before validation. Retain the validated logs in the external
candidate-specific verification namespace and exclude them from Git: renderer
logs embed machine-local paths and wall-clock text even when the rendered PDF
and source-bound receipts use a deterministic epoch. The public artifact tree
also excludes transient LaTeX/Beamer sidecars (`aux`, `bbl`, `blg`, `lof`,
`lot`, `nav`, `out`, `snm`, `toc`, and `vrb`).

Validate `manuscript/references.bib` and pandoc cite keys in manuscript sections.

## Coverage below 90%

**Fix:**

```bash
uv run --locked pytest tests/ \
  --cov=src \
  --cov-report=term-missing
```

Add tests in `tests/fedference/` for uncovered branches.

## Analysis script Python error

**Fix:**

```bash
uv run --locked python scripts/02_run_analysis.py 2>&1 | tee /tmp/analysis.log
uv run --locked python -c "import yaml; yaml.safe_load(open('manuscript/config.yaml'))"
```

## Tests pass locally but the template pipeline fails

**Cause:** Template-side project resolution or linked-project state.

**Fix:** Use explicit paths as in [`../reference/verification-commands.md`](../reference/verification-commands.md).

## See also

- [`faq.md`](faq.md)
- [`../application-guide.md`](../application-guide.md)
- [`../development/quickstart.md`](../development/quickstart.md)
