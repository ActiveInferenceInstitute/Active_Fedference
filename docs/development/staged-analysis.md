# Bounded, resumable analysis

Publication compute uses a frozen external workspace. Repository leases cover
source preparation and validated imports; they do not cover the numerical
subprocesses. The cancelled monolithic 2026-10-30 run remains incomplete, and
its partial reports are preserved. They are not silently adopted as checkpoints.

## Prepare, execute, import

Use the existing pinned project environment, with both Python prefixes and all
TMP/cache paths on GitVault. Set `AF_REPO` to the integration checkout and
`WORKSPACE` to a **new** directory outside it, under the mounted vault's Local
storage. Preparation refuses an existing directory or a workspace inside the
project. Run preparation under the supported repository lease:

```bash
uv run --locked python scripts/analysis_stages.py prepare \
  --project-root "$AF_REPO" --workspace "$WORKSPACE" --profile publication
```

After that command exits and its lease releases, execute from the frozen
source using the same qualified environment. Choose one stage and its wall-time
budget explicitly. This example runs only the belief-sharing report:

```bash
cd "$WORKSPACE/source"
uv run --locked --no-sync python scripts/analysis_stages.py run \
  --workspace "$WORKSPACE" --stage belief_sharing \
  --max-seconds 60 --max-rss-mb 1024
uv run --locked --no-sync python scripts/analysis_stages.py status \
  --workspace "$WORKSPACE"
```

The catalog exposes each existing report independently. Calling `run` again
reuses its admitted completion without starting another producer. No command
implicitly schedules every report. Do not wrap numerical execution in
`vault with-repo`: the source and output namespace are already independent.
Report-only workers defer figure generators and their plotting dependencies
until the figure-assembly boundary. The optional Torch report therefore uses
its RSS allowance for its numerical backend instead of loading every plot
module first. The selected memory and wall-time limits still apply unchanged;
installed backend overhead can itself exhaust a budget and remains a failure.
An explicit selected-stage budget must account for the study's frozen design;
reaching a budget is an incomplete attempt, never permission to shrink seeds
or change the estimand.

When all report stages are complete, `run --stage figures` assembles the
report-derived artifacts. It admits the entire report set before rendering,
and does not recompute a missing report. Give this subprocess its own explicit
`--max-seconds` and memory budget. Finally, under a short destination repository
lease, import the checked bundle:

```bash
cd "$AF_REPO"
uv run --locked --no-sync python scripts/analysis_stages.py import \
  --project-root "$AF_REPO" --workspace "$WORKSPACE" --stage figures
```

An individual report may also be imported with its stage name. That action
does **not** issue an analysis-stage acceptance receipt. Complete publication
bundle import records the existing analysis receipt only after rechecking
source identities and every declared output. Hydration, the full test receipt,
rendering, science, native qualification and publication remain separate gates.
The ordinary `02_run_analysis.py` publication path refuses monolithic compute;
its explicit checkpoint assembly is restricted to the frozen workspace.
Direct library and temporary smoke workflows remain available for tests.

## Identity and atomic completion

`workspace.json` binds exact source/configuration/lock/manuscript bytes, the
profile and therefore all configured seeds and budgets, actual Python/platform,
interpreter binary/ABI, installed distribution versions, and numerical thread selectors. Preparation verifies
its pre/post input snapshot. Every execution, reuse and import revalidates it.
The installed-distribution roster comes from the selected interpreter's
`purelib` and `platlib` directories, with an identical directory counted once.
Caller-only metadata and duplicate `sys.path` entries do not describe a changed
installed environment. Actual installed name/version changes, interpreter/ABI
changes and numerical thread-selector changes still invalidate reuse. Source
bytes are bound separately; this does not permit an old workspace to adopt
changed code or an unqualified interpreter.
Changed code, config, seed or runtime requires a new workspace; it cannot reuse
old reports by filename. Completion is not an authenticity or scientific-validity
claim about caller-controlled files.

Each attempt owns a unique directory with its original stdout/stderr, start
record, terminal status, sampled peak RSS and elapsed time. The worker validates
the report through the existing schema/write boundary. The supervisor requires
exit zero, revalidates inputs and the report, binds report and terminal-receipt
hashes, and atomically publishes the completed directory. Readers recheck all
those predicates. A missing/corrupt marker, report, receipt, schema or inventory
fails rather than recomputing over the suspect evidence.

Imports reject destination outputs changed since preparation, while an already
identical destination is idempotent. Previous admitted bytes that are replaced
are retained by hash under the workspace's `prior-outputs/`. Source/WIP is never
reset or stashed. Imports replace individual artifacts atomically; a multi-file
import is not a filesystem-wide transaction. Interruption before the complete
publication receipt leaves an incomplete import that must be inspected/resumed.

## Cancellation and resource boundaries

```bash
uv run --locked --no-sync python scripts/analysis_stages.py cancel \
  --workspace "$WORKSPACE"
```

Cancellation targets the recorded attempt token. The supervisor sends normal
termination to its own newly created worker group, allows two seconds of cleanup,
and reaps it; only that group may receive a subsequent kill if needed. SIGINT or
SIGTERM delivered to the supervisor also follows its cleanup path. Existing
completed stages remain reusable, while the interrupted attempt and partial
files remain separate. Checkpoints are at **report-stage boundaries**: resuming
an interrupted report recomputes that report, not earlier completed stages.
There is no claim of per-trial or per-seed checkpoint recovery.

One workspace permits one writer. Its kernel lock descriptor passes directly
to the worker during process creation, before any active-PID record is written.
The lock remains held if the supervisor dies or that record is lost. The wall
budget is finite, positive, at most one hour, and required on each invocation.
Workers admit their inherited descriptor and arm their own wall alarm,
CPU-time limit and 16 MiB per-file limit before numerical imports or full
source/runtime admission. Python may defer the wall alarm while a native call runs; the CPU limit
still bounds CPU work. RSS is sampled by the live supervisor and compared with
the requested limit; this is not a hard allocation ceiling or a GPU-memory
bound, and sampling stops if that supervisor dies. Filesystem/process groups and
signals currently require POSIX (the exercised environments are Darwin/Linux).
These cooperative local-workspace controls do not qualify the unfinished native
custody or hostile-filesystem race guarantees.

An active worker surviving a crashed supervisor blocks a concurrent resume;
the recorded PID/birth is observed before a new producer starts. Unknown
liveness fails closed. This observation does not authorize signalling a prior
or unrelated process. `status` distinguishes its last attempt record from the
current observed worker state.

`stage_timings.json` from cached assembly measures assembly/cache retrieval,
not a repetition of scientific compute. Original per-stage wall time remains
in each validated completion receipt. Compare timing diagnostics only under
matching execution and instrumentation conditions.

## Verification

`tests/analysis/test_stage_process.py`, `test_stage_workspace.py` and
`test_stage_bundle.py` use actual processes and numerical
producers to check cancellation, timeout, resume, reuse, writer exclusion,
malformed/corrupt markers and reports, and changed-input/output refusal.
`test_stage_catalog.py` checks all producer call contracts and small real
payloads. No full publication rerun is implied by these smoke checks.

## GNN model consumption within a moving-world trial

Each moving-world trial admits its GNN source and native parameterization once. Both action selection and observation updates consume that admitted per-position observation map throughout the trial. Observation updates retain independent likelihood copies; they do not repeatedly reload and revalidate the same source for each agent and step. Admission still occurs for every new trial, so there is no process-global source cache or skipped model validation. Seeds, RNG draw order, trial counts, solver settings and stage deadlines are unchanged.


Parameter-recovery trials likewise share a fixed candidate likelihood grid
within one invocation. Each candidate GNN model is admitted once for that
invocation, and each true-acuity world is still constructed separately. The
next invocation rebuilds the grid; there is no global source cache. Candidate
values, observation RNG draws, likelihood arithmetic, trial counts, estimator
selection and stage deadlines stay unchanged.
