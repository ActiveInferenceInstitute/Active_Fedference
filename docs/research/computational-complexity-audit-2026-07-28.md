# Computational-complexity and scaling audit — 2026-07-28

This audit records the complexity claim boundary and the evidence required for
the first-release computational diagnostic. It is deliberately separate from
the machine-specific timing values in
[`output/reports/complexity_scaling.json`](../../output/reports/complexity_scaling.json),
which is the generated source of manuscript tokens and figure data.

## Scope and sources

The accounting was derived by reading the public call paths rather than by
assigning complexity from method names:

| Path | Source | Declared dimensions |
|---|---|---|
| log-linear, iterative robust, and variational pooling | `src/fedference/aggregation.py` | agents `N`, states `S`, solver iterations `I`, starts `B` |
| self-excluding belief sharing | `src/fedference/belief_sharing.py` | leave-one-out fan-out `N`, states `S`, robust iterations `I` |
| one-step state inference | `src/fedference/belief_updating.py` | modalities `M`, states `S` |
| local federation server round | `src/fedference/federation/server.py`, transport, aggregation | worker count `N`, states `S`, robust iterations `I` |

The dense categorical implementation is the object being characterized. The
orders are not claimed for a sparse, streamed, GPU, accelerated, or alternate
FedGVI implementation. The server accounting excludes queue wait and network
latency; incoming belief payload serialization is separately recognized as
linear in the number of agent-state values. Physical per-recipient broadcast
volume is a separate transport quantity: because each result includes an
`N`-length agent-weight vector, it is `Theta(N S + N^2)` when replicated across
`N` recipients.

## Accounting review

- `log_linear_pool`: one dense belief matrix and one weighted log-matrix
  reduction give `Theta(N S)` dominant time and `Theta(N S)` retained storage.
- `robust_aggregate`: each of `I` iterations evaluates agent-wise divergences
  and a dense reduction, giving `Theta(I N S)` time. The returned history makes
  the current storage `Theta(N S + I S)`.
- `variational_aggregate`: `B` starts each run the iterative objective, giving
  `Theta(B I N S)` time and `Theta(N S + I S + B S)` retained/peak storage.
- `share_round` with `exclude_self=True`: one global fusion plus one
  leave-one-out fusion for each agent gives `Theta(N^2 S)` naive time. Replacing
  the fusion with the iterative robust rule gives `Theta(I N^2 S)`. The current
  implementation retains only the per-call result while iterating over agents,
  so the peak dense storage remains `Theta(N S + I S)`, not `Theta(N^2 S)`.
- `infer_states`: independent modality messages add over the state vector, so
  time is `Theta(M S)` and working storage is `Theta(S)`.
- the federation server round includes worker ordering, belief deserialization,
  one result serialization, queue puts, and robust aggregation, summarized as
  `Theta(N log N + I N S)` local compute and `Theta(N S)` dense working storage.
  Queue/network wait is excluded; physical per-recipient broadcast volume is
  `Theta(N S + N^2)` when the serialized agent-weight vector is sent to every
  worker.

These are dominant interaction counts under the stated dense representation,
not hardware-independent FLOP totals. Constants, vectorization, BLAS choice,
cache behavior, allocation, validation, Python dispatch, and concurrency can
materially affect observed time.

## Measurement design

`src/fedference/experiments/complexity.py` calls the public implementations on
fixed `np.random.default_rng(seed)` inputs. It records the input SHA-256 digest,
the full repeated timing samples, median/minimum/maximum, the dimensionless
work proxy, and an ordinary-least-squares slope of log median time against log
problem size. The figure draws normalized asymptotic guides and min–max bars;
the bars are not confidence intervals and the fitted slopes are not inferential
statistics.

The publication grid and machine receipt are generated from
`manuscript/config.yaml` and written to the report. Smoke runs use a clipped
grid and clipped fixed dimensions so tests exercise the real paths without
rewriting a publication-scale snapshot. Publication outputs must be regenerated
through `scripts/02_run_analysis.py` before hydration, rendering, or release
verification. The configurable `max_iter` controls direct aggregation and
server timings; the robust self-excluding sharing measurement records the
public `share_round` path's default robust solver budget separately.

## Claim boundary

The supported claim is: the release exposes a reproducible, source-bound
complexity accounting and a machine-local scaling diagnostic for the declared
dense categorical implementation. It does not support a universal runtime
law, cross-machine ranking, network-performance claim, asymptotic theorem for
all FedGVI variants, or a statistical claim that an observed finite-grid slope
must equal its symbolic exponent. Sublinear finite-grid slopes are compatible
with the accounting when fixed validation/allocation/cache/interpreter costs
are material.

The corresponding manuscript and visual claim are governed by:

- [`manuscript/13_methods_statistics.md`](../../manuscript/13_methods_statistics.md),
  section `sec:methods-complexity`;
- [`manuscript/SYNTAX.md`](../../manuscript/SYNTAX.md), figure registry entry
  `fig:complexity-scaling`;
- [`docs/research/manuscript-claim-audit.md`](manuscript-claim-audit.md); and
- [`docs/research/visual-claim-audit.md`](visual-claim-audit.md).

The acceptance probes are the complexity unit tests, the typed report and
figure-dependency validators, the generated report/figure pair, the pipeline
freshness receipts, the full project gate, and `build_release.py --verify`.

## Current machine receipt (publication profile)

The regenerated report records macOS arm64, Python 3.13.11, NumPy 2.4.2,
10 logical CPUs, `time.perf_counter`, seed `20260728`, five measured repeats,
one warmup, and the publication grids in the report itself. The observed
log--log slopes were:

| Axis | Path | Expected order | Observed slope |
|---|---|---:|---:|
| agents | log-linear pool | 1 | 0.89 |
| agents | robust aggregate | 1 | 0.96 |
| agents | variational aggregate | 1 | 0.71 |
| agents | naive self-excluding sharing | 2 | 1.77 |
| agents | robust self-excluding sharing | 2 | 1.97 |
| states | log-linear pool | 1 | 0.42 |
| states | robust aggregate | 1 | 0.37 |
| states | variational aggregate | 1 | 0.44 |
| modalities | one-step state inference | 1 | 0.66 |

The state-axis slopes are intentionally reported rather than hidden: the finite
grid is implementation timing evidence and fixed overhead is visible at these
sizes. The values above are a receipt for this machine-local run, not a new
asymptotic theorem; the JSON report remains authoritative for exact samples,
digests, work proxies, and environment metadata.
