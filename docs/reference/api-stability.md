# API and schema stability

Active Fedference uses additive evolution throughout the current release line.

## Stable public surface

- Top-level aggregation functions keep their current return behavior:
  `log_linear_pool` returns an array; `robust_aggregate` and
  `variational_aggregate` return `AggregationResult`; legacy `aggregate`
  returns the consensus array.
- `AggregationConfig` plus `aggregate_result` is the canonical rich interface.
  `AggregationResult.as_dict()` and `solver_status` add finite JSON diagnostics
  without changing existing function return types. Status is exactly
  `nominal`, `converged_with_fallback`, `not_converged`, or
  `not_converged_with_fallback` according to convergence and fallback presence.
- `AgentPosterior`, `LabeledAggregationRequest`,
  `LabeledAggregationResult`, and `aggregate_labeled` are the strict labeled
  application surface. Request/result schemas are version `1.0`; semantic
  hashes include expanded default weights but not JSON formatting. The result
  retains labels and normalized rows without inferring a decision or scientific
  interpretation.
- `share_round_result`, `FederationServer.run_round_result`,
  `run_multiprocess_round_result`, and `run_socket_round_result` propagate the
  canonical `AggregationResult`. Existing `share_round`, consensus-returning
  process helpers, and the legacy socket dictionary retain their exact return
  types. Frozen rich results copy stored arrays and mark them read-only.
- `inspect_socket_replay` returns `ReplayValidationResult` with stable
  integrity/solver `ReplayFinding` values. `validate_socket_replay` remains the
  integrity-only Boolean adapter. Recorded configuration is the default;
  explicit tuning values assert equality rather than replacing it.
- `RuntimeProvenance`, `SourceProvenance`, `runtime_provenance`, and
  `collect_source_provenance` expose installed runtime/source identity without
  retaining machine-local installation URLs or inventing source revisions.
- `CalibrationEpisode.local_posteriors` is the canonical calibration input;
  `beliefs` remains a warned compatibility alias. `CalibrationResult` binds its
  selected configuration, candidate declarations/scores, episode IDs/content
  hashes/world families, and primary estimand in a self-verifying canonical
  digest. A malformed or tampered frozen selection fails closed.
- Canonical aggregation vocabulary is `local_posteriors`, `base_weights`,
  `raw_effective_weights`, and `normalized_effective_weights`; belief/weight
  spellings remain warned compatibility adapters. `shared_posteriors`,
  `global_posterior`, `site_factor`, and `loss_by_state` are the corresponding
  canonical names in sharing, cavity, and generalized-Bayes APIs.
- `fedference.federation.ReplayGuard` remains the in-memory compatibility
  guard. `PersistentReplayGuard` is the additive restart-durable local option;
  its SQLite `user_version` is validated and unknown versions fail closed.
  Neither class is a multi-host security claim.
- `ExperimentSpec`, `DatasetSpec`, and `RunReceipt` are versioned research
  evidence contracts. Their serialized `schema_version` is independent of the
  package version. Receipt schema 1.2 requires runtime provenance; exact legacy
  schema-1.1 fields remain readable and round-trip with unavailable provenance
  represented explicitly.
- `ApplicationReceipt` schema 1.0 is distinct from a research receipt. It binds
  exactly canonical `request.json` and `result.json`, runtime/source provenance,
  semantic/configuration hashes, timestamps, and solver health. Completed
  execution does not imply nominal health, scientific validity, or a decision.
- The installed `fedference` command exposes `aggregate`, `list`, `run`,
  `benchmark`, `verify`, and `replay`. `verify --require-nominal-solver` is
  application-only. `replay --json` exposes structured findings and
  `--require-nominal-solver` can strengthen its numerical-health exit policy.
- The installed CLI's stable public facade is `fedference_cli.main`; the
  internal `_parser`, `_commands`, and `_support` modules are responsibility
  boundaries rather than public API. Existing imports of
  `fedference_cli._report_fallbacks` remain available for compatibility, but
  new callers should use typed domain/evidence APIs instead of private helpers.
- Torch is optional through the `bnn` extra. The default `fedference` import
  remains NumPy/SciPy-only. `fedference.__version__` exactly reflects the
  installed `active_fedference` distribution, and `py.typed` declares inline
  typing to installed consumers.
- `publication.identifiers.manuscript_pdf_filename` is the canonical
  version-and-DOI-bound top-level manuscript filename helper; clean-checkout
  and release tooling use it instead of embedding a historical release name.
- Publication identity has two valid states: an `X.Y.Z.devN` development build
  has no DOI or release date, while a final `X.Y.Z` build has both and requires
  `paper.date` to exactly match `publication.date_released`. Package and
  manuscript versions must match. Historical version-named PDFs remain
  tracked; a development state does not require a new PDF, while a final state
  requires the exact version/DOI-derived file.

Experimental comparators, theory witnesses, recovery fixtures, and planned
registry runners may evolve while their roadmap item remains open. Their
documentation must say when they are outside the stable dispatch.

## Compatibility and deprecation

Within the current 1.x major release line, new minor or patch releases may add
fields with compatibility defaults, methods, experiments, profiles, or receipt
schema versions. They must not silently reinterpret an existing field or change
a top-level return type. A future major release requires an explicit migration
plan rather than silently inheriting this 1.x promise.

The v1.1 shape correction is the one intentional fail-closed compatibility
change: an externally supplied categorical PMF or weight must already be one-
dimensional. Row matrices, column matrices, and higher-rank tensors are no
longer flattened implicitly. Valid lists, one-dimensional NumPy arrays, and an
outer iterable yielding valid row vectors continue to work. Callers with a
known singleton dimension must remove it explicitly and assert the resulting
shape before entering the library.

The report-schema migration used by this review is explicit: `robustness_sweep`
is schema `2.0` and uses `d_equivalent` plus headline tie, mean-difference, and
worst-rate disclosure fields; `robust_influence_weights` is schema `2.0` and
uses `normalized_effective_weights`; the heuristic-characterization report is
schema `2.0` and binds scoped no-go metadata; the review-grid report is schema
`1.1`. Its schema-`1.0` predecessor is retained only as historical evidence and
is not accepted by the current reader. Readers fail closed on unsupported
declared versions. The legacy `cohens_d`
report field and `agent_weights` output key may be read only as compatibility
data while their deprecation path remains documented. The federation wire key
`agent_weights` is intentionally retained and is not silently reinterpreted; a
wire migration requires a versioned envelope change.

```python
from fedference import AggregationConfig, aggregate_result

local_posteriors = [[0.70, 0.20, 0.10], [0.60, 0.30, 0.10]]
# Raw coefficients are log-pooling exponents; they are not normalized first.
base_weights = [0.75, 0.25]
config = AggregationConfig(method="robust", robustness=1.5)
result = aggregate_result(
    local_posteriors,
    config=config,
    base_weights=base_weights,
)
influence = result.normalized_effective_weights
raw = result.raw_effective_weights
```

Scaling every raw `base_weights` coefficient can change consensus concentration;
only `normalized_effective_weights` is a normalized diagnostic. Process and
loopback convenience adapters currently use uniform base weights. The executable
interface comparison and transport examples are in
[`examples/README.md`](../../examples/README.md).

Callers using `robust_aggregate(beliefs=..., weights=...)` or reading
`result.agent_weights` receive a `DeprecationWarning` and parity is tested.

A caller uses either `AggregationConfig` or legacy method/tuning arguments.
Supplying both raises `ValueError`; no precedence rule is inferred.

Any later deprecation requires:

1. a migration note with old and new examples;
2. a runtime warning for at least one minor release;
3. parity tests during the warning period; and
4. removal no earlier than the next allowed release boundary.

Serialized schemas fail closed on missing, wrong-typed, unsupported-version,
digest-mismatched, or non-standard non-finite JSON data (`NaN` / `Infinity`).
Supported schema versions may carry additive fields so older readers can remain
forward-compatible; a versioned reader must reject an unsupported declared
version and must not guess how to coerce or reinterpret its fields. Required
fields remain explicit at the write boundary.

## Output and evidence stability

CLI write commands require an explicit empty `--output-dir` and never overwrite
the committed reviewer snapshot. `fedference aggregate` validates request,
provenance requirements, and destination safety before creating its output and
atomically writes request, result, then receipt. Smoke and pilot profiles may test mechanics or
select budgets, but only a locked confirmatory profile can enter a citable
evidence pack. Negative scientific outcomes are compatible with release;
malformed reports, failed controls, stale receipts, or provenance mismatches are
not.

Benchmark reports record solver-health counts (`*_fallback_predictions`,
`*_nonconverged_predictions`, and `*_max_iterations`) separately from
predictive estimands. Receipts summarize affected dataset/seed/method cells,
without treating prediction rows or nested seeds as independent replications.

Development receipts may record a dirty tree and still verify their exact
config/output bytes. Application verification reports artifact integrity and
source equivalence as separate levels; source equivalence is checked only when
requested and available. Research publication verification uses `fedference
verify --require-clean-git`; the recorded and live full commit, clean tree
state, and `uv.lock` digest must then match. An application or research receipt
never establishes scientific validity by itself.
