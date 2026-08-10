# API and schema stability

Active Fedference uses additive evolution throughout the v0.x series.

## Stable public surface

- Top-level aggregation functions keep their current return behavior:
  `log_linear_pool` returns an array; `robust_aggregate` and
  `variational_aggregate` return `AggregationResult`; legacy `aggregate`
  returns the consensus array.
- `AggregationConfig` plus `aggregate_result` is the canonical rich interface.
  `share_round`, `FederationServer`, process adapters, and socket adapters accept
  the same optional configuration. `AggregationResult.fallback_events` is an
  additive, default-empty diagnostic: a returned base-weight-substitution
  trajectory cannot silently count as convergence. Multi-start may also record
  a fallback observed in a discarded start while returning a separately
  converged start.
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
- `ExperimentSpec`, `DatasetSpec`, and `RunReceipt` are versioned evidence
  contracts. Their serialized `schema_version` is independent of the package
  version. Receipt schema 1.1 adds explicit Git tree state and requires a
  receipt-bound `config` artifact whose canonical hash matches
  `config_sha256`.
- The installed `fedference` command exposes `list`, `run`, `benchmark`,
  `verify`, and `replay`.
- Torch is optional through the `bnn` extra. The default `fedference` import
  remains NumPy/SciPy-only.

Experimental comparators, theory witnesses, recovery fixtures, and planned
registry runners may evolve while their roadmap item remains open. Their
documentation must say when they are outside the stable dispatch.

## Compatibility and deprecation

New v0.x releases may add fields with compatibility defaults, methods,
experiments, profiles, or receipt schema versions. They must not silently
reinterpret an existing field or change a top-level return type.

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
result = robust_aggregate(local_posteriors, base_weights=base_weights)
influence = result.normalized_effective_weights
raw = result.raw_effective_weights
```

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
the committed reviewer snapshot. Smoke and pilot profiles may test mechanics or
select budgets, but only a locked confirmatory profile can enter a citable
evidence pack. Negative scientific outcomes are compatible with release;
malformed reports, failed controls, stale receipts, or provenance mismatches are
not.

Benchmark reports record solver-health counts (`*_fallback_predictions`,
`*_nonconverged_predictions`, and `*_max_iterations`) separately from
predictive estimands. Receipts summarize affected dataset/seed/method cells,
without treating prediction rows or nested seeds as independent replications.

Development receipts may record a dirty tree and still verify their exact
config/output bytes. Publication verification uses `fedference verify
--require-clean-git`; the recorded and live full commit, clean tree state, and
`uv.lock` digest must then match. Verification from outside the source checkout
must also pass `--project-root /path/to/checkout`.
