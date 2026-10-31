# API retirement inventory — 2026-10-27

This is the machine-consumed migration contract for Plan Task 3. It freezes
current runtime surfaces and source caller locations; it does not remove an API,
a serialized format, or a scientific control. The [stability policy](../reference/api-stability.md)
and [modularity contract](modularity.md) remain applicable until the corresponding
migration is implemented and reviewed. Release authorization is separate.

## Resolved decisions and current implementation

The explicit breaking-change authorization resolves all 23 owner decisions.
Schema 2 retains all 94 initial rows and all 61 previously retired statuses.
The owner_decisions metadata records each OD disposition, canonical route,
assigned task and rationale. OD01–03 are implemented by Task 52. The independent
OD21 leaf removes only the CLI fallback-helper facade reexport; the identical
internal helper and public main entry point remain. Full Task 54 still depends
on Task 53 and remains pending. Other Task 53/54/55/56/58 surfaces retain their
separately assigned migration and verification gates. The original owner-decision
classification is retained as source evidence; it no longer denotes an unanswered question.

Task 52 removes aggregate, share_round, SharingDiagnostics, friston_belief_share
and run_tabular_benchmark. Canonical config/result paths retain all six scientific
controls, exact solver budgets, and None for unavailable truth metrics.
Benchmark callers explicitly load the synthetic mechanics fixture or their CSV
before running run_benchmark and binding report metadata. Supplementary aliases
and removed rich-selector keywords are recorded separately from the 94 rows.

HD01 refers physically to
src/fedference/evidence_contracts/run_schema.py::RunReceipt.from_dict.
Its current run1.2 constructor remains. Live1.1 dispatch may retire only after
Task55 verifies the full available corpus through fedference_archive, preserving
unavailable provenance and explicit losses. HD02–04 retain current wire-v1
decoding and its normalized-influence semantics. No archive or published bytes
are rewritten here. Current caller locators are conservative AST references.

## Dispositions and migration

- `remove-deprecated`: migrate active callers and guidance to `replacement`, then
  remove the warned alias at the authorized release boundary. Mark its row
  `retired`, retain the row, and refresh its caller locations. Historical evidence
  and dated records remain unchanged. Tasks 12 and 13 own implementation.
- `owner-decision`: supported convenience APIs, unannounced private aliases,
  configuration paths, facades, and serialized fields require an explicit scope
  decision. Capture old data, converter behavior, and loss detection before any
  versioned schema or return-type migration.
- `retain-science`: retain naive/log-linear, robust, variational, comparator and
  exploratory proxy controls. A deprecated keyword is distinct from its method.
  The BNN proxy's KLD/AR selector denotes L2 coefficients, not evaluated
  weight-space divergences or a FedGVI replication.
- `retain-history`: keep evidence-schema 1.1 and protocol-v1 readers. Schema 1.1
  explicitly lacks runtime provenance; protocol `agent_weights` has its existing
  normalized-influence semantics. Python aliases can retire without changing wire
  keys. Unsupported formats continue to fail closed.

For example, migrate `robust_aggregate(beliefs=rows, weights=weights)` to
`robust_aggregate(local_posteriors=rows, base_weights=weights)` and
`result.agent_weights` to `result.normalized_effective_weights`. Preserve raw
`base_weights` as exponents; normalizing them changes the model. Migrate
`cohens_d_from_rank_biserial` to `d_equivalent_from_rank_biserial`; this changes
an inaccurate name, not the effect-size estimand. Canonical numeric behavior,
negative controls and historical decoding require their own runtime gates.

## Gate and coverage limits

Run `uv run --locked --extra dev pytest tests/test_retirement_inventory.py -q`.
The test parses this JSON and the current Python AST. Discovery covers
`DeprecationWarning`/`PendingDeprecationWarning`, literal `legacy.pop`, shared
`_resolve_compat_alias` calls, the current tuple-driven update-factor aliases,
one-hop `**legacy` forwarding, and reexports of discovered deprecated symbols.
The explicitly classified supported surfaces below supplement warning discovery.
The gate also requires the scientific controls and historical readers to exist,
with evidence 1.1 and protocol-v1 constants preserved.

`callsites` are current `path::qualified-caller` syntactic references in `src`, `tests`,
`examples`, and `scripts`, excluding this inventory test. Keyword references
match callable leaf name plus literal keyword, with constructor keywords matched
to the class name. Qualified properties match attribute accesses only; declared
class/static methods match explicit `Class.member` accesses. Other functions and
imports match their leaf name. Thus these are conservative caller candidates, not
Python type resolution: same-named attributes can overmatch, while indirect class
receivers such as `cls`, renamed imports, dynamic `getattr`, string-built keywords,
nested dictionary expansion, notebooks, and
external downstream callers are outside coverage. Empty arrays mean no matching
static reference, not proof of no downstream consumers. Add newly discovered
patterns to the gate and inventory rather than assuming this bounded scan is a
complete dynamic call graph. It does not validate documentation wording, count
mentions, or certify scientific validity. All historical prose, sealed outputs,
and immutable evidence are preserved; active guidance migration belongs to Task 34.

When source moves, review and refresh exact caller locations. Do not blindly
regenerate dispositions. A retired alias must disappear from discovery; retained
readers cannot be relabeled retired. Real disposable source copies prove that
an injected unclassified warning and a removed protocol reader fail the gate.

## Machine-readable inventory

```json
{
  "schema_version": 2,
  "entries": [
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "tau",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/agents.py::SentinelEnsemble.assimilate(learning_rate=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "symbol",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "normalized_effective_weights",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/aggregation.py::AggregationResult.agent_weights"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "warning-helper",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "remove compatibility warning helper after caller migration",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/aggregation.py::_resolve_compat_alias"
    },
    {
      "callsites": [
        "tests/fedference/test_federation_health.py::test_structured_replay_integrity_finding_codes",
        "tests/fedference/test_socket_transport.py::test_malformed_replay_worker_orders_fail_closed",
        "tests/fedference/test_socket_transport.py::test_replay_log_rejects_tampered_consensus_digest",
        "tests/fedference/test_socket_transport.py::test_replay_rejects_duplicate_worker_events_and_config_tamper",
        "tests/fedference/test_socket_transport.py::test_replay_rejects_top_level_aggregate_config_tamper",
        "tests/figures/test_visual_flow_figures.py::test_application_flow_failure_routes_use_clear_dedicated_lanes"
      ],
      "disposition": "owner-decision",
      "historical_reader": "none-supported-interface",
      "kind": "retained-interface",
      "owner": "breaking-52",
      "rationale": "Redundant array projection and method/kwargs dispatch; no distinct mathematics.",
      "replacement": "aggregate_result(..., config=AggregationConfig(...)).consensus",
      "scope": "Task 52",
      "status": "retired",
      "symbol": "src/fedference/aggregation.py::aggregate"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/aggregation.py::aggregate(beliefs=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "base_weights",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/aggregation.py::aggregate(weights=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/aggregation.py::aggregate_result(beliefs=)"
    },
    {
      "callsites": [
        "tests/fedference/test_review_hardening.py::test_aggregation_validation_alias_and_dispatch_paths"
      ],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "base_weights",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/aggregation.py::aggregate_result(weights=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "raw_effective_weights",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/aggregation.py::aggregation_free_energy(agent_weights=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/aggregation.py::aggregation_free_energy(beliefs=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "consensus_posterior",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/aggregation.py::aggregation_free_energy(consensus=)"
    },
    {
      "callsites": [
        "src/__init__.py::<module>",
        "src/analysis/producers/belief.py::<module>",
        "src/analysis/producers/belief.py::_heatmap_inputs",
        "src/fedference/__init__.py::<module>",
        "src/fedference/aggregation.py::aggregate_result",
        "src/fedference/benchmark.py::<module>",
        "src/fedference/benchmark.py::run_benchmark",
        "src/fedference/experiments/_common.py::<module>",
        "src/fedference/experiments/_common.py::_consensus_accuracy",
        "src/fedference/experiments/complexity.py::<module>",
        "src/fedference/experiments/complexity.py::_aggregation_call",
        "src/fedference/experiments/conditional_world.py::<module>",
        "src/fedference/experiments/conditional_world.py::_trial",
        "src/fedference/experiments/heuristic_characterization.py::<module>",
        "src/fedference/experiments/heuristic_characterization.py::_log_pool_aggregator",
        "src/fedference/experiments/navigation.py::<module>",
        "src/fedference/experiments/navigation.py::run_disjoint_fov_world",
        "src/fedference/experiments/navigation.py::run_efe_navigation_test",
        "src/fedference/experiments/worlds.py::<module>",
        "src/fedference/experiments/worlds.py::run_3level_world",
        "src/fedference/experiments/worlds.py::run_hierarchical_world",
        "src/fedference/experiments/worlds.py::run_moving_world",
        "src/fedference/experiments/worlds.py::run_nlevel_world",
        "src/fedference/trials.py::<module>",
        "src/fedference/trials.py::compare_flat_vs_nlevel",
        "src/figures/hierarchical_pomdp.py::generate_hierarchical_pomdp",
        "src/figures/system_overview.py::<module>",
        "src/figures/system_overview.py::naive_pool",
        "src/invariants.py::<module>",
        "src/invariants.py::check_pmf_normalization",
        "src/invariants.py::check_robust_recovers_naive",
        "src/manuscript_vars/loaders.py::<module>",
        "src/manuscript_vars/loaders.py::_recovery_residuals",
        "tests/fedference/test_agents.py::<module>",
        "tests/fedference/test_agents.py::test_assimilate_fuses_shared_location_as_log_product",
        "tests/fedference/test_agents.py::test_assimilate_leaves_private_gaze_unchanged",
        "tests/fedference/test_agents.py::test_assimilate_rejects_implicit_consensus_flattening",
        "tests/fedference/test_agents.py::test_assimilate_rejects_negative_tau",
        "tests/fedference/test_agents.py::test_assimilate_zero_learning_rate_is_identity_on_shared",
        "tests/fedference/test_agents.py::test_full_round_drives_colony_toward_a_shared_cell",
        "tests/fedference/test_aggregation_comparators.py::<module>",
        "tests/fedference/test_aggregation_comparators.py::test_identical_beliefs_recover_that_belief_and_differ_from_no_rule",
        "tests/fedference/test_aggregation_config.py::<module>",
        "tests/fedference/test_aggregation_config.py::test_rich_dispatch_matches_each_low_level_method",
        "tests/fedference/test_aggregation_objective.py::<module>",
        "tests/fedference/test_aggregation_objective.py::test_robust_aggregate_iterative_path_recovers_pool_in_the_small_c_limit",
        "tests/fedference/test_aggregation_objective.py::test_variational_and_heuristic_share_the_friston_corner",
        "tests/fedference/test_aggregation_objective.py::test_variational_consensus_is_more_conservative_than_naive",
        "tests/fedference/test_aggregation_objective.py::test_variational_recovers_log_linear_pool_at_zero_robustness",
        "tests/fedference/test_aggregation_objective.py::test_variational_recovers_pool_in_the_small_c_limit",
        "tests/fedference/test_core_edges.py::<module>",
        "tests/fedference/test_core_edges.py::test_aggregate_validation_and_dispatch",
        "tests/fedference/test_core_edges.py::test_aggregation_rejects_all_zero_weights",
        "tests/fedference/test_core_edges.py::test_degenerate_one_state_pmf_has_zero_divergence_and_pool",
        "tests/fedference/test_core_edges.py::test_log_linear_pool_honours_weights",
        "tests/fedference/test_core_edges.py::test_log_linear_pool_same_beliefs_sharpens",
        "tests/fedference/test_core_identities.py::<module>",
        "tests/fedference/test_core_identities.py::test_log_linear_pool_is_product_of_experts",
        "tests/fedference/test_core_identities.py::test_pool_uses_admitted_posteriors_for_small_positive_raw_masses",
        "tests/fedference/test_core_identities.py::test_robust_aggregate_downweights_the_outlier",
        "tests/fedference/test_core_identities.py::test_robust_aggregate_recovers_naive_at_zero_robustness",
        "tests/fedference/test_heuristic_characterization.py::<module>",
        "tests/fedference/test_heuristic_characterization.py::test_influence_at_zero_robustness_is_the_naive_pool",
        "tests/fedference/test_property_based.py::test_aggregate_dispatch_naive_consistent",
        "tests/fedference/test_property_based.py::test_all_aggregators_return_valid_pmf",
        "tests/fedference/test_property_based.py::test_robust_aggregate_zero_robustness_equals_log_linear_pool",
        "tests/fedference/test_property_based.py::test_share_round_exclude_self_uses_each_recipients_leave_one_out_pool",
        "tests/fedference/test_property_based.py::test_variational_aggregate_zero_robustness_equals_log_linear_pool",
        "tests/fedference/test_review_hardening.py::<module>",
        "tests/fedference/test_review_hardening.py::test_aggregation_validation_alias_and_dispatch_paths",
        "tests/fedference/test_review_hardening.py::test_canonical_aggregation_keywords_and_retired_alias_rejection",
        "tests/fedference/test_strict_shape_boundaries.py::<module>",
        "tests/fedference/test_strict_shape_boundaries.py::test_advertised_aggregation_and_federation_boundaries_reject_matrix_rows",
        "tests/fedference/test_strict_shape_boundaries.py::test_advertised_weight_boundaries_reject_matrix_weights",
        "tests/fedference/test_strict_shape_boundaries.py::test_direct_aggregation_preserves_outer_generator_support",
        "tests/fedference/test_tempered_aggregation.py::<module>",
        "tests/fedference/test_tempered_aggregation.py::test_tempered_log_linear_recovery",
        "tests/fedference/test_tempered_aggregation_objective.py::<module>",
        "tests/fedference/test_tempered_aggregation_objective.py::test_c_to_zero_recovery_tempered",
        "tests/figures/test_frontmatter.py::<module>",
        "tests/figures/test_frontmatter.py::test_frontmatter_claims_match_owners_and_actual_recovery",
        "tests/test_domain_retirement.py::<module>",
        "tests/test_domain_retirement.py::test_canonical_calls_and_scientific_comparators",
        "tests/test_domain_retirement.py::test_retired_assimilation_keyword_leaves_state_unchanged",
        "tests/test_unified_domain_api.py::<module>",
        "tests/test_unified_domain_api.py::test_six_substantive_scientific_controls_remain"
      ],
      "disposition": "retain-science",
      "historical_reader": "none-scientific-control",
      "kind": "retained-interface",
      "owner": "project-owner",
      "rationale": "Substantive method or comparator; removing aliases does not remove the experiment control or alter scientific claims.",
      "replacement": "same scientific operation through canonical arguments",
      "scope": "preservation contract",
      "status": "present",
      "symbol": "src/fedference/aggregation.py::log_linear_pool"
    },
    {
      "callsites": [
        "tests/fedference/test_review_hardening.py::test_aggregation_validation_alias_and_dispatch_paths"
      ],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/aggregation.py::log_linear_pool(beliefs=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "base_weights",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/aggregation.py::log_linear_pool(weights=)"
    },
    {
      "callsites": [
        "src/__init__.py::<module>",
        "src/fedference/__init__.py::<module>",
        "src/fedference/aggregation.py::aggregate_result",
        "src/fedference/benchmark.py::<module>",
        "src/fedference/benchmark.py::run_benchmark",
        "src/fedference/bnn_baseline_torch.py::<module>",
        "src/fedference/bnn_baseline_torch.py::_consensus_accuracy",
        "src/fedference/bnn_baseline_torch.py::federated_bnn_round",
        "src/fedference/experiments/complexity.py::<module>",
        "src/fedference/experiments/complexity.py::_aggregation_call",
        "src/fedference/experiments/conditional_world.py::<module>",
        "src/fedference/experiments/conditional_world.py::_trial",
        "src/fedference/experiments/heuristic_characterization.py::<module>",
        "src/fedference/experiments/heuristic_characterization.py::empirical_breakdown",
        "src/fedference/experiments/heuristic_characterization.py::numerical_influence_function",
        "src/figures/system_overview.py::<module>",
        "src/figures/system_overview.py::robust_pool",
        "src/invariants.py::<module>",
        "src/invariants.py::check_pmf_normalization",
        "src/invariants.py::check_robust_recovers_naive",
        "src/manuscript_vars/loaders.py::<module>",
        "src/manuscript_vars/loaders.py::_federation_variables",
        "src/manuscript_vars/loaders.py::_recovery_residuals",
        "src/manuscript_vars/loaders.py::_tempered_variables",
        "tests/fedference/test_aggregation_config.py::<module>",
        "tests/fedference/test_aggregation_config.py::test_legacy_dispatch_and_config_path_are_compatible",
        "tests/fedference/test_aggregation_config.py::test_rich_dispatch_matches_each_low_level_method",
        "tests/fedference/test_aggregation_objective.py::<module>",
        "tests/fedference/test_aggregation_objective.py::test_robust_aggregate_iterative_path_recovers_pool_in_the_small_c_limit",
        "tests/fedference/test_contamination.py::<module>",
        "tests/fedference/test_contamination.py::test_confident_wrong_agent_is_downweighted_by_robust_pool",
        "tests/fedference/test_core_edges.py::<module>",
        "tests/fedference/test_core_edges.py::test_aggregate_validation_and_dispatch",
        "tests/fedference/test_core_edges.py::test_aggregation_rejects_all_zero_weights",
        "tests/fedference/test_core_edges.py::test_aggregation_rejects_nonfinite_controls_and_terminal_weights_match_consensus",
        "tests/fedference/test_core_edges.py::test_all_mass_on_one_state_into_robust_and_variational_aggregate",
        "tests/fedference/test_core_edges.py::test_robust_aggregate_all_agents_vetoed_fallback",
        "tests/fedference/test_core_edges.py::test_robust_aggregate_reports_iterations_when_engaged",
        "tests/fedference/test_core_edges.py::test_single_agent_robust_aggregate_is_that_agent",
        "tests/fedference/test_core_identities.py::<module>",
        "tests/fedference/test_core_identities.py::test_pool_uses_admitted_posteriors_for_small_positive_raw_masses",
        "tests/fedference/test_core_identities.py::test_robust_aggregate_downweights_the_outlier",
        "tests/fedference/test_core_identities.py::test_robust_aggregate_recovers_naive_at_zero_robustness",
        "tests/fedference/test_federation_end_to_end.py::<module>",
        "tests/fedference/test_federation_end_to_end.py::test_federation_bit_identical_to_inprocess",
        "tests/fedference/test_federation_end_to_end.py::test_federation_robust_bit_identical",
        "tests/fedference/test_federation_end_to_end.py::test_federation_robust_bit_identical_when_workers_arrive_out_of_order",
        "tests/fedference/test_federation_end_to_end.py::test_multiprocess_round_bit_identical_to_inprocess",
        "tests/fedference/test_federation_end_to_end.py::test_multiprocess_round_robust_bit_identical_to_inprocess",
        "tests/fedference/test_heuristic_characterization.py::<module>",
        "tests/fedference/test_heuristic_characterization.py::test_capture_threshold_actually_flips_the_argmax",
        "tests/fedference/test_heuristic_characterization.py::test_influence_at_zero_robustness_is_the_naive_pool",
        "tests/fedference/test_property_based.py::test_all_aggregators_return_valid_pmf",
        "tests/fedference/test_property_based.py::test_robust_aggregate_agent_weights_sum_to_one",
        "tests/fedference/test_property_based.py::test_robust_aggregate_zero_robustness_equals_log_linear_pool",
        "tests/fedference/test_review_hardening.py::<module>",
        "tests/fedference/test_review_hardening.py::test_aggregation_validation_alias_and_dispatch_paths",
        "tests/fedference/test_review_hardening.py::test_raw_and_normalized_effective_weights_have_distinct_contracts",
        "tests/fedference/test_socket_transport.py::<module>",
        "tests/fedference/test_socket_transport.py::test_every_worker_receives_the_same_broadcast",
        "tests/fedference/test_socket_transport.py::test_socket_round_is_bit_identical_to_in_process",
        "tests/fedference/test_strict_shape_boundaries.py::<module>",
        "tests/fedference/test_strict_shape_boundaries.py::test_advertised_aggregation_and_federation_boundaries_reject_matrix_rows",
        "tests/fedference/test_strict_shape_boundaries.py::test_advertised_weight_boundaries_reject_matrix_weights",
        "tests/fedference/test_strict_shape_boundaries.py::test_outer_generators_cross_aggregate_and_sharing_boundaries",
        "tests/fedference/test_tempered_aggregation_objective.py::<module>",
        "tests/fedference/test_tempered_aggregation_objective.py::test_lambda_star_empirical",
        "tests/figures/test_frontmatter.py::<module>",
        "tests/figures/test_frontmatter.py::test_frontmatter_claims_match_owners_and_actual_recovery",
        "tests/test_domain_retirement.py::<module>",
        "tests/test_domain_retirement.py::test_canonical_calls_and_scientific_comparators",
        "tests/test_domain_retirement.py::test_retired_result_properties_are_absent",
        "tests/test_token_sourcing.py::<module>",
        "tests/test_token_sourcing.py::test_federation_bit_identical_token_matches_direct_recomputation",
        "tests/test_unified_domain_api.py::<module>",
        "tests/test_unified_domain_api.py::test_six_substantive_scientific_controls_remain"
      ],
      "disposition": "retain-science",
      "historical_reader": "none-scientific-control",
      "kind": "retained-interface",
      "owner": "project-owner",
      "rationale": "Substantive method or comparator; removing aliases does not remove the experiment control or alter scientific claims.",
      "replacement": "same scientific operation through canonical arguments",
      "scope": "preservation contract",
      "status": "present",
      "symbol": "src/fedference/aggregation.py::robust_aggregate"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/aggregation.py::robust_aggregate(beliefs=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "base_weights",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/aggregation.py::robust_aggregate(weights=)"
    },
    {
      "callsites": [
        "src/fedference/__init__.py::<module>",
        "src/fedference/aggregation.py::aggregate_result",
        "src/fedference/experiments/complexity.py::<module>",
        "src/fedference/experiments/complexity.py::_aggregation_call",
        "src/fedference/experiments/diagnostics.py::<module>",
        "src/fedference/experiments/diagnostics.py::run_variational_aggregation_report",
        "src/fedference/experiments/heuristic_characterization.py::<module>",
        "src/fedference/experiments/heuristic_characterization.py::empirical_breakdown",
        "src/manuscript_vars/loaders.py::_tempered_variables",
        "tests/fedference/test_aggregation_config.py::<module>",
        "tests/fedference/test_aggregation_config.py::test_rich_dispatch_matches_each_low_level_method",
        "tests/fedference/test_aggregation_objective.py::<module>",
        "tests/fedference/test_aggregation_objective.py::test_aggregate_dispatch_variational",
        "tests/fedference/test_aggregation_objective.py::test_converged_iterate_is_a_stationary_point",
        "tests/fedference/test_aggregation_objective.py::test_diverging_agent_influence_collapses_to_zero",
        "tests/fedference/test_aggregation_objective.py::test_effective_weights_never_exceed_base",
        "tests/fedference/test_aggregation_objective.py::test_free_energy_descends_monotonically",
        "tests/fedference/test_aggregation_objective.py::test_free_energy_matches_independent_recomputation",
        "tests/fedference/test_aggregation_objective.py::test_free_energy_supports_zero_base_weight_by_zeroing_effective_weight",
        "tests/fedference/test_aggregation_objective.py::test_multistart_false_uses_single_pool_seed_and_can_be_captured",
        "tests/fedference/test_aggregation_objective.py::test_multistart_prefers_a_converged_start_over_lower_unfinished_trace",
        "tests/fedference/test_aggregation_objective.py::test_multistart_reaches_lower_or_equal_free_energy_than_pool_seed",
        "tests/fedference/test_aggregation_objective.py::test_multistart_vetoes_near_vertex_adversary",
        "tests/fedference/test_aggregation_objective.py::test_negative_robustness_rejected",
        "tests/fedference/test_aggregation_objective.py::test_raw_effective_weights_are_bounded_by_base_weights",
        "tests/fedference/test_aggregation_objective.py::test_single_agent_consensus_recovers_that_agent_at_c_zero",
        "tests/fedference/test_aggregation_objective.py::test_variational_and_heuristic_share_the_friston_corner",
        "tests/fedference/test_aggregation_objective.py::test_variational_consensus_is_more_conservative_than_naive",
        "tests/fedference/test_aggregation_objective.py::test_variational_recovers_log_linear_pool_at_zero_robustness",
        "tests/fedference/test_aggregation_objective.py::test_variational_recovers_pool_in_the_small_c_limit",
        "tests/fedference/test_aggregation_objective.py::test_variational_weight_collapse_is_receipted_and_not_convergence",
        "tests/fedference/test_core_edges.py::<module>",
        "tests/fedference/test_core_edges.py::test_all_mass_on_one_state_into_robust_and_variational_aggregate",
        "tests/fedference/test_core_edges.py::test_share_round_variational_dispatches_objective_backed_rule",
        "tests/fedference/test_core_edges.py::test_variational_aggregate_max_iter_zero_returns_valid_consensus",
        "tests/fedference/test_property_based.py::test_all_aggregators_return_valid_pmf",
        "tests/fedference/test_property_based.py::test_variational_aggregate_zero_robustness_equals_log_linear_pool",
        "tests/fedference/test_review_hardening.py::<module>",
        "tests/fedference/test_review_hardening.py::test_raw_and_normalized_effective_weights_have_distinct_contracts",
        "tests/fedference/test_strict_shape_boundaries.py::<module>",
        "tests/fedference/test_strict_shape_boundaries.py::test_advertised_aggregation_and_federation_boundaries_reject_matrix_rows",
        "tests/fedference/test_strict_shape_boundaries.py::test_advertised_weight_boundaries_reject_matrix_weights",
        "tests/fedference/test_tempered_aggregation.py::<module>",
        "tests/fedference/test_tempered_aggregation.py::test_tempered_bit_identical_at_default",
        "tests/fedference/test_tempered_aggregation.py::test_tempered_log_linear_recovery",
        "tests/fedference/test_tempered_aggregation.py::test_tempered_scenario_retains_true_state_majority_across_lambda",
        "tests/fedference/test_tempered_aggregation.py::test_tempered_sharpens_with_lower_lambda",
        "tests/fedference/test_tempered_aggregation_objective.py::<module>",
        "tests/fedference/test_tempered_aggregation_objective.py::test_aggregation_free_energy_entropy_scaling",
        "tests/fedference/test_tempered_aggregation_objective.py::test_c_to_zero_recovery_tempered",
        "tests/fedference/test_tempered_aggregation_objective.py::test_entropy_weight_1_bit_identical",
        "tests/fedference/test_tempered_aggregation_objective.py::test_entropy_weight_must_be_non_negative",
        "tests/fedference/test_tempered_aggregation_objective.py::test_lambda_star_empirical",
        "tests/fedference/test_tempered_aggregation_objective.py::test_low_entropy_weight_sharpens_consensus",
        "tests/fedference/test_tempered_aggregation_objective.py::test_tempered_free_energy_monotone_descent",
        "tests/fedference/test_tempered_aggregation_objective.py::test_tempered_scenario_controls_adversary_weight",
        "tests/fedference/test_tempered_aggregation_objective.py::test_variational_low_level_controls_reject_coercive_types",
        "tests/fedference/test_tempered_aggregation_objective.py::test_zero_entropy_weight_selects_the_linear_objective_vertex",
        "tests/figures/test_aggregation_descent.py::<module>",
        "tests/figures/test_aggregation_descent.py::test_descent_from_real_aggregator",
        "tests/figures/test_bounded_influence.py::<module>",
        "tests/figures/test_bounded_influence.py::test_bounded_influence_from_real_aggregator",
        "tests/figures/test_descent_comparison.py::<module>",
        "tests/figures/test_descent_comparison.py::test_comparison_from_real_aggregator",
        "tests/test_domain_retirement.py::<module>",
        "tests/test_domain_retirement.py::test_canonical_calls_and_scientific_comparators",
        "tests/test_unified_domain_api.py::<module>",
        "tests/test_unified_domain_api.py::test_six_substantive_scientific_controls_remain"
      ],
      "disposition": "retain-science",
      "historical_reader": "none-scientific-control",
      "kind": "retained-interface",
      "owner": "project-owner",
      "rationale": "Substantive method or comparator; removing aliases does not remove the experiment control or alter scientific claims.",
      "replacement": "same scientific operation through canonical arguments",
      "scope": "preservation contract",
      "status": "present",
      "symbol": "src/fedference/aggregation.py::variational_aggregate"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/aggregation.py::variational_aggregate(beliefs=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "base_weights",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/aggregation.py::variational_aggregate(weights=)"
    },
    {
      "callsites": [
        "tests/fedference/test_aggregation_comparators.py::<module>",
        "tests/fedference/test_aggregation_comparators.py::test_clr_median_is_valid_deterministic_and_permutation_invariant",
        "tests/fedference/test_aggregation_comparators.py::test_duplicate_majority_is_recognized_as_an_exact_weighted_median",
        "tests/fedference/test_aggregation_comparators.py::test_identical_beliefs_recover_that_belief_and_differ_from_no_rule",
        "tests/test_unified_domain_api.py::<module>",
        "tests/test_unified_domain_api.py::test_six_substantive_scientific_controls_remain"
      ],
      "disposition": "retain-science",
      "historical_reader": "none-scientific-control",
      "kind": "retained-interface",
      "owner": "project-owner",
      "rationale": "Substantive method or comparator; removing aliases does not remove the experiment control or alter scientific claims.",
      "replacement": "same scientific operation through canonical arguments",
      "scope": "preservation contract",
      "status": "present",
      "symbol": "src/fedference/aggregation_comparators.py::clr_geometric_median_pool"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/aggregation_comparators.py::clr_geometric_median_pool(beliefs=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "base_weights",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/aggregation_comparators.py::clr_geometric_median_pool(weights=)"
    },
    {
      "callsites": [
        "tests/fedference/test_aggregation_comparators.py::<module>",
        "tests/fedference/test_aggregation_comparators.py::test_linear_pool_is_weighted_arithmetic_mixture",
        "tests/test_domain_retirement.py::<module>",
        "tests/test_domain_retirement.py::test_canonical_calls_and_scientific_comparators",
        "tests/test_unified_domain_api.py::<module>",
        "tests/test_unified_domain_api.py::test_six_substantive_scientific_controls_remain"
      ],
      "disposition": "retain-science",
      "historical_reader": "none-scientific-control",
      "kind": "retained-interface",
      "owner": "project-owner",
      "rationale": "Substantive method or comparator; removing aliases does not remove the experiment control or alter scientific claims.",
      "replacement": "same scientific operation through canonical arguments",
      "scope": "preservation contract",
      "status": "present",
      "symbol": "src/fedference/aggregation_comparators.py::linear_opinion_pool"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/aggregation_comparators.py::linear_opinion_pool(beliefs=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "base_weights",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/aggregation_comparators.py::linear_opinion_pool(weights=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "symbol",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "normalized_effective_weights",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/belief_sharing.py::SharingDiagnostics.agent_weights"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "symbol",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "shared_posteriors",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/belief_sharing.py::SharingDiagnostics.shared_beliefs"
    },
    {
      "callsites": [],
      "disposition": "owner-decision",
      "historical_reader": "none-supported-interface",
      "kind": "retained-interface",
      "owner": "breaking-52",
      "rationale": "Historical diagnostics projection duplicates rich result and invents NaN for unavailable metrics; retain typed None and explicit fields.",
      "replacement": "share_round_result; consume shared_posteriors/global_aggregation",
      "scope": "Task 52",
      "status": "retired",
      "symbol": "src/fedference/belief_sharing.py::share_round"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/belief_sharing.py::share_round(agent_beliefs=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "base_weights",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/belief_sharing.py::share_round(weights=)"
    },
    {
      "callsites": [
        "tests/fedference/test_review_hardening.py::test_share_round_alias_conflicts_and_fail_closed_inputs"
      ],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/belief_sharing.py::share_round_result(agent_beliefs=)"
    },
    {
      "callsites": [
        "tests/fedference/test_review_hardening.py::test_share_round_alias_conflicts_and_fail_closed_inputs"
      ],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "base_weights",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/belief_sharing.py::share_round_result(weights=)"
    },
    {
      "callsites": [
        "tests/test_unified_domain_api.py::test_explicit_benchmark_inputs_preserve_reports",
        "tests/test_unified_domain_api.py::test_same_runtime_migration_oracle_rejects_wrong_current_parameters"
      ],
      "disposition": "owner-decision",
      "historical_reader": "none-supported-interface",
      "kind": "retained-interface",
      "owner": "breaking-52",
      "rationale": "Implicit bundled path is a redundant convenience route; preserve synthetic smoke dataset through explicit package-resource loading.",
      "replacement": "load_tabular_csv(explicit_path) then run_benchmark(features, labels, ...); registered external loader for real data",
      "scope": "Task 52",
      "status": "retired",
      "symbol": "src/fedference/benchmark.py::run_tabular_benchmark"
    },
    {
      "callsites": [
        "src/fedference/experiments/diagnostics.py::<module>",
        "src/fedference/experiments/diagnostics.py::run_bnn_robustness_report._seeded_curve",
        "tests/fedference/test_bnn_baseline.py::<module>",
        "tests/fedference/test_bnn_baseline.py::test_ar_compatibility_shrinkage_path_runs",
        "tests/fedference/test_bnn_baseline.py::test_clean_baseline_learns_a_useful_boundary",
        "tests/fedference/test_bnn_baseline.py::test_deterministic_under_fixed_seed",
        "tests/fedference/test_bnn_baseline.py::test_different_seeds_give_different_runs",
        "tests/fedference/test_bnn_baseline.py::test_fed_gvi_rejects_rcce_param_out_of_range",
        "tests/fedference/test_bnn_baseline.py::test_fed_gvi_rejects_unknown_loss_and_divergence",
        "tests/fedference/test_bnn_baseline.py::test_fed_gvi_rejects_zero_clients",
        "tests/fedference/test_bnn_baseline.py::test_rcce_advantage_holds_across_seeds",
        "tests/fedference/test_bnn_baseline.py::test_rcce_beats_nll_under_contamination",
        "tests/fedference/test_bnn_baseline.py::test_rcce_separation_is_not_a_knife_edge_in_loss_param",
        "tests/fedference/test_bnn_baseline.py::test_rcce_with_zero_param_recovers_nll",
        "tests/fedference/test_bnn_baseline.py::test_returns_accuracy_in_unit_interval",
        "tests/fedference/test_bnn_baseline.py::test_weights_shape_is_features_plus_bias",
        "tests/test_unified_domain_api.py::<module>",
        "tests/test_unified_domain_api.py::test_six_substantive_scientific_controls_remain"
      ],
      "disposition": "retain-science",
      "historical_reader": "none-scientific-control",
      "kind": "retained-interface",
      "owner": "project-owner",
      "rationale": "The compatibility-named divergence selector KLD/AR selects L2 coefficients 0.05/0.10. It does not evaluate Alpha-Renyi or KLD weight-space divergences, variational posterior inference, or source-scale FedGVI replication.",
      "replacement": "same point-estimate loss/L2 proxy with explicit scientific caveats",
      "scope": "preservation contract",
      "status": "present",
      "symbol": "src/fedference/bnn_baseline.py::fed_gvi_logreg"
    },
    {
      "callsites": [
        "src/fedference/bnn_baseline_torch.py::<module>",
        "src/fedference/bnn_baseline_torch.py::PointMassMLP.__init__",
        "src/fedference/bnn_baseline_torch.py::federated_bnn_round",
        "src/fedference/bnn_baseline_torch.py::run_bnn_torch_experiment",
        "src/fedference/bnn_variational_torch.py::<module>",
        "src/fedference/bnn_variational_torch.py::VariationalMLP.__init__",
        "src/manuscript_vars/loaders.py::<module>",
        "src/manuscript_vars/loaders.py::_bnn_torch_variables",
        "tests/test_token_sourcing.py::<module>",
        "tests/test_token_sourcing.py::test_bnn_fallback_tokens_are_the_experiment_defaults"
      ],
      "disposition": "owner-decision",
      "historical_reader": "none-supported-interface",
      "kind": "retained-interface",
      "owner": "breaking-54",
      "rationale": "One torch-free default owner; backend references the defaults module without reexport.",
      "replacement": "fedference.bnn_defaults.BNN_BETA_DEFAULT",
      "scope": "Task 54",
      "status": "present",
      "symbol": "src/fedference/bnn_baseline_torch.py::BNN_BETA_DEFAULT"
    },
    {
      "callsites": [
        "src/fedference/bnn_baseline_torch.py::<module>",
        "src/fedference/bnn_baseline_torch.py::federated_bnn_round",
        "src/fedference/bnn_baseline_torch.py::run_bnn_torch_experiment",
        "src/fedference/bnn_variational_torch.py::<module>",
        "src/fedference/bnn_variational_torch.py::VariationalMLP.__init__",
        "src/manuscript_vars/loaders.py::<module>",
        "src/manuscript_vars/loaders.py::_bnn_torch_variables",
        "tests/test_token_sourcing.py::<module>",
        "tests/test_token_sourcing.py::test_bnn_fallback_tokens_are_the_experiment_defaults"
      ],
      "disposition": "owner-decision",
      "historical_reader": "none-supported-interface",
      "kind": "retained-interface",
      "owner": "breaking-54",
      "rationale": "Duplicate reexport removed; value unchanged.",
      "replacement": "fedference.bnn_defaults.BNN_HIDDEN_DIM_DEFAULT",
      "scope": "Task 54",
      "status": "present",
      "symbol": "src/fedference/bnn_baseline_torch.py::BNN_HIDDEN_DIM_DEFAULT"
    },
    {
      "callsites": [
        "src/fedference/bnn_baseline_torch.py::<module>",
        "src/fedference/bnn_baseline_torch.py::run_bnn_torch_experiment",
        "src/manuscript_vars/loaders.py::<module>",
        "src/manuscript_vars/loaders.py::_bnn_torch_variables",
        "tests/test_token_sourcing.py::<module>",
        "tests/test_token_sourcing.py::test_bnn_fallback_tokens_are_the_experiment_defaults"
      ],
      "disposition": "owner-decision",
      "historical_reader": "none-supported-interface",
      "kind": "retained-interface",
      "owner": "breaking-54",
      "rationale": "Duplicate reexport removed; value unchanged.",
      "replacement": "fedference.bnn_defaults.BNN_N_CLIENTS_DEFAULT",
      "scope": "Task 54",
      "status": "present",
      "symbol": "src/fedference/bnn_baseline_torch.py::BNN_N_CLIENTS_DEFAULT"
    },
    {
      "callsites": [
        "src/fedference/bnn_baseline_torch.py::<module>",
        "src/fedference/bnn_baseline_torch.py::PointMassMLP.fit",
        "src/fedference/bnn_baseline_torch.py::federated_bnn_round",
        "src/fedference/bnn_baseline_torch.py::run_bnn_torch_experiment",
        "src/manuscript_vars/loaders.py::<module>",
        "src/manuscript_vars/loaders.py::_bnn_torch_variables",
        "tests/test_token_sourcing.py::<module>",
        "tests/test_token_sourcing.py::test_bnn_fallback_tokens_are_the_experiment_defaults"
      ],
      "disposition": "owner-decision",
      "historical_reader": "none-supported-interface",
      "kind": "retained-interface",
      "owner": "breaking-54",
      "rationale": "Duplicate reexport removed; value unchanged.",
      "replacement": "fedference.bnn_defaults.BNN_N_STEPS_DEFAULT",
      "scope": "Task 54",
      "status": "present",
      "symbol": "src/fedference/bnn_baseline_torch.py::BNN_N_STEPS_DEFAULT"
    },
    {
      "callsites": [
        "src/fedference/bnn_baseline_torch.py::<module>",
        "src/fedference/bnn_baseline_torch.py::federated_bnn_round",
        "src/fedference/bnn_baseline_torch.py::run_bnn_torch_experiment",
        "src/manuscript_vars/loaders.py::<module>",
        "src/manuscript_vars/loaders.py::_bnn_torch_variables",
        "tests/test_token_sourcing.py::<module>",
        "tests/test_token_sourcing.py::test_bnn_fallback_tokens_are_the_experiment_defaults"
      ],
      "disposition": "owner-decision",
      "historical_reader": "none-supported-interface",
      "kind": "retained-interface",
      "owner": "breaking-54",
      "rationale": "Duplicate reexport removed; value unchanged.",
      "replacement": "fedference.bnn_defaults.BNN_ROBUSTNESS_DEFAULT",
      "scope": "Task 54",
      "status": "present",
      "symbol": "src/fedference/bnn_baseline_torch.py::BNN_ROBUSTNESS_DEFAULT"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/calibration.py::CalibrationEpisode.__init__(beliefs=)"
    },
    {
      "callsites": [
        "src/fedference_cli/_commands.py::_replay_command",
        "tests/integration/test_gnn_pymdp.py::test_real_agent_inference_policy_and_action",
        "tests/integration/test_gnn_pymdp.py::test_seeded_replay_and_preference_sensitivity"
      ],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "symbol",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/calibration.py::CalibrationEpisode.beliefs"
    },
    {
      "callsites": [
        "src/fedference/evidence_contracts/custody.py::load_receipt",
        "src/fedference/evidence_contracts/custody.py::load_run_receipt",
        "tests/fedference/test_evidence.py::test_legacy_run_receipt_round_trips_exact_fields_with_unavailable_runtime",
        "tests/fedference/test_evidence_modularity.py::test_run_decoder_invalid_shapes",
        "tests/test_boundary_retirement.py::test_canonical_transport_figures_and_historical_corpus",
        "tests/test_boundary_retirement.py::test_retired_runtime_alias_and_unknown_schema"
      ],
      "disposition": "retain-history",
      "historical_reader": "evidence-schema-1.1",
      "kind": "retained-interface",
      "owner": "project-owner",
      "rationale": "Historical artifacts must remain interpretable with exact semantics. Unsupported versions still fail closed; missing runtime provenance remains explicitly unavailable.",
      "replacement": "retain supported decoder; versioned converter and loss detection required before any future retirement",
      "scope": "preservation contract",
      "status": "present",
      "symbol": "src/fedference/evidence_contracts/run_schema.py::RunReceipt.from_dict"
    },
    {
      "callsites": [],
      "disposition": "owner-decision",
      "historical_reader": "none-supported-interface",
      "kind": "retained-interface",
      "owner": "breaking-54",
      "rationale": "A coherent study interface is meaningful; remove compatibility-only private exports, not the entire package interface.",
      "replacement": "Documented public study/configuration exports only; private helpers imported from their owning modules",
      "scope": "Task 54",
      "status": "present",
      "symbol": "src/fedference/experiments/__init__.py::__all__"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "reexport",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "_finite_d_equivalent",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/experiments/__init__.py::_finite_cohens_d"
    },
    {
      "callsites": [
        "tests/fedference/test_experiments.py::<module>",
        "tests/fedference/test_experiments.py::test_soft_colony_rows_are_pmfs_peaked_on_truth",
        "tests/figures/test_belief_heatmap.py::test_belief_heatmap_happy_path",
        "tests/figures/test_belief_heatmap.py::test_belief_heatmap_rejects_consensus_mismatch"
      ],
      "disposition": "owner-decision",
      "historical_reader": "none-supported-interface",
      "kind": "retained-interface",
      "owner": "breaking-54",
      "rationale": "Unwarned equality alias has no distinct behavior.",
      "replacement": "fedference.colonies.soft_colony",
      "scope": "Task 54",
      "status": "present",
      "symbol": "src/fedference/experiments/__init__.py::_soft_colony"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "symbol",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "_finite_d_equivalent",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/experiments/_common.py::_finite_cohens_d"
    },
    {
      "callsites": [
        "src/analysis/stage_catalog.py::report_producers",
        "src/analysis/workflow.py::<module>",
        "src/analysis/workflow.py::run_analysis_pipeline",
        "src/fedference/experiments/__init__.py::<module>",
        "src/figures/cross_study_summary.py::_load_cross_study_report",
        "tests/analysis/test_stage_catalog.py::_input_contract"
      ],
      "disposition": "owner-decision",
      "historical_reader": "schema-or-return-migration",
      "kind": "retained-interface",
      "owner": "breaking-56",
      "rationale": "Summary is substantive; the old-profile fallback silently changes the independent unit/estimand and must disappear.",
      "replacement": "Require validated matched-trial per_rate_summary; reject its absence",
      "scope": "Task 56",
      "status": "present",
      "symbol": "src/fedference/experiments/cross_study.py::summarize_cross_study"
    },
    {
      "callsites": [
        "src/analysis/producers/robustness.py::<module>",
        "src/analysis/producers/robustness.py::_influence_weights",
        "src/fedference/experiments/__init__.py::<module>"
      ],
      "disposition": "owner-decision",
      "historical_reader": "schema-or-return-migration",
      "kind": "retained-interface",
      "owner": "breaking-56",
      "rationale": "Report computation remains; remove duplicate agent_weights report key through explicit current schema migration.",
      "replacement": "Current normalized_effective_weights report with exactly one field spelling",
      "scope": "Task 56",
      "status": "present",
      "symbol": "src/fedference/experiments/diagnostics.py::run_influence_weights_report"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/experiments/heuristic_characterization.py::numerical_influence_function(beliefs=)"
    },
    {
      "callsites": [
        "src/analysis/stage_catalog.py::report_producers",
        "src/analysis/workflow.py::<module>",
        "src/analysis/workflow.py::run_analysis_pipeline",
        "src/fedference/experiments/__init__.py::<module>",
        "tests/analysis/test_report_schemas.py::<module>",
        "tests/analysis/test_report_schemas.py::_base_valid_payload_for_name",
        "tests/analysis/test_stage_catalog.py::_input_contract",
        "tests/analysis/test_stage_catalog.py::test_call_contract_accepts_real_helpers_and_partials_and_rejects_wrong_seed",
        "tests/analysis/test_stage_catalog.py::test_observation_restores_profile_after_real_producer_failure",
        "tests/fedference/test_hierarchical_bmr.py::<module>",
        "tests/fedference/test_hierarchical_bmr.py::test_report_identifies_surprise_threshold_as_primary_control",
        "tests/fedference/test_hierarchical_bmr.py::test_report_rejects_invalid_control_configuration",
        "tests/fedference/test_hierarchical_bmr.py::test_report_seed_is_compatibility_provenance_not_replication",
        "tests/fedference/test_hierarchical_bmr.py::test_report_threshold_controls_nested_prunable_flags"
      ],
      "disposition": "owner-decision",
      "historical_reader": "schema-or-return-migration",
      "kind": "retained-interface",
      "owner": "breaking-56",
      "rationale": "Name the actual configured deterministic surprise-threshold lesion/control; preserve values and no-generalization boundary, not an old BMR alias.",
      "replacement": "run_hierarchical_surprise_control; hierarchical_surprise_control report/figure stems",
      "scope": "Task 56",
      "status": "present",
      "symbol": "src/fedference/experiments/worlds.py::run_hierarchical_bmr"
    },
    {
      "callsites": [
        "examples/03_federation_boundaries.py::<module>",
        "examples/03_federation_boundaries.py::main",
        "src/fedference/federation/__init__.py::<module>",
        "tests/fedference/test_federation_end_to_end.py::<module>",
        "tests/fedference/test_federation_end_to_end.py::test_federation_adapters_reject_non_configuration_objects",
        "tests/fedference/test_federation_end_to_end.py::test_federation_timeouts_reject_boolean_values",
        "tests/fedference/test_federation_end_to_end.py::test_multiprocess_round_bit_identical_to_inprocess",
        "tests/fedference/test_federation_end_to_end.py::test_multiprocess_round_rejects_empty_beliefs",
        "tests/fedference/test_federation_end_to_end.py::test_multiprocess_round_robust_bit_identical_to_inprocess",
        "tests/fedference/test_federation_health.py::<module>",
        "tests/fedference/test_federation_health.py::test_multiprocess_rich_result_is_ordered_and_legacy_is_an_array",
        "tests/fedference/test_strict_shape_boundaries.py::<module>",
        "tests/fedference/test_strict_shape_boundaries.py::test_advertised_aggregation_and_federation_boundaries_reject_matrix_rows",
        "tests/test_boundary_retirement.py::<module>",
        "tests/test_boundary_retirement.py::test_canonical_transport_figures_and_historical_corpus"
      ],
      "disposition": "owner-decision",
      "historical_reader": "none-supported-interface",
      "kind": "retained-interface",
      "owner": "breaking-53",
      "rationale": "Redundant array wrapper; MultiprocessRoundResult.consensus is the canonical projection (process.py:45-48).",
      "replacement": "run_multiprocess_round_result(...).consensus; full diagnostics in server_aggregation",
      "scope": "Task 53",
      "status": "present",
      "symbol": "src/fedference/federation/process.py::run_multiprocess_round"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-boundaries",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 13",
      "status": "retired",
      "symbol": "src/fedference/federation/process.py::run_multiprocess_round(beliefs=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-boundaries",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 13",
      "status": "retired",
      "symbol": "src/fedference/federation/process.py::run_multiprocess_round_result(beliefs=)"
    },
    {
      "callsites": [
        "src/fedference/federation/__init__.py::<module>",
        "src/fedference/federation/process.py::<module>",
        "src/fedference/federation/process.py::run_multiprocess_round_result",
        "src/manuscript_vars/loaders.py::_federation_variables",
        "tests/fedference/test_federation_end_to_end.py::<module>",
        "tests/fedference/test_federation_end_to_end.py::test_federation_adapters_reject_non_configuration_objects",
        "tests/fedference/test_federation_end_to_end.py::test_federation_bit_identical_to_inprocess",
        "tests/fedference/test_federation_end_to_end.py::test_federation_robust_bit_identical",
        "tests/fedference/test_federation_end_to_end.py::test_federation_robust_bit_identical_when_workers_arrive_out_of_order",
        "tests/fedference/test_federation_end_to_end.py::test_federation_timeouts_reject_boolean_values",
        "tests/fedference/test_federation_end_to_end.py::test_server_rejects_unknown_worker_and_response_queue_sets",
        "tests/fedference/test_federation_end_to_end.py::test_server_run_round_rejects_duplicate_worker_id",
        "tests/fedference/test_federation_end_to_end.py::test_server_run_round_times_out_when_a_worker_never_sends",
        "tests/fedference/test_federation_end_to_end.py::test_worker_receive_consensus_matches_server_broadcast",
        "tests/fedference/test_federation_health.py::<module>",
        "tests/fedference/test_federation_health.py::test_nonunit_positive_masses_are_bit_identical_across_every_boundary",
        "tests/fedference/test_federation_health.py::test_server_rich_round_retains_result_but_wire_payload_is_unchanged",
        "tests/fedference/test_transport_envelope.py::<module>",
        "tests/fedference/test_transport_envelope.py::test_config_and_legacy_robustness_conflicts_fail",
        "tests/fedference/test_transport_envelope.py::test_queue_server_dispatches_variational_configuration",
        "tests/test_token_sourcing.py::<module>",
        "tests/test_token_sourcing.py::test_federation_bit_identical_token_matches_direct_recomputation"
      ],
      "disposition": "owner-decision",
      "historical_reader": "none-supported-interface",
      "kind": "retained-interface",
      "owner": "breaking-53",
      "rationale": "Constructor is necessary. Migrate selector by keyword, not by deleting the class or constructor.",
      "replacement": "Remove only robustness compatibility argument; use config=AggregationConfig(...)",
      "scope": "Task 53",
      "status": "present",
      "symbol": "src/fedference/federation/server.py::FederationServer.__init__"
    },
    {
      "callsites": [
        "src/manuscript_vars/loaders.py::_federation_variables",
        "tests/fedference/test_federation_end_to_end.py::test_federation_bit_identical_to_inprocess",
        "tests/fedference/test_federation_end_to_end.py::test_federation_robust_bit_identical",
        "tests/fedference/test_federation_end_to_end.py::test_federation_robust_bit_identical_when_workers_arrive_out_of_order",
        "tests/fedference/test_federation_end_to_end.py::test_server_rejects_unknown_worker_and_response_queue_sets",
        "tests/fedference/test_federation_end_to_end.py::test_server_run_round_rejects_duplicate_worker_id",
        "tests/fedference/test_federation_end_to_end.py::test_server_run_round_times_out_when_a_worker_never_sends",
        "tests/fedference/test_federation_end_to_end.py::test_worker_receive_consensus_matches_server_broadcast",
        "tests/fedference/test_transport_envelope.py::test_queue_server_dispatches_variational_configuration",
        "tests/test_token_sourcing.py::test_federation_bit_identical_token_matches_direct_recomputation"
      ],
      "disposition": "owner-decision",
      "historical_reader": "none-supported-interface",
      "kind": "retained-interface",
      "owner": "breaking-53",
      "rationale": "Redundant consensus projection.",
      "replacement": "FederationServer.run_round_result(...).consensus",
      "scope": "Task 53",
      "status": "present",
      "symbol": "src/fedference/federation/server.py::FederationServer.run_round"
    },
    {
      "callsites": [
        "src/fedference/federation/__init__.py::<module>",
        "src/fedference/federation/socket_transport.py::PersistentReplayGuard",
        "src/fedference/federation/socket_transport.py::run_socket_round",
        "src/fedference/federation/socket_transport.py::run_socket_round_result",
        "tests/fedference/test_socket_transport.py::<module>",
        "tests/fedference/test_socket_transport.py::test_shared_replay_guard_rejects_round_id_reuse"
      ],
      "disposition": "owner-decision",
      "historical_reader": "none-supported-interface",
      "kind": "retained-interface",
      "owner": "breaking-53",
      "rationale": "Distinct lifecycle semantics, not an alias; never claim ephemeral state is restart-durable or multi-host secure.",
      "replacement": "ReplayGuard for explicitly ephemeral scope; PersistentReplayGuard for durable local scope",
      "scope": "Task 53",
      "status": "present",
      "symbol": "src/fedference/federation/socket_transport.py::ReplayGuard"
    },
    {
      "callsites": [],
      "disposition": "owner-decision",
      "historical_reader": "none-supported-interface",
      "kind": "retained-interface",
      "owner": "breaking-53",
      "rationale": "Private boolean projection hides findings; migrate caller predicate explicitly.",
      "replacement": "inspect_socket_replay(...).integrity_valid when that precise predicate is intended",
      "scope": "Task 53",
      "status": "present",
      "symbol": "src/fedference/federation/socket_transport.py::_validate_socket_replay"
    },
    {
      "callsites": [
        "src/fedference/federation/__init__.py::<module>",
        "src/fedference/federation/socket_transport.py::inspect_socket_replay",
        "tests/fedference/test_federation_health.py::<module>",
        "tests/fedference/test_federation_health.py::test_replay_uses_recorded_config_and_partial_flags_are_assertions"
      ],
      "disposition": "retain-history",
      "historical_reader": "protocol-v1",
      "kind": "retained-interface",
      "owner": "project-owner",
      "rationale": "Historical artifacts must remain interpretable with exact semantics. Unsupported versions still fail closed; missing runtime provenance remains explicitly unavailable.",
      "replacement": "retain supported decoder; versioned converter and loss detection required before any future retirement",
      "scope": "preservation contract",
      "status": "present",
      "symbol": "src/fedference/federation/socket_transport.py::aggregation_config_from_replay"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-boundaries",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 13",
      "status": "retired",
      "symbol": "src/fedference/federation/socket_transport.py::inspect_socket_replay(beliefs=)"
    },
    {
      "callsites": [
        "examples/03_federation_boundaries.py::<module>",
        "examples/03_federation_boundaries.py::main",
        "src/fedference/federation/__init__.py::<module>",
        "tests/fedference/test_federation_health.py::<module>",
        "tests/fedference/test_federation_health.py::test_socket_rich_result_and_legacy_dictionary_contract",
        "tests/fedference/test_socket_transport.py::<module>",
        "tests/fedference/test_socket_transport.py::test_auth_key_must_be_non_empty",
        "tests/fedference/test_socket_transport.py::test_authenticated_socket_round_preserves_bit_identity",
        "tests/fedference/test_socket_transport.py::test_every_worker_receives_the_same_broadcast",
        "tests/fedference/test_socket_transport.py::test_malformed_replay_worker_orders_fail_closed",
        "tests/fedference/test_socket_transport.py::test_persistent_replay_guard_rejects_reuse_after_reopen",
        "tests/fedference/test_socket_transport.py::test_rejects_empty_beliefs",
        "tests/fedference/test_socket_transport.py::test_reordering_agnostic_consensus_holds",
        "tests/fedference/test_socket_transport.py::test_replay_log_persists_and_reloads_from_disk",
        "tests/fedference/test_socket_transport.py::test_replay_log_reconstructs_consensus",
        "tests/fedference/test_socket_transport.py::test_replay_log_rejects_inconsistent_broadcast_digest",
        "tests/fedference/test_socket_transport.py::test_replay_log_rejects_tampered_belief_digest",
        "tests/fedference/test_socket_transport.py::test_replay_log_rejects_tampered_consensus_digest",
        "tests/fedference/test_socket_transport.py::test_replay_log_rejects_tampered_frame_size",
        "tests/fedference/test_socket_transport.py::test_replay_log_rejects_tampered_listen_metadata",
        "tests/fedference/test_socket_transport.py::test_replay_log_rejects_wrong_consensus",
        "tests/fedference/test_socket_transport.py::test_replay_rejects_duplicate_worker_events_and_config_tamper",
        "tests/fedference/test_socket_transport.py::test_replay_rejects_top_level_aggregate_config_tamper",
        "tests/fedference/test_socket_transport.py::test_replay_rejects_unknown_fields_and_out_of_order_events",
        "tests/fedference/test_socket_transport.py::test_round_id_and_timeout_types_fail_before_socket_use",
        "tests/fedference/test_socket_transport.py::test_save_socket_replay_creates_parent_directory",
        "tests/fedference/test_socket_transport.py::test_shared_replay_guard_rejects_round_id_reuse",
        "tests/fedference/test_socket_transport.py::test_single_worker_round",
        "tests/fedference/test_socket_transport.py::test_socket_round_accepts_localhost_alias",
        "tests/fedference/test_socket_transport.py::test_socket_round_is_bit_identical_to_in_process",
        "tests/fedference/test_socket_transport.py::test_socket_round_rejects_non_configuration_objects",
        "tests/fedference/test_socket_transport.py::test_socket_round_rejects_non_loopback_hosts",
        "tests/fedference/test_strict_shape_boundaries.py::<module>",
        "tests/fedference/test_strict_shape_boundaries.py::test_advertised_aggregation_and_federation_boundaries_reject_matrix_rows",
        "tests/fedference/test_transport_envelope.py::<module>",
        "tests/fedference/test_transport_envelope.py::test_config_and_legacy_robustness_conflicts_fail",
        "tests/fedference/test_transport_envelope.py::test_socket_round_carries_config_hash_and_variational_result",
        "tests/test_boundary_retirement.py::<module>",
        "tests/test_boundary_retirement.py::test_canonical_transport_figures_and_historical_corpus",
        "tests/test_cli_surface_failure_matrix.py::<module>",
        "tests/test_cli_surface_failure_matrix.py::test_cli_real_replay_reports_integrity_and_solver_health_separately",
        "tests/test_fedference_cli.py::<module>",
        "tests/test_fedference_cli.py::test_cli_replay_accepts_positive_nonunit_belief_masses",
        "tests/test_fedference_cli.py::test_cli_replay_can_require_nominal_solver_health",
        "tests/test_fedference_cli.py::test_cli_replay_maps_malformed_numeric_json_to_parser_errors",
        "tests/test_fedference_cli.py::test_cli_replay_maps_malformed_worker_ids_to_usage_errors",
        "tests/test_fedference_cli.py::test_cli_replay_reads_recorded_config_and_emits_structured_findings",
        "tests/test_fedference_cli.py::test_cli_replay_verifies_a_real_socket_round"
      ],
      "disposition": "owner-decision",
      "historical_reader": "schema-or-return-migration",
      "kind": "retained-interface",
      "owner": "breaking-53",
      "rationale": "Redundant dictionary/array return projection; protocol decoding remains a necessary current boundary.",
      "replacement": "run_socket_round_result(...).server_aggregation and .replay",
      "scope": "Task 53",
      "status": "present",
      "symbol": "src/fedference/federation/socket_transport.py::run_socket_round"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-boundaries",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 13",
      "status": "retired",
      "symbol": "src/fedference/federation/socket_transport.py::run_socket_round(beliefs=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-boundaries",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 13",
      "status": "retired",
      "symbol": "src/fedference/federation/socket_transport.py::run_socket_round_result(beliefs=)"
    },
    {
      "callsites": [
        "examples/03_federation_boundaries.py::<module>",
        "examples/03_federation_boundaries.py::main",
        "src/fedference/federation/__init__.py::<module>",
        "tests/fedference/test_federation_health.py::<module>",
        "tests/fedference/test_federation_health.py::test_nonconvergence_and_fallback_are_solver_findings_not_integrity_failures",
        "tests/fedference/test_federation_health.py::test_nonunit_positive_masses_are_bit_identical_across_every_boundary",
        "tests/fedference/test_federation_health.py::test_replay_uses_recorded_config_and_partial_flags_are_assertions",
        "tests/fedference/test_socket_transport.py::<module>",
        "tests/fedference/test_socket_transport.py::test_malformed_replay_worker_orders_fail_closed",
        "tests/fedference/test_socket_transport.py::test_replay_log_persists_and_reloads_from_disk",
        "tests/fedference/test_socket_transport.py::test_replay_log_reconstructs_consensus",
        "tests/fedference/test_socket_transport.py::test_replay_log_rejects_inconsistent_broadcast_digest",
        "tests/fedference/test_socket_transport.py::test_replay_log_rejects_tampered_belief_digest",
        "tests/fedference/test_socket_transport.py::test_replay_log_rejects_tampered_consensus_digest",
        "tests/fedference/test_socket_transport.py::test_replay_log_rejects_tampered_frame_size",
        "tests/fedference/test_socket_transport.py::test_replay_log_rejects_tampered_listen_metadata",
        "tests/fedference/test_socket_transport.py::test_replay_log_rejects_wrong_consensus",
        "tests/fedference/test_socket_transport.py::test_replay_rejects_duplicate_worker_events_and_config_tamper",
        "tests/fedference/test_socket_transport.py::test_replay_rejects_top_level_aggregate_config_tamper",
        "tests/fedference/test_socket_transport.py::test_replay_rejects_unknown_fields_and_out_of_order_events",
        "tests/fedference/test_socket_transport.py::test_socket_round_rejects_non_configuration_objects",
        "tests/test_boundary_retirement.py::<module>",
        "tests/test_boundary_retirement.py::test_canonical_transport_figures_and_historical_corpus"
      ],
      "disposition": "owner-decision",
      "historical_reader": "none-supported-interface",
      "kind": "retained-interface",
      "owner": "breaking-53",
      "rationale": "Do not collapse solver fallback/nonconvergence into transport-integrity failure during migration.",
      "replacement": "inspect_socket_replay with separate integrity_valid and numerical-health findings",
      "scope": "Task 53",
      "status": "present",
      "symbol": "src/fedference/federation/socket_transport.py::validate_socket_replay"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-boundaries",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 13",
      "status": "retired",
      "symbol": "src/fedference/federation/socket_transport.py::validate_socket_replay(beliefs=)"
    },
    {
      "callsites": [
        "src/fedference/federation/__init__.py::<module>",
        "src/fedference/federation/socket_transport.py::<module>",
        "src/fedference/federation/socket_transport.py::_worker",
        "src/fedference/federation/socket_transport.py::run_socket_round_result",
        "tests/fedference/test_transport_envelope.py::<module>",
        "tests/fedference/test_transport_envelope.py::test_protocol_envelope_rejects_duplicate_and_nonfinite_json_members",
        "tests/fedference/test_transport_envelope.py::test_protocol_envelope_rejects_unknown_version",
        "tests/fedference/test_transport_envelope.py::test_protocol_envelope_requires_an_exact_integer_version",
        "tests/fedference/test_transport_envelope.py::test_protocol_envelope_round_trip_and_digest_tamper_detection",
        "tests/test_boundary_retirement.py::<module>",
        "tests/test_boundary_retirement.py::test_canonical_transport_figures_and_historical_corpus",
        "tests/test_boundary_retirement.py::test_retired_runtime_alias_and_unknown_schema"
      ],
      "disposition": "retain-history",
      "historical_reader": "protocol-v1",
      "kind": "retained-interface",
      "owner": "project-owner",
      "rationale": "Historical artifacts must remain interpretable with exact semantics. Unsupported versions still fail closed; missing runtime provenance remains explicitly unavailable.",
      "replacement": "retain supported decoder; versioned converter and loss detection required before any future retirement",
      "scope": "preservation contract",
      "status": "present",
      "symbol": "src/fedference/federation/transport.py::deserialize_envelope"
    },
    {
      "callsites": [
        "src/fedference/federation/__init__.py::<module>",
        "src/fedference/federation/socket_transport.py::<module>",
        "src/fedference/federation/socket_transport.py::_worker",
        "src/fedference/federation/worker.py::FederationWorker.receive_consensus",
        "tests/fedference/test_federation_end_to_end.py::<module>",
        "tests/fedference/test_federation_end_to_end.py::test_result_serialization_round_trip",
        "tests/fedference/test_transport_envelope.py::<module>",
        "tests/fedference/test_transport_envelope.py::test_result_transport_rejects_malformed_or_non_probability_archives",
        "tests/test_boundary_retirement.py::<module>",
        "tests/test_boundary_retirement.py::test_canonical_transport_figures_and_historical_corpus"
      ],
      "disposition": "retain-history",
      "historical_reader": "protocol-v1",
      "kind": "retained-interface",
      "owner": "project-owner",
      "rationale": "Historical artifacts must remain interpretable with exact semantics. Unsupported versions still fail closed; missing runtime provenance remains explicitly unavailable.",
      "replacement": "retain supported decoder; versioned converter and loss detection required before any future retirement",
      "scope": "preservation contract",
      "status": "present",
      "symbol": "src/fedference/federation/transport.py::deserialize_result"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-boundaries",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "normalized_effective_weights",
      "scope": "Task 13",
      "status": "retired",
      "symbol": "src/fedference/federation/transport.py::serialize_result(agent_weights=)"
    },
    {
      "callsites": [
        "tests/fedference/test_review_hardening.py::test_cavity_and_update_factor_fail_closed_compatibility_paths"
      ],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "site_factor",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/generalized_bayes.py::cavity(factor=)"
    },
    {
      "callsites": [
        "tests/fedference/test_review_hardening.py::test_cavity_and_update_factor_fail_closed_compatibility_paths"
      ],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "global_posterior",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/generalized_bayes.py::cavity(posterior=)"
    },
    {
      "callsites": [
        "tests/fedference/test_review_hardening.py::test_generalized_bayes_validation_and_alpha_branches"
      ],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "tau",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/generalized_bayes.py::generalized_posterior(learning_rate=)"
    },
    {
      "callsites": [
        "tests/fedference/test_review_hardening.py::test_generalized_bayes_validation_and_alpha_branches"
      ],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "loss_by_state",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/generalized_bayes.py::generalized_posterior(loss_vec=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "new_global_posterior",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/generalized_bayes.py::update_factor(new_posterior=)"
    },
    {
      "callsites": [
        "tests/fedference/test_review_hardening.py::test_cavity_and_update_factor_fail_closed_compatibility_paths"
      ],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "old_site_factor",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/generalized_bayes.py::update_factor(old_factor=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "old_global_posterior",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/generalized_bayes.py::update_factor(old_posterior=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "symbol",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "normalized_effective_weights",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/hybrid.py::HybridAggregationResult.agent_weights"
    },
    {
      "callsites": [
        "tests/fedference/test_hybrid.py::test_hybrid_aggregate_canonical_inputs_and_retired_arguments"
      ],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/hybrid.py::hybrid_aggregate(beliefs=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "base_weights",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/hybrid.py::hybrid_aggregate(weights=)"
    },
    {
      "callsites": [
        "tests/fedference/test_hybrid.py::test_hybrid_canonical_inputs_and_retired_arguments"
      ],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/hybrid.py::hybrid_log_linear_pool(beliefs=)"
    },
    {
      "callsites": [
        "tests/fedference/test_hybrid.py::test_hybrid_canonical_inputs_and_retired_arguments"
      ],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "base_weights",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/hybrid.py::hybrid_log_linear_pool(weights=)"
    },
    {
      "callsites": [
        "tests/fedference/test_core_edges.py::test_rcce_deprecated_q_alias_is_explicit_and_unambiguous"
      ],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "q_loss",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/losses.py::rcce(q=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/pomdp.py::efe_policy_select(beliefs=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/server_theory.py::construct_orientation_witness(beliefs=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/server_theory.py::heuristic_weight_block(beliefs=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/server_theory.py::objective_weight_block(beliefs=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "symbol",
      "owner": "retirement-domain",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "d_equivalent_from_rank_biserial",
      "scope": "Task 12",
      "status": "retired",
      "symbol": "src/fedference/statistics.py::cohens_d_from_rank_biserial"
    },
    {
      "callsites": [
        "src/fedference_cli/_support.py::_write_evidence_run",
        "tests/test_cli_surface_failure_matrix.py::<module>",
        "tests/test_cli_surface_failure_matrix.py::test_support_validation_matrix_and_receipt_extraction",
        "tests/test_fedference_cli.py::<module>",
        "tests/test_fedference_cli.py::test_cli_facade_preserves_public_imports_and_delegates_by_responsibility",
        "tests/test_fedference_cli.py::test_report_fallbacks_summarizes_solver_health_without_pseudoreplication"
      ],
      "disposition": "owner-decision",
      "historical_reader": "none-supported-interface",
      "kind": "retained-interface",
      "owner": "breaking-54",
      "rationale": "Private facade reexport is unnecessary; canonical internal implementation remains.",
      "replacement": "fedference_cli._support._report_fallbacks internally; typed evidence for public callers",
      "scope": "Task 54",
      "status": "retired",
      "symbol": "src/fedference_cli/__init__.py::_report_fallbacks"
    },
    {
      "callsites": [
        "examples/01_minimal_aggregation.py::<module>",
        "examples/02_compare_aggregation_methods.py::<module>",
        "examples/03_federation_boundaries.py::<module>",
        "examples/04_cli_receipt_workflow.py::<module>",
        "examples/05_labeled_application.py::<module>",
        "examples/06_gnn_sentinel.py::<module>",
        "scripts/00_preflight.py::<module>",
        "scripts/01_run_invariants.py::<module>",
        "scripts/02_run_analysis.py::<module>",
        "scripts/_generate_api_docs.py::<module>",
        "scripts/_generate_api_docs.py::main",
        "scripts/analysis_stages.py::<module>",
        "scripts/bootstrap_gnn.py::<module>",
        "scripts/build_release.py::<module>",
        "scripts/ci_shards.py::<module>",
        "scripts/ci_shards.py::main",
        "scripts/emit_metadata.py::<module>",
        "scripts/generate_api_docs.py::<module>",
        "scripts/prepare_web_package.py::<module>",
        "scripts/record_pipeline_stage.py::<module>",
        "scripts/summarize_tokens.py::<module>",
        "scripts/validate_all.py::<module>",
        "scripts/validate_clean_checkout.py::<module>",
        "scripts/validate_mermaid.py::<module>",
        "scripts/validate_outputs.py::<module>",
        "scripts/validate_pipeline_freshness.py::<module>",
        "scripts/validate_rendered_surfaces.py::<module>",
        "scripts/validate_test_coverage.py::<module>",
        "scripts/validate_web_package.py::<module>",
        "scripts/z_generate_manuscript_variables.py::<module>",
        "scripts/zenodo_release.py::<module>",
        "src/analysis/__init__.py::<module>",
        "src/fedference_cli/__init__.py::<module>",
        "src/fedference_cli/__main__.py::<module>",
        "src/figures/graphical_abstract.py::<module>",
        "tests/analysis/test_workflow.py::<module>",
        "tests/analysis/test_workflow.py::test_main_prints_paths_and_returns_mapping",
        "tests/test_cli_surface_failure_matrix.py::<module>",
        "tests/test_cli_surface_failure_matrix.py::_expect_usage_error",
        "tests/test_cli_surface_failure_matrix.py::test_cli_clean_checkout_application_receipt_and_tamper_paths",
        "tests/test_cli_surface_failure_matrix.py::test_cli_executes_all_lightweight_research_adapters_and_tamper_fails",
        "tests/test_cli_surface_failure_matrix.py::test_cli_executes_real_torch_bnn_smoke_and_pilot_profiles",
        "tests/test_cli_surface_failure_matrix.py::test_cli_real_replay_reports_integrity_and_solver_health_separately",
        "tests/test_cli_surface_failure_matrix.py::test_verify_distinguishes_unrequested_source_and_invalid_checkout",
        "tests/test_fedference_cli.py::<module>",
        "tests/test_fedference_cli.py::test_cli_aggregate_nonconvergence_is_retained_and_policy_visible",
        "tests/test_fedference_cli.py::test_cli_aggregate_rejects_malformed_input_and_unsafe_destinations_before_write",
        "tests/test_fedference_cli.py::test_cli_aggregate_rejects_wrong_or_version_mismatched_project_root",
        "tests/test_fedference_cli.py::test_cli_aggregate_writes_canonical_application_artifacts_and_verifies",
        "tests/test_fedference_cli.py::test_cli_application_source_equivalence_is_checkout_path_independent",
        "tests/test_fedference_cli.py::test_cli_calibration_pilot_writes_a_verifiable_receipt",
        "tests/test_fedference_cli.py::test_cli_facade_preserves_public_imports_and_delegates_by_responsibility",
        "tests/test_fedference_cli.py::test_cli_installed_aggregate_allows_output_tree_in_unrelated_repo",
        "tests/test_fedference_cli.py::test_cli_list_is_machine_readable",
        "tests/test_fedference_cli.py::test_cli_refuses_committed_reviewer_output",
        "tests/test_fedference_cli.py::test_cli_rejects_nominal_solver_policy_for_research_receipt",
        "tests/test_fedference_cli.py::test_cli_replay_accepts_positive_nonunit_belief_masses",
        "tests/test_fedference_cli.py::test_cli_replay_can_require_nominal_solver_health",
        "tests/test_fedference_cli.py::test_cli_replay_maps_malformed_numeric_json_to_parser_errors",
        "tests/test_fedference_cli.py::test_cli_replay_maps_malformed_worker_ids_to_usage_errors",
        "tests/test_fedference_cli.py::test_cli_replay_reads_recorded_config_and_emits_structured_findings",
        "tests/test_fedference_cli.py::test_cli_replay_verifies_a_real_socket_round",
        "tests/test_fedference_cli.py::test_cli_server_theory_smoke_writes_and_verifies_receipt",
        "tests/test_fedference_cli.py::test_cli_strict_verification_rejects_dirty_receipt",
        "tests/test_fedference_cli.py::test_cli_validates_before_creating_output_directory",
        "tests/test_fedference_cli.py::test_confirmatory_profile_is_blocked_until_pilot_freezes_it",
        "tests/test_fedference_cli.py::test_nested_project_root_still_protects_reviewer_snapshot",
        "tests/test_model_validation_cli.py::<module>",
        "tests/test_model_validation_cli.py::test_ambiguous_family_and_missing_file",
        "tests/test_model_validation_cli.py::test_gnn_requires_explicit_root",
        "tests/test_model_validation_cli.py::test_native_model_rejection",
        "tests/test_model_validation_cli.py::test_native_model_validation_preserves_canonical_identity",
        "tests/test_model_validation_cli.py::test_native_text_and_explicit_format",
        "tests/test_release_preflight.py::test_render_receipt_cli_refuses_an_unvalidated_render_tree",
        "tests/test_release_preflight.py::test_render_receipt_cli_requires_producer_logs",
        "tests/test_zenodo.py::<module>"
      ],
      "disposition": "owner-decision",
      "historical_reader": "none-supported-interface",
      "kind": "retained-interface",
      "owner": "breaking-54",
      "rationale": "This is the current CLI boundary, not an alias kept only for obsolete callers.",
      "replacement": "fedference_cli.main remains the pyproject installed entrypoint",
      "scope": "Task 54",
      "status": "present",
      "symbol": "src/fedference_cli/__init__.py::main"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-boundaries",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "local_posteriors",
      "scope": "Task 13",
      "status": "retired",
      "symbol": "src/figures/belief_heatmap.py::generate_belief_heatmap(agent_beliefs=)"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "keyword",
      "owner": "retirement-boundaries",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "normalized_effective_weights",
      "scope": "Task 13",
      "status": "retired",
      "symbol": "src/figures/robust_influence_weights.py::generate_robust_influence_weights(agent_weights=)"
    },
    {
      "callsites": [],
      "disposition": "owner-decision",
      "historical_reader": "none-supported-interface",
      "kind": "retained-interface",
      "owner": "breaking-54",
      "rationale": "Old module facade duplicates the current package; migrate scripts/tests and remove the facade.",
      "replacement": "manuscript_vars public generate_variables/render_manuscript_tree/save_variables; owning internal modules for private helpers",
      "scope": "Task 54",
      "status": "present",
      "symbol": "src/manuscript_variables.py::__all__"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "reexport",
      "owner": "retirement-boundaries",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "_format_d_equivalent",
      "scope": "Task 13",
      "status": "retired",
      "symbol": "src/manuscript_variables.py::_format_cohens_d"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "symbol",
      "owner": "retirement-boundaries",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "_federation_demo_local_posteriors",
      "scope": "Task 13",
      "status": "retired",
      "symbol": "src/manuscript_vars/loaders.py::_federation_demo_beliefs"
    },
    {
      "callsites": [],
      "disposition": "remove-deprecated",
      "historical_reader": "none-runtime-alias",
      "kind": "symbol",
      "owner": "retirement-boundaries",
      "rationale": "Warned compatibility surface; migrate active callers to the named canonical replacement before removal. Preserve historical records.",
      "replacement": "_format_d_equivalent",
      "scope": "Task 13",
      "status": "retired",
      "symbol": "src/manuscript_vars/tokens.py::_format_cohens_d"
    }
  ],
  "owner_decisions": [
    {
      "id": "OD01",
      "symbol": "src/fedference/aggregation.py::aggregate",
      "disposition": "remove/migrate",
      "canonical": "aggregate_result(..., config=AggregationConfig(...)).consensus",
      "task": 52,
      "rationale": "Redundant array projection and method/kwargs dispatch; no distinct mathematics.",
      "implementation_status": "implemented"
    },
    {
      "id": "OD02",
      "symbol": "src/fedference/belief_sharing.py::share_round",
      "disposition": "remove/migrate",
      "canonical": "share_round_result; consume shared_posteriors/global_aggregation",
      "task": 52,
      "rationale": "Historical diagnostics projection duplicates rich result and invents NaN for unavailable metrics; retain typed None and explicit fields.",
      "implementation_status": "implemented"
    },
    {
      "id": "OD03",
      "symbol": "src/fedference/benchmark.py::run_tabular_benchmark",
      "disposition": "remove/migrate",
      "canonical": "load_tabular_csv(explicit_path) then run_benchmark(features, labels, ...); registered external loader for real data",
      "task": 52,
      "rationale": "Implicit bundled path is a redundant convenience route; preserve synthetic smoke dataset through explicit package-resource loading.",
      "implementation_status": "implemented"
    },
    {
      "id": "OD04",
      "symbol": "src/fedference/bnn_baseline_torch.py::BNN_BETA_DEFAULT",
      "disposition": "remove/migrate",
      "canonical": "fedference.bnn_defaults.BNN_BETA_DEFAULT",
      "task": 54,
      "rationale": "One torch-free default owner; backend references the defaults module without reexport.",
      "implementation_status": "pending"
    },
    {
      "id": "OD05",
      "symbol": "src/fedference/bnn_baseline_torch.py::BNN_HIDDEN_DIM_DEFAULT",
      "disposition": "remove/migrate",
      "canonical": "fedference.bnn_defaults.BNN_HIDDEN_DIM_DEFAULT",
      "task": 54,
      "rationale": "Duplicate reexport removed; value unchanged.",
      "implementation_status": "pending"
    },
    {
      "id": "OD06",
      "symbol": "src/fedference/bnn_baseline_torch.py::BNN_N_CLIENTS_DEFAULT",
      "disposition": "remove/migrate",
      "canonical": "fedference.bnn_defaults.BNN_N_CLIENTS_DEFAULT",
      "task": 54,
      "rationale": "Duplicate reexport removed; value unchanged.",
      "implementation_status": "pending"
    },
    {
      "id": "OD07",
      "symbol": "src/fedference/bnn_baseline_torch.py::BNN_N_STEPS_DEFAULT",
      "disposition": "remove/migrate",
      "canonical": "fedference.bnn_defaults.BNN_N_STEPS_DEFAULT",
      "task": 54,
      "rationale": "Duplicate reexport removed; value unchanged.",
      "implementation_status": "pending"
    },
    {
      "id": "OD08",
      "symbol": "src/fedference/bnn_baseline_torch.py::BNN_ROBUSTNESS_DEFAULT",
      "disposition": "remove/migrate",
      "canonical": "fedference.bnn_defaults.BNN_ROBUSTNESS_DEFAULT",
      "task": 54,
      "rationale": "Duplicate reexport removed; value unchanged.",
      "implementation_status": "pending"
    },
    {
      "id": "OD09",
      "symbol": "src/fedference/experiments/__init__.py::__all__",
      "disposition": "retain-current",
      "canonical": "Documented public study/configuration exports only; private helpers imported from their owning modules",
      "task": 54,
      "rationale": "A coherent study interface is meaningful; remove compatibility-only private exports, not the entire package interface.",
      "implementation_status": "pending"
    },
    {
      "id": "OD10",
      "symbol": "src/fedference/experiments/__init__.py::_soft_colony",
      "disposition": "remove/migrate",
      "canonical": "fedference.colonies.soft_colony",
      "task": 54,
      "rationale": "Unwarned equality alias has no distinct behavior.",
      "implementation_status": "pending"
    },
    {
      "id": "OD11",
      "symbol": "src/fedference/experiments/cross_study.py::summarize_cross_study",
      "disposition": "retain-current",
      "canonical": "Require validated matched-trial per_rate_summary; reject its absence",
      "task": 56,
      "rationale": "Summary is substantive; the old-profile fallback silently changes the independent unit/estimand and must disappear.",
      "implementation_status": "pending"
    },
    {
      "id": "OD12",
      "symbol": "src/fedference/experiments/diagnostics.py::run_influence_weights_report",
      "disposition": "retain-current",
      "canonical": "Current normalized_effective_weights report with exactly one field spelling",
      "task": 56,
      "rationale": "Report computation remains; remove duplicate agent_weights report key through explicit current schema migration.",
      "implementation_status": "pending"
    },
    {
      "id": "OD13",
      "symbol": "src/fedference/experiments/worlds.py::run_hierarchical_bmr",
      "disposition": "remove/migrate",
      "canonical": "run_hierarchical_surprise_control; hierarchical_surprise_control report/figure stems",
      "task": 56,
      "rationale": "Name the actual configured deterministic surprise-threshold lesion/control; preserve values and no-generalization boundary, not an old BMR alias.",
      "implementation_status": "pending"
    },
    {
      "id": "OD14",
      "symbol": "src/fedference/federation/process.py::run_multiprocess_round",
      "disposition": "remove/migrate",
      "canonical": "run_multiprocess_round_result(...).consensus; full diagnostics in server_aggregation",
      "task": 53,
      "rationale": "Redundant array wrapper; MultiprocessRoundResult.consensus is the canonical projection (process.py:45-48).",
      "implementation_status": "pending"
    },
    {
      "id": "OD15",
      "symbol": "src/fedference/federation/server.py::FederationServer.__init__",
      "disposition": "retain-current",
      "canonical": "Remove only robustness compatibility argument; use config=AggregationConfig(...)",
      "task": 53,
      "rationale": "Constructor is necessary. Migrate selector by keyword, not by deleting the class or constructor.",
      "implementation_status": "pending"
    },
    {
      "id": "OD16",
      "symbol": "src/fedference/federation/server.py::FederationServer.run_round",
      "disposition": "remove/migrate",
      "canonical": "FederationServer.run_round_result(...).consensus",
      "task": 53,
      "rationale": "Redundant consensus projection.",
      "implementation_status": "pending"
    },
    {
      "id": "OD17",
      "symbol": "src/fedference/federation/socket_transport.py::ReplayGuard",
      "disposition": "retain-current",
      "canonical": "ReplayGuard for explicitly ephemeral scope; PersistentReplayGuard for durable local scope",
      "task": 53,
      "rationale": "Distinct lifecycle semantics, not an alias; never claim ephemeral state is restart-durable or multi-host secure.",
      "implementation_status": "pending"
    },
    {
      "id": "OD18",
      "symbol": "src/fedference/federation/socket_transport.py::_validate_socket_replay",
      "disposition": "remove/migrate",
      "canonical": "inspect_socket_replay(...).integrity_valid when that precise predicate is intended",
      "task": 53,
      "rationale": "Private boolean projection hides findings; migrate caller predicate explicitly.",
      "implementation_status": "pending"
    },
    {
      "id": "OD19",
      "symbol": "src/fedference/federation/socket_transport.py::run_socket_round",
      "disposition": "remove/migrate",
      "canonical": "run_socket_round_result(...).server_aggregation and .replay",
      "task": 53,
      "rationale": "Redundant dictionary/array return projection; protocol decoding remains a necessary current boundary.",
      "implementation_status": "pending"
    },
    {
      "id": "OD20",
      "symbol": "src/fedference/federation/socket_transport.py::validate_socket_replay",
      "disposition": "remove/migrate",
      "canonical": "inspect_socket_replay with separate integrity_valid and numerical-health findings",
      "task": 53,
      "rationale": "Do not collapse solver fallback/nonconvergence into transport-integrity failure during migration.",
      "implementation_status": "pending"
    },
    {
      "id": "OD21",
      "symbol": "src/fedference_cli/__init__.py::_report_fallbacks",
      "disposition": "remove/migrate",
      "canonical": "fedference_cli._support._report_fallbacks internally; typed evidence for public callers",
      "task": 54,
      "rationale": "Private facade reexport is unnecessary; canonical internal implementation remains.",
      "implementation_status": "implemented"
    },
    {
      "id": "OD22",
      "symbol": "src/fedference_cli/__init__.py::main",
      "disposition": "retain-current",
      "canonical": "fedference_cli.main remains the pyproject installed entrypoint",
      "task": 54,
      "rationale": "This is the current CLI boundary, not an alias kept only for obsolete callers.",
      "implementation_status": "pending"
    },
    {
      "id": "OD23",
      "symbol": "src/manuscript_variables.py::__all__",
      "disposition": "remove/migrate",
      "canonical": "manuscript_vars public generate_variables/render_manuscript_tree/save_variables; owning internal modules for private helpers",
      "task": 54,
      "rationale": "Old module facade duplicates the current package; migrate scripts/tests and remove the facade.",
      "implementation_status": "pending"
    }
  ],
  "historical_decisions": [
    {
      "id": "HD01",
      "symbol": "src/fedference/evidence_contracts/run_schema.py::RunReceipt.from_dict",
      "disposition": "Keep the current run1.2 constructor; remove its live1.1 dispatch only after the full available old corpus is verified through `fedference_archive`. Missing runtime provenance remains unavailable and cannot be fabricated or presented as an equivalent current run.",
      "tasks": [
        55
      ],
      "implementation_status": "pending"
    },
    {
      "id": "HD02",
      "symbol": "fedference.federation.socket_transport.aggregation_config_from_replay",
      "disposition": "Retain-current: it interprets the single current wire-v1 replay configuration; no legacy alternate selector route.",
      "tasks": [
        53,
        55
      ],
      "implementation_status": "pending"
    },
    {
      "id": "HD03",
      "symbol": "fedference.federation.transport.deserialize_envelope",
      "disposition": "Retain-current: required current wire-v1 boundary, with strict version rejection.",
      "tasks": [
        53
      ],
      "implementation_status": "pending"
    },
    {
      "id": "HD04",
      "symbol": "fedference.federation.transport.deserialize_result",
      "disposition": "Retain-current: decode current wire-v1 `agent_weights` as normalized effective influence into canonical Python semantics. Never conflate it with raw exponent/base weights.",
      "tasks": [
        53
      ],
      "implementation_status": "pending"
    }
  ],
  "additional_retirements": [
    {
      "symbol": "src/fedference/aggregation.py::friston_belief_share",
      "replacement": "log_linear_pool",
      "task": 52,
      "status": "retired"
    },
    {
      "symbol": "src/fedference/belief_sharing.py::SharingDiagnostics",
      "replacement": "SharingRoundResult",
      "task": 52,
      "status": "retired"
    },
    {
      "symbol": "src/fedference/belief_sharing.py::share_round_result(method=)",
      "replacement": "config=AggregationConfig(method=...)",
      "task": 52,
      "status": "retired"
    },
    {
      "symbol": "src/fedference/belief_sharing.py::share_round_result(robustness=)",
      "replacement": "config=AggregationConfig(robustness=...)",
      "task": 52,
      "status": "retired"
    }
  ]
}
```
