| Module | Name | Kind | Summary |
|---|---|---|---|
| `analysis.report_schemas` | `BeliefQualityReport` | class | Top-level payload written to ``belief_quality.json``. |
| `analysis.report_schemas` | `BeliefSharingReport` | class | Top-level payload written to ``belief_sharing.json``. |
| `analysis.report_schemas` | `BnnRobustnessReport` | class | Top-level payload written to ``bnn_robustness.json``. |
| `analysis.report_schemas` | `BnnTorchOkReport` | class | Executed PyTorch complement payload written to ``bnn_torch.json``. |
| `analysis.report_schemas` | `BnnTorchSkippedReport` | class | Degradation payload when the PyTorch optional extra is unavailable. |
| `analysis.report_schemas` | `ComplexityScalingReport` | class | Top-level payload written to ``complexity_scaling.json``. |
| `analysis.report_schemas` | `ConditionalWorldReport` | class | Top-level payload written to ``conditional_world.json``. |
| `analysis.report_schemas` | `ContaminationGalleryCell` | class | One seed-aggregated contamination-gallery mechanism cell. |
| `analysis.report_schemas` | `ContaminationGalleryReport` | class | Top-level payload written to ``contamination_gallery.json``. |
| `analysis.report_schemas` | `CrossStudySummaryReport` | class | Top-level payload written to ``cross_study_summary.json``. |
| `analysis.report_schemas` | `DisjointFovWorldReport` | class | Top-level payload written to ``disjoint_fov_world.json``. |
| `analysis.report_schemas` | `EfeDecompositionReport` | class | Top-level payload written to ``efe_decomposition.json``. |
| `analysis.report_schemas` | `EmergenceReport` | class | Top-level payload written to ``emergence.json``. |
| `analysis.report_schemas` | `FigureDependencyContract` | class | Declared top-level report fields consumed by one figure generator. |
| `analysis.report_schemas` | `FigureMetadataEntry` | class | Per-figure metadata payload written inside ``figure_registry.json``. |
| `analysis.report_schemas` | `FigureRegistryPayload` | class | Top-level figure registry payload. |
| `analysis.report_schemas` | `HeuristicCharacterizationReport` | class | Top-level payload written to ``heuristic_characterization.json``. |
| `analysis.report_schemas` | `HierarchicalBmrReport` | class | Top-level payload written to ``hierarchical_bmr.json``. |
| `analysis.report_schemas` | `HierarchicalWorldReport` | class | Top-level payload written to ``hierarchical_world.json``. |
| `analysis.report_schemas` | `LanguageAcquisitionReport` | class | Top-level payload written to ``language_acquisition.json``. |
| `analysis.report_schemas` | `MovingWorldReport` | class | Top-level payload written to ``moving_world.json``. |
| `analysis.report_schemas` | `NLevel3WorldReport` | class | Top-level payload written to ``nlevel3_world.json``. |
| `analysis.report_schemas` | `ParameterRecoveryReport` | class | Top-level payload written to ``parameter_recovery.json``. |
| `analysis.report_schemas` | `ReportSchemaError` | class | Raised when a report or figure-registry payload violates its schema. |
| `analysis.report_schemas` | `ReviewGridReport` | class | Top-level payload written to ``robustness_review_grid.json``. |
| `analysis.report_schemas` | `RobustInfluenceWeightsReport` | class | Top-level payload written to ``robust_influence_weights.json``. |
| `analysis.report_schemas` | `RobustnessOnsetCell` | class | One pooled-method, seed-aggregated robustness-onset rate cell. |
| `analysis.report_schemas` | `RobustnessOnsetReport` | class | Top-level payload written to ``robustness_onset.json``. |
| `analysis.report_schemas` | `RobustnessSweepReport` | class | Top-level payload written to ``robustness_sweep.json``. |
| `analysis.report_schemas` | `SchemaDefinition` | class | Shallow top-level schema definition. |
| `analysis.report_schemas` | `VariationalAggregationReport` | class | Top-level payload written to ``variational_aggregation.json``. |
| `analysis.report_schemas` | `check_figure_contract` | function | Validate the declared report fields consumed by one figure generator. |
| `analysis.report_schemas` | `validate_report` | function | Validate one report or figure-registry payload before it is written. |
| `analysis.workflow` | `BnnTorchOptions` | class | Validated optional keyword arguments for the Torch complement. |
| `analysis.workflow` | `main` | function | Run the analysis pipeline and print every artifact path to stdout. |
| `analysis.workflow` | `run_analysis_pipeline` | function | Run every fedference experiment, write reports + figures, return paths. |
| `documentation` | `build_api_reference_markdown` | function | Return markdown API reference for Active Fedference. |
| `documentation` | `run_api_doc_generation` | function | Generate the glossary-style API index and the static API reference. |
| `experiment_config` | `ExperimentConfig` | class | Frozen Active Fedference parameters from ``config.yaml`` -> ``experiment:``. |
| `experiment_config` | `load_experiment_config` | function | Load Active Fedference parameters from ``manuscript/config.yaml``. |
| `fedference.agents` | `Observation` | class | One multi-factor observation for a sentinel. |
| `fedference.agents` | `Sentinel` | class | A single active-inference sentinel with a Markov-blanket partition. |
| `fedference.agents` | `SentinelEnsemble` | class | A colony of sentinels with private gaze and shared location/proximity/pose. |
| `fedference.aggregation` | `AggregationConfig` | class | Validated, serializable configuration shared by every aggregation adapter. |
| `fedference.aggregation` | `AggregationResult` | class | Consensus plus per-agent influence diagnostics. |
| `fedference.aggregation` | `AggregatorProtocol` | class | Callable server-side aggregation boundary used by federation adapters. |
| `fedference.aggregation` | `aggregate` | function | Convenience dispatch returning just the consensus pmf. |
| `fedference.aggregation` | `aggregate_result` | function | Canonical aggregation dispatcher returning consensus and diagnostics. |
| `fedference.aggregation` | `aggregation_free_energy` | function | Variational free energy minimized by :func:`variational_aggregate`. |
| `fedference.aggregation` | `log_linear_pool` | function | Product-of-experts consensus = Friston (2024) Eq. 7. |
| `fedference.aggregation` | `robust_aggregate` | function | Robustly fuse agent beliefs by iterative divergence-reweighting. |
| `fedference.aggregation` | `variational_aggregate` | function | Objective-backed conservative fusion on the stated finite-simplex free energy. |
| `fedference.aggregation_comparators` | `ComparatorResult` | class | Consensus and convergence diagnostics for an experimental comparator. |
| `fedference.aggregation_comparators` | `clr_geometric_median_pool` | function | Pool beliefs by a weighted geometric median in CLR coordinates. |
| `fedference.aggregation_comparators` | `linear_opinion_pool` | function | Weighted arithmetic mixture of categorical beliefs. |
| `fedference.bayesian_model_reduction` | `greedy_reduce` | function | Greedy multi-hypothesis structure learning by iterated model reduction. |
| `fedference.bayesian_model_reduction` | `hierarchical_reduce` | function | Score which levels of a hierarchical POMDP earn their structural keep. |
| `fedference.bayesian_model_reduction` | `log_beta` | function | Return ``lnB(a) = sum_k gammaln(a_k) - gammaln(sum_k a_k)``. |
| `fedference.bayesian_model_reduction` | `reduce` | function | Score a Dirichlet model reduction via the Beta-function free energy. |
| `fedference.bayesian_model_reduction` | `reduced_posterior` | function | Closed-form reduced posterior ``post + reduced_prior - prior``. |
| `fedference.belief_sharing` | `SharingDiagnostics` | class | Per-round outcome of belief sharing. |
| `fedference.belief_sharing` | `share_round` | function | Run one federated belief-sharing round over a shared factor. |
| `fedference.belief_updating` | `infer_states` | function | One-step variational posterior over hidden states (Friston Eq. 4). |
| `fedference.belief_updating` | `vfe` | function | Variational free energy ``F[q]`` for a belief ``qs`` (Friston 2024). |
| `fedference.benchmark` | `load_tabular_csv` | function | Load a numeric-feature + integer-``label`` CSV into ``(features, labels)``. |
| `fedference.benchmark` | `run_benchmark` | function | Federated tabular classification with contaminated clients. |
| `fedference.benchmark` | `run_external_benchmark_pack` | function | Run the preregistered three-dataset pack without pooling nested seeds. |
| `fedference.benchmark` | `run_external_dataset_benchmark` | function | Fetch, verify, and run one registered UCI benchmark. |
| `fedference.benchmark` | `run_tabular_benchmark` | function | Load the bundled synthetic tabular CSV (or a user CSV) and stress it. |
| `fedference.bnn_baseline` | `contaminate` | function | Flip a ``fraction`` of binary labels, biased toward high-leverage outliers. |
| `fedference.bnn_baseline` | `fed_gvi_logreg` | function | Run the federated logistic-regression baseline and return its result. |
| `fedference.bnn_baseline` | `make_blobs` | function | Synthetic 2-D, 2-class Gaussian blobs. |
| `fedference.bnn_baseline_torch` | `PointMassMLP` | class | Deterministic point-estimate MLP with a beta-loss FedGVI objective. |
| `fedference.bnn_baseline_torch` | `federated_bnn_round` | function | One round of federated point-mass MLP training. |
| `fedference.bnn_baseline_torch` | `run_bnn_torch_experiment` | function | Run the PyTorch point-mass MLP FedGVI complement end-to-end. |
| `fedference.bnn_fedgvi` | `DiagonalGaussian` | class | Normalized diagonal Gaussian posterior parameters. |
| `fedference.bnn_fedgvi` | `FedGVIServerState` | class | Prior, client sites, and round counter for resumable FedGVI updates. |
| `fedference.bnn_fedgvi` | `GaussianSiteFactor` | class | Possibly unnormalized client factor in Gaussian natural coordinates. |
| `fedference.bnn_fedgvi` | `load_server_checkpoint` | function | Load and validate a round-level server checkpoint. |
| `fedference.bnn_fedgvi` | `save_server_checkpoint` | function | Atomically write one round-level server checkpoint. |
| `fedference.bnn_variational_torch` | `VariationalMLP` | class | Mean-field variational MLP: diagonal-Gaussian ``q(w)`` over every weight. |
| `fedference.bnn_variational_torch` | `gaussian_kl_reference` | function | One-weight KL via the tested categorical-sibling :func:`gaussian_kl` — the independent reference the module's summed KL is checked against. |
| `fedference.calibration` | `CalibrationEpisode` | class | One independent world used only for hyperparameter calibration. |
| `fedference.calibration` | `CalibrationResult` | class | Frozen selected configuration and calibration provenance. |
| `fedference.calibration` | `CandidateScore` | class | Calibration score for one candidate configuration. |
| `fedference.calibration` | `calibrate_aggregation` | function | Select and freeze the highest-log-score candidate on calibration data. |
| `fedference.calibration` | `evaluate_locked_aggregation` | function | Evaluate a frozen configuration and reject calibration/evaluation overlap. |
| `fedference.colonies` | `healthy_colony` | function | Build a colony of healthy beliefs; optional jitter when ``rng`` is given. |
| `fedference.colonies` | `soft_colony` | function | Build ``n_agents`` soft healthy beliefs peaked on ``true_state``. |
| `fedference.complexity` | `ComplexityBenchmarkConfig` | class | Seeded, bounded grid for the machine scaling experiment. |
| `fedference.complexity` | `ComplexityEstimate` | class | Concrete dominant-work proxy for one operation and parameter setting. |
| `fedference.complexity` | `ComplexitySpec` | class | One source-bound asymptotic accounting row. |
| `fedference.complexity` | `complexity_catalog` | function | Return the immutable catalog of implementation-derived complexity rows. |
| `fedference.complexity` | `estimate_complexity` | function | Calculate a concrete dominant-work proxy for a catalogued operation. |
| `fedference.contamination` | `contaminate` | function | Corrupt a sentinel belief into an adversarial / miscalibrated report. |
| `fedference.continuous_recovery` | `conjugate_gaussian_posterior` | function | Closed-form Normal-Normal posterior for a mean with known ``obs_var``. |
| `fedference.continuous_recovery` | `recovery_residuals` | function | Off-corner recovery witness: the robust-vs-conjugate posterior gap as a function of ``beta``. |
| `fedference.continuous_recovery` | `robust_gaussian_posterior` | function | Density-power (beta) robust Normal-Normal posterior via weighted fixed point. |
| `fedference.dirichlet_learning` | `DirichletLearningResult` | class | Trajectory of a Dirichlet likelihood-learning run. |
| `fedference.dirichlet_learning` | `expected_likelihood` | function | Expected likelihood ``E[A] = a / sum_o(a)`` (column-normalized). |
| `fedference.dirichlet_learning` | `learn_likelihood` | function | Learn the likelihood ``A`` via conjugate Dirichlet updates (Eqs. 9-12). |
| `fedference.divergences` | `alpha_renyi_divergence` | function | Return FedGVI's Alpha-Rényi divergence. |
| `fedference.divergences` | `divergence` | function | Dispatch a named categorical divergence (``KLD``, ``RKL``, ``AR``, ``TV``). |
| `fedference.divergences` | `gaussian_alpha_renyi` | function | Return the Gaussian Alpha-Rényi divergence used by FedGVI. |
| `fedference.divergences` | `gaussian_kl` | function | Closed-form ``KL(N(mu_q, var_q) || N(mu_p, var_p))`` for 1-D Gaussians. |
| `fedference.divergences` | `gaussian_renyi` | function | Closed-form standard Rényi divergence between two 1-D Gaussians. |
| `fedference.divergences` | `kl_divergence` | function | Return ``KL(q || p) = sum_k q_k log(q_k / p_k)`` in nats (>= 0). |
| `fedference.divergences` | `renyi_divergence` | function | Return the standard Rényi divergence ``D_alpha(q || p)``. |
| `fedference.divergences` | `reverse_kl` | function | Return the reverse KL ``KL(p || q)`` — FedGVI's ``RKL`` client divergence. |
| `fedference.divergences` | `total_variation` | function | Return total-variation distance ``0.5 * sum_k |q_k - p_k|`` in [0, 1]. |
| `fedference.evidence` | `ArtifactRecord` | class | One output file bound into a run receipt. |
| `fedference.evidence` | `DatasetSpec` | class | Legally and byte-level reproducible external dataset declaration. |
| `fedference.evidence` | `ExperimentSpec` | class | Decision-complete declaration for one research experiment family. |
| `fedference.evidence` | `RunReceipt` | class | Content-bound receipt for one executed experiment profile. |
| `fedference.evidence` | `SourceReference` | class | Pinned scholarly or implementation source used by an experiment. |
| `fedference.evidence` | `canonical_sha256` | function | Hash a JSON-compatible value using canonical key and separator order. |
| `fedference.evidence` | `load_run_receipt` | function | Load and validate a JSON run receipt. |
| `fedference.evidence` | `make_artifact_record` | function | Create a receipt record for an existing file below ``root``. |
| `fedference.evidence` | `sha256_file` | function | Return the SHA-256 digest of ``path`` without loading it all into memory. |
| `fedference.evidence` | `verify_run_receipt` | function | Return exact provenance/artifact findings; empty means verified. |
| `fedference.evidence` | `write_run_receipt` | function | Atomically persist a canonical run receipt. |
| `fedference.expected_free_energy` | `EFETerms` | class | The four EFE terms for one policy, summed over its horizon. |
| `fedference.expected_free_energy` | `decompose` | function | Closed-form EFE decomposition for one policy over a categorical POMDP. |
| `fedference.expected_free_energy` | `preferred_outcomes` | function | Preferred-outcome distribution ``p(o) = softmax(C)`` from log-preferences ``C``. |
| `fedference.experiments.belief_sharing` | `run_belief_sharing` | function | Run the categorical belief-sharing source-mechanism analogue. |
| `fedference.experiments.belief_sharing` | `run_emergence` | function | Run the categorical BMR source-mechanism analogue related to Fig. 9. |
| `fedference.experiments.belief_sharing` | `run_language_acquisition` | function | Run one categorical language-learning trajectory. |
| `fedference.experiments.belief_sharing` | `summarize_language_acquisition` | function | Summarize language learning across independent seeded trajectories. |
| `fedference.experiments.complexity` | `run_complexity_scaling` | function | Run and return the source-bound complexity calculation and measurements. |
| `fedference.experiments.conditional_world` | `ConditionalScenario` | class | One preregistered world/target/observability/weighting cell. |
| `fedference.experiments.conditional_world` | `conditional_scenario_grid` | function | Return the source-owned finite grid before results are inspected. |
| `fedference.experiments.conditional_world` | `run_belief_quality_sensitivity` | function | Score naive and robust consensus beliefs on a fixed conditional subset. |
| `fedference.experiments.conditional_world` | `run_conditional_world_generalization` | function | Run the pre-registered world/target/observability attack grid. |
| `fedference.experiments.cross_study` | `summarize_cross_study` | function | Collect per-study federation benefit across multiple seeds. |
| `fedference.experiments.diagnostics` | `run_bnn_robustness_report` | function | BNN held-out accuracy vs label contamination for standard vs robust clients. |
| `fedference.experiments.diagnostics` | `run_efe_decomposition_report` | function | Closed-form EFE decomposition of one sentinel policy. |
| `fedference.experiments.diagnostics` | `run_influence_weights_report` | function | Server-side robust pooling influence weights on a contaminated colony. |
| `fedference.experiments.diagnostics` | `run_variational_aggregation_report` | function | Diagnostics for the objective-backed variational aggregator. |
| `fedference.experiments.gallery` | `run_contamination_gallery` | function | Seed-robust robustness verdict under each contamination *mechanism*. |
| `fedference.experiments.gallery` | `run_robustness_onset` | function | Per-mechanism robustness *onset*: the rate at which a pooled display robust member exceeds naive. |
| `fedference.experiments.heuristic_characterization` | `characterization_grid` | function | Run a small declared MAJ-1 scenario grid. |
| `fedference.experiments.heuristic_characterization` | `empirical_breakdown` | function | Measure the breakdown point of both server aggregators on the same colony. |
| `fedference.experiments.heuristic_characterization` | `numerical_influence_function` | function | Finite-difference empirical influence of one agent under ``robust_aggregate``. |
| `fedference.experiments.heuristic_characterization` | `run_heuristic_characterization` | function | JSON-serialisable characterization report: breakdown points plus a numerical influence sweep of one contaminating agent at the study settings. |
| `fedference.experiments.navigation` | `run_disjoint_fov_world` | function | Multi-agent moving world: each agent sees only ``fov_width`` consecutive positions. |
| `fedference.experiments.navigation` | `run_efe_navigation_test` | function | Compare EFE-guided vs. random movement combined with belief sharing. |
| `fedference.experiments.parameter_recovery` | `run_parameter_recovery` | function | Validate generative-model identifiability by fitting acuity from synthetic data. |
| `fedference.experiments.report_bundle` | `disjoint_fov_report` | function | Disjoint-FOV necessity contrast plus EFE-navigation multi-seed statistics. |
| `fedference.experiments.report_bundle` | `hierarchical_world_report` | function | Point estimate at ``seed`` plus multi-seed location-accuracy statistics. |
| `fedference.experiments.report_bundle` | `moving_world_report` | function | Moving-world report with multi-seed accuracy and EFE-vs-isolated statistics. |
| `fedference.experiments.report_bundle` | `nlevel3_world_report` | function | Point estimate at ``seed`` plus multi-seed 3-level location-accuracy statistics. |
| `fedference.experiments.review_grid` | `run_review_grid` | function | Run the expanded finite review grid with selection-free payloads. |
| `fedference.experiments.robustness` | `run_robustness_sweep` | function | Sweep contamination rate x divergence; earn the robust-vs-naive verdict. |
| `fedference.experiments.sensitivity` | `run_belief_sharing_sensitivity` | function | 2-D location-accuracy sweep over sensor acuity x colony size. |
| `fedference.experiments.sensitivity` | `run_hierarchical_sensitivity` | function | 2-D location-accuracy sweep for the hierarchical POMDP. |
| `fedference.experiments.worlds` | `run_3level_world` | function | Study 7 — 3-level hierarchical federation (L3=meta-context → L2=context → L1=location). |
| `fedference.experiments.worlds` | `run_hierarchical_bmr` | function | Hierarchical structure learning by Bayesian model reduction (companion to the N-level study). |
| `fedference.experiments.worlds` | `run_hierarchical_world` | function | Study 6 — hierarchical federation at L1 (location) and L2 (context). |
| `fedference.experiments.worlds` | `run_moving_world` | function | Moving-world federation: isolated vs communicating vs EFE-guided (V4). |
| `fedference.experiments.worlds` | `run_nlevel_world` | function | Study 7 variant — generic N-level hierarchical federation. |
| `fedference.external_data` | `ExternalDataset` | class | Parsed external dataset plus byte-level provenance. |
| `fedference.external_data` | `fetch_external_dataset` | function | Download-if-needed, verify, and parse a registered external dataset. |
| `fedference.external_data` | `load_dataset_archive` | function | Verify and parse one archive according to its declared registry spec. |
| `fedference.federation.process` | `run_multiprocess_round` | function | Run one real spawned-process federation round and return its consensus. |
| `fedference.federation.server` | `FederationServer` | class | Collects beliefs from n_workers, aggregates, broadcasts consensus. |
| `fedference.federation.socket_transport` | `PersistentReplayGuard` | class | SQLite-backed round-id guard that survives process restarts. |
| `fedference.federation.socket_transport` | `ReplayGuard` | class | In-memory guard against round-id reuse within one running process. |
| `fedference.federation.socket_transport` | `load_socket_replay` | function | Load a persisted digest-only socket replay log. |
| `fedference.federation.socket_transport` | `run_socket_round` | function | Run one federation round over real loopback TCP sockets. |
| `fedference.federation.socket_transport` | `save_socket_replay` | function | Persist a digest-only socket replay log as deterministic JSON. |
| `fedference.federation.socket_transport` | `validate_socket_replay` | function | Validate a digest-only replay, returning ``False`` for malformed input. |
| `fedference.federation.transport` | `ProtocolEnvelope` | class | Versioned metadata binding for a serialized federation payload. |
| `fedference.federation.transport` | `deserialize_belief` | function | Deserialize and validate one exact float64 categorical belief. |
| `fedference.federation.transport` | `deserialize_envelope` | function | Validate and unpack bytes produced by :func:`serialize_envelope`. |
| `fedference.federation.transport` | `deserialize_result` | function | Deserialize and validate a result serialized by :func:`serialize_result`. |
| `fedference.federation.transport` | `serialize_belief` | function | Lossless numpy float64 serialization of a 1-D pmf belief array. |
| `fedference.federation.transport` | `serialize_envelope` | function | Bind payload bytes to deterministic, versioned protocol metadata. |
| `fedference.federation.transport` | `serialize_result` | function | Serialize consensus and normalized influence losslessly. |
| `fedference.federation.worker` | `FederationWorker` | class | Sends a belief to the server and receives the consensus. |
| `fedference.generalized_bayes` | `cavity` | function | Return the cavity ``q_{-n}`` with a site factor ``t_n`` removed. |
| `fedference.generalized_bayes` | `generalized_posterior` | function | Closed-form generalized posterior over hidden states. |
| `fedference.generalized_bayes` | `softmax` | function | Numerically stable softmax returning a categorical pmf. |
| `fedference.generalized_bayes` | `update_factor` | function | Return the refreshed local factor ``t_i`` after a client update. |
| `fedference.hybrid` | `HybridAggregationResult` | class | Consensus and reweighting diagnostics for a hybrid aggregation round. |
| `fedference.hybrid` | `HybridBelief` | class | A categorical mixture with a Gaussian conditional per component. |
| `fedference.hybrid` | `hybrid_aggregate` | function | Aggregate hybrid beliefs with exact zero-robustness recovery. |
| `fedference.hybrid` | `hybrid_log_linear_pool` | function | Combine hybrid beliefs by a categorical log pool and Gaussian precision pool. |
| `fedference.hybrid_tracking` | `HybridTrackingConfig` | class | Deterministic task and observation settings. |
| `fedference.hybrid_tracking` | `run_hybrid_tracking` | function | Run one seeded closed-loop hybrid tracking episode. |
| `fedference.losses` | `beta_loss` | function | Density-power (beta) loss for a categorical likelihood. |
| `fedference.losses` | `loss_vector` | function | Return the per-state loss vector ``L(p(o|s), o)`` over hidden states ``s``. |
| `fedference.losses` | `nll` | function | Standard negative log-likelihood loss ``-log p(o | s)``. |
| `fedference.losses` | `rcce` | function | Robust categorical cross-entropy (generalized cross-entropy, GCE). |
| `fedference.pomdp` | `LayerSpec` | class | Specification for one level of an N-level hierarchical POMDP. |
| `fedference.pomdp` | `build_3level_world` | function | Construct a 3-level hierarchical POMDP (L3=meta-context → L2=context → L1=location). |
| `fedference.pomdp` | `build_hierarchical_world` | function | Construct a 2-level hierarchical POMDP (V2). |
| `fedference.pomdp` | `build_moving_world` | function | Construct a moving sentinel POMDP with disjoint fields-of-view (V4). |
| `fedference.pomdp` | `build_nlevel_world` | function | Construct a generic N-level hierarchical POMDP. |
| `fedference.pomdp` | `build_sentinel_world` | function | Construct the categorical sentinel POMDP specialization (Friston et al. 2024, Figs. 1/4). |
| `fedference.pomdp` | `efe_policy_select` | function | Select one information-seeking action per agent by expected free energy. |
| `fedference.pomdp` | `hierarchical_infer` | function | Alternating minimization for the 2-level hierarchical POMDP (V2). |
| `fedference.pomdp` | `nlevel_infer` | function | Alternating minimization for a generic N-level hierarchical POMDP. |
| `fedference.pomdp` | `normalise_columns` | function | Return a copy of ``matrix`` with every column renormalized to sum to 1. |
| `fedference.protocol_parity` | `ParityRow` | class | One source parameter, project value, and explicit disposition. |
| `fedference.protocol_parity` | `ProtocolParityMatrix` | class | Versioned comparison for one named source protocol. |
| `fedference.protocol_parity` | `fedgvi_bnn_parity_matrix` | function | Current parity disposition for the portable FedGVI BNN lane. |
| `fedference.protocol_parity` | `friston_protocol_parity_matrices` | function | Current explicit unknowns for Eq. 2 and source Figures 5, 7, and 9. |
| `fedference.research_registry` | `get_dataset_spec` | function | Return one declared dataset or raise a stable lookup error. |
| `fedference.research_registry` | `get_experiment_spec` | function | Return one declared experiment or raise a stable lookup error. |
| `fedference.research_registry` | `registry_fingerprint` | function | Return the canonical SHA-256 of :func:`registry_manifest`. |
| `fedference.research_registry` | `registry_manifest` | function | Return the canonical machine-readable research registry. |
| `fedference.scoring` | `brier_score` | function | Return per-observation multiclass Brier scores (smaller is better). |
| `fedference.scoring` | `categorical_log_score` | function | Return per-observation categorical log scores in nats. |
| `fedference.scoring` | `deterministic_score_controls` | function | Return oracle, uniform, and confidently-wrong control beliefs. |
| `fedference.scoring` | `expected_calibration_error` | function | Return equal-width expected calibration error. |
| `fedference.scoring` | `reliability_curve` | function | Return equal-width confidence/reliability bins for multiclass beliefs. |
| `fedference.scoring` | `summarize_scores` | function | Return mean primary/secondary scores and their declared sample count. |
| `fedference.scoring` | `validate_score_summary` | function | Fail closed if a serialized score summary loses its core fields. |
| `fedference.server_theory` | `ObjectiveOrientationWitness` | class | Forward-objective and reverse-heuristic weight blocks at one consensus. |
| `fedference.server_theory` | `construct_orientation_witness` | function | Construct the finite-simplex witness used by theory tests and reports. |
| `fedference.server_theory` | `heuristic_weight_block` | function | Reverse-KL weight update used by ``robust_aggregate``. |
| `fedference.server_theory` | `objective_weight_block` | function | Exact ``a`` update for the declared forward-KL block objective. |
| `fedference.statistics` | `bh_fdr` | function | Benjamini-Hochberg (1995) step-up FDR control over a family of p-values. |
| `fedference.statistics` | `bootstrap_ci` | function | Percentile bootstrap confidence interval for the mean of ``samples``. |
| `fedference.statistics` | `cohens_d_from_rank_biserial` | function | Deprecated compatibility alias for :func:`d_equivalent_from_rank_biserial`. |
| `fedference.statistics` | `d_equivalent_from_rank_biserial` | function | Return the monotone ``d``-equivalent display transform of rank-biserial ``r``. |
| `fedference.statistics` | `interpret_effect_size` | function | Label a secondary ``d``-equivalent display value by magnitude. |
| `fedference.statistics` | `minimum_detectable_effect` | function | Approximate a two-sided normal-mean minimum detectable effect. |
| `fedference.statistics` | `multiseed_summary` | function | Summary statistics for a vector of per-seed scalar values. |
| `fedference.statistics` | `paired_test` | function | Wilcoxon signed-rank test on the matched pairs ``(a_i, b_i)``. |
| `fedference.statistics` | `per_group_test` | function | Run :func:`paired_test` independently on each ``(a, b)`` group. |
| `fedference.statistics` | `power_analysis` | function | Statistical power of the paired Wilcoxon signed-rank test. |
| `fedference.statistics` | `rank_stability` | function | Mean Spearman rank correlation of agent rankings across folds. |
| `fedference.statistics` | `sample_size_for_power` | function | Smallest ``n`` whose paired-Wilcoxon power reaches ``target_power``. |
| `fedference.statistics` | `summary_statistics` | function | Summarize independent simulation replicates with MCSE and MDE. |
| `fedference.torch_bnn` | `TorchDeviceReceipt` | class | Resolved device and any explicit portability fallback. |
| `fedference.torch_bnn` | `bnn_protocol_profile` | function | Return a defensive copy of one declared BNN execution profile. |
| `fedference.torch_bnn` | `configure_torch_determinism` | function | Seed Torch and request deterministic algorithms where implemented. |
| `fedference.torch_bnn` | `resolve_torch_device` | function | Resolve ``cpu``, ``mps``, or ``auto`` without a silent fallback. |
| `fedference.trials` | `FlatVsNlevelMetrics` | class | Per-trial flat vs hierarchical location accuracy and free energy. |
| `fedference.trials` | `compare_flat_vs_nlevel` | function | Compare flat log-linear pooling against an N-level infer + pool path. |
| `fedference_cli` | `main` | function | CLI entry point; return a process-compatible status code. |
| `figures.aggregation_descent` | `generate_aggregation_descent` | function | Render the variational free energy ``F`` against descent iteration. |
| `figures.belief_heatmap` | `generate_belief_heatmap` | function | Render a colony's per-agent beliefs plus consensus as a heatmap. |
| `figures.belief_quality` | `generate_belief_quality` | function | Render control log scores and reliability curves from the score report. |
| `figures.bnn_robustness` | `generate_bnn_robustness` | function | Render held-out accuracy curves vs contamination for client configurations. |
| `figures.bounded_influence` | `generate_bounded_influence` | function | Render outlier influence vs divergence: variational curve vs naive line. |
| `figures.complexity_scaling` | `generate_complexity_scaling` | function | Generate the four-panel complexity/scaling diagnostic figure. |
| `figures.conditional_world` | `generate_conditional_world` | function | Render per-cell seed contrasts and finite-grid attack summaries. |
| `figures.contamination_gallery` | `generate_contamination_gallery` | function | Render naive vs pooled display robust accuracy per contamination mechanism. |
| `figures.cross_study_summary` | `generate_cross_study_summary` | function | Render native-unit facets for the nine-study summary. |
| `figures.descent_comparison` | `generate_descent_comparison` | function | Render the single-start vs multi-start free-energy descent on one axis. |
| `figures.disjoint_fov_world` | `generate_disjoint_fov_figure` | function | Generate the disjoint-FOV two-panel figure for the V4 manuscript. |
| `figures.efe_decomposition` | `generate_efe_decomposition` | function | Render the EFE identity as an additive stack and signed waterfall. |
| `figures.emergence_bmr` | `generate_emergence_bmr` | function | Render the redundant-vs-supported model-reduction free-energy contrast. |
| `figures.free_energy_comparison` | `generate_free_energy_comparison` | function | Render the free-energy gap between incommunicado and communicating colonies. |
| `figures.generative_model_schema` | `generate_generative_model_schema` | function | Generate the formal categorical generative-model schematic. |
| `figures.graphical_abstract` | `generate_graphical_abstract` | function | Generate the refreshed graphical abstract and manuscript cover. |
| `figures.graphical_abstract` | `main` | function | Generate the graphical abstract and print its output paths. |
| `figures.heuristic_breakdown` | `generate_heuristic_breakdown` | function | Render the influence + breakdown characterization of ``robust_aggregate``. |
| `figures.hierarchical_bmr` | `generate_hierarchical_bmr` | function | Render the per-level Bayesian-surprise comparison for two worlds. |
| `figures.hierarchical_pomdp` | `generate_hierarchical_pomdp` | function | 2x3 six-panel figure for the V2 hierarchical POMDP study (2-level + 3-level). |
| `figures.language_kl_decay` | `generate_language_kl_decay` | function | Render the language-acquisition KL learning curve. |
| `figures.message_passing` | `generate_message_passing` | function | Generate the claim-bounded belief-sharing message-passing schematic. |
| `figures.moving_world` | `generate_moving_world` | function | Three-panel bar chart for the moving sentinel world. |
| `figures.parameter_recovery` | `generate_parameter_recovery` | function | Generate a two-panel parameter-recovery figure for sensor acuity. |
| `figures.pomdp_loop` | `generate_pomdp_loop` | function | Generate the sentinel-world and active-inference loop schematic. |
| `figures.robust_influence_weights` | `generate_robust_influence_weights` | function | Render per-agent robust influence weights with saboteurs highlighted. |
| `figures.robustness_onset` | `generate_robustness_onset` | function | Render naive vs robust accuracy-vs-rate panels with onset markers. |
| `figures.robustness_review_grid` | `generate_robustness_review_grid` | function | Render conditional-cell contrasts and pooled rate-profile contrasts. |
| `figures.robustness_sweep` | `generate_robustness_sweep` | function | Render consensus-accuracy curves over the contamination-rate sweep. |
| `figures.sensitivity_heatmap` | `generate_sensitivity_heatmap` | function | 2-panel heatmap of federation accuracy gain over acuity x colony size. |
| `figures.system_overview` | `SystemOverviewData` | class | Numerical arrays drawn by the system-overview and cover figures. |
| `figures.system_overview` | `SystemOverviewMetadata` | class | Scalar provenance exported to manuscript tokens and the cover. |
| `figures.system_overview` | `adversarial_belief` | function | Peaked categorical at wrong state. |
| `figures.system_overview` | `build_data` | function | Return the derived schematic colony (posteriors, consensuses, weights). |
| `figures.system_overview` | `generate_system_overview` | function | Generate the three-panel system-overview figure (beliefs, weights, recovery). |
| `figures.system_overview` | `honest_belief` | function | Peaked categorical near true_state with mild uncertainty. |
| `figures.system_overview` | `naive_pool` | function | Log-linear pool: softmax of weighted sum of log-posteriors. |
| `figures.system_overview` | `robust_pool` | function | Return the heuristic robust consensus and normalized influence weights. |
| `invariants` | `InvariantResult` | class | Witness record for one numerical invariant. |
| `invariants` | `all_invariants` | function | Every fedference invariant the analysis report should display. |
| `invariants` | `check_efe_identity` | function | EFE decomposition identity: ``(risk+ambiguity)+(pragmatic+epistemic)==0``. |
| `invariants` | `check_kl_monotonicity` | function | KL(true A || learned A) declines monotonically as Dirichlet counts accrue. |
| `invariants` | `check_pmf_normalization` | function | Every fused consensus is a valid categorical pmf (non-negative, sums to 1). |
| `invariants` | `check_robust_recovers_naive` | function | ``robust_aggregate(robustness=0)`` is bit-identical to ``log_linear_pool``. |
| `invariants` | `write_invariants_report` | function | Run :func:`all_invariants` and serialise the witnesses to JSON. |
| `manuscript_vars.generate` | `generate_variables` | function | Resolve every manuscript token placeholder to a string value. |
| `manuscript_vars.render` | `render_manuscript_tree` | function | Hydrate manuscript tokens into a guarded ``output/manuscript`` tree. |
| `manuscript_vars.render` | `save_variables` | function | Persist *variables* as JSON for downstream rendering and debugging. |
| `project_paths` | `project_output_dirs` | function | Return common output directories for Active Fedference. |
| `project_paths` | `resolve_env_project_root` | function | Return the effective project root, honoring ``ACTIVE_FEDFERENCE_PROJECT_ROOT``. |
| `project_paths` | `resolve_project_root` | function | Resolve the project root directory from a loaded package or default to parent. |
| `publication.clean_checkout` | `CleanCheckoutReport` | class | Results of the clean-checkout tracking and import probe. |
| `publication.clean_checkout` | `inspect_clean_checkout` | function | Inspect Git cleanliness, required tracking, and package imports. |
| `publication.metadata` | `build_metadata` | function | Return ``{relative_path: exact_file_content}`` for every surface. |
| `publication.metadata` | `check_metadata` | function | Read-only drift check: return the surfaces whose on-disk content differs. |
| `publication.metadata` | `write_metadata` | function | Write every generated surface; return the paths written (explicit only). |
| `publication.pipeline_freshness` | `PipelineStageSpec` | class | Declared content boundary for one pipeline stage. |
| `publication.pipeline_freshness` | `record_pipeline_stage` | function | Record one successful stage after its inputs and outputs exist. |
| `publication.pipeline_freshness` | `validate_pipeline_freshness` | function | Return fail-closed freshness findings for the requested stage closure. |
| `publication.release_manifest` | `build_release` | function | Write ``output/release/`` and return the manifest mapping. |
| `publication.release_manifest` | `compute_fingerprint` | function | SHA-256 over the sorted ``(path, content-sha256)`` set of the inputs. |
| `publication.release_manifest` | `timestamp_from_source_date_epoch` | function | Convert ``SOURCE_DATE_EPOCH`` seconds to the canonical UTC timestamp. |
| `publication.release_manifest` | `validate_utc_timestamp` | function | Validate an optional canonical UTC timestamp. |
| `publication.release_manifest` | `verify_release` | function | Verify the exact artifact set and every digest in ``manifest.json``. |
| `publication.surface_validation` | `SurfaceValidation` | class | Aggregate result for generated reviewer-facing surfaces. |
| `publication.surface_validation` | `validate_rendered_surfaces` | function | Validate the combined manuscript, slide PDFs, logs, and HTML package. |
| `publication.web_package` | `WebPackageValidation` | class | Asset, reference, markup, and accessibility result for generated HTML. |
| `publication.web_package` | `mirror_web_figures` | function | Mirror every generated figure into the web package, removing stale files. |
| `publication.web_package` | `normalize_web_xrefs` | function | Replace raw citation/cross-reference markup with self-contained HTML links. |
| `publication.web_package` | `validate_web_package` | function | Check generated HTML assets, links, markup, and accessibility structure. |
