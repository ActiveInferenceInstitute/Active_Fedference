---
project: active_fedference
effort: E5
phase: verify
progress: 257/259
iteration: 47-single-machine-research-pilot-hardening
updated_by: iteration-47-2026-08-08
mode: ALGORITHM
started: 2026-06-24
updated: 2026-08-08
algorithm_config:
  forge_not_applicable: "direct local Codex review; external cross-vendor verification was not rerun and ISC-89 remains deferred"
  cato_truncates_after_background_launch: true
  advisor_completed: false
  blocker: "The source-bound changed-tree ladder is green, but the uncommitted overlay is not release evidence. Exact-commit fresh-clone reproduction, a structured cross-vendor verdict, confidentiality/license/author approval, DOI creation, and public release remain unclosed gates."
---

# Active Fedference — Ideal State Artifact

> Robust federated active inference: a discrete-categorical reimplementation of
> **FedGVI** (Federated Generalised Variational Inference; Mildner, Hamelijnck,
> Giampouras & Damoulas, ICML 2025 — arXiv:2502.00846) applied to the federated
> belief-sharing scenario of **Friston et al. (2024)**, *Federated inference and
> belief sharing* (Neurosci. Biobehav. Rev. 156:105500).

## Problem

Friston et al. (2024) model distributed cognition as *belief-sharing*: agents
broadcast posterior beliefs over shared latent states and reach a "hive-mind"
consensus through message passing (Eqs 6–8). Under the documented shared
support, posterior-log-potential, and fixed-weight bridge assumptions, the
project maps that message to a categorical log-linear product pool and its
interior-simplex reverse-KL minimizer. This is not a reconstruction of the
source's complete protocol. A sufficiently confident wrong broadcast can
dominate the project pool. FedGVI addresses a related robustness problem for
federated *learning* through generalized losses and divergences under stated
assumptions; its client result does not automatically certify a server
belief-pooling rule.
At project inception, the reviewed source bundle contained no artifact that
both exposed the categorical recovery relationship and evaluated the separate
client and server mechanisms under one reproducible evidence contract.

## Vision

A researcher runs one command and sees: the exact recovery identities at the
`robustness=0` / `β=0` / `α=1` limits, alongside reduced categorical
source-mechanism analogues and a robustness sweep where a contaminated sentinel
can separate declared methods under some attack geometries and reverse their
ordering under others — with paired statistics and negative controls keeping
the claims at the level of the observed estimand. The useful connection is
precise but bounded: the project log-linear pool is an executable recovery
corner of the categorical construction, and its relationship to Friston Eq. 7
is a posterior-log-potential specialization rather than a complete
message-passing reconstruction. The client theorem, server heuristic, and
objective-backed alternative retain different assumptions and guarantees.

## External or future lanes

- Exact source-scale CUDA BNN replication remains external. The optional Torch
  namespace and M4/MPS portability contracts do not turn a local profile into
  the source RTX-5090 experiment.
- The hybrid recovery fixture is not the planned continuous-control benchmark;
  the full hybrid and hierarchical task families remain gated by their declared
  recovery and preregistration criteria.
- Loopback sockets and processes are local transport evidence. Docker
  multi-node emulation is MAJ-4A; receipts from physically distinct hosts are
  MAJ-4B and remain external.
- Friston protocol work is a Python paper-constrained reconstruction unless
  every source parameter and routine is recovered. No MATLAB or Octave
  dependency is introduced.
- New linguistic theory remains out of scope. The reduced categorical
  source-mechanism analogue does not establish source-protocol identity.

## Principles

- **The limit proves project-local recovery only.** Every robust generalisation
  must reduce, in a named limit, to the exact non-robust project quantity.
  The categorical comparison to Friston Eq. 7 requires its separately stated
  posterior-log-potential bridge assumptions and is never a claim of complete
  source-protocol reconstruction. Recovery is tested, not asserted, and is
  never promoted into a theorem about behavior away from the limit.
  (Substrate-independent: this is the Deutsch hard-to-vary spine.)
- **Real numbers only.** No mocks; every test is a genuine computation on small
  categorical distributions or seeded simulations under this repository's
  explicit no-mocks policy.
- **Thin orchestrators.** All math lives in `src/fedference/`; scripts only do
  I/O, figures, and orchestration. Named domain boundary adapters own explicit
  evidence, data/cache, checkpoint, packaged-resource, and replay I/O.
- **Claims follow statistics.** No "robust beats naive" verdict is written
  before a paired test + multiple-comparison deflation has produced it
  (Algorithm Gate I).

## Constraints

- Python ≥3.10, typed, NumPy/SciPy only for the mathematical core; deterministic
  (seeded RNG). Torch is optional and absent from the default import graph.
- Two-layer architecture; `src/fedference/` may not import
  `infrastructure.*`. Explicit I/O stays in named package-boundary adapters and
  the declared manuscript layer contract records the separation.
- Project test coverage gate ≥90% on `src/`; infra contract unaffected.
- Standalone private repository at `docxology/active_fedference`, separate from
  the public template checkout; publication artifacts and metadata must preserve
  that boundary.
- Forge/Cato unavailable this session (codex = ChatGPT account → 401 on GPT-5.x);
  cross-vendor coverage substituted by inline RedTeam per Algorithm Rule 2a.

## Goal

Deliver a tested, reproducible Active Fedference research project that (1) reimplements the
FedGVI robust generalised-Bayes primitives in the discrete-categorical setting,
(2) proves the project-local recovery limits and documents the categorical
posterior-log-potential bridge to Friston et al. (2024) Eq. 7, rather than
claiming complete source-protocol recovery, and (3) extensively validates
robust federated active inference across reduced source-mechanism analogues and
a contaminated-sentinel robustness sweep with paired statistics, anchored by a
small FedGVI classification baseline.

## Criteria

### Core: FedGVI primitives (VERIFIED — tests/test_fedference_core.py, 14/14)
- [x] ISC-1: `divergences.kl_divergence` is 0 for identical pmfs and >0 otherwise.
- [x] ISC-2: `divergences.renyi_divergence(·,α)` → `kl_divergence` as α→1 (probe: test_renyi_recovers_kl).
- [x] ISC-3: `divergences.divergence` dispatches KLD/RKL/AR/TV by name.
- [x] ISC-4: `losses.rcce(·,q)` → `nll` as q→0 (probe: test_rcce_recovers_nll).
- [x] ISC-5: `losses.beta_loss(·,β)` → `nll` as β→0 (probe: test_beta_loss_recovers_nll).
- [x] ISC-6: `losses.rcce(q=1)` is bounded ≤1 where `nll` diverges (robustness witness).
- [x] ISC-7: `generalized_posterior(KLD, NLL)` equals closed-form prior×likelihood Bayes.
- [x] ISC-8: `cavity` then re-multiply restores the posterior (factor invertibility).
- [x] ISC-9: `log_linear_pool` equals the normalised product-of-experts. Under
  the documented categorical posterior-log-potential assumptions, it is a
  specialization of Friston Eq. 7; it is not a complete source-protocol
  reconstruction.
- [x] ISC-10: `robust_aggregate(robustness=0)` is bit-identical to `log_linear_pool` (CENTRAL identity).
- [x] ISC-11: `robust_aggregate(robustness>0)` raises true-state mass above naive and min-weights the liar.
- [x] ISC-12: `share_round(exclude_self=True)` gives per-agent distinct consensus (sensory attenuation).
- [x] ISC-13: robust `share_round` beats naive on accuracy AND surprise under a contaminated sentinel.
- [x] ISC-14: `loss_vector` maps an `(n_o,n_s)` A-tensor + outcome to a per-state loss vector.

### Active-inference machinery (PENDING — fan-out workflow)
- [x] ISC-15: `pomdp.build_sentinel_world()` returns A/B/C/D for Friston's location×proximity×pose×gaze factors with normalised columns. Probe: `pytest tests/test_pomdp.py`.
- [x] ISC-16: `belief_updating.infer_states` softmax message passing matches a hand-computed 1-step posterior. Probe: test_belief_updating.
- [x] ISC-17: `dirichlet_learning.learn_likelihood` KL-to-target is monotone-decreasing to <1e-2 in the categorical source-mechanism analogue related to Friston Fig. 7. Probe: test_dirichlet_learning.
- [x] ISC-18: `dirichlet_learning` honours the η forgetting hyperprior (Eq 12): total counts saturate at η. Probe: test_dirichlet_forgetting.
- [x] ISC-19: `expected_free_energy.decompose` satisfies risk+ambiguity == −(pragmatic+epistemic) to 1e-9 (the EFE identity). Probe: test_efe_identity.
- [x] ISC-20: `bayesian_model_reduction.reduce` ΔF matches the Beta-function formula (Eq 13) on a known reduced prior. Probe: test_bmr.
- [x] ISC-21: Anti: no active-inference module imports `infrastructure.*` (layer_contract). Probe: grep + check_template_drift.

### Federated active-inference ensemble & experiments (PENDING — fan-out workflow)
- [x] ISC-22: `agents.SentinelEnsemble` keeps gaze private and location/proximity/pose shared (Markov-blanket separation). Probe: test_agents.
- [x] ISC-23: `experiments.run_belief_sharing` implements the reduced categorical belief-sharing analogue related to Friston Fig. 5: communicating agents reach lower mean free energy than incommunicado. Probe: test_exp_belief_sharing.
- [x] ISC-24: `experiments.run_language_acquisition` provides the reduced categorical KL-decline analogue related to Fig. 7; seed is the independent aggregation unit, not a claim of exact source-protocol replication. Probe: test_exp_language.
- [x] ISC-25: `experiments.run_emergence` provides the categorical BMR diagnostic related to Fig. 9, with no claim of source-protocol identity. Probe: test_exp_emergence.
- [x] ISC-26: `contamination.contaminate` produces a confidently-wrong / label-noised broadcast at a configurable rate. Probe: test_contamination.
- [x] ISC-27: robustness sweep: across {KLD, RKL, AR(α), β, γ/rcce} the naive consensus accuracy degrades with contamination rate while ≥1 robust member stays above a threshold. Probe: test_robustness_sweep.
- [x] ISC-28: `statistics.paired_test` (Wilcoxon signed-rank) on robust-vs-naive accuracy across seeds returns p and effect size. Probe: test_statistics.
- [x] ISC-29: `statistics.bh_fdr` deflates the family of divergence comparisons (Gate I). Probe: test_statistics_fdr.
- [x] ISC-30: the "robust beats naive" verdict in the manuscript is emitted by the stats module, never hardcoded. Probe: test_manuscript_variables (token provenance).

### BNN / classification baseline (PENDING — fan-out workflow)
- [x] ISC-31: `bnn_baseline.fed_gvi_logreg` runs a small NumPy mean-field logistic-regression FedGVI on a synthetic 2-class task, deterministic under seed. Probe: test_bnn_baseline.
- [x] ISC-32: under label contamination the β/rcce baseline retains higher held-out accuracy than the NLL/KLD baseline (anchors the paper's qualitative result). Probe: test_bnn_robustness.

### Project contract (PENDING — fan-out workflow + my edits)
- [x] ISC-33: `uv run --extra dev pytest tests/` passes with project coverage ≥90% on `src/`. Probe: pytest --cov.
- [x] ISC-34: `scripts/02_run_analysis.py`-equivalent thin orchestrator runs experiments and prints output paths. Probe: bash run.
- [x] ISC-35: `manuscript/config.yaml` identifies the project (title/authors/keywords) and `manuscript_variables` hydrates every {{TOKEN}}. Probe: test_manuscript_variables.
- [x] ISC-36: top-level `README.md`/`AGENTS.md`/`TODO.md`/`STANDALONE.md` describe Active Fedference, not the optimization template. Probe: grep absence of "optimization template".
- [x] ISC-37: Anti: project lives in its own standalone git repository (``docxology/active_fedference``); it is NOT part of the template repo's git tree. Probe: ``git remote -v`` from the project root shows ``docxology/active_fedference``; ``git check-ignore`` from the template checkout confirms the path is not tracked there.

### Iteration 2 — comprehensive polish (PENDING — fan-out workflow)
- [x] ISC-38: every display equation across the manuscript carries a `{#eq:...}` label and is cross-referenced at least once via `[@eq:...]`. Probe: grep label-count == reference-coverage.
- [x] ISC-39: key results are stated as numbered formalisms — `\newtheorem` Definition/Theorem/Lemma/Proposition environments defined in preamble.md and used for the generalised posterior, the three recovery limits, and the central aggregation identity. Probe: grep `\begin{theorem}`/`\begin{definition}` in manuscript + `\newtheorem` in preamble.
- [x] ISC-39.1: the three recovery limits (β→0, α→1, robustness→0) appear as numbered Theorems whose statements match the tested ISCs 5/2/10. Probe: grep + cross-read.
- [x] ISC-40: every H2 subsection in the methodology/results carries a `{#sec:...}` label. Probe: grep H2 vs label count.
- [x] ISC-41: ≥6 figures are generated by `src/figures/` AND referenced via `[@fig:...]` with complete captions (axes, n, error bars, test). Probe: count generators, `[@fig:` refs, caption lines.
- [x] ISC-42: every `[@fig:...]`/`[@tbl:...]`/`[@eq:...]` reference resolves to a defined label (no dangling cross-refs). Probe: set-difference grep.
- [x] ISC-43: each referenced figure PNG is produced when `scripts/02_run_analysis.py` runs (output/figures/ contains every referenced file). Probe: run pipeline + ls.
- [x] ISC-44: `src/figures` gains generators for language-KL-decay, emergence-BMR, EFE-decomposition, robust-influence-weights, and BNN-robustness (beyond the original 3), each with a test. Probe: pytest tests/figures.
- [x] ISC-45: `statistics.py` gains bootstrap confidence intervals, raw (pre-FDR) p-values, per-rate tests, and a standardized effect-size (Cohen's-d-equivalent) — each tested. Probe: pytest tests/.../test_statistics.
- [x] ISC-46: every new statistic surfaces as a `{{TOKEN}}` emitted by `manuscript_variables.generate_variables`; no hardcoded number in prose. Probe: token cross-check test.
- [x] ISC-47: the manuscript is MORE MODULAR — methodology/results split into focused per-topic / per-study section files, render order preserved by numeric prefixes. Probe: file count + ordered prefixes.
- [x] ISC-48: `tests/` mirrors `src/` structure — `tests/fedference/test_<m>.py` for each `src/fedference/<m>.py`, `tests/figures/`, `tests/analysis/`, plumbing tests at root; every mirror dir has `__init__.py`. Probe: tree-diff src vs tests.
- [x] ISC-49: every script in `scripts/` is a strict thin orchestrator — imports from `src/`, contains no domain logic (no numpy/scipy math, no algorithm). Probe: grep for math ops / ast check + line budget.
- [x] ISC-50: `TODO.md` has all completed items removed and is rescoped into deeply-specified minor / medium / major upcoming work. Probe: read; no `[x]`-done-as-open; three tiers present.
- [x] ISC-51: preamble gains caption styling + theorem environments; aesthetics improved (titleformat/caption/booktabs). Probe: grep preamble.
- [x] ISC-52: Anti: no enhancement weakens the locked core or drops coverage below 90%; full suite stays green. Probe: pytest --cov.
- [x] ISC-53: Anti: the two-robustness-axes honesty is preserved — no new figure/stat/claim lets the aggregation heuristic inherit the β/rcce guarantees. Probe: read manuscript scope section.

### Iteration 3 — IMRAD, American English, community framing (PENDING — fan-out workflow)
- [x] ISC-54: All manuscript prose is American English; a British-spelling scan over `manuscript/[0-9]*.md` returns 0 (references.bib titles exempt). Probe: grep British wordlist.
- [x] ISC-55: `src/`, `tests/`, `scripts/` are American English; suite stays green after conversion. Probe: grep + pytest (DONE pre-workflow: 91+17+27 subs, 395 green).
- [x] ISC-56: Framing is community-positioned — the introduction/related-work state what the active-inference community AND the federated-learning / robust-Bayes literature have and have NOT done; no single author is name-dropped as the spine. Probe: read; count distinct cited threads ≥8; Friston-mentions are proportionate.
- [x] ISC-57: "Yes, and" — prior work is extended generatively, not dismissed. Probe: read related-work for build-on (not just critique) framing.
- [x] ISC-58: "Show, not tell" — every quantitative claim is backed by a referenced figure/table/equation/token, not a bare adjective. Probe: results sections cite a `[@fig:/@tbl:]` per claim.
- [x] ISC-59: Comprehensive IMRAD across many modular files — Introduction, Methods, Results, Discussion each split into multiple numbered section files (≥24 section files total). Probe: ls manuscript/[0-9]*.md | wc.
- [x] ISC-60: Every equation has `{#eq:}` + ≥1 `[@eq:]` ref; every figure `{#fig:}` + `[@fig:]`; every table `{#tbl:}` + `[@tbl:]`. Probe: label-vs-ref set diff == 0 dangling.
- [x] ISC-61: Zero hardcoded data/config numerals in prose — every number is a `{{TOKEN}}`; math constants in `$...$` exempt. Probe: bare-numeral scan over rendered output/manuscript == 0.
- [x] ISC-62: All config values are auto-injected tokens emitted by `generate_variables` (n_agents, rates, seed, thresholds, sample sizes). Probe: token coverage test.
- [x] ISC-63: Methods deepened — explicit federation protocol, generative-model spec, and a sample-size / statistical-power justification section. Probe: read 12_methods_experimental_design + grep power token.
- [x] ISC-64: Sample sizes increased and justified (n_seeds, n_trials) with a reported power analysis token; analysis pipeline runtime stays bounded. Probe: config + statistics.power_analysis test.
- [x] ISC-65: references.bib expanded with real community citations (pymdp, RxInfer, FedAvg, partitioned VI, collective active inference, structure learning, GVI) — no fabricated entries. Probe: key count ≥18; spot-check.
- [x] ISC-66: The combined PDF still builds (no missing LaTeX package; preamble stays install-free). Probe: run the template repository's `scripts/pipeline/stage_03_render.py` for `working/active_fedference`; combined PDF present.
- [x] ISC-67: Anti: no hardcoded number sneaks back; the token cross-check test fails the build on any unresolved `{{`. Probe: grep output/manuscript.
- [x] ISC-68: Anti: the two-robustness-axes honesty survives the rewrite (server aggregation = heuristic; β/rcce = FedGVI-faithful). Probe: read limitations section.
- [x] ISC-69: Anti: coverage stays ≥90% and all tests green after backend deepening. Probe: pytest --cov.
- [x] ISC-70: Figures/tables carry complete captions (axes, n, error bars/CI, test) — show-not-tell at the caption level. Probe: read figure embeds.

### Iteration 15 — injection audit + visualization/cover review (2026-07-06)
- [x] ISC-72: token-provenance gate scans ALL manuscript sections including `S[0-9]*.md` supplements (currently `[0-9]*.md` only). Probe: read glob in test_token_provenance.py + gate green.
- [x] ISC-73: token-table arity gate likewise scans `S[0-9]*.md`. Probe: read glob in test_token_tables.py + gate green.
- [x] ISC-74: `BELIEF_SHARING_ACUITY` token is derived from the single source-of-truth constant used by `run_belief_sharing`, not a duplicated `0.55` literal. Probe: grep generate.py imports the constant; call-site sweep shows one definition.
- [x] ISC-75: `SYSTEM_OVERVIEW_METADATA` accuracy/contamination percentages are COMPUTED from the figure's own pooled beliefs (true-state mass), never hand-typed. Probe: read system_overview.py; a test asserts pct == round(100*consensus[TRUE_STATE]).
- [x] ISC-76: `ISC_EFE_TOLERANCE` binds to a shared code constant consumed by the EFE identity test, not a free string. Probe: grep constant def + import in both sites.
- [x] ISC-77: no remaining hand-typed data numeral in any `src/manuscript_vars/*.py` token assignment (excluding format widths, N/A sentinels, config-metadata defaults). Probe: rg sweep clean.
- [x] ISC-78: full-suite green after gate widening — any S-file offender surfaced by ISC-72/73 is tokenized. Probe: pytest.
- [x] ISC-79: every figure PNG regenerated fresh at HEAD before visual audit (stale-PNG anti-pattern). Probe: pipeline run log + mtime check.
- [x] ISC-80: all ~21 figures visually reviewed against their manuscript captions; every finding (overlap, illegibility, color, caption-pixel mismatch) either fixed or explicitly dispositioned. Probe: review table in Verification.
- [x] ISC-81: figure improvements re-render clean; figure tests + caption gate + palette gate stay green. Probe: pytest tests/figures + gates.
- [x] ISC-82: cover image (`graphical_abstract.py` → `manuscript/cover_image.png`) reviewed at render size and improved (contrast, font size, layout); regenerated deterministically by the pipeline. Probe: fresh render + visual read + config.yaml cover binding intact.
- [x] ISC-83: combined PDF rebuilds with improved figures + cover; no LaTeX errors; cover appears on title page. Probe: render log + pdftoppm page-1 visual.
- [x] ISC-84: Anti: no figure "improvement" changes any reported number or scientific claim — data values in reports/tokens are untouched by figure-only edits. Probe: manuscript_variables.json diff pre/post limited to timestamps/artifact counts.
- [x] ISC-85: Anti: no token test is circular (token asserted equal to the same literal it copies) — new provenance bindings chain token → shared constant → consuming code. Probe: read new tests.
- [x] ISC-86: Anti: audit of PNGs happens only on fresh renders (no stale-PNG false positives). Probe: ISC-79 ordering in transcript.
- [x] ISC-87: ruff + mypy clean after all edits. Probe: ruff check + mypy exit 0.
- [x] ISC-88: coverage stays ≥90% on src/. Probe: pytest --cov.
- [DEFERRED-VERIFY] ISC-89: Advisor + cross-vendor (Forge; Cato best-effort) run against the final artifact set at VERIFY. Probe: logged invocations + verdicts in Verification. STATUS: Advisor ran (gaps surfaced → all addressed); Forge Batch A ran (verdict `concerns` → both findings fixed; items 2/3/5/9 not refuted); Forge Batch B (items 4/6/7/8: dirichlet consumers, overclaim sweep, three-axes sweep, green-by-construction check on new tests) BLOCKED by codex quota ("You've hit your usage limit... try again at 8:22 PM") — follow-up: re-run staged prompt `scratchpad/forge_audit_B.md` as `fedference-audit-B` after 20:22 2026-07-06. In-family coverage of those surfaces exists (methods + claims audit lanes); only the cross-vendor guarantee is deferred.
- [x] ISC-90: every `output/reports/*.json` consumed by a token loader is regenerated by the current pipeline run (no stale-orphan reports feeding tokens). Probe: mtime sweep after fresh `02_run_analysis.py`.
- [x] ISC-91: targeted scan for hand-typed plain integers and spelled-out numbers matching config values ("6 agents", "thirty seeds") in manuscript prose returns 0 undispositioned hits. Probe: rg sweep + review.
- [x] ISC-92: `manuscript/config.yaml` free-text fields carry no numeric scientific claims (config is not token-substituted). Probe: read config.yaml.
- [x] ISC-93: cover + representative figures verified at rendered PDF page scale (pdftoppm), not only standalone PNG. Probe: page raster reads.

### Iteration 16 — comprehensive claims/docs/methods/viz review (2026-07-06, this session)
- [x] ISC-94: every CRITICAL/MAJOR finding of the four-lane audit (methods-vs-code, docs-vs-repo, claims-vs-reports, figure-visual) is fixed or explicitly dispositioned. Probe: findings lists vs applied-fix reports in Verification.
- [x] ISC-95: `fdr_alpha` config knob is consumed by the executed sweep (not decorative) and the manuscript token reads the executed report value. Probe: test_sweep_fdr_alpha_is_consumed_not_decorative + STATISTICS_FDR_ALPHA from sweep report.
- [x] ISC-96: `sample_size_for_power` floors at the minimal feasible Wilcoxon n (5 one-sided at α=0.05), never an infeasible n=1. Probe: test_sample_size_for_power_saturated_effect_returns_feasible_floor.
- [x] ISC-97: Dirichlet `kl_trajectory` records the post-final-batch point (num_steps+1 points); `final_kl` is truly after the final batch. Probe: test_language_acquisition_* length asserts + LANGUAGE_FINAL_KL from post-final state.
- [x] ISC-98: `_project_root(None)` resolves to the repo root, not `src/`; a missing stage_timings.json degrades tokens to N/A and writes nothing (no fabricated 0.0 durations). Probe: test_default_project_root_is_repo_root_not_src + test_missing_stage_timings_degrades_to_na_and_writes_nothing.
- [x] ISC-99: Anti: "bit-identical" appears in the manuscript only where the residual is exactly zero (aggregate identity, federation transport, same-code-path rounds) — the posterior recovery is described as machine-precision. Probe: pdftotext context sweep of all 12 occurrences.

### Iteration 18 — real slices of every MAJOR-tier item (2026-07-06)
Each MAJOR is a multi-week research item; this iteration lands a genuine, tested, non-overclaiming SLICE of each and keeps the MAJOR itself OPEN. A 6-designer + 6-adversarial-challenge feasibility workflow classified all six as "partial, low faking-risk" (reproducing the key numbers) before any implementation.
- [x] ISC-107: MAJ-5 slice — `run_nlevel_world` builds/federates any `depth >= 2` (the `{2,3}` cap is lifted); hierarchical_reduce's collapse/keep structure learning holds at depth 4 (non-gating top surprise 0.0 exact → prunable; informative kept, positive control). Probe: test_nlevel_depth (8).
- [x] ISC-108: MAJ-1 slice — `heuristic_characterization.py`: numerical influence function (c=0 reproduces the naive pool's flat 1/n exactly) + finite capture witness (robust captured at k={{HCHAR_ROBUST_BREAKDOWN_K}} colluders, variational at k={{HCHAR_VARIATIONAL_BREAKDOWN_K}} in the declared fixture). This is evidence against unconditional truth recovery, not an estimator-level influence theorem or global breakdown bound. Figure + §sec:results-heuristic-characterization. Probe: test_heuristic_characterization (5).
- [x] ISC-109: MAJ-3 slice — `continuous_recovery.py`: 1-D Gaussian density-power robust generalized Bayes; beta=0 recovers the conjugate posterior EXACTLY; off-corner (beta 1e-1..1e-4) gap shrinks monotonically O(beta); a genuine outlier is down-weighted (robust mean 0.995 vs conjugate 2.388). Probe: test_continuous_recovery (5).
- [x] ISC-110: MAJ-2 slice — `bnn_variational_torch.VariationalMLP`: genuine mean-field q(w)=N(mu,sigma), reparam forward, closed-form KL (== the tested per-weight gaussian_kl summed), MC-ELBO. sigma→0 recovers PointMassMLP EXACTLY (independent reference); ELBO decomposes into data + exact KL. Probe: test_bnn_variational_torch (5, requires_torch).
- [x] ISC-111: MAJ-4 slice — `federation/socket_transport.run_socket_round`: real loopback-TCP with length-prefixed framing; consensus BIT-IDENTICAL (atol=0) to in-process robust_aggregate; all workers get the broadcast. Probe: test_socket_transport (6).
- [x] ISC-115: MAJ-4 replay/auth slice — `federation/socket_transport.run_socket_round(..., auth_key=...)` authenticates each frame with HMAC-SHA256, returns a digest-only replay log, and `validate_socket_replay` reconstructs the consensus from the logged worker order. Probe: test_socket_transport replay/auth cases (5).
- [x] ISC-116: MAJ-4 durable replay slice — `run_socket_round(..., replay_path=...)` persists the digest-only replay log as deterministic JSON; `load_socket_replay` reloads it and `validate_socket_replay` verifies the consensus from disk-backed event order. Probe: test_socket_transport durable replay cases (3).
- [x] ISC-117: MAJ-4 replay tamper slice — `validate_socket_replay` rejects mutated belief digests, frame digests/byte counts, consensus digests, and inconsistent broadcast digests before accepting a persisted replay. Probe: test_socket_transport tamper cases (4).
- [x] ISC-112: MAJ-6 slice — `benchmark.py` + source-owned `data/synthetic_tabular.csv`: tabular Gaussian-NB federated harness; recovery identity holds at robustness=0; a MEASURED (not universal) robust edge under a 4/5 adversarial majority (naive 0.822 vs robust 0.844). Honestly SYNTHETIC stand-in, harness accepts any user CSV. Probe: test_benchmark (5).
- [x] ISC-113: Anti — every MAJOR stays marked OPEN (these are slices, not completions); no slice claims a theorem, a guarantee robust_aggregate lacks, GPU/multi-machine scale, or a real external dataset it does not have. Probe: TODO MAJ rows read "slice landed, item open"; grep confirms no overclaim tokens.
- [x] ISC-114: full suite + coverage gate green after all six slices; ruff + mypy clean; combined PDF rebuilds with S17 + Figure 23. Probe: `pytest --cov` → 770 passed, 95.21% (final tree 771 with an added error-path test); ruff/mypy clean across 83 source files; render below.

### Iteration 17 — improvements & additions from the forward roster (2026-07-06)
- [x] ISC-100: MED-2 canonical metadata emitter — one config source (`manuscript/config.yaml` publication block) emits CITATION.cff/.zenodo.json/codemeta.json; the three surfaces are drift-checkable and the shipped repo is emitter-consistent. Probe: `scripts/emit_metadata.py --check` exit 0 + test_publication_metadata (5, incl. tamper-detection).
- [x] ISC-101: MED-1 release manifest — `scripts/build_release.py` writes `output/release/{manifest.json,sha256sums.txt,README.md}` with real SHA-256 digests; `--verify` and `shasum -a 256 -c` both detect a tampered/missing artifact. Probe: test_release_manifest (5, incl. shasum-c agreement).
- [x] ISC-102: MED-3 remote CI — `.github/workflows/ci.yml` maps the local profile (ruff, mypy, full suite + ≥90% coverage, metadata drift check, invariants) to GitHub Actions. Probe: read workflow; steps mirror the local gate commands verbatim.
- [x] ISC-103: MAJ-7 hierarchical structure learning — `bayesian_model_reduction.hierarchical_reduce` scores per-level Bayesian surprise; on a degenerate (non-gating) meta-context it flags the level prunable (surprise {{HBMR_DEGEN_TOP_SURPRISE}}, recommends prune), on an informative one it keeps it (surprise {{HBMR_INFORM_TOP_SURPRISE}}). Probe: test_hierarchical_bmr (6, both directions) — the SAME machinery gives opposite verdicts, differing only in L3's conditioned priors (not green-by-construction).
- [x] ISC-104: MAJ-7 figure + report + tokens + supplement — `run_hierarchical_bmr` writes `hierarchical_bmr.json`, `generate_hierarchical_bmr` draws the per-level surprise chart, HBMR_* tokens hydrate the new §sec:results-hierarchical-bmr supplement; all wired through `02_run_analysis.py`. Probe: pipeline run emits both artifacts + manuscript gates green with S16.
- [x] ISC-105: Anti: adding the structure-learning companion does not inflate the study count — N_STUDIES stays {{N_STUDIES}} (derived from cross_study_summary), and the BMR study is framed as a companion, not a numbered federation-benefit study. Probe: grep "Study 10" == 0; N_STUDIES token unchanged.
- [x] ISC-106: full suite + coverage gate stays green after all iteration-17 additions; ruff + mypy clean. Probe: pytest --cov ≥90; ruff/mypy exit 0.

### Iteration 19 — FirstPrinciples/Science/RedTeam deep review of docs/ + manuscript/ (2026-07-12)
- [x] ISC-118: FirstPrinciples pass on `docs/` clusters (core, development, operations, reference, todo) — every claim/procedure is deconstructed to hard constraint vs soft convention vs unvalidated assumption; findings captured with file:line. Probe: workflow agent findings schema, non-empty per cluster.
- [x] ISC-119: FirstPrinciples pass on `manuscript/` clusters (intro/methods 00–13, formalism/results 14–20+S-results, discussion/supplement/reproducibility 21–29+S-supp+meta) — same deconstruction, respecting the fixed SYNTAX.md label/citation registry as a hard constraint (not reinterpretable). Probe: workflow agent findings schema, non-empty per cluster.
- [x] ISC-120: Science pass on `docs/` clusters — every procedural/empirical claim checked for falsifiability and evidence-backing (tied to an actual gate/test/script, not assertion). Probe: findings schema.
- [x] ISC-121: Science pass on `manuscript/` clusters — every empirical claim checked against its `{{TOKEN}}` provenance and the actual generator that emits it (R8); no new hardcoded numbers proposed. Probe: findings schema + zero proposed-fixes containing bare decimals/percentages.
- [x] ISC-122: RedTeam pass on `docs/` clusters — adversarial stress-test for inconsistency, staleness, missing limitations, contradictions with current code/tests. Probe: findings schema.
- [x] ISC-123: RedTeam pass on `manuscript/` clusters — adversarial stress-test for overclaiming against the project's own Baseline Contract (TODO.md: client-FedGVI rigorous / `robust_aggregate` heuristic / `variational_aggregate` rigorous-conservative); explicitly checks the Jul-10→Jul-11 uncommitted `aggregation.py`/`web_package.py` drift for unreconciled prose. Probe: findings schema + explicit Baseline Contract conformance check per finding.
- [x] ISC-124: Adversarial verification (skeptic pass) run on every raw finding from ISC-118..123 before any edit — refutes false positives, especially re-litigation of already-closed REDTEAM_REVIEW.md items. Probe: verification schema, refuted vs survived counts logged.
- [x] ISC-125: Anti: no survived finding recommends a fix that (a) hardcodes a manuscript number outside the `{{TOKEN}}` protocol, (b) invents a citation key outside the fixed `references.bib`/SYNTAX.md allowlist, (c) adds an `{#eq:}`/`{#fig:}`/`{#sec:}`/`{#tbl:}` label not registered in SYNTAX.md, or (d) weakens/removes the three-axis honesty framing. Probe: grep implemented diffs for new bare decimals in prose, new `@`-cite keys not in the fixed list, new `{#...}` labels absent from SYNTAX.md.
- [x] ISC-126: Confirmed, in-scope findings (no src/ or data/ changes required) are implemented directly in `docs/` and `manuscript/` files. Probe: `git diff --stat -- docs/ manuscript/` shows non-zero touched files beyond the pre-existing uncommitted set, each traceable to a survived finding.
- [x] ISC-127: Findings that require code/data changes (new tokens, new experiments) are NOT implemented directly (out of the user's named scope: docs/manuscript only) — instead written up as new/extended `docs/todo/*.md` roadmap entries matching the existing MINOR/MEDIUM/MAJOR pattern. Probe: Read new/modified `docs/todo/*.md`.
- [x] ISC-128: Antecedent: the manuscript's machine-checked gates (`test_xref_integrity.py`, `test_caption_completeness.py`, `test_docs_contract.py`, `test_token_provenance.py`) stay green after all `docs/`+`manuscript/` edits — a precondition for any edit to count as a real (non-regressive) improvement. Probe: `uv run pytest tests/test_xref_integrity.py tests/test_caption_completeness.py tests/test_docs_contract.py tests/test_token_provenance.py -q`.
- [x] ISC-129: static grep of `manuscript/*.md` for stray unresolved `{{` tokens introduced by new edits returns none beyond pre-existing state. Probe: grep.
- [x] ISC-130: R10 baseline gate (`pytest tests/ --cov=src --cov-fail-under=90`) launched at OBSERVE completes and its pass/fail + coverage number is recorded in Decisions before `phase: complete` (src/ untouched by this iteration, so a pre-existing failure is not this iteration's regression, but must be honestly reported, not silently absorbed). Probe: background log tail + exit code.
- [x] ISC-131: Antecedent (added post-Advisor, ID-stability rule — new criterion, not a renumber): the fresh combined PDF (not just source-level gates) is rendered from the current `docs/`+`manuscript/` tree and page-scale-checked for 0 dangling `??` refs and 0 unresolved `{{TOKEN}}` — text-extraction-only verification on prose edits is sufficient per Rule 1 (no raw-LaTeX theorem/formalism block was touched); a raster read was not required since no theorem-block content was edited. Probe: `pdftotext` sweep of the freshly rendered `active_fedference_combined.pdf`.
- [x] ISC-132: Antecedent (added post-Advisor): every one of the three code-deferred findings (RKL/rcce constant duplication, hierarchical_layers.yaml acuity mismatch, cross-study reduced trial count) has its reader-facing manuscript claim actually corrected or explicitly hedged in the same edit pass — deferring the *fix* to `docs/todo/` does not license leaving the *claim* standing unqualified. Probe: read the three edited passages + the three new todo pages side by side.

### Iteration 19b — user-directed scope expansion: land the 3 deferred code fixes (2026-07-12, same day)
- [x] ISC-133: `_DIVERGENCE_ROBUSTNESS`'s 5 constants become pairwise-distinct (MED-2), with a real fail-pre-fix regression test, and `robustness_sweep.json` regenerated to confirm `RKL != rcce` post-fix. Probe: `accuracy_by_method_and_rate['RKL'] == ['rcce']` → `False`.
- [x] ISC-134: `cross_study.py`'s hardcoded Study-8 trial count becomes a named constant exposed as a real manuscript token (MED-3), with a test binding the token to the source constant. Probe: `{{CROSS_STUDY_SENS_N_TRIALS}}` resolves to `'3'`, distinct from `{{SENS_N_TRIALS}}='20'`.
- [x] ISC-135: `hierarchical_layers.yaml`'s `sensor.acuity` corrected to match `build_3level_world`'s actual default (MIN-2), with every other field audited against the same function's defaults for silent drift. Probe: diff read + new regression test file.
- [x] ISC-136: the three interim manuscript hedge sentences added for ISC-132 are removed (not left stale) now that their underlying defects are fixed, and `docs/todo/*.md` + `TODO.md` + `docs/todo/README.md` reflect the removals per the project's own Removal Rule. Probe: grep for the hedge phrasing returns 0; `ls docs/todo/` shows 8 pages (was 11).
- [x] ISC-137: Antecedent: the full source gate, ruff, mypy, and the manuscript provenance gate all stay green at the post-fix tree, and a fresh PDF render confirms every hedge is gone and every new token renders as a real (not literal-`{{`) value. Probe: gate output + `pdftotext` sweep of the freshly rendered PDF.
- [x] ISC-138: Anti: no new src/ change beyond the 3 named fixes' direct blast radius (confirmed via `stat` mtime sweep — only `_common.py`, `cross_study.py`, `hierarchical_layers.yaml`, `manuscript_vars/{loaders,generate,tokens}.py`, plus test files, carry today's mtime in `src/`/`tests/`; the pre-existing Jul-10/11 stale diff elsewhere was left untouched).

### Iteration 20 — warning-free large-sample rerun and manuscript truth audit (2026-07-12)
- [x] ISC-139: The PyTorch variational-KL regression test emits no tensor-to-scalar autograd warning while retaining its independent numerical assertion. Probe: targeted pytest with warnings promoted to errors.
- [x] ISC-140: The executed manuscript profile uses 240 independent seeds and 480 paired robustness trials, with both values sourced from `manuscript/config.yaml` and reflected in generated reports/tokens. Probe: config/report/token equality.
- [x] ISC-141: All analysis reports, figures, manuscript tokens, web/release artifacts, and the combined PDF are freshly regenerated from the 240/480 profile. Probe: pipeline logs, mtimes, and artifact validation.
- [x] ISC-142: Every declarative results and discussion sentence remains supported by the regenerated reports or is corrected/qualified before rendering. Probe: sentence-level claims audit against report JSON and token provenance.
- [x] ISC-143: The combined manuscript PDF has zero unresolved tokens, dangling references, literal citation syntax, or LaTeX undefined-reference warnings, and receives page-scale visual inspection. Probe: log grep, `pdftotext`, page raster/contact-sheet review.
- [x] ISC-144: The full source/structure gate remains green at at least 90% coverage with warnings treated as defects or explicitly classified. Probe: pytest coverage, ruff, mypy, layer grep.
- [x] ISC-145: Anti — the larger sample does not silently change estimands, seeds, contamination mechanisms, robustness labels, or the three-axis claim boundary. Probe: code/config diff and manuscript claim audit.

### Iteration 27 — reconciliation, verifier hardening, and MAJ-1 characterization (2026-07-15)
- [x] ISC-146: The live canonical remote tips are fetched and reconciled without destructive reset, force-push, or direct work on `main`; the feature branch descends from the canonical `docxology-private/main` tip. Probe: `git ls-remote`, `git fetch --prune`, branch ancestry, and preserved preflight status/patch manifests.
- [x] ISC-147: The dirty worktree is audited by layer before staging; reviewer scratch under `.tmp/` is ignored without deleting it, while stale nested sidecar paths remain explicitly governed by the existing docs-contract negative control. Probe: preflight manifests under `/tmp`, `.gitignore`, and `test_local_review_artifacts_are_ignored_and_test_profiles_are_declared`.
- [x] ISC-148: Unit, integration, publication, slow, and optional-Torch test profiles are registered, applied to representative modules, and documented with real-computation/no-mocks constraints. Probe: `pyproject.toml`, `tests/README.md`, marker declarations, and docs-contract tests.
- [x] ISC-149: `robust_aggregate` receives a deterministic finite characterization grid spanning simplex dimension, honest-agent count, robustness, attack mechanism, and weight scenario, with explicit independent-unit and negative-control metadata. Probe: `characterization_grid()` tests and serialized report schema.
- [x] ISC-150: The MAJ-1 figure and manuscript caption identify the estimand, finite-search scope, and uncertainty boundary; the server heuristic remains a scoped implementation fact and does not inherit a theorem or global breakdown guarantee. Probe: figure generator, S17 caption, report `no_claim`, and claim-audit docs.
- [x] ISC-151: Anti — the reconciliation and MAJ-1 extension do not promote MAJ-2–MAJ-6, secure transport, larger models, or positive attack results into universal robustness claims; the objective theorem ladder remains an explicit open research question. Probe: `REDTEAM_REVIEW.md`, `TODO.md`, `docs/research/`, and manuscript claim audit.

### Iteration 28 — verifier profile and roadmap reassessment (2026-07-15)
- [x] ISC-152: The analysis entry point accepts an explicit `publication`/`smoke` profile override, validates it, and preserves publication as the default; the smoke subprocess exercises the same real code path with bounded work. Probe: `test_analysis_profile_is_explicit_and_validated` and the `02_run_analysis.py` smoke subprocess test → **2 passed**.
- [x] ISC-153: The verifier no longer reruns the publication-scale analysis from a smoke-marked test; the publication release remains an explicit separate gate, and the false-certification/runtime path is recorded in `REDTEAM_REVIEW.md`. Probe: script-smoke profile wiring, invalid-profile negative control, and roadmap/audit reconciliation.
- [x] ISC-154: The token-table arity fixture uses the bounded `smoke` profile in its temporary project while retaining the same real pipeline and token-generation path; it no longer hides a publication-scale rerun inside a schema check. Probe: `tests/test_token_tables.py` fixture and targeted token-table test; targeted verifier controls → **3 passed in 82.94s**.
- [x] ISC-155: The full source gate is rerunnable within a bounded verifier budget after both publication-default escape paths are repaired; no test profile silently changes publication artifacts into the release snapshot. Probe: full suite **865 passed**, **93.53%** source coverage, publication profile **34 passed**, and post-gate publication regeneration/render/validation/copy all passed.

### Iteration 29 — first-principles reconstruction and verifier closure (2026-07-16)
- [x] ISC-156: Shared finite-simplex validation rejects non-finite mass, negative mass, empty/ragged rows, invalid weights, and invalid categorical indices while preserving valid zero-probability log-domain behavior. Probe: `tests/fedference/test_validation.py` and core edge negative controls → **green targeted tests**.
- [x] ISC-157: `robust_aggregate` and `variational_aggregate` recompute terminal effective-weight/free-energy diagnostics at the returned consensus, and solver controls reject non-finite or negative values. Probe: terminal-weight equality and control-validation tests → **green targeted tests**.
- [x] ISC-158: Standard Rényi and FedGVI Alpha-Rényi are separately named and dispatched; the finite categorical AR posterior solves the stated objective, including active-set boundary solutions. Probe: divergence normalization, alpha-objective, and recovery tests; FedGVI source link in review record → **green targeted tests**.
- [x] ISC-159: Release verification rejects tampered, missing, duplicated, unlisted, resized, and metadata-inconsistent artifacts. Probe: real-file release-manifest tamper, extra-file, and malformed-byte tests → **8 passed**.
- [x] ISC-160: The `source` validation profile runs lint, typing, invariants, layer isolation, and exact-set release checks, and `full` includes it. Probe: dry-run profile contract plus live source gate → **source profile green**.
- [x] ISC-161: Thin-orchestration and visualization documentation distinguishes client rigor, server heuristic evidence, variational objective evidence, and transport scope; README, architecture, method prose, and figure-source Ruff checks agree. Probe: docs contract, figure registry, 88 figure tests, Ruff, and architecture Mermaid boundary map → **docs contract, figures, and Ruff green**.
- [x] ISC-162: The review record and TODO preserve all unfinished MAJ/MED/MIN research scope without converting a landed categorical fix into a continuous, cross-host, or theorem claim. Probe: `docs/research/first-principles-redteam-review-2026-07-16.md`, `TODO.md`, and linked scoped pages → **review/TODO reconciliation complete**.

### Iteration 30 — runtime-surface cleanup and composability (2026-07-17)
- [x] ISC-163: Executable Python contains no retired placeholder markers or test-double APIs, the no-op Torch type-checking block is removed, and the formerly skippable report-source test is required. Probe: `tests/test_runtime_surface.py`, `tests/test_token_sourcing.py`, and focused federation/token tests → **22 passed**.
- [x] ISC-164: Point-mass and variational Torch models are standard composable `torch.nn.Module` objects with registered state, optimizer-compatible parameters, and fail-closed dimension/hyperparameter controls. Probe: `test_bnn_baseline_torch.py` and `test_bnn_variational_torch.py` → **18 passed**.

### Iteration 31 — publication-gate closure (2026-07-17)
- [x] ISC-165: The authoritative post-change gate and explicit profiles complete without coverage/storage failure, and the reviewer snapshot is regenerated from the publication configuration after the gate. Probe: full suite **865 passed in 34:07**, **93.53%** coverage; `not slow` **821 passed**, integration **14 passed**, publication **34 passed**; publication analysis, token hydration, web package, output inventory, template PDF/HTML/slides render, output validation, and copy all passed. Rendered snapshot: 42 manuscript sections, 42 slide decks, 1 combined PDF, 43 web HTML files, 33 figures, and 487 copied output files.

### Iteration 32 — comprehensive final audit and release synchronization (2026-07-17)
- [x] ISC-166: The executable-surface verifier covers the complete delivered `src/` and `scripts/` trees, and the public-API docstring gate covers the complete delivered source/script surface; no retired placeholder markers or test-double APIs remain in runtime code. Probe: `tests/test_runtime_surface.py`, `tests/test_docstrings.py`, and the authoritative suite → **867 passed**, **93.47%** coverage.
- [x] ISC-167: The hierarchical-POMDP figure generator is evidence-first by default: it requires two executed reports, while the seeded synthetic path is explicit and isolated as an illustrative fallback. Probe: `test_hierarchical_pomdp_requires_executed_reports_by_default` and the full figure suite.
- [x] ISC-168: Manuscript stage-timing hydration fails closed on malformed metadata and emits `N/A` rather than fabricated durations; generated API documentation contains no unresolved manuscript-token literal. Probe: malformed-timing regression, token/output scans, and regenerated API reference/glossary.
- [x] ISC-169: The final source, explicit test profiles, package, and publication surfaces are synchronized after the last changes. Probe: full **867 passed**, **93.47%** coverage; `not slow` **823 passed**, integration **14 passed**, publication **34 passed**; Ruff, mypy, invariants, layer check, exact-set release verification, output validation, and web-package validation all passed. Current snapshot: 75-page PDF, 42 manuscript sections, 42 slide decks, 43 web HTML files, 33 figures, 487 output files, and 404 release artifacts. ISC-89 remains the sole explicitly deferred cross-vendor criterion.

### Iteration 33 — comprehensive manuscript review + first-principles TODO reconciliation (2026-07-17)

Source gates:
- [x] ISC-170: `uv run mypy src/` reports 0 errors (fix `language_kl_decay.py:89-91` shade_ci ndarray args, `disjoint_fov_world.py:124,165` fontsize float). Probe: mypy output.
- [x] ISC-171: Ruff remains green after all edits. Probe: `uv run ruff check src/ scripts/ tests/`.
- [x] ISC-172: Full suite + coverage ≥90% passes at the final tree, run exactly once (Rule 4). Probe: pytest --cov tail.

TODO surfaces (forward-only reconciliation):
- [x] ISC-173: `TODO.md` contains no "SLICE LANDED" markers, no "Publication-polish closure" section, no iteration retrospectives or stale result counts. Probe: grep.
- [x] ISC-174: `TODO.md` is a concise forward index: baseline contract, gates, item tables, parked tracks, removal rule; per-item scope detail lives only in `docs/todo/` pages (duplicated MAJ sections removed). Probe: read + docs contract.
- [x] ISC-175: Every scoped page carries the contract headings plus standardized fields (primary estimand, independent/replication unit, falsifier, prohibited claims/no-claim boundary). Probe: extended `test_scoped_todo_pages_are_complete`.
- [x] ISC-176: An ENG-1 typed report/figure-schema page exists, bidirectionally linked from `TODO.md` and `docs/todo/README.md`. Probe: `test_todo_index_and_scoped_pages_are_bidirectionally_linked`.
- [x] ISC-177: Contract tests reject "SLICE LANDED" and iteration-retrospective/stale-count markers in TODO surfaces; proof-of-detection performed (injected known-bad marker makes the test fail, then removed). Probe: pytest run pre/post.
- [x] ISC-178: `docs/todo/README.md` index and totals match the actual page set on disk. Probe: contract test + read.

Manuscript comprehensive review:
- [x] ISC-179: Every manuscript section file reviewed by ≥1 finder agent and every finding adversarially verified before application; verified findings applied or explicitly declined with a reason. Probe: workflow journal + git diff.
- [x] ISC-180: Three-axis honesty preserved after all edits: no sentence grants `robust_aggregate` an objective/bounded-influence guarantee, no accuracy claim for `variational_aggregate` beyond statistics. Probe: `tests/test_manuscript_claim_audit.py`.
- [x] ISC-181: No unresolved `{{TOKEN}}` in `output/manuscript/` after regeneration. Probe: grep.
- [x] ISC-182: Cross-reference integrity green (`test_xref_integrity.py`). Probe: pytest.
- [x] ISC-183: Every `[@citekey]` used in manuscript prose resolves to a `references.bib` entry. Probe: xref/citation gate or key-set diff.
- [x] ISC-184: Manuscript contract battery green: docs contract, caption completeness, token provenance, manuscript variables. Probe: pytest -q.
- [x] ISC-185: No genuine retired/stub/fake/legacy implementation residue in runtime or doc surfaces; legitimate "synthetic world"/"no mocks"/deterministic-default terminology preserved. Probe: rg classification sweep + `test_runtime_surface.py`.
- [x] ISC-186: Regenerated PDF theorem blocks and display math verified by raster page inspection, not text extraction alone. Probe: page-image Read of rendered PDF.
- [x] ISC-187: Publication artifacts regenerated only after source+doc gates green: analysis reports, tokens, PDF/HTML/slides, output validation. Probe: pipeline logs.
- [x] ISC-188: Anti: no completion claim while any deliverable D1..D7 lacks artifact-backed evidence (scope-miss fingerprint guard). Probe: DELIVERABLE COMPLIANCE + RE-READ.
- [x] ISC-189: Anti: no public filename, API, manuscript label, or output location changed. Probe: figure-registry contract test + git diff review.
- [x] ISC-190: Anti: no edit introduces a scientific claim exceeding its declared estimand, uncertainty, independent unit, or conditional world. Probe: claim audit + adversarial verify stage.

Wave 2 (user escalation: substantive additions/improvements everywhere):
- [x] ISC-191: Every manuscript group receives a substantive improvement pass (methods, statistics, formalism, results, discussion, supplements, captions) by an improver agent, adversarially audited by a diff-reviewing second agent that repairs contract violations. Probe: workflow journal + git diff.
- [x] ISC-192: Any figure-generator change keeps its `tests/figures/` suite green and analysis artifacts are regenerated afterward. Probe: pytest tests/figures + fresh registry mtime.
- [x] ISC-193: Post-wave-2 oracle battery green (claim audit, captions, tokens, xref, docs contract, variables). Probe: pytest -q.
- [x] ISC-194: Anti: wave 2 adds no unresolvable token, no hardcoded result number, no new label, no claim-boundary violation. Probe: battery + unresolved-token grep.
- [x] ISC-195: Web surface renders all theorem environments as `.theorem-box` Divs after the same-line-`\label` infra fix (template `web_renderer._THEOREM_BLOCK_RE`); regression covered by a new template unit test. Probe: grep theorem-box in `output/web/index.html` (>4 = boxes beyond CSS) + template pytest.

Wave 3 (user escalation: ambitiously pursue all improvements — ENG-1 implementation, web body fix, Cato closure):
- [x] ISC-196: A typed schema module under `src/` defines report payload and figure-metadata schemas covering every report written to `output/reports/`; central validation wired at the write boundary in `src/analysis/workflow.py`. Probe: Read module + Grep wiring.
- [x] ISC-197: Write-boundary validation rejects a deliberately malformed report with a named-report/named-field error (proof-of-detection negative control run and captured). Probe: pytest negative-control test output.
- [x] ISC-198: Each figure generator's required report fields are an explicit checkable contract; a missing required field is reported against a named report+field. Probe: stage-contract tests.
- [x] ISC-199: Fixed-seed parity — post-change analysis reproduces the same report values as pre-change (volatile fields like timestamps excluded and documented). Probe: JSON diff.
- [x] ISC-200: Full suite green with the new schema tests at the final wave-3 tree (single definitive run). Probe: pytest --cov tail.
- [x] ISC-201: Template web theorem-box bodies preserve `\texttt{...}` content (as code) and inline `\(math\)` (as MathJax-renderable), covered by template unit tests; regenerated web verified. Probe: template pytest + grep of a rendered box body.
- [x] ISC-202: Cross-vendor audit obtains a structured verdict via direct `codex exec` (bypassing the truncating subagent harness), or the second failure mode is captured verbatim. Probe: captured codex stdout JSON.
- [x] ISC-203: Anti: wave 3 changes no report key, filename, figure path, token value, or public interface (parity + registry + full battery green). Probe: ISC-199 diff + contract tests.
- [x] ISC-204: If ENG-1's full page scope lands, the roadmap honors the Removal Rule: page deleted, TODO.md/README rows removed, link tests still green; otherwise residual scope re-recorded honestly. Probe: docs contract battery + grep.

- [x] ISC-205: Slide decks resolve cross-deck \ref{} to the combined PDF's numbers via an aux-map pre-pass (template feature + unit tests); formalism deck raster shows no "??". Probe: template pytest + slide raster Read.
- [x] ISC-206: Release manifest carries a deterministic source/config fingerprint; --verify rejects a stale-but-consistent bundle (negative control proven); the 8 existing tamper tests stay green. Probe: pytest test_release_manifest.py + quoted verify failure.
- [x] ISC-207: Anti: wave-3c keeps docs contract + release round-trip green and the MIN-1 page forward-only (no landed-slice language). Probe: docs contract battery + grep.
- [x] ISC-208: Two consecutive publication-profile analysis runs with schema validation active produce value-identical reports (run-vs-run determinism, 24 reports). Probe: JSON diff.

- [x] ISC-209: Every documentation group (root, core, dev, ops/pipeline, code-adjacent) plus a manuscript last-mile pass is improved and adversarially audited; docs contract + documentation + claim-audit oracles green after. Probe: workflow journal + pytest.
- [x] ISC-210: Docs accurately reflect the session's landed capabilities: report write-boundary schema validation, release provenance fingerprint, *_MATH token convention, web theorem-box and slide aux-map rendering. Probe: grep of the named docs.
- [x] ISC-211: Anti: wave 4 introduces no forward-only marker violation, no hardcoded live count, no broken relative link. Probe: contract battery + auditor verdicts.


Wave 5 (E5 tri-lens deep review — FirstPrinciples / Science / RedTeam):
- [x] ISC-212: FirstPrinciples claim-chain map: every link of the central scientific chain reduced to its pinning test, each test read and verified non-vacuous; weakest link named. Probe: agent report with file:line per link.
- [x] ISC-213: Science falsifier audit: each headline claim's falsifier demonstrated ABLE to fire via cheap temp-scope perturbation (no repo mutation). Probe: quoted perturbation outputs.
- [x] ISC-214: Science negative-control census: all negative controls enumerated and classified real-vs-vacuous; ≥3 verified by inversion in temp copies. Probe: census table + inversion outputs.
- [x] ISC-215: RedTeam gate-bypass attempts: ≥3 in-place injections (hardcoded number, caption strip, forward-only marker) each caught by its gate, with byte-exact md5-verified restores. Probe: pre/post outputs + md5.
- [x] ISC-216: RedTeam contamination-channel census: silent-corruption channels enumerated with per-channel detectability (incl. MIN-1 fingerprint coverage); guard proposals recorded. Probe: census report.
- [x] ISC-217: RedTeam clone-correctness: load-bearing untracked files enumerated (R12); exact tracking set for clone-correct suite named. Probe: git ls-files diff analysis.
- [x] ISC-218: Synthesis: every CONFIRMED finding fixed by primary or explicitly dispositioned with reason; post-review probes green (n_seeds=240, docs battery). Probe: disposition table + probe outputs.

Wave 6 (user escalation: provenance and clean-checkout hardening):
- [x] ISC-219: Publication analysis, hydration, and external render stages carry content-hashed upstream/downstream receipts; stale or missing dependencies fail closed. Probe: `tests/test_pipeline_freshness.py`, real receipt chain, and `validate_pipeline_freshness.py`.
- [x] ISC-220: A real subprocess clean-checkout probe checks Git cleanliness, required tracking, and package importability without promoting a dirty development tree to release evidence. Probe: `tests/test_clean_checkout.py` plus the expected dirty-tree negative control.

Wave 7 (implementation complexity and scaling diagnostic):
- [x] ISC-221: Dense categorical complexity orders are derived from the actual aggregation, sharing, inference, and server call paths; naive and iterative-robust sharing receive separate measured scaling rows; the report, figure, manuscript tokens, claim ledger, and release provenance are wired and verified without cross-host or network overclaim. Probe: complexity unit/figure tests, publication report and figure, full validation ladder, freshness receipts, and `build_release.py --verify`.

Wave 8 (finite-world evidence broadening and release closure):
- [x] ISC-227: Proper categorical log score, multiclass Brier score, reliability curves, ECE, and deterministic oracle/uniform/confident-wrong controls are implemented with clipping and row/label validation; score tests cover identities, failure modes, and control ordering. Probe: `tests/fedference/test_scoring.py` and the score-control integration report.
- [x] ISC-228: The conditional-world experiment expands the declared mechanism over 40 hidden-state/observability/attack/weight cells, reduces 12 nested trials to 16 seed-level units, and exposes mixed signs rather than promoting universal robustness. Probe: `conditional_world.json`, deterministic rerun, zero-robustness recovery, distinct-target, and seed-unit controls.
- [x] ISC-229: A typed hybrid discrete/Gaussian representation adds categorical log pooling and Gaussian precision pooling plus a finite robust fixed-point diagnostic; zero robustness recovers the hybrid pool exactly and positive robustness is explicitly not called objective-backed. Probe: `tests/fedference/test_hybrid.py` and module claim boundaries.
- [x] ISC-230: The two new reports and figures are validated at the write boundary, consumed only through declared figure contracts, registered from source manuscript labels even when a stale hydrated tree exists, and asserted by the real workflow smoke test. Probe: report-schema, workflow, figure-contract, and registry tests; 29-entry rendered registry.
- [x] ISC-231: The claim ledger, statistical audit, visual audit, experiment/artifact documentation, manuscript supplement, and forward roadmap reconcile the mixed attack-geometry and negative proper-score findings without universal, calibration, or continuous-task overclaim. Probe: docs contract, caption/xref/token gates, and source-bound audit text.
- [x] ISC-232: The publication-default analysis, hydration, external template render, surface validation, PDF structural checks, native-page raster QA, and release manifest verification are regenerated after the new evidence surfaces. Probe: current receipts, rendered outputs, `qpdf --check`, and `build_release.py --verify`.
- [x] ISC-233: The release-tracking set is extended for the new modules, tests, reports, figures, and source-bound audit surfaces; the remaining clean-checkout/fresh-clone and cross-vendor gate is kept open until a committed checkout is independently rerun. Probe: `REQUIRED_TRACKED_PATHS`, clean-tree subprocess result after commit, and explicit no-signal boundary.

Wave 9 (public research platform foundation):
- [x] ISC-234: A validated public `AggregationConfig`, canonical rich-result dispatcher, and typed `AggregatorProtocol` preserve legacy return behavior while threading one conflict-checked configuration through belief sharing, in-process servers, processes, and sockets. Probe: aggregation-config, federation end-to-end, process, and socket tests.
- [x] ISC-235: Transport frames carry a versioned envelope with round/worker/config identity, payload digest, authentication disposition, and bounded frame sizes. Duplicate, tampered, wrong-key, out-of-order, and incomplete rounds fail closed; a caller-shared in-memory guard rejects process-local reuse, while an optional SQLite guard rejects reuse across local process restarts. Neither defines a shared multi-host replay domain. Probe: transport-envelope and real loopback-socket tests.
- [x] ISC-236: Versioned experiment, dataset, artifact, and run-receipt contracts back an installed `fedference` CLI with `list`, `run`, `benchmark`, `verify`, and `replay`; write-producing commands require an explicit isolated output directory. Receipts bind a canonical configuration artifact and Git tree disposition, while wheel/sdist builds include the packaged synthetic resource and retain a Torch-free default core. Probe: evidence, registry, CLI, package-build, and isolated-install tests.
- [x] ISC-237: The registered CC BY 4.0 UCI pack binds official URLs/DOIs/licenses/archive hashes/schema/preprocessing/splits; held-out log score is primary, calibration is separate and frozen, and duplicate datasets or nested seeds cannot manufacture independent units. Probe: live archive verification plus external-data, benchmark, calibration, and receipt tests.
- [x] ISC-238: The optional Torch/MPS lane records device and fallback disposition without making Torch a core dependency; the protocol primitive represents FedGVI as prior plus site factors, cavity construction, and factor replacement with strict round checkpoints. Probe: actual MPS/CPU tests, site/cavity/replacement identities, malformed-checkpoint negatives, and resume equivalence.
- [x] ISC-239: Linear pooling, the current heuristic, the finite-simplex variational family, and an experimental CLR geometric-median comparator have explicit mathematical boundaries; executable KL-orientation witnesses do not masquerade as the still-open MAJ-1 theorem or no-go proposition. Probe: comparator, objective, randomized-simplex, and server-theory tests plus the claim ledger.
- [x] ISC-240: Machine-readable FedGVI and Friston parity matrices derive the permitted replication label from unresolved/deviating rows, while the hybrid tracking fixture pins discrete, one-component Gaussian, zero-robustness, and covariance-handling recovery boundaries without claiming the full benchmark. Probe: protocol-parity and hybrid-tracking tests.
- [x] ISC-241: `TODO.md`, scoped roadmap pages, future-work prose, architecture/reference docs, API policy, clean-checkout tracking set, and claim/literature audits agree on v0.1–v1.x sequencing, MAJ-2A/B, MAJ-4A/B, MAJ-7, MAJ-8, null-result policy, local-compute limits, and external release authority. Probe: docs/manuscript contract and cross-reference suites.
- [ ] ISC-242: The final reviewed commit passes the full coverage/type/lint/layer/package/publication-freshness/render/release ladder twice from isolated fresh clones with at least 40 GiB of safe headroom. A local pass closes only the reproducibility subgate: a structured cross-vendor verdict, confidentiality/license/author approval, DOI creation, and public release remain independent external gates and must not be inferred. Probe: two commit-bound clone receipts plus explicit external-gate dispositions in the verification record.

Wave 10 (comprehensive public-platform correctness and claim audit):
- [x] ISC-243: Run receipts use schema 1.1, bind one canonical configuration artifact plus Git tree disposition, validate UTC ordering and full revisions, reject path/symlink escape, and write atomically; the installed CLI validates runner/profile/seeds/output policy before creating files and supports strict clean-tree verification. Probe: evidence and installed-CLI tests plus a wheel-installed run/verify round trip.
- [x] ISC-244: The packaged synthetic benchmark is byte-identical to its source fixture and works outside the checkout; strict parsers reject malformed rows and fractional labels; Gaussian-NB fits handle absent shard classes without invalid priors; calibration selections own immutable data and reject forged or duplicate evaluation units. Probe: benchmark, external-data, calibration, wheel/sdist, and live three-archive checks.
- [x] ISC-245: FedGVI protocol state owns immutable prior/site arrays, validates every cavity and client update, deep-copies named profiles, and records checkpoint/device boundaries; the hybrid fixture reports on-policy score, known-context component RMSE, and a predictive-risk surrogate without relabeling them as held-out, oracle, or expected-free-energy evidence. Probe: BNN, Torch, parity, and hybrid tests.
- [x] ISC-246: Queue, process, envelope, and socket paths validate worker sets and types, fail closed on malformed replay structures, clean up partial process starts, enforce loopback-only socket binding, and optionally reject round reuse through a caller-shared in-memory or SQLite-backed local guard; cross-host identity, confidentiality, replay-domain design, and security remain open. Probe: transport-envelope, process, end-to-end, and loopback-socket tests.
- [x] ISC-247: Every repository `AGENTS.md` and `README.md`, the manuscript, roadmap, claim ledger, architecture, and testing/reference guides agree on source-conditional client results, the server heuristic boundary, finite-capture versus B-robustness, paper-constrained Friston reconstruction, local-only transport, and stale reviewer artifacts. Probe: documentation/manuscript contract suites, Ruff, mypy, link/xref/caption/token checks, and repository-wide claim sweeps.

Wave 11 (transport threat boundary and publication accessibility):
- [x] ISC-248: The socket helper rejects wildcard/non-loopback hosts before binding; `PersistentReplayGuard` atomically retains claimed round IDs across local process restarts; the registry pins TLS/X.509/Python-SSL/Docker design authorities; and a repository-grounded threat model separates shared-key integrity, per-worker identity, confidentiality, poisoning, availability, persistent state, container-host trust, and physical-host claims. Probe: loopback and restart-replay negatives, registry-source test, docs contract, and security claim sweep.
- [x] ISC-249: Web-package validation fails on missing language/title, skip/main navigation, image alternatives, image-figure captions, full-size-link labels, or duplicate IDs in addition to existing asset/reference failures. All 43 tracked HTML pages and 58 assets pass the strengthened read-only gate. The HTML surface is labelled accessibility-enhanced rather than WCAG-conformant, and the untagged manuscript/slide PDFs remain explicitly non-PDF/UA convenience surfaces. Probe: accessibility reject fixture, current web-package validation, `pdfinfo`, docs/manuscript contracts, and surface-validation integration.
- [x] ISC-250: Unreleased release-manifest builds omit volatile wall-clock metadata and are byte-idempotent; schema 3 validates the nullable `generated_at` and its policy, while approved releases may opt into canonical UTC metadata through `--timestamp` or `SOURCE_DATE_EPOCH`. Probe: repeated-build byte comparison, timestamp rejection/conversion tests, source gate, and isolated-clone rebuild comparison.
- [x] ISC-251: Pipeline-freshness schema 2 omits volatile `recorded_at` metadata by default, validates the nullable timestamp/policy contract, resets legacy receipts only from the analysis boundary, and supports explicit canonical UTC or `SOURCE_DATE_EPOCH` metadata. The analysis producer writes a publication-profile sidecar with a pre-run input snapshot; only the canonical configured/effective publication run can mint the receipt required by non-draft hydration, and any during-run input drift is rejected. Provisional hydration never records its stage; the public render recorder validates current PDF/slide/web surfaces before accepting the external boundary. External render commands pin one epoch across stages 03–05 and record the renderer commit, dirty-diff digest, and epoch without claiming the external overlay is content-verified here. Probe: receipt byte-idempotence, smoke/custom-config/input-drift rejection, provisional-hydration negative, malformed timestamp and legacy-schema tests, deterministic render rebuild, and clone receipts.
- [x] ISC-252: Manuscript hydration never reads wall-clock time: `GENERATION_TIMESTAMP` is derived only from validated `SOURCE_DATE_EPOCH`, or emits an explicit unreleased sentinel. The same epoch is carried across analysis, hydration, PDF/web/slide render, and validation. Probe: pure timestamp conversion/sentinel tests, token hydration, deterministic rerender, and source sweep for wall-clock manuscript producers.
- [x] ISC-253: PEP 517 build isolation exactly pins setuptools and routes wheel/sdist output through a source-shipped backend that normalizes member order, owner, and time metadata under `SOURCE_DATE_EPOCH`. Both archive formats are byte-identical across repeated builds, the backend and manifest template enter release provenance and clean-checkout tracking, and CI installs only after the reproducibility comparison passes. Probe: real tar/ZIP drift fixtures, paired `uv build` plus `cmp`, wheel/sdist install smoke, and release-fingerprint tests.

## Test Strategy

| isc | type | check | threshold | tool |
|-----|------|-------|-----------|------|
| 1–14 | numeric identity | limit/equality on small pmfs | exact/rel 1e-2 | pytest (PASSING) |
| 15–20 | numeric identity | active-inference math vs hand value / Friston figure trend | 1e-9 / monotone | pytest |
| 22–27 | simulation | seeded ensemble run reproduces figure-level trend | direction + margin | pytest |
| 28–29 | statistics | Wilcoxon p, BH-FDR on seed replicates | p<0.05 after FDR | pytest + scipy |
| 31–32 | simulation | baseline accuracy under contamination | robust > NLL | pytest |
| 33 | coverage | src/ line+branch coverage | ≥90% | pytest --cov |
| 21,37 | structural | import/grep/git invariants | binary | grep / git |

## Features

| name | satisfies | depends_on | parallelizable |
|------|-----------|------------|----------------|
| fedgvi-core (divergences,losses,generalized_bayes,aggregation,belief_sharing) | 1–14 | — | DONE (hand-authored, verified) |
| pomdp + belief_updating | 15–16 | core | yes |
| dirichlet_learning | 17–18 | pomdp | yes |
| expected_free_energy | 19 | pomdp | yes |
| bayesian_model_reduction | 20 | dirichlet | yes |
| contamination | 26 | core | yes |
| agents (SentinelEnsemble) | 22 | pomdp,belief_updating,belief_sharing | after pomdp |
| experiments | 23–25,27,30 | agents,dirichlet,bmr,contamination,statistics | after deps |
| statistics | 28–29 | — | yes |
| bnn_baseline | 31–32 | core | yes |
| plumbing rewire (experiment_config,invariants,manuscript_variables,analysis/workflow,figures) | 33–35 | experiments | after experiments |
| meta + manuscript | 35–37 | experiments | partly parallel |

## Decisions

- 2026-06-24: Placement = private standalone Active Fedference repository,
  separate from the public template checkout. Publication boundary is explicit
  in `STANDALONE.md` and metadata surfaces.
- 2026-06-24: FedGVI brought in by **reimplementing core primitives** in typed NumPy for the discrete-categorical setting, not vendoring PyTorch. User choice. Rationale: fits no-mocks/90%/deterministic gates; the discrete reimplementation IS the contribution.
- 2026-06-24: Validation = full suite (three Friston-related reduced categorical
  source-mechanism analogues + robustness sweep) + small NumPy FedGVI logreg
  baseline. User choice.
- 2026-06-24: Cloned `template_code_project` for the full project contract (meta/manuscript/docs/CI), pruned its optimization domain, added a self-contained `src/fedference/` domain package rather than retrofitting the optimization-coupled `analysis/`+`figures/` plumbing.
- 2026-06-24: `robust_aggregate` reweighting = `exp(-c·KL(s_n‖consensus))`
  iterated to a fixed point; `c=0` recovers the project log-linear pool
  exactly (ISC-10). It is the selected discrete analogue of FedGVI
  client-weighting; under the documented bridge assumptions the pool is a
  categorical specialization of Friston Eq. 7, not a full protocol recovery.
- 2026-06-24: refined β-loss to the *recentred* density-power score so the β→0 scalar limit is exactly NLL (initial form diverged as −1/β); caught by ISC-5 on first run.
- 2026-06-24: Forge/Cato unavailable (codex ChatGPT-account 401). Per Rule 2a, substitute inline RedTeam for cross-vendor coverage in VERIFY. `forge_unavailable: true`, `cato_unavailable: true`, `substituted: redteam-inline`.

- 2026-06-24: advisor (CLAUDECODE-blocked) AND Cato (codex 401) both down in VERIFY. Per Rule 2a, ran an inline RedTeam QuickAttack instead. `advisor_unavailable: true`, `cato_unavailable: true`, `substituted: redteam-inline`.
- 2026-06-24 (RedTeam finding — SCIENTIFIC-VALIDITY FLAG): there are TWO robustness axes and only one is FedGVI-faithful. (1) RIGOROUS: robust per-agent generalised-Bayes update via β/rcce losses in `generalized_posterior` — derived from a stated objective, limits to NLL/Bayes (ISC-5,7). (2) HEURISTIC: server-side `robust_aggregate` divergence-reweighting — only its naive-recovery limit (ISC-10) is proven; it is NOT the minimiser of a closed-form FedGVI objective. Remediation: the manuscript must carry the scientific claim on axis (1) and present axis (2) honestly as a complementary heuristic; the robustness sweep should report both; no experiment may claim axis (2) inherits FedGVI's bounded-influence guarantees. Added as a constraint on the breadth manuscript.

- 2026-06-24 (iteration 2 — comprehensive polish): the baseline (ISC-1..37) is green and locked at `312 passed, coverage 97.08%`. Iteration-2 polish surface (theorem environments, eq/sec labels, 8 figure generators, modular per-study manuscript, mirrored `tests/` tree, caption/booktabs aesthetics) is in place. `TODO.md` rewritten to track only *upcoming* work: completed items removed (no `[x]`-as-open), rescoped into three deeply-specified tiers — MINOR (caption completeness, docstring gate, edge tests, palette/theme, xref integrity, token-provenance lint), MEDIUM (continuous-state Gaussian divergence stub, three new contamination models, HTML/slides render polish, greedy multi-hypothesis BMR, sweep power analysis), MAJOR (faithful GPU/BNN FedGVI port, real multi-machine federation, closed-form objective to upgrade axis-2 from heuristic to rigorous, moving multi-factor sentinel world). The MAJOR closed-form-objective item is the designated headline scientific upgrade; landing it is the one path that rewrites the axis-2 honesty caveat from "heuristic" to "rigorous".

- 2026-06-24 (iteration 2 — comprehensive polish): ran a 5-phase workflow (w758x737y, 11 agents) then independently re-verified on HEAD. Manuscript made modular (14 numbered section files 00–12+99), every display equation `{#eq:}`-labelled (12 labels / 13 refs), 10 numbered formal environments incl. 3 Theorems for the recovery limits, 30 `{#sec:}` subsection labels, 8 figures generated + referenced with full captions (all PNGs produced by the pipeline), statistics extended (bootstrap CIs, raw p, per-rate tests, standardized effect size), tests reorganised to mirror `src/`, scripts confirmed strict thin orchestrators, TODO rescoped into MINOR/MEDIUM/MAJOR. Suite: 395 passed, 98.10% coverage.
- 2026-06-24 (post-workflow fixes, hand-authored): (1) replaced the hardcoded manuscript signposts `_ISC_TOTAL="35"`/`_TEST_COUNT="TBD"`/`_COVERAGE_PERCENT="90"` in `src/manuscript_variables.py` with LIVE derivation (`_count_isc` parses ISA.md, `_count_tests` counts `def test_`, `_coverage_percent` reads a coverage artifact else the honest `≥90` gate floor) and updated the `_make_project` test fixture to carry an ISA + tests/ so the contract test exercises it; (2) rewrote the wholesale-stale `manuscript/AGENTS.md` (it still described the optimization template — wrong sections, 6 wrong figures, wrong script) to the real 14-section / 8-figure / `02_run_analysis.py` structure. Both caught by Gate-J re-probing of the workflow's self-report.

- 2026-06-24 (iteration 3 — IMRAD/American/community): converted src+tests+scripts to American English (boundary-safe converter, 135 subs, 395 green) and saved the durable preference to memory; ran workflow wqmb5dpvq (8 agents) rebuilding the manuscript as 28-file American-English IMRAD — community-positioned (22 refs; active inference + federated/robust-Bayes "what has/hasn't been done"), yes-and / show-not-tell, full pandoc-crossref auto-numbering (19 eq / 8 fig / tables, 0 dangling), zero hardcoded numbers, deeper methods + sample-size/power (n_trials 12→40, n_seeds added). 434 tests, 98.11% coverage, combined PDF rebuilt (672 KB). Committed + pushed to docxology/active_fedference (cd88402).
- 2026-06-24 (Gate-J HEAD re-probe fixes, hand-authored): (1) the token-coverage test was scanning a DELETED file (03_results.md) — rewired to glob all 28 sections, so "every prose number is a token" is now enforced manuscript-wide (passes); (2) deduped a double-defined {#sec:two-axes} label (kept canonical in 02_gap.md, renamed the results restatement to {#sec:two-axes-results}); (3) refreshed the stale SYNTAX two-axes/contributions registry lines.
- 2026-06-24 (known follow-up): the SYNTAX.md label registry and manuscript/AGENTS.md file inventory are hand-maintained and drift on each manuscript restructure; they are non-rendered dev guides (the rendered manuscript's cross-refs verify clean). A MINOR follow-up is to auto-generate these registries from the live section/label set rather than hand-syncing.

- 2026-07-06 (iteration 15 — OBSERVE/BUILD decisions): (1) R10 baseline at HEAD bf9fe52: 705 passed BUT configured coverage gate FAILED — 85.70% vs fail_under=90 (`--cov=src`, branch=true, matches pyproject scope). The manuscript's COVERAGE_PERCENT token reads a STALE July-2 root artifact claiming 94.90. Caveat: the baseline run overlapped this session's fix-agent edits, so 85.70 needs a clean re-measure at VERIFY before any remediation decision. (2) Gate-scope hole closed: token-provenance + token-table gates globbed only `[0-9]*.md`, missing all 11 S-supplements; widening fired on 12 real hardcoded "95%" CI literals (proof-of-detection satisfied organically) — all tokenized to {{CI_PERCENT}}. (3) SYSTEM_OVERVIEW_METADATA hand-typed 69%/99%/40% replaced by derivation from the schematic's own pooled beliefs — true values are 26%/70% ("accuracy" was also the wrong word: both argmaxes were correct pre-fix; adversarial concentration raised 4→6 so Panel B's "fails" claim is now literally true — argmax flips to state 6); captions rewritten to "true-state consensus mass", fabricated "60 trials, 95% bootstrap CI" caption tail deleted; unbacked "***" significance stars removed from the cover (deterministic schematic — no test ran); derivation-chain test added. (4) BELIEF_SHARING_ACUITY bound to new DEFAULT_ACUITY constant; ISC_EFE_TOLERANCE bound to new EFE_IDENTITY_ATOL. (5) Injection audit (agent) found 2 live prose-vs-data contradictions (S11 latent-recovery inversion; S05 "not significant" vs p=0.0078) + hardcoded FEDERATION_BIT_IDENTICAL verdict + duplicated-literal token class — delegated to a fix agent with pinned invariants. (6) 21-figure visual review (2 agents, fresh renders): 10 FIX-REQUIRED figures (incl. moving_world's empty signed-gap panel and system_overview hiding its own headline) — delegated to 2 fix agents on disjoint file sets; system_overview + graphical_abstract fixed by hand.

- 2026-07-06 (iteration 16 — comprehensive review, four-lane audit + fixes): (1) R10 baseline at the 98-change working tree: **712 passed, coverage 94.69%** (`uv run --extra dev pytest tests/ --cov=src --cov-fail-under=90` → "712 passed in 1612.62s", "Total coverage: 94.69%"); the prior session's mid-session 85.70% was contamination, not real. `coverage_project.json` regenerated from the fresh `.coverage` (was a stale Jul-2 artifact claiming 94.90 with no freshness check). (2) Four read-only audit lanes (methods-vs-code, docs-vs-repo, claims-vs-reports, plus fresh-render visual checks) produced 7 CRITICAL + 22 MAJOR + ~20 MINOR verified findings; two write agents + hand edits applied all of them. Defect distribution matched the quiet-strong-claims prior: per-study results sections were clean; defects clustered in abstract/conclusion/S13 summarizing glosses and in the strongest quiet words ("bit-identical", "mean-field", "hand-computed", "read from config"). (3) Headline claim fixes: S05+abstract hand-typed "(not significant)" on a p=0.0078 rejecting contrast → corrected direction (isolated significantly higher) using the derived {{MOVING_SIGNIFICANCE_VERDICT}} token; conclusion's 3-level latent swap (meta-context↔context) fixed; worst-rate accuracies no longer attributed to the verdict-rate q-value; S13 sensitivity patterns rewritten to match the actual grids (benefit lives at LOW acuity, colony-2 column exactly zero, hierarchical ≈0); "closes most of the gap" → "roughly a quarter"; disjoint half-views no longer called "overlapping". (4) Structural code fixes with proof-of-detection tests: fdr_alpha plumbed config→sweep→report→token (was three coincidentally-equal constants); sample_size_for_power floored at feasible Wilcoxon n=5 (was recommending n=1); Dirichlet kl_trajectory off-by-one (final_kl was pre-final-batch; now num_steps+1 points); _project_root(None) resolved to src/ and generate_variables FABRICATED an all-zeros stage_timings.json when missing (now: correct root, read-only N/A degradation, 2 regression tests); EFE_IDENTITY_ATOL now consumed by the identity test AND invariants.py (was hardcoded 1e-9 in both); V4 loader silent numeric fallbacks → strict key access; "mean-field BNN" renamed to point-mass-family deterministic MLP throughout. (5) Docs drift: 19 findings fixed incl. quickstart/experiments-and-artifacts still asserting "hierarchy beats flat" and "EFE widens gap" (both contradicted by reports), handoff notes with already-done open items, README's "stub" federation (it is a real tested multiprocess transport), stale counts (8→21 figures, ISC-1..37→93, 12→22 reports). (6) Fresh pipeline + tokens (313) + combined PDF (78 pp, 2.76 MB, 0 dangling refs, 0 unresolved tokens); cover + previously-broken figures verified at pdftoppm page scale.

- 2026-07-12 (iteration 19 — OBSERVE, R10 + scope decisions): (1) Working tree carries 332 uncommitted changes at session start (`git status -s`), spanning src/tests/manuscript/docs/output; mtimes cluster at 2026-07-11T13:13 (aggregation.py, most manuscript sections) with no file touched again since — stale, single-actor, no R15 co-actor signal. Per R10 (6+ changes), launched the full baseline gate `uv run --extra dev pytest tests/ --cov=src --cov-fail-under=90` in background at OBSERVE; result to be logged before `phase: complete` (ISC-130). (2) The ISA frontmatter claimed `phase: complete, iteration: 18c-formalism-verification, updated: 2026-07-06`, but the uncommitted diff (331 files, +8319/-7166) postdates that closure and is NOT reconciled into Decisions/Changelog — treating that stale "complete" claim as ground truth for current file content would violate R8 (inherited-premise gate); this iteration reads actual current `docs/`+`manuscript/` file contents directly rather than trusting the ISA's iteration-18 narrative of what they contain. Full archaeology of the unreconciled src/ diff is out of scope for this iteration (user named only `docs/` and `manuscript/`) and is left as-is (not touched, not reverted). (3) ISC count for iteration 19 (ISC-118..130, 13 criteria) is deliberately below the E4 soft floor of 128: a full per-file×per-lens atomization (77 files × 3 lenses ≈ 231) would be administrative padding, not distinct atomic claims, since files within a thematic cluster (e.g. the 6 discussion/supplement manuscript files) genuinely share findings. Real atomic decomposition is cluster×lens (6 clusters × 3 lenses = 18, folded into ISC-118..123) plus verify/implement/gate criteria; concrete per-finding ISC-N.M splits are added during BUILD once the Workflow's findings schema returns real findings (ID-stability rule: parent ISC-118..123 preserved, splits appended, never renumbered). (4) Scope boundary: only `docs/` and `manuscript/` are edited directly; any finding requiring `src/`/`data/` changes (new tokens, new experiments, new generators) is deferred to `docs/todo/*.md` roadmap entries rather than implemented, per the user's explicit two-directory scope and to avoid destabilizing the already-uncommitted 332-change tree further.

- 2026-07-12 (iteration 19 — resumed session, cold-start reconciliation): a prior instance of this same iteration had already executed a real (unreconciled) first BUILD/EXECUTE pass before ending without updating ISA checkboxes/Verification: `git status -s -- docs/ manuscript/` shows 58 changed paths; mtime analysis (`stat -f %Sm`) splits them into 20 stale Jul-10/11 files (the pre-existing iteration-18 diff, untouched by iteration 19) and **35 files genuinely modified 2026-07-12T15:38–15:47** (7 docs/core+development+operations+reference/todo files, 2 brand-new `docs/todo/*.md` pages, 24 manuscript files, `manuscript/SYNTAX.md`), `git diff --stat` on the 33 tracked ones showing 328 insertions / 231 deletions of real prose fixes (spot-checked `manuscript/14_formalism.md`: bare "Theorem 5"/"Lemma 3"/"Proposition 4"/"Corollary 6" prose references converted to `\label{}`/`\ref{}` pairs so prose can't drift from the LaTeX-rendered theorem number). Per R8, this prior session's own claims are treated as an inherited premise needing re-derivation, not fact: (a) confirmed `src/`/`tests/`/`scripts`/`data/` carry **zero** files with today's mtime — scope boundary was honestly held; (b) ran the four ISC-128 gates fresh on the current (already-edited) tree — `uv run pytest tests/test_xref_integrity.py tests/test_caption_completeness.py tests/test_docs_contract.py tests/test_token_provenance.py -q` → **30 passed** (real, this session); (c) grepped for the same hardcoded-theorem-number pattern across the whole manuscript to check for a missed call-site (R14 sweep) — found 3 residual hits, all in `manuscript/SYNTAX.md`'s "Modular section decomposition" table (lines ~363-366): on inspection these are NOT a bug — that table is an explicit historical record of a past file-split decision ("Decision: split the 8 monolithic section files...") using the labels as they were AT THAT TIME, and `SYNTAX.md` itself is a non-rendered dev-only registry per its own "Preamble Injection" section (pandoc never renders it) — a real false-positive caught and refuted before any edit, not implemented. (d) The two new `docs/todo/*.md` pages (`wire-sensitivity-noise-floor-token.md`, `fix-moving-world-figure-extension-mismatch.md`) and their `TODO.md`/`docs/todo/README.md` entries follow the established MINOR/MEDIUM template exactly (Status/Rationale/Scope/Implementation Notes/Acceptance Criteria/Verification Probes/Claim-Boundary Constraints/Dependencies) — this is genuine ISC-127 output from the prior pass, kept as-is. (5) Launched the R10 full baseline gate fresh in background (`nohup uv run --extra dev pytest tests/ --cov=src --cov-fail-under=90`, log at session scratchpad) since a background job from a prior, now-ended session cannot still be running. (6) Launched a 10-cluster Workflow (`wf_0bffa1bd-fd6`) to (i) freshly review the ~9 clusters/files NOT touched in the 15:38-15:47 pass with the same three-lens FirstPrinciples/Science/RedTeam methodology, and (ii) adversarially audit the 33 already-edited files' diffs for regressions or half-applied fixes, each cluster pipelined through find→verify→implement so genuinely new in-scope findings are fixed directly and any requiring `src/`/`data/` changes are collected for `docs/todo/*.md` roadmap entries rather than implemented.

- 2026-07-12 (iteration 19 — VERIFY, Advisor + Cato + render/gate closure): (1) Advisor (`Inference.ts --mode advisor --auto-state`) ran; its STATE block resolved an unrelated global PAI-hooks task ISA (known `--auto-state` failure mode, memory `gotcha-advisor-autostate-resolves-wrong-isa`) — ignored per that memory, its substantive critique was not affected. Advisor's core demand: "a disclosure is not a remediation" for the RKL/rcce finding — does the defect collapse the headline claim's *outputs*, not just its labels? Re-derived directly against `output/reports/robustness_sweep.json`: `AR` (mean_accuracy_diff 0.0767, effect_size 1.0, q≈2e-21) and `beta` (0.0685, effect_size 1.0, q≈2e-21) are genuinely independent of the RKL/rcce duplication and each alone fully supports "robust beats naive" with a saturated effect size and vanishing q-value — the headline `best_robust_method` selection (`headline_method: "RKL"`, chosen by a strict `>` loop over `effect_size`, which is tied at 1.0 across all four robust methods — a saturation/ceiling artifact the manuscript's own caption already flags as "signed-saturation") reports a real, non-fabricated number either way. Conclusion: the manuscript-level disclosure (already added, stating "four distinct operating points, not five") is the correct-strength fix — the duplication is real and now honestly disclosed, but does not invalidate or require withdrawing the headline robust-vs-naive verdict, which AR/beta support independently of RKL/rcce. (2) Advisor's stronger, actually-actionable catch: a first render attempt (`uv run python scripts/pipeline/stage_03_render.py --project active_fedference`, run twice) silently verified against a STALE mirror at `/Users/4d/Documents/GitHub/template/output/working/active_fedference/` (unchanged since 2026-07-11T13:44, byte-identical across both attempts) rather than the real render target. The actual current render lands at the private repo's own path (`/Users/4d/Documents/GitHub/projects/working/active_fedference/output/pdf/active_fedference_combined.pdf`, confirmed fresh at 16:51 today) — re-verified there: 73 pages, 0 `??`, 0 unresolved `{{`, and every one of the iteration's new disclosure sentences (S12 "canonical 3-level topology", S14 Study-8 trial-budget caveat, both RKL/rcce disclosures) present verbatim in the rendered text. Learning: for this project, always resolve the render output path via the private repo's real (non-symlinked) root, not the template mirror under `output/working/<project>/` — the latter can be stale and silently look successful. (3) Repo-wide grep for "bounded-influence"/"bounded influence"/"B-robust" (Advisor demand 5) found the manuscript, `SYNTAX.md`, and `docs/` were ALREADY correctly hedged (matching the already-updated `aggregation.py` docstring's "not by itself a proof that the normalized consensus estimator is B-robust") — the only two stale outliers were `TODO.md`'s "Baseline Contract" (still called `variational_aggregate` "the rigorous server rule: bounded-influence") and `REDTEAM_REVIEW.md`'s "Claim Boundaries" (same unqualified phrasing) — both corrected to match the code's and manuscript's already-walked-back framing ("a proven raw effective-weight bound with empirical redescending behavior... not an estimator-level B-robustness proof"). (4) Cato (`codex-cli 0.144.1`, confirmed live via a fresh `codex --version` re-probe — the ISA frontmatter's `cato_unavailable: true` from iteration 1 (2026-06-24) was a stale inherited premise per R8 and is corrected below) was invoked for the mandatory E4 VERIFY cross-vendor audit; its background run returned `status: completed` but its final message was a mid-sentence fragment ("Claim 2's docstring disclaimer confirmed verbatim at line 258-259 ... Let me run the gates (macOS has no `timeout`). ") with no structured verdict — matching the newly-recorded memory `gotcha-cato-subagent-truncates-after-background-launch` (2x occurrence, judged structural, not a prompting issue). Per Rule 2a this is NO-SIGNAL, and per the memory's own "structural not prompting" classification a same-shape retry is not expected to resolve it, so no retry was spent. **This leaves the Cato cross-vendor VERIFY gate genuinely unsatisfied for this iteration** — substituted with (a) the Advisor's full, completed, substantive critique (Rule 2 hard gate, satisfied) and (b) independent re-derivation of every one of its claims directly against real artifacts (the sweep report JSON, the fresh PDF, live `git diff`/`grep` reads) rather than trusting any agent's narrative — but this is not the same as a passing cross-vendor signal, and is reported honestly rather than papered over. (5) Final gates at the true final tree: manuscript/docs provenance (`test_xref_integrity`, `test_caption_completeness`, `test_docs_contract`, `test_token_provenance`, `test_manuscript_variables`) → 61 passed; R10/ISC-130 full source baseline (`uv run --extra dev pytest tests/ --cov=src --cov-fail-under=90`) → **803 passed, 94.54% coverage** (≥90% required, met); `src/` was untouched throughout this iteration so this baseline reflects the pre-existing state, not a regression introduced here.

- 2026-07-17 (iteration 33 — OBSERVE/PLAN): (1) Scope = user's original "comprehensively review and make all improvements to the manuscript" merged with the mid-turn first-principles roadmap plan (TODO reconciliation, mypy fixes, contract-test hardening, terminology audit, regeneration). effort_source: classifier (E4). (2) R10: 615 uncommitted deltas at session start (409 = `output/` regeneration from the committed iteration-32 pass); full baseline suite launched in background at OBSERVE. (3) R8 generator-preexec: ruff green and mypy = exactly the 5 claimed figure-layer errors, verified by running both this session; `codex --version` → 0.144.1 live (frontmatter `cato_truncates_after_background_launch` retained as known risk, availability premise refreshed). (4) Decision: mypy fixed at the ingestion point — `_common.py` `shade_ci` widened to `Sequence[float] | np.ndarray` and `annotate_stats_box` fontsize to `float` — 2 signature edits instead of 5 call-site casts (root-cause-at-ingestion; the narrow signatures were the bug). (5) Decision: TODO.md's per-item MAJ sections were a second copy of the docs/todo scoped pages and had already drifted (R14-in-prose); rewrite makes the scoped page the single owner of item scope and TODO.md a pure forward index; "SLICE LANDED" markers removed under the plan's false-assumption verdict ("a landed slice means the TODO is done"). (6) Orchestration: Workflow wf_b42fd381-d90 (8 finder→verifier→fixer manuscript groups, 12 TODO-page agents, 1 terminology auditor); Forge delegated the contract-test hardening with mandatory proof-of-detection injection; page/fixer agents own disjoint file sets, README/TODO.md index edits reserved to the primary.

- 2026-07-17 (iteration 33 — EXECUTE, cross-repo web-render regression caught by raster+grep verification): wave-1/wave-2 fixers added the standard amsthm same-line labels (`\begin{proposition}[Name]\label{prop:...}`) so PDF `\ref{}`s resolve — the PDF renders correctly (raster-verified pages 1/13/25/42: shared-counter numbering, clean math, resolved refs). But the template repo's web-only theorem rewriter (`web_renderer._THEOREM_BLOCK_RE`) required a newline immediately after `\begin{env}[name]`, so the freshly regenerated web surface silently dropped ALL 11 theorem environments (HEAD web: 11 boxes + 4 CSS hits = 15 occurrences; fresh: CSS only). Root-caused to the regex, fixed in infrastructure per the iteration-7 precedent (accept the standard idiom, don't contort project markdown): regex now tolerates and captures a same-line `\label`, which becomes the Div anchor id. Proof-of-detection: new template unit test `test_theorem_block_with_same_line_label_is_rewritten_with_anchor`; A/B on real hydrated files pre-fix boxes=0 → post-fix 14_formalism=2, 06=2, 29=1; template renderer suite 41 passed. Learned: text-extraction validation of HTML would have missed this — the drop is only visible by counting rendered boxes (same class of failure as the pdftotext-masks-theorem-math memory).

## Changelog

- conjectured: The project categorical posterior-log-potential bridge places
  its log-linear pool in correspondence with Friston belief-sharing (Eqs. 6–8).
  refuted_by: Full source-protocol identity is excluded; the documented bridge
  remains a categorical specialization only.
  learned: `robust_aggregate(robustness=0)` is bit-identical to the project
  log-linear pool (ISC-10), while `generalized_posterior(KLD,NLL)` equals
  closed-form Bayes (ISC-7). Those are separate project-local recovery
  statements, not a certification of the complete source protocol.
  criterion_now: ISC-7, ISC-9, ISC-10 pin these project-local statements.

- conjectured (iteration 2): the remaining open contract was the iteration-2 polish list (ISC-38..53) and the backlog should keep tracking those rows.
  refuted_by: re-reading the repo on HEAD — the polish surface is already built (preamble carries 5 `\newtheorem` environments, 17 eq labels and 10 distinct `@fig:` refs in the manuscript, 8 figure generators in `src/figures/`, the manuscript is split into per-study section files, and `tests/` mirrors `src/`). Tracking those as "open" would list done work as todo, the exact failure mode the rewrite forbids.
  learned: the backlog's correct contents are NOT the polish ISCs but genuinely *new* surface beyond the green baseline. Rescoped `TODO.md` into MINOR/MEDIUM/MAJOR tiers of upcoming work, each with an acceptance probe; completed items are recorded only in this ISA's Verification, never re-listed as open. The one structural lever that changes the project's scientific standing is the MAJOR closed-form aggregation objective (axis-2 heuristic → rigorous).
  criterion_now: TODO.md carries three tiers and zero done-as-open rows; ISA Verification remains the single record of completed contract.

- conjectured (iteration 19): a manuscript-level prose disclosure ("two of the five labels currently collide") is sufficient remediation for the RKL/rcce divergence-constant duplication in the headline robustness sweep, without needing to determine whether the duplication changes any reported number.
  refuted_by: the Advisor — a disclosure is not a remediation; the real question is whether the sweep's *outputs* (not just labels) collapsed, which is a read-only, in-scope determination that must actually be made, not deferred.
  learned: re-derived directly against `output/reports/robustness_sweep.json` that `AR` and `beta` (the two genuinely non-duplicated robust methods) each independently show effect_size=1.0 and q≈2e-21 against naive — the headline "robust beats naive" verdict does not depend on RKL/rcce being distinct, so the disclosure-level fix is correctly scoped once that independence is actually verified, not assumed. Also learned that the `headline_method` selection (`"RKL"`) is chosen by a strict-greater-than loop over `effect_size`, which saturates at 1.0 across all four robust methods on this task — a ceiling artifact the manuscript's own caption already names ("signed-saturation") — so naming "RKL" as headline is not itself a false claim of superiority.
  criterion_now: ISC-123/ISC-125/ISC-132 require this exact style of independent re-derivation (re-run the real generator/report, don't trust the finding's narrative) before treating any deferred-to-todo finding's manuscript mitigation as adequate.

- conjectured (iteration 19): running `stage_03_render.py --project active_fedference` and finding `0 ??`/`0 unresolved {{` in the resulting PDF is sufficient proof the day's 29 prose edits rendered cleanly.
  refuted_by: the checked PDF (`/Users/4d/Documents/GitHub/template/output/working/active_fedference/...`) was a stale mirror unchanged since 2026-07-11T13:44 — byte-identical across two separate render invocations — while the render log's own "Location:" line pointed at the private repo's real, non-symlinked output root the whole time.
  learned: for a project checked out as a symlink into the template tree, the render pipeline's actual write target is the private repo's own `output/` (resolved via the real, non-symlinked path), not the template's `output/working/<project>/` mirror — that mirror can be stale and silently look like a successful, current render. Always resolve and check the render log's stated "Location:" line, not an assumed conventional path.
  criterion_now: ISC-131 pins this — future render verification for this project must resolve the actual write path from the render log, not assume the template-tree mirror.

## Verification

- 2026-07-16 (iteration 29 — first-principles reconstruction and RedTeam verifier closure): the source profile passed Ruff, mypy, invariants, the `src/fedference` layer check, release build, and exact-set release verification; targeted core/validation/manifest/profile tests passed; the corrected renderer fallback passed the complete figure suite (**88 passed**), and the release-manifest suite passed (**8 passed**). The authoritative full command reached 749 passed tests before coverage SQLite failed with `database or disk is full`, cascading into the remaining publication tests. A bounded non-publication coverage retry was stopped when the same host condition fell below a safe free-space margin. This keeps ISC-155 open rather than claiming a coverage pass; the storage condition and unaffected source/profile evidence are recorded in `docs/research/first-principles-redteam-review-2026-07-16.md`.
- 2026-07-17 (iteration 30 — runtime-surface cleanup and composability): runtime/static inspection found no active stub/fake/dummy/legacy/deprecated markers after cleanup; the only actual no-op was an empty `TYPE_CHECKING` block. The federation boundary test was corrected to assert the landed loopback-TCP adapter rather than describing the package as a scaffold. Both Torch model classes now subclass `torch.nn.Module`, expose registered `state_dict`/parameter surfaces, use standard module optimizers, and reject invalid controls. Focused runtime/federation/token tests → **22 passed**; Torch composability/configuration tests → **18 passed**; Ruff and mypy remained green. The detailed record is `docs/research/runtime-surface-composability-review-2026-07-17.md`; ISC-155 remains open pending the full coverage/storage and publication-regeneration gate.
- 2026-07-17 (iteration 31 — publication-gate closure): the authoritative `uv run --extra dev pytest tests/ --cov=src --cov-fail-under=90` completed **865 passed in 34:07** with **93.53%** source coverage after the prior host-level storage failure. Explicit profiles also passed: `not slow` **821**, integration **14**, publication **34**. Publication-default analysis, token hydration, web preparation/validation, output inventory, template stages 03–05, and copied-output validation all passed; the regenerated reviewer snapshot contains 42 manuscript sections, 42 slide decks, 1 combined PDF, 43 web HTML files, 33 figures, and 487 copied files. ISC-155 is closed; the remaining TODO is scientific/release-integrity scope only.
- 2026-07-17 (iteration 32 — comprehensive final audit and release synchronization): the executable scan covers all `src/` and `scripts/` Python with no retired placeholder markers or test-double APIs; the public API docstring gate covers the delivered source/script surface; and the hierarchical-POMDP figure now requires executed reports unless its illustrative fallback is explicitly requested. Malformed stage timing metadata degrades to `N/A` without writing fabricated state, and regenerated API docs contain no unresolved manuscript-token literal. Final gates: **867 passed**, **93.47%** source coverage; `not slow` **823**, integration **14**, publication **34**; Ruff, mypy, invariants, layer isolation, output validation, web-package validation, and exact-set release verification all passed. The current reviewer snapshot is a 75-page PDF with 42 manuscript sections, 42 slide decks, 43 web HTML files, 33 figures, 487 output files, and 404 release artifacts. ISC-89 remains the sole explicitly deferred cross-vendor criterion; TODO remains forward-only.
- 2026-07-15 (iteration 28 — verifier profile and roadmap reassessment): RedTeam found two publication-scale escape paths in verifier fixtures: the script smoke oracle and the token-table schema fixture. Added an explicit profile override with publication default and bounded real smoke mode, then changed the temporary token-table fixture to use that same smoke profile. The pre-repair full gate completed **843 passed, 94.39% coverage** in 38:43; final post-repair source/release gates remain pending. Targeted workflow/profile and token-table controls, Ruff, publication regeneration, render, package, and release evidence must be recorded after the repair.
- 2026-07-15 (iteration 27 — reconciliation, verifier hardening, and MAJ-1 characterization): fetched live `origin/main` and `docxology-private/main` (both at `22af88f`), created `codex/reconcile-redteam-research` from the preserved worktree tip, and captured preflight status/patch/untracked manifests under `/tmp` before editing. Added `.tmp/` ignore coverage without deleting local review artifacts; registered/documented `slow`, `integration`, `publication`, and existing optional-Torch profiles; applied real-computation markers to representative suites; and added a docs-contract regression test for the profile/scratch policy. Focused contract/characterization/figure gates → **27 passed**; Ruff → **All checks passed**. `characterization_grid()` now emits 64 deterministic scenario rows over two simplex dimensions, two honest-agent counts, two robustness values, four attack mechanisms, and two weight scenarios, with declared independent unit, finite-search negative control, and no theorem/global-breakdown claim metadata. Full source coverage, pipeline, render, package, and release gates remain pending in this cycle and must be recorded here before closure.

- 2026-07-14 (iteration 25 — FirstPrinciples/Science/RedTeam phase-plan and verifier closure): shared `SENSITIVITY_NOISE_FLOOR` across the experiment config, heatmap, and token pipeline; changed moving-world's canonical generator return to the embedded PNG while retaining a deterministic PDF companion; added real registry/embed/generator consistency and data-bearing caption gates; and added explicit `publication`/`smoke` workflow profiles so temporary pipeline fixtures remain real but bounded. Targeted gates → **33 passed**, Ruff clean. The exact `validate_all full` wrapper completed with quick **30/30**, manuscript **51/51**, package PASS, and the embedded source gate **840/840**; the live coverage.py artifact reports **94.48%** total (96.10% statements; 87.71% branches), above the 90% floor. Invariants, publication analysis, token hydration, refreshed coverage artifact, and template stages 03–05 completed; the template output validator passed for the 76-page PDF, 42 HTML sections, 42 slide decks, and 33 figure artifacts. The scholarship-indexed phase plan (`docs/todo/scholarship-and-phase-plan.md`) records source bundles, estimands, independent units, falsifiers, render requirements, and no-claim boundaries for MAJ-1..6 plus the parked tracks.
- 2026-07-14 (iteration 26 — extended statistical and claim-boundary audit): independently reconciled 12 live report families against their stated estimands, nesting, intervals, and token-facing claims; verified finite numerics and recomputed key differences; documented the mechanism-specific negative controls (confident-wrong/drift positive, Byzantine inconclusive, label-noise/uniform negative), non-monotone onset, observability-dependent communication, hierarchy location costs, BNN endpoint reversal, rank-effect saturation, and mixed cross-study units. Corrected the stale 74-page claim in `docs/research/manuscript-claim-audit.md` to the current 76-page render and added `docs/research/extended-statistical-audit-2026-07-14.md`. No code-level verifier blocker was introduced; the full render/test evidence remains the iteration-25 closure record above.

- 2026-07-12 (iteration 20 — warning-free large-sample rerun and manuscript truth/render audit):
  - ISC-139: replaced every test-only direct Python scalar conversion of a gradient-tracked PyTorch KL/ELBO tensor with `.detach().item()`; `uv run pytest tests/fedference/test_bnn_variational_torch.py -W error -q` → **5 passed, 0 warnings**. The first full `-W error` probe deliberately exposed a second call site, so the repair was widened to every conversion in that file before the final gate.
  - ISC-140/145: raised only the executed sampling budgets in `manuscript/config.yaml` and its example, from 60→240 independent seeds and 120→480 paired robustness trials. No estimand, rate grid, colony size, contamination mechanism, robustness label, or core algorithm changed. Regenerated evidence records `belief_sharing.n_seeds=240`, `robustness_sweep.n_trials=n=480`, and tokens `CONFIG_N_SEEDS=240`, `CONFIG_N_TRIALS=480`, `SWEEP_N_TRIALS=480`.
  - ISC-141: reran invariants, the complete analysis pipeline, token hydration, web preparation/validation, template render/validation, release build, and output validation. The 240/480 analysis completed in 116.98 s on its explicit run; all 23 manuscript figures, 29 reports, 43 web HTML files, and the 392-artifact release bundle were regenerated and validated.
  - ISC-142: the result/claim contract suite (`test_manuscript_variables`, token sourcing/tables, xrefs, captions, docs contract, experiment and sensitivity assertions) → **154 passed**. The larger run preserves every qualitative conclusion: communication lowers mean free energy (13.0916 vs 16.4676; gap 3.3760), and all four robust server settings remain positive BH-rejected winners at the verdict rate (480 paired trials; q=2.33e-80; achieved power 1.0). No results/discussion hedge required numerical reversal.
  - ISC-143: the first page-text audit caught a pre-existing renderer interaction: all 62 theorem-like `Name~\\ref{...}` references rendered as literal `Nameabout N`. Replaced the unsafe tilde convention manuscript-wide with renderer-safe spacing, updated `SYNTAX.md`, and added a fail-before-fix regression test. Also removed a literal citation-marker example and shortened one overfull list label. Fresh 73-page PDF: 0 `Nameabout`, 0 `{{`, 0 `??`, 0 literal `[@...]`, 0 undefined-reference warnings, 0 LaTeX errors, and 0 horizontal overflows above 1 pt. Four contact sheets covering all 73 page rasters were visually inspected: no clipping, collision, missing figure/table, or malformed theorem math. One LaTeX overfull-vbox diagnostic remains from float pagination; page inspection confirms no visible defect.
  - ISC-144: `uv run --extra dev pytest tests/ --cov=src --cov-fail-under=90 -W error` → **807 passed, 0 warnings, 94.54% coverage** in 1796.93 s; ruff clean, mypy clean across 84 source files, and the `src/fedference` layer-import probe clean.

- 2026-07-12 (iteration 19 — FirstPrinciples/Science/RedTeam review of `docs/`+`manuscript/`):
  - ISC-118/120/122 (docs/ 3-lens review): 10-cluster Workflow `wf_0bffa1bd-fd6` (28 agents, find→verify→implement pipeline) covered every `docs/` file not already touched earlier the same day (core/development/manuscript-docs/operations/reference/todo remainders) plus a diff-audit of 33 files edited earlier today. Journal: `/Users/4d/.claude/projects/-Users-4d-Documents-GitHub-template/ef4bb42e-2863-46b0-86ff-7d93ce256b63/subagents/workflows/wf_0bffa1bd-fd6/journal.jsonl`. Findings schema returned non-empty results for 6 of 10 clusters (empty = genuinely clean, not skipped).
  - ISC-119/121/123 (manuscript/ 3-lens review): same workflow, 6 manuscript clusters (intro-methods, results, supplement, discussion-repro, meta remainders + the diff-audit). CRITICAL finding independently re-confirmed by hand: `uv run python -c "..."` against `output/reports/robustness_sweep.json` → `accuracy_by_method_and_rate['RKL'] == accuracy_by_method_and_rate['rcce']` is `True` (exact dict equality); verdict block's `mean_accuracy_diff`/`cohens_d`/`effect_size`/p·q-values also identical for both labels. Baseline Contract conformance: `TODO.md`'s own Baseline Contract and `REDTEAM_REVIEW.md`'s Claim Boundaries were found stale against the already-corrected manuscript-wide "bounded-influence" framing (see Decisions) — fixed.
  - ISC-124 (adversarial verify): each cluster's raw findings passed through a dedicated verify-stage agent before any implement stage ran; the workflow's `allDeferred`/edit lists show findings refuted for re-litigating closed items or violating anti-criteria were dropped before implementation (see workflow result JSON, `/private/tmp/.../tasks/w1acpfz54.output`).
  - ISC-125 (anti-criteria): manual review of all 29 docs/manuscript/root-doc edits this session — zero new hardcoded manuscript numerals outside `{{TOKEN}}` (the RKL/rcce and S14 disclosures deliberately state facts qualitatively, not the literal 1.5/3 values); zero new `@citekey`s; one new label `{#tbl:nlevel3-params}` added and confirmed registered in `SYNTAX.md`'s table-label registry in the same edit; zero honesty-framing weakenings (the two TODO.md/REDTEAM_REVIEW.md fixes are corrections, i.e. *tightening* the framing to match code, not loosening it).
  - ISC-126: 29 concrete docs/manuscript/root-doc edits applied directly this session (22 via the workflow + 7 by hand: 2 RKL/rcce disclosures, 1 `SWEEP_NAIVE_ACCURACY` wording fix, S12 softening, S14 caveat, TODO.md Baseline Contract fix, REDTEAM_REVIEW.md fix), each traceable to a survived finding in the workflow result or the Advisor-driven follow-up.
  - ISC-127: 3 new `docs/todo/*.md` pages filed (`disambiguate-duplicate-divergence-robustness-constants.md` MED-2, `expose-cross-study-sensitivity-trial-count.md` MED-3, `fix-hierarchical-layers-yaml-acuity-mismatch.md` MIN-2), each following the established Status/Rationale/Scope/Implementation Notes/Acceptance Criteria/Verification Probes/Claim-Boundary Constraints/Dependencies template; `TODO.md` tier tables and `docs/todo/README.md` index both updated (11 total scoped pages, was 8).
  - ISC-128: `uv run pytest tests/test_xref_integrity.py tests/test_caption_completeness.py tests/test_docs_contract.py tests/test_token_provenance.py tests/test_manuscript_variables.py -q` → **61 passed** (0.20s first pass at 30 tests before the manuscript-variables addition, 437.89s / 61 passed at the final tree including `test_manuscript_variables.py`).
  - ISC-129: token-provenance/manuscript-variables gates above passing is the reciprocal check that no stray unresolved `{{TOKEN}}` was introduced; independently confirmed via `pdftotext` on the freshly-rendered combined PDF (ISC-131): 0 occurrences of `{{`.
  - ISC-130: `uv run --extra dev pytest tests/ --cov=src --cov-fail-under=90` (backgrounded at OBSERVE, `nohup`, log at session scratchpad `fedference_r10_baseline.log`) → **"803 passed, 1 warning in 1762.91s (0:29:22)"**, **"Total coverage: 94.54%"**, "Required test coverage of 90% reached." `src/` carried zero files with today's mtime throughout the iteration (confirmed via `stat` sweep), so this is the honest pre-existing baseline, not a regression this iteration caused or fixed.
  - ISC-131: `uv run --extra dev python scripts/z_generate_manuscript_variables.py` then `uv run python scripts/pipeline/stage_03_render.py --project active_fedference`, real output resolved to `/Users/4d/Documents/GitHub/projects/working/active_fedference/output/pdf/active_fedference_combined.pdf` (mtime 16:51 today, confirmed via the render log's own "Location:" line after an initial false read against a stale template-tree mirror — see Decisions). `pdftotext` sweep: 73 pages, 0 `??`, 0 unresolved `{{`; grepped and confirmed present verbatim: the S12 "canonical 3-level topology" softening, the S14 Study-8 trial-budget caveat, and both RKL/rcce disclosure sentences (12_methods line ~1190-1192, 19_results line ~1598).
  - ISC-132: for each of the 3 deferred (code-requiring) findings, the exact reader-facing sentence it contradicted was identified and corrected/hedged in the same pass (not merely filed): RKL/rcce → 2 manuscript disclosure sentences (12_methods_experimental_design.md, 19_results_robustness.md); hierarchical_layers.yaml mismatch → S12's "mirrors the canonical 3-level defaults" softened to "topology and priors... not read by any code path"; cross-study trial count → S14 caveat sentence added distinguishing the Study-8 reduced budget from the documented grid.
  - Advisor (Rule 2, HARD at E4): `bun Inference.ts --mode advisor --auto-state` ran to completion with a full substantive critique; every demand addressed as documented in Decisions (RKL/rcce independence re-derived, render-path gotcha caught and fixed, repo-wide bounded-influence sweep completed, per-finding reader-facing-sentence check done).
  - Cato (Rule 2a, MANDATORY at E4): invoked via `Agent(subagent_type="Cato", ...)`; `codex-cli 0.144.1` confirmed live (fresh `--version` probe, superseding the stale iteration-1 `cato_unavailable: true` premise), but the agent's final result was a truncated mid-sentence fragment with no structured verdict — **NO-SIGNAL per Rule 2a**, matching the now-recorded structural memory `gotcha-cato-subagent-truncates-after-background-launch`. Not retried (documented as structural, not transient). **This VERIFY gate is honestly reported as unsatisfied** — see Decisions and the closing summary.
  - Reflection Ledger v2: `entry_id 8fe1e25e-8fe1-4b0a-afb7-dbb94bf43da9`, `timestamp 2026-07-12T23:56:30.977Z` (schema_version 2), criteria 15/15 passed/0/0/0, `within_budget: false` (E4's <30min budget was exceeded — the R10 baseline alone ran 29:22).

- 2026-07-06 (iteration 18c — user-directed formalism/claims/visualization verification, 31-agent workflow: 11 per-formalism statement-vs-proof-vs-code checks + claims-completeness + visualization-functional lanes, adversarially verified):
  - **CRITICAL-class rendering bug found and fixed (had silently broken ALL 11 formalisms across every prior render):** inline math inside the raw-LaTeX theorem/lemma/proposition/definition/corollary environments was rendering as MANGLED text in the PDF — `$\alpha\to1$` → "alpha->1", `$L_\beta$` → "Lb eta", every `\alpha`→"alpha", `\to`→"->". Root cause traced to `infrastructure/rendering/_pdf_unicode_remap.py`: its `_MATH_BOUNDARY_RE` protects `\(...\)`/`\[...\]`/math-envs but NOT `$...$`, and pandoc leaves theorem-block math as literal `$...$` (raw-LaTeX passthrough), so the remap ASCII-fied the bare LaTeX commands. pdftotext always mangles math, which is why no prior text-scan caught it — only a page-scale RASTER read of a theorem page exposed it. Fixed project-locally (works WITH the existing remap, no shared-infra risk): converted all 106 theorem-block inline `$...$` → `\(...\)` (which the remap protects). Verified by rasterizing the lemma/proposition page: "Lemma 3 (KL is the α → 1 limit of the Rényi family) ... D_α(q‖p) = (α−1)^{-1} log Σ_k q_k^α p_k^{1−α} tends to KL(q‖p) as α → 1" and "Proposition 4 (β-loss and rcce recover NLL) ... as β → 0" now render CORRECTLY. Rendered PDF: 0 "alpha lpha", 0 "beta- >".
  - MAJOR: Figure 22 (`fig:hierarchical-bmr`) was LABELED but never `[@fig:]`-cross-referenced in the body (the xref gate only catches dangling refs, not orphan labels) → added a `[@fig:hierarchical-bmr]` reference in the S16 results prose.
  - MINOR (was MAJOR): citation `[@ashman2022partitioned; @bui2018partitioned]` inside Definition 2's raw-LaTeX block rendered as literal markdown → moved out of the block (it was already cited in the adjacent prose).
  - MINOR (green-by-construction test, exactly the class this project forbids): `test_update_factor_roundtrip` asserted only `t.sum()==1.0` (a softmax triviality) despite its name — replaced with the real PVI identity (re-multiplying the refreshed factor onto the cavity of the old posterior reconstructs the new posterior, atol 1e-12) PLUS a negative control (a wrong factor must NOT reconstruct it). The code was already correct; the test now actually binds the named property.
  - MINOR: `[@eq:tempered-softmax]` (a bare-`\label` equation inside a definition block) collapsed to the definition's number, making "the tempered softmax of eq. 1 is the minimizer of eq. 1" self-referential in 3 places → reworded to descriptive references.
  - The 11-formalism statement-vs-proof-vs-code audit found NO false or overclaiming formalism — every theorem/lemma/proposition/corollary substantively establishes its named statement against the cited code+test (verifier re-derived each); the only defects were the rendering/citation/test-hygiene issues above. Claims-completeness lane: token→report→code chains verified. Visualization lane: all generators tested + invoked; the only gap was the orphan Figure 22 label (fixed).
  - Manuscript gates green (47 passed after fixes); ruff+mypy clean across 84 files; PDF 80 pp, 0 `??`, 0 unresolved tokens, 0 literal `[@refs]`, 0 citation leaks, 0 mangled theorem math. Definitive full gate re-running.

- 2026-07-06 (iteration 18b — user-directed comprehensive quality pass, 37-agent workflow: 23 figure caption-vs-pixel re-audits + proofread + new-slice honesty re-check + token audit, each adversarially verified):
  - **All 23 figures re-audited clean** at PDF page scale (0 findings). **The six new MAJOR slices passed the adversarial honesty re-check with ZERO overclaims** — the verifier confirmed each recovery-limit/negative-control binding (c=0 influence == naive pool, sigma→0 == PointMassMLP, beta=0 == conjugate, socket bit-identity, synthetic-not-Iris labeling) and that every parent MAJOR stays OPEN.
  - 8 confirmed prose/consistency findings fixed (7 MINOR + 1 MAJOR): (MAJOR) ~20 `[@eq:]/[@sec:]` cross-refs were leaking into the PDF as LITERAL text inside raw-LaTeX theorem/definition blocks (pandoc-crossref can't reach inside raw LaTeX) → converted to LaTeX `(\ref{...})` in the 5 files with theorem environments; literal-`[@ref]` count in the rendered PDF dropped from ~20 to **0**. (MINOR) S13 "Study 3"→"Study 1" for belief sharing; 22 "five new"→"six novel studies" count match; S16 figure caption L-index collision (L0/L1 reduction indices vs the L1=location convention) bridged explicitly; 07 local redundancy tightened; internal V3/V4 tags dropped from reader-facing prose; 23 limitations rewrote a FALSE "mean-field Laplace diagonal-covariance BNN" claim (the code computes no covariance) to an honest point-estimate description that references the new VariationalMLP slice; S17 (previously an unreferenced supplement) now linked from the limitations discussion of robust_aggregate's heuristic status.
  - R15 co-actor handling: the template repo's convergent automation edited `workflow.py` (figure-registry crediting), `disjoint_fov_world.py` (str→Path return + stray `{#fig:}` docstring label removal) and others mid-session; behaviour-diffed (20 affected tests pass) and KEPT as genuine improvements, completing the co-actor's incomplete str→Path change (return annotation was left `-> str`, fixed to `-> Path` for mypy).
  - Manuscript gates green (53 passed); ruff + mypy clean across 84 source files; PDF 80 pp, 0 `??`, 0 unresolved tokens, 0 literal refs, 0 `§sec:`. Committed 979f90a (amended for lint). Definitive full gate re-running at the final tree.

- 2026-07-06 (iteration 18 — a real, tested, non-overclaiming SLICE of every MAJOR-tier item; each MAJOR stays OPEN):
  - Feasibility grounding: a 12-agent workflow (6 designers + 6 adversarial challengers) classified all six MAJORs as "partial, low faking-risk" BEFORE implementation, reproducing the key numbers (robust breakdown k=2, variational k=4; depth-4 collapse=0.0 exact; Jacobian match 4e-9). This prevented fake-yes (crippled/cosmetic slices) and fake-no (giving up on achievable work).
  - MAJ-5 (ISC-107): `run_nlevel_world` lifts the `depth in {2,3}` cap to any `depth >= 2` (each extra level a 2-state meta-context); `hierarchical_reduce` collapse/keep structure learning holds at depth 4 — non-gating top surprise 0.0 exact → prunable, informative kept (positive control). test_nlevel_depth 8 passed.
  - MAJ-1 (ISC-108, the CRITICAL headline): `heuristic_characterization.py` MEASURES robust_aggregate, proves nothing. numerical_influence_function: c=0 reproduces the naive pool's flat 1/n exactly (bound to log_linear_pool). empirical_breakdown: robust captured by k=2 colluders, variational by k=4 in the declared fixture — evidence against unconditional truth recovery, not a B-robustness theorem or global breakdown bound. Figure 23 + §sec:results-heuristic-characterization. HCHAR_* tokens. test_heuristic_characterization 5 passed; the non-monotone-at-tiny-eps behavior is reported, not asserted away.
  - MAJ-3 (ISC-109): `continuous_recovery.py` — 1-D Gaussian density-power robust generalized Bayes. beta=0 recovers the conjugate posterior bit-identically; off-corner gap shrinks monotonically O(beta) (9.9e-6→9.9e-9 over beta 1e-1..1e-4); a genuine outlier down-weighted to zero (robust mean 0.995 vs conjugate 2.388). test_continuous_recovery 6 passed.
  - MAJ-2 (ISC-110): `bnn_variational_torch.VariationalMLP` — genuine mean-field q(w)=N(mu,softplus(rho)), reparam forward, closed-form KL (== the tested per-weight gaussian_kl summed, 1976.72==1976.72), MC-ELBO. sigma→0 recovers PointMassMLP EXACTLY (independent reference); ELBO(kl=1)-ELBO(kl=0)==KL with MC draws held fixed. test_bnn_variational_torch 5 passed (requires_torch). Also fixed a stale "Mean-field Gaussian BNN" docstring on PointMassMLP the earlier rename missed.
  - MAJ-4 (ISC-111/115/116/117): `federation/socket_transport.run_socket_round` — real loopback-TCP, 4-byte length-prefixed framing, one worker thread per belief over its own TCP connection. Consensus BIT-IDENTICAL (np.array_equal, atol=0) to in-process robust_aggregate at robustness 0 and 1.5; every worker receives the broadcast. The socket slice now also supports optional HMAC-SHA256 frame integrity and persisted digest-verified replay validation via `save_socket_replay`, `load_socket_replay`, and `validate_socket_replay`, including tamper rejection for belief, frame, consensus, and broadcast digest evidence. test_socket_transport 18 passed.
  - MAJ-6 (ISC-112): `benchmark.py` + source-owned `data/synthetic_tabular.csv` (deterministic 3-class Gaussian-blob stand-in, honestly labeled SYNTHETIC — no sklearn/local Iris existed, and transcribing 150 exact rows from memory would be fabrication). Gaussian-NB federated harness: recovery identity holds at robustness=0; a MEASURED (not universal) robust edge under a 4/5 adversarial majority (naive 0.822 vs robust 0.844). Harness accepts any user CSV. test_benchmark 5 passed.
  - Honesty (ISC-113): every MAJOR row in TODO reads "SLICE LANDED … still open"; N_STUDIES stays 9 (the characterization + slices are companions, not numbered studies); no slice claims a theorem, a guarantee robust_aggregate lacks, GPU/multi-machine scale, or a real external dataset it doesn't have.
  - DEFINITIVE GATE (ISC-114): `uv run --extra dev pytest tests/ --cov=src --cov-fail-under=90` → "770 passed", "Total coverage: 95.21%" (final tree 771 with one added error-path test; a confirming run completes below). New-module coverage: heuristic_characterization 96.6%, continuous_recovery 90%+ (after error-path test), bnn_variational_torch 97.9%, socket_transport 97.6%, benchmark 97.9%. ruff + mypy clean across 83 source files (was 78). Combined PDF 80 pp, 0 dangling refs, 0 unresolved tokens, 0 literal `§sec:`; renders "113 of 114 acceptance criteria" + coverage 95.21. Release bundle rebuilt + verified; metadata emitter-consistent. Work checkpointed to git before the wave (commit a3a88e6).

- 2026-07-06 (iteration 17 — improvements & additions from the forward roster):
  - MED-2 (ISC-100): `src/publication/metadata.py` emits CITATION.cff/.zenodo.json/codemeta.json from the single `manuscript/config.yaml` publication block + pyproject version. `scripts/emit_metadata.py --check` → "consistent: CITATION.cff, .zenodo.json, codemeta.json" (exit 0); `--write` regenerated all three (they were hand-drift before). Tests: test_publication_metadata 5 passed incl. `test_check_detects_a_tampered_surface` (a version tamper is flagged) and `test_real_repository_surfaces_are_emitter_consistent`. The .zenodo.json `related_identifiers` direction fix from iteration 16 is now emitter-owned (can't silently regress).
  - MED-1 (ISC-101): `src/publication/release_manifest.py` + `scripts/build_release.py` write `output/release/{manifest.json,sha256sums.txt,README.md}`. Real run: "output/release written: 379 artifacts, 14403386 bytes"; `--verify` → "release bundle verified". Tests: test_release_manifest 5 passed incl. `test_verify_detects_tamper_and_shasum_agrees` (both our verify AND `shasum -a 256 -c` fail on a tampered artifact) and `test_manifest_json_is_sha256sum_c_compatible`. README counts derived from the walk, not hand-typed.
  - MED-3 (ISC-102): `.github/workflows/ci.yml` maps the local profile to GitHub Actions — ruff, mypy, `pytest --cov=src --cov-fail-under=90`, `emit_metadata.py --check`, invariants — each step is the verbatim local command.
  - MAJ-7 (ISC-103/104/105): `bayesian_model_reduction.hierarchical_reduce` scores per-level Bayesian surprise KL(posterior‖empirical prior). Directional oracle (test_hierarchical_bmr, 6 passed): on a degenerate world whose top meta-context is non-gating (identical conditioned priors) the top level earns surprise 0.000 → prunable → `recommended_prune == 0`; on an informative world (distinct conditioned priors) the same level earns 0.328 → kept. The two worlds differ ONLY in L3's conditioned priors, so the opposite verdict is not green-by-construction — `test_prune_flag_separates_the_two_worlds` pins the discrimination. A count-collapse ΔF and marginal-likelihood-from-prior probe were tried and rejected (timeboxed per R9) before landing on the surprise signal; the dead ends are noted. `run_hierarchical_bmr` writes `hierarchical_bmr.json`; `generate_hierarchical_bmr` draws Figure 22 (verified at pdftoppm page scale — L0 prunable at 0, kept at 0.328, both worlds keep L1); HBMR_* tokens hydrate §sec:results-hierarchical-bmr (S16). Framed as a structure-learning companion, NOT a numbered study — N_STUDIES stays 9 (grep "Study 10" == 0). tests/figures/test_hierarchical_bmr 2 passed.
  - Manuscript gates green with S16: xref/caption/token-provenance/token-tables/manuscript-variables/docs-contract 53 passed. Final PDF: 79 pp, 0 `??`, 0 unresolved tokens, 0 literal `§sec:`/`@fig:`/`@eq:` (232 resolved `sec. N`; the only literals were pre-existing raw-LaTeX theorem-block refs). A literal `§sec:results-emergence` I introduced in S16 (a `str.replace` fake-pass) was caught by a page-scale raster read and fixed → re-rendered clean; learning saved to memory.
  - Stale-contract gate fixed: `test_scoped_todo_pages_are_complete` asserted every scoped page says "State: Open" forever (forbade recording completion) → relaxed to `State: (Open|Done)`; the four completed pages (MED-1/2/3, MAJ-7) now marked Done.
  - ruff + mypy clean across 78 source files after all additions.
  - DEFINITIVE GATE at the final tree: `uv run --extra dev pytest tests/ --cov=src --cov-fail-under=90` → "735 passed in 1558.13s", "Total coverage: 94.96%". `coverage_project.json` regenerated (94.96). New-module coverage: bayesian_model_reduction 98.4%, publication/metadata 97.3%, publication/release_manifest 95.7%, figures/hierarchical_bmr 100%, experiments/worlds 94.0%. Final render: PDF 79 pp, 0 `??`, 0 unresolved tokens, 0 literal `§sec:`; renders live "105 of 106 acceptance criteria" + coverage 94.96 + 735 tests. Release bundle rebuilt (379 artifacts) + verified; metadata emitter-consistent. This iteration delivered the entire remaining MEDIUM tier (MED-1/2/3) plus MAJ-7 — the only MAJOR with crisp falsifiable acceptance criteria; MAJ-1/2/3/4/5/6 remain genuine multi-week research (GPU BNN port, continuous state, real multi-machine networking, external benchmark) out of a single session's scope.

- 2026-07-06 (iteration 16b — user-directed workflow close-out, 49 agents: 21 per-figure caption-vs-pixel audits at PDF page scale + 3 sweep lanes + 24 adversarial verifiers + completeness critic; 23/24 findings survived refutation, all fixed):
  - Figures: 14/21 audited fully clean. Fixed: hierarchical_pomdp panel 5 replayed a single scalar as a fake "cumulative trajectory" when reports were supplied (CRITICAL) → honest two-bar measured-gap comparison + caption rewrite; disjoint_fov_world right-panel title asserted a "Benefit" its own annotation negates → "EFE vs random navigation (null result)" + caption now states the right panel's separate 2-agent/4-position configuration via V4_EFE_* tokens; unattributed in-figure "(Fig. 5)"/"(Fig. 7)"/"(Eq. 2)" (colliding with the document's own numbering) → attributed "Friston et al. 2024, …"; language_kl_decay caption off-by-one vs the 25 drawn markers → "{{LANGUAGE_N}} recorded points" wording; system_overview Panel C label collision moved; 01_introduction cross-ref described a different figure than the schematic → rewritten to the actual panels; robustness_sweep caption notes robust-curve occlusion.
  - Supplements sweep (5 findings fixed): S13 pattern-1 overshoot from this session's own rewrite (grid max lives in the SECOND-lowest acuity row, 0.586 at n=10) → "low-to-moderate acuity, peaks in the second-lowest row"; S13 pattern-3 now names the tails; S14 seeding-formula claim scoped to the belief-sharing sweep only (hierarchical passes the constant base seed — its own docstring says so); supplement 27 "holds up to (and at) rate 1" → measured-on-the-swept-grid wording + qualitative vertex pin; 27's rounded-token factor inconsistency (0.143/0.001≠267.1) → "below …, computed from unrounded influences".
  - Rendered surfaces lane: CLEAN — web/slides carry all corrected claims, 0 unresolved tokens, figure assets byte-matched.
  - Tests-bind lane: system-overview colony counts were NOT bound to the drawn colony → test now asserts n_agents == len(beliefs), n_adversarial == suppressed-weight count, n_honest consistency; ISC_EFE_TOLERANCE now falls back to a literal for non-power-of-ten tolerances.
  - Completeness critic: (1) output/docs/api_glossary.md was 10 days stale (generator ImportError silently swallowed in the standalone env) → regenerated from the template env (67 diff lines; includes the experiments/ subpackage + renamed class); (2) .zenodo.json related_identifiers claimed "isCitedBy" for FedGVI/Friston (false direction) → "cites"; (3) README "pure NumPy/SciPy" scoped, torch complement + colonies.py added to the module map; (4) TODO baseline 712→716 tests; (5) CITATION.cff abstract etc. remain open per the project's own MIN-1/MIN-2 roster (acknowledged, not silently claimed).
  - Root-of-misnomer fix: bnn_baseline_torch.py itself claimed "mean-field Gaussian BNN" (class MeanFieldBNN) while implementing a deterministic point-estimate MLP → class renamed PointMassMLP (all 9 call-sites), docstrings rewritten honestly, README/API-glossary synced; tests 6/6 green.
  - ISC-count honesty: _count_isc ignored [DEFERRED-VERIFY] rows, rendering "98 of 98 verified" while ISC-89 is deferred → parser counts deferred toward total (98 of 99), pinned by test_count_isc_deferred_rows_stay_in_the_denominator.
  - Final artifacts: combined PDF 78 pp, 0 dangling refs, 0 unresolved tokens; all pre-fix defect phrases scan to 0 in pdftotext.
  - DEFINITIVE GATE at the final source tree: `uv run --extra dev pytest tests/ --cov=src --cov-fail-under=90` → "717 passed in 1606.40s (0:26:46)", "Total coverage: 94.83%". `coverage_project.json` regenerated from this run (94.83). Corrected-path coverage: statistics.py 100%, experiments/robustness.py 100%, figures/hierarchical_pomdp.py 100%, manuscript_vars/generate.py 96.4%, loaders.py 91.9%. Final render post-gate: PDF renders live-derived "98 of 99 acceptance criteria", coverage 94.83, 717 tests; `ruff check` "All checks passed!", `mypy src/` "Success: no issues found in 75 source files".

- 2026-07-06 (iteration 16 — comprehensive review, artifact tokens):
  - ISC-72/73: `tests/test_token_provenance.py:53` + `tests/test_token_tables.py:49` both read `sorted(_MANUSCRIPT.glob("[0-9]*.md")) + sorted(_MANUSCRIPT.glob("S[0-9]*.md"))`; gates green in the 712-pass run.
  - ISC-74: `src/manuscript_vars/generate.py:247-249` `from fedference.experiments.belief_sharing import DEFAULT_ACUITY` (single def at `belief_sharing.py:23`).
  - ISC-75/85: `tests/test_manuscript_variables.py:270-309` asserts `SYSTEM_OVERVIEW_METADATA["naive_acc_pct"] == round(100.0 * float(naive[TRUE_STATE]))` (chain token→metadata→recomputed-from-beliefs; non-circular).
  - ISC-76: `EFE_IDENTITY_ATOL` defined `expected_free_energy.py:47`, consumed by `tests/fedference/test_expected_free_energy.py` (6 uses), `src/invariants.py` (tol + passed), and the `ISC_EFE_TOLERANCE` token (`generate.py`).
  - ISC-77: python sweep over `src/manuscript_vars/*.py` token assignments → 0 result-shaped literals; V4 + moving-world loaders converted to strict `report[...]` access.
  - ISC-78/88: FINAL gate `uv run --extra dev pytest tests/ --cov=src --cov-fail-under=90` → `715 passed` (one draft-mode expectation updated to the new honest N/A, re-run green), `Total coverage: 94.86%`; last full gate re-running post-strict-access fix (log `/tmp/fedference_final_gate2.log`).
  - ISC-79/86: all 21 PNGs regenerated this session before any visual read (pipeline log `/tmp/fedference_pipeline.log`; newest PNG 09:23).
  - ISC-80: prior iteration-15 session reviewed all 21 figures (2 agents) and fixed 10; this session re-verified the fixed keystones on fresh renders at PDF page scale — cover (26%/70%/+44pp derived values, no fabricated stars), Figure 16 (previously-empty gap panel now populated: bars −0.91/−1.10 vs zero line), Figure 20 (9 studies, Studies 5/6 at ≈0 as captioned); caption-completeness + palette + xref gates green.
  - ISC-81/82/83/93: combined PDF 78 pp / 2.76 MB; `pdftotext` scan: 0 `??`, 0 unresolved `{{TOKEN}}`; cover on page 1 verified by `pdftoppm` raster read.
  - ISC-84: number changes this session are accuracy fixes with dispositions (LANGUAGE_FINAL_KL 0.0030→0.0027 off-by-one fix; SWEEP_HEADLINE_N_FOR_TARGET_POWER 1→5 feasibility floor; COVERAGE_PERCENT 94.90-stale→94.69/94.86-fresh); figure-only edits changed none.
  - ISC-87: `ruff check src/ tests/ scripts/` → "All checks passed!"; `mypy src/` → "Success: no issues found in 75 source files".
  - ISC-90: mtime sweep post-pipeline: 22 reports, all token-consumed reports fresh; 4 stale files (artifact_manifest, evidence_registry, output_statistics, validation_report) consumed by NO token loader, refreshed anyway via `validate_outputs.py` + `validate_all.py package` → "validation_result: PASS (52 files)", "web_package: PASS".
  - ISC-91: absolute-path scan (40 files asserted scanned — non-vacuous) → 6 hits, 3 tokenized to `{{N_STUDIES}}`, 3 dispositioned structural ("three levels" = the fixed 3-level design).
  - ISC-92: config.yaml free-text carries no numeric scientific claims (numbers present are consumed `experiment:` parameters).
  - ISC-94..99: see iteration-16 Decisions entry; proof-of-detection tests: `test_sweep_fdr_alpha_is_consumed_not_decorative` (strict α=1e-12 kills all rejections), `test_sample_size_for_power_saturated_effect_returns_feasible_floor` (n=5/6, pre-fix returned 1), `test_default_project_root_is_repo_root_not_src` + `test_missing_stage_timings_degrades_to_na_and_writes_nothing` (pre-fix: root=src/, fabricated zeros file).
  - Determinism probe (advisor demand): two consecutive full `02_run_analysis.py` + token runs → 313 tokens, **0 non-volatile diffs** (volatile set = timestamp + 5 stage durations).
  - Negative controls (advisor demand): `test_paired_test_no_effect_for_symmetric_offsets`, `test_bh_fdr_rejects_nothing_when_all_large`, near-null group not rejected (`test_statistics.py:254`), fdr strict-α kill test — the verdict machinery demonstrably CAN return null.
  - Advisor (Rule 2): `Inference.ts --mode advisor` ran against the final artifact set; demands (reciprocal token check, negative control, determinism probe, metadata/slides sign-off, corrected-path coverage) each addressed: token-provenance gate IS the reciprocal check (green); slides/web regenerated 09:28-09:29 this render; CITATION.cff/.zenodo/codemeta/pyproject cross-consistent (docs-audit probes); corrected paths exercised by the new consumption tests.
  - Forge (Rule 2a): Batch A verdict `concerns` — items 2/3/5/9 explicitly not refuted with file:line evidence; item 1 concerns (abstract not token-bound; loader silent `.get(...,0)`) BOTH FIXED this session (abstract now renders `{{MOVING_SIGNIFICANCE_VERDICT}}`; moving-world loader strict). Batch B deferred (codex quota) — see ISC-89.

- 2026-06-25 (iteration 4 — server-side rigor + breadth, hand-authored then RedTeam-verified): landed the **headline Tier-3 scientific upgrade** and four Tier-2 breadth items. (1) `aggregation.variational_aggregate` + `aggregation_free_energy`: the server-side aggregator is now the exact block-coordinate minimizer of a stated free energy `F(q,a)=Σ a_n CE(q,s_n) − H(q) + (1/c)KL_gen(a‖w)` — proven monotone descent (max single-step ascent 4.4e-16, machine zero), stationary-point convergence (gradient <1e-6), exact log-linear-pool recovery at c=0 (shares ISC-10 corner), and what this historical entry originally called a bounded-influence guarantee (a diverging agent's normalized weight fell 267× on the tested path). **Superseding correction (iteration 19):** the theorem proves the raw effective-weight bound; the normalized-weight trace is an empirical redescending diagnostic, not an estimator-level B-robustness proof. HONESTY: this is a DIFFERENT, conservative (max-entropy) aggregator — it does NOT beat naive on peak accuracy (verified ~0.14 vs 0.90) and does NOT retroactively give the sharp reverse-KL heuristic an objective. The two-axes caveat is rewritten as a THREE-way honest triangle (axis 1 source-conditional client theorem; axis 2 accurate-but-unguaranteed server heuristic; axis 3 objective-backed, raw-weight-bounded, but conservative server variational). (2) Breadth: `contamination` gains byzantine (multiplicative log-odds tilt) + drift (round-scaled) models; `divergences` gains out-of-scope Gaussian KL/Rényi closed forms; `bayesian_model_reduction` gains `greedy_reduce` multi-hypothesis structure learning. (3) Two new figures (`aggregation_descent`, legacy-named `bounded_influence`) wired through the analysis workflow + `VARIATIONAL_*` tokens. (4) Sample sizes raised n_seeds 20→30, n_trials 40→60; verdict holds stronger (all robust win, q=1.6e-11, power 1.0); pipeline 7s. (5) Manuscript: new methods subsection {#sec:method-variational}, two modular supplements (27 derivation+Theorem, 28 extended methods), results subsection {#sec:results-variational}, triangle rewrite of limitations/future, 6th contribution. Suite: **482 passed, 98.11% coverage**; 0 dangling cross-refs; token gate green.
- 2026-06-25 (iteration 5 — all MINOR + one MEDIUM, hand-authored, tool-verified): closed the entire Tier-1 MINOR backlog and the contamination-gallery Tier-2 item. (1) New gate tests: `test_caption_completeness` (every figure caption states axes + uncertainty; fixed 6 pre-existing captions), `test_docstrings` (100% public-API docstrings in `src/fedference/`; backfilled 3 dirichlet properties), `test_xref_integrity` (0 dangling `@eq/@fig/@tbl/@sec` refs), `test_token_provenance` (no hardcoded decimal/percentage in prose — tokenized the CI level as `CI_PERCENT`), `tests/figures/test_palette` (no inline hex outside `_common.py`; moved the one `#7F8C8D` literal into `COLOR_MUTED`). (2) Additional numerical edge tests: Rényi α→0⁺/∞ limits, degenerate one-state + all-mass-on-one pmfs into both aggregators, single-agent robust pool, dispatch errors. (3) Contamination gallery: `run_contamination_gallery` (reuses sweep helpers; `run_robustness_sweep` untouched) + `generate_contamination_gallery` figure + `GALLERY_*` tokens + supplement §sec:supp-gallery, with the honest directional-vs-entropy verdict (robust beats naive under confident_wrong/byzantine/drift; entropy attacks uniform/label_noise leave naive undegraded). Suite: **512 passed, 98.20% coverage**; ruff check + mypy clean; combined PDF 0.88 MB; TODO deeply re-scoped (MINOR exhausted, new MINOR/MEDIUM surface added with acceptance probes, three-axis standing rules).
- 2026-06-25 (iteration 5b — audit-response, hand-authored from workflow wiwa3kbxp): a parallel audit workflow (4 lenses) found the iteration-5 contamination gallery OVERCLAIMED — "robust beats naive under every directional attack" was a single-seed boolean that held for only ~21/30 seeds (byzantine's multiplicative tilt escalates to a veto cliff). FIXED HONESTLY: `run_contamination_gallery` is now **seed-aggregated** (default 16 seeds × 20 trials) and reports, per mechanism, the mean robust-minus-naive difference, a bootstrap CI over seeds, and the across-seed *win fraction*; a mechanism is `reliably_beats` only if win fraction ≥ 0.95 AND the difference-CI excludes zero. Result is now seed-stable: `reliable_kinds = [confident_wrong, drift]` (win 1.00, CI>0) across starting seeds 0/100/500; byzantine correctly NOT reliable (win ~0.62, CI spans 0); entropy attacks conservative. Manuscript §sec:supp-gallery + figure rewritten to the honest claim (no categorical "every"; byzantine veto-cliff caveat stated). The audit also flagged gate under-strictness (letter vs spirit); HARDENED: provenance lint now catches comma/bare-dot/scientific spellings + documents its scope; docstring gate requires ≥12 substantive chars (not a "." placeholder); palette gate catches 3/4/8-digit hex + rgb tuples/strings (caught 6 pre-existing `#222`/`#555` greys → added `COLOR_AXIS`/`COLOR_GRID`); caption gate adds an anti-stub length + axis-description check. The xref gate was judged the strongest (no bypass). Suite stayed green throughout.
- 2026-06-25 (iteration 6 — new MINOR + 3/4 MEDIUM, hand-authored, tool-verified): closed the four new Tier-1 MINOR items and three Tier-2 MEDIUM items. MINOR: (1) exhaustive contamination-class gate (`_DIRECTIONAL_KINDS ∪ _ENTROPY_KINDS == contamination._KINDS`, disjoint); (2) docstring gate widened to `src/analysis` + `scripts` (backfilled 5 script `main()` docstrings); (3) palette **greyscale-safety** assertion — which FORCED A REAL FIX: COLOR_NAIVE/COLOR_ROBUST (red #C0392B / blue #1F77B4, ΔL≈0.025, indistinguishable in greyscale) recoloured to deep-brick-red #922B21 / sky-blue #5DADE2 (ΔL≈0.30); added COLOR_AXIS/COLOR_GRID for the `#222`/`#555` greys the stricter palette gate surfaced; (4) token-table arity lint (every `*_TABLE_ROWS` matches its header column count). MEDIUM: (MD4) `run_robustness_sweep` gains an opt-in `kind` parameter (byzantine/drift; `confident_wrong` default bit-identical, verified); (MD2) `generate_descent_comparison` figure + report + tokens + supplement embed — single-start capture (F≈high) vs multi-start escape (F≈low) on a near-vertex colony, making the iteration-4 fix visible; (MD3) `run_robustness_onset` + `generate_robustness_onset` small-multiples figure + tokens + supplement §sec:supp-onset — per-mechanism naive-vs-robust accuracy curves over the rate grid with the onset rate, showing byzantine's transient window vs the sustained additive-attack robustness. MD1 (HTML theorem-box styling) PROBED and found genuinely infra-level (pandoc drops the raw-LaTeX `\newtheorem` blocks in HTML; a project-markdown fix would break the working PDF) — re-scoped honestly in TODO, NOT faked, primary PDF surface unaffected. Combined PDF 1.05 MB; ruff + mypy clean; all manuscript gates green.
- 2026-06-25 (iteration 6b — audit-response, from workflow w9p4z6iv4): a 4-lens audit confirmed the `kind`-param default is bit-identical (verified structurally: the confident_wrong branch consumes zero RNG), the byzantine/onset claims are honest and seed-stable, and the MD1 HTML-theorem deferral is legitimate (pandoc genuinely drops the raw-LaTeX theorem blocks; PDF carries them). It found ONE hard blocker + four lower issues, all FIXED: (H) the descent-comparison figure shipped with `capture_gap = 0.0` (the full n_agents colony does NOT capture single-start, so the "capture basin vs vetoing basin" figure showed two identical curves while the prose claimed a gap) → rebuilt on a capture-prone 2-honest + near-vertex-liar colony (now SINGLE_F≈1.31 vs MULTI_F≈−0.23, gap≈1.54) + a guard test asserting `capture_gap > 0.05` so it cannot regress silently. (M1) palette gate missed non-hex inline colours (named CSS, `tab:`, `C0-9`, greyscale strings, rgb tuples) → hardened to catch all of them while allowing structural white/black/none and colormap names; verified against the audit's exact bypasses. (M2) the gallery default `kinds` was a hardcoded literal, not registry-derived → now `kinds=contamination._KINDS`, and the partition test asserts `by_kind.keys() == set(_KINDS)`, so a new model cannot silently drop. (L1) ROBUST_CYCLE luminance check was adjacency-only → now all-pairs (≥0.025). (L2 docstring) substance bar raised to ≥2 words. ruff + mypy clean; combined PDF 1.05 MB.
- 2026-06-25 (iteration 7 — last MEDIUM resolved via infrastructure, not project code): the lone remaining MEDIUM (HTML theorem rendering) was fixed UPSTREAM in the template repo (`docxology/template`): `infrastructure/rendering/web_renderer.py` now rewrites raw-LaTeX `\begin{theorem|lemma|proposition|corollary|definition}` blocks **web-only** into numbered, shared-counter `.theorem-box` Divs with embedded CSS (`_html_theorem_blocks` + `TestTheoremBlocks`, 5 cases; rendering infra suite 771 passed, web_renderer coverage 73%, ruff+mypy clean). This is a GENERIC, reusable infra capability — this project required NO markdown change. Verified: §sec:supp-variational's Theorem + Definition (previously absent from `output/web/*.html`) now render with `.theorem-box` styling; the PDF path is unchanged (still uses the LaTeX preamble) and beamer slides already rendered theorems natively. Per the user's directive, the higher-order change was made in infrastructure (modular/documented/tested), not as a project-specific hack. The project's regenerated `output/web/*.html` is committed here. All MINOR + MEDIUM tiers are now exhausted; only Tier-3 MAJOR remains (deferred).
- ISC-1..14: `uv run --extra dev pytest tests/test_fedference_core.py -q` → `14 passed in 0.14s` (2026-06-24). Tool-verified.
- ISC-15..37: authored by background workflow `w4nhq9rpb` (13 agents), then **independently re-run on HEAD** (Gate J): `uv run --extra dev pytest tests/ --cov=src` → `312 passed, 0 failed, coverage 97.08%` (2026-06-24). Project ≥90% gate met. Core modules 95–100% after supplementary edge tests (tests/test_fedference_core_edges.py). Workflow fixed 3 breadth defects (agents marginal pushforward, EFE deterministic-zero entropy, a degenerate contamination test) without touching the locked core. Tool-verified.
- Caveat carried forward (RedTeam finding): the robustness-sweep verdict rests rigorously on axis (1) β/rcce generalised-Bayes; axis (2) aggregation reweighting is a complementary heuristic. Manuscript must not conflate the two.

### Iteration 33 verification (2026-07-17)

- ISC-170: Bash — `uv run mypy src/` → "Success: no issues found in 89 source files" (was 5 errors in language_kl_decay.py:89-91, disjoint_fov_world.py:124,165; fixed at ingestion point in `_common.py`).
- ISC-173/174: Grep + pytest — `rg "SLICE LANDED|Publication-polish closure|iteration \d" TODO.md` → no matches; rewritten TODO.md is index-only (per-item MAJ sections removed as R14-in-prose duplication).
- ISC-175/176/178: pytest — `uv run pytest tests/test_docs_contract.py -q -k "todo or forward or scoped or roadmap"` → "5 passed, 17 deselected in 0.76s" (covers hardened completeness fields, TODO↔pages and README↔pages bidirectional links, 12-page set incl. new typed-report-and-figure-schemas.md).
- ISC-177: Forge proof-of-detection — injected "- [x] SLICE LANDED …/649 passed" into TODO.md → `1 failed … AssertionError: assert ['TODO.md'] == []`; byte-exact restore (md5 831df96fc0c11cf0e6266338f76da98c) → "1 passed, 21 deselected in 0.17s".
- ISC-185: terminology audit agent (read-only, whole-tree sweep) → genuine residue list EMPTY; all stub/fake/legacy/dummy hits are negative-control guard machinery (`tests/test_runtime_surface.py` `_RETIRED_MARKER`), no-mocks policy prose, or synthetic-world seeded-computation terminology — preserved per plan.

- ISC-171: Bash — final-tree `uv run ruff check src/ scripts/ tests/` → "All checks passed!"; `uv run mypy src/` → "Success: no issues found in 89 source files".
- ISC-179/191: Workflow journals — wf_b42fd381-d90 (36 agents: 8 finders → verifiers → fixers; 41 applied, 12 declined-with-reason, all declined items applied by primary or SYNTAX agent) and wf_db9fdc22-498 (15 agents: 7 improver→auditor pipelines, verdicts clean/repaired, 0 unresolved concerns after primary fixes to 21/23).
- ISC-180/182/184/193: pytest — post-edit battery "36 passed" (claim audit + xref + docs contract) and "41 passed in 210.13s" (captions + token provenance + variables).
- ISC-181: Bash — `grep -r "{{" output/manuscript/` → only BibTeX brace-protection in references.bib; zero unresolved tokens.
- ISC-183: SYNTAX agent — 4 missing keys (koehler2009mcse, mildner2025rates, morris2019simulation, loy2021lmeresampler) confirmed in references.bib + cited, appended to fixed key registry; xref suite green.
- ISC-186: Read (raster) — pages 1/13/25/35/42 of the 77-page PDF read as images: shared-counter theorem numbering (Lemma 3, Prop 4, Thm 5, Cor 6, Prop 7), recovery-residual table (0 / 5.55e-17), sweep figure with hydrated tokens (0.6719/0.7846) and CIs, zero "??" in full text layer.
- ISC-187: Bash — 02_run_analysis (exit 0, 29 reports + 33+ figures + registry), z_generate_manuscript_variables (exit 0), template 03_render_pdf → "Combined manuscript PDF successfully generated! 6.44 MB, Valid PDFs: 1/1", 04_validate_output → "VALIDATION COMPLETE - All checks passed!", 05_copy_outputs complete; web+slides mtimes fresh.
- ISC-189: contract — figure-registry test green (registry↔embeds↔generator metadata), no label renames (xref green), no filename changes in git diff.
- ISC-190/194: audits — wave-2 auditors repaired 3 overclaim/precision drifts (13 estimand sentence, 14 falsification-surface wording, 23 terminal-reversal attribution); primary softened 21's cross-study rows claim to match the caption; claim-audit suite green after all edits.
- ISC-192: pytest — figure suite "97 passed in 35.95s" after 3 font-floor fixes (pomdp_loop 8.0→8.5, heuristic_breakdown 9→9.5, belief_heatmap labelsize 9→9.5); analysis regenerated after.
- ISC-195: Bash + pytest — template renderer fix: regression A/B pre-fix boxes=0 → post-fix 11 boxes in output/web/index.html (HEAD had 10; the federation proposition broken-at-HEAD now renders); template suite "41 passed" incl. new test_theorem_block_with_same_line_label_is_rewritten_with_anchor; box-body diff vs HEAD byte-similar (degradation pre-existing, not amplified).

### Governance (iteration 33)

| field | value |
|---|---|
| authoritative_baseline | full suite 876 passed / 2 transient link failures (mid-edit ENG-1 page race) / 93.09% cov, 941.88s, R10 background run bw6ve22lb |
| environment_probe | codex-cli 0.144.1 live; uv/pytest/ruff/mypy via project venv; template repo render probe TEMPLATE_OK |
| observed_failures | caption test vs stale hydration (fixed by regeneration); web theorem-box drop (fixed in template renderer) |
| change_surface_manifest | manuscript/*.md (41+wave2 edits), TODO.md, docs/todo/* (12 pages+README), tests/test_docs_contract.py, src/figures/{_common,pomdp_loop,heuristic_breakdown,belief_heatmap}.py, ISA.md; template repo: web_renderer.py + its test |
| residue_scan | terminology audit genuine=0; unresolved-token grep clean |
| known_bad_case | "SLICE LANDED"/"649 passed" injection into TODO.md |
| pre_result | test FAILED (AssertionError: ['TODO.md'] != []) |
| post_result | byte-exact restore → 1 passed; final surfaces green |
| production_entrypoint | scripts/02_run_analysis.py → z_generate_manuscript_variables.py → template 03/04/05 render/validate/copy |
| coactor_isolation | disjoint file ownership per agent; index/TODO.md edits reserved to primary; template repo touched only web_renderer.py+test |
| owned_paths | ["manuscript", "docs/todo", "TODO.md", "tests/test_docs_contract.py", "src/figures", "ISA.md", "output"] |
| visual_verification | raster Reads of PDF pages 1/13/25/35/42 + web theorem-box body diff vs HEAD |
| long_pole_command | uv run pytest tests/ --cov=src --cov-fail-under=90 |
| verifier_failure_count | 2 (stale-hydration caption test; web theorem-box regression) — both fixed and re-proven |
| final_gate_run_count | 1 (definitive run at final tree, in flight at governance-write time; result recorded at ISC-172) |
| premise_provenance | see table below |

### Premise Provenance (iteration 33)

| Premise | Generator | Observed At | Evidence Token | Status |
|---|---|---|---|---|
| mypy has exactly 5 figure-layer errors | uv run mypy src/ | 2026-07-17 session | "Found 5 errors in 2 files (checked 89 source files)" | verified |
| ruff green pre-change | uv run ruff check | 2026-07-17 session | "All checks passed!" | verified |
| contract heading set + markers | Read tests/test_docs_contract.py | 2026-07-17 session | file read in full, lines 280-298 | verified |
| SLICE LANDED sites (6 TODO.md rows + 4 page State lines) | rg sweep | 2026-07-17 session | grep output quoted in transcript | verified |
| codex live | codex --version | 2026-07-17 session | "codex-cli 0.144.1" | verified |
| verdict-accuracy tokens exist | rg on output/data/manuscript_variables.json | 2026-07-17 session | SWEEP_BEST_VERDICT_ACCURACY_MEAN et al. listed | verified |
- ISC-172: Bash — definitive full suite at final tree (single run, Rule 4): "880 passed in 894.34s (0:14:54)", "Total coverage: 93.33%", exit 0 (task bacjjvn6u).
- ISC-188: RE-READ + DELIVERABLE COMPLIANCE — D1..D7 each cite captured artifact tokens (see governance + per-ISC entries); no unbacked ✓.
- Reflection Ledger v2: entry_id 9195bcde-d304-4585-aca3-7102181f096a, 2026-07-18T01:55:32.727Z (26/26/0/0/0, E4 classifier).
- Cato (Rule 2a): NO SIGNAL ×2 — first run truncated mid-audit after back-link verification ("scholarship-and-phase-plan.md:150 'iteration' hit... doesn't trip the regex"); bounded resume-retry truncated again after confirming claim 4 ("manuscript repeatedly denies robust_aggregate a bounded-influence guarantee and frames variational_aggregate as conservative"). No structured JSON verdict; gate recorded UNSATISFIED per R18, not substituted.

- conjectured (iteration 33): a manuscript polished across two adversarially-audited agent waves would have no remaining cross-surface rendering defects.
  refuted_by: the freshly regenerated web surface silently dropped all 11 theorem environments — the standard amsthm same-line \label idiom (added so PDF \ref{}s resolve) fell outside the template web renderer's block regex; PDF perfect, HTML lossy, validators green throughout.
  learned: transformed-content surfaces need countable invariants (rendered-box count vs source-environment count), because extraction- and validator-level checks cannot see silent structural drops; fixed at the infrastructure level (regex + anchor id + unit test) per the iteration-7 precedent.
  criterion_now: ISC-195 (web theorem-box count > CSS baseline, template unit test green).
- ISC-196: Read/Grep — src/analysis/report_schemas.py (31K, TypedDict + explicit checker, 18 table schemas + bnn_torch union + figure_registry) wired at workflow.py:117-118 `if schema is not None: report_schemas.validate_report(schema, payload)` inside `_write_json`, covering all 19 report writes + registry; 15 `check_figure_contract` call sites. Provenance note: implementer narrative claimed nothing landed (codex mechanism down) while artifacts exist and pass — recorded as muddy provenance, code verified directly.
- ISC-197: pytest — proof-of-detection probe: `ReportSchemaError: belief_sharing payload missing required field 'n_agents' (expected int)` raised from the real `_write_json` before any disk write; 173-test suite tests/analysis/test_report_schemas.py → "191 passed in 167.33s" with tests/analysis/.
- ISC-198: pytest — all 20 FIGURE_DEPENDENCY_CONTRACTS pairs validated against real committed reports; stripped-copy negatives name generator+report+field; contract-vs-schema consistency test included.
- ISC-199/208: Bash — run-vs-run determinism: two consecutive publication-profile `02_run_analysis.py` runs with write-boundary validation active produced 24/24 value-identical reports (JSON deep-diff NONE). Validator purity confirmed by code read (pure check before serialization, no payload mutation). Note: the naive pre/post-schema snapshot comparison was confounded — a wave-3 actor had overwritten output/reports with smoke-profile data (n_seeds=4); the parity rerun restored publication scale (n_seeds=240) and the contamination is documented here rather than hidden. ENG-1 full acceptance met → page deleted under the Removal Rule; evidence lives in ISC-196..199/208.
- ISC-204: pytest — Removal Rule honored for ENG-1: page deleted, TODO.md Engineering table and README row/total removed; `uv run pytest tests/test_docs_contract.py -q` → "22 passed in 1.97s" (bidirectional link + completeness gates derive from disk and stay green).
- ISC-206: pytest + Bash — release fingerprint slice: `tests/test_release_manifest.py` → "11 passed in 1.47s" (8 pre-existing tamper tests untouched + 3 new); real-tree `build_release.py` → "444 artifacts" and `--verify` → "release bundle verified"; negative control quoted: "MISMATCH: manifest: provenance fingerprint mismatch — bundle records 646f3bb318f7 but the current source/config tree computes d077764800a3 (...) the bundle is stale relative to the tree".
- ISC-207: pytest — docs contract green after MIN-1 page neutral update (quoted tail below); banned-marker regexes find nothing; page retains State: Open + all contract fields.
- ISC-209/210/211: Workflow wf_5d981049-fd3 (12 agents, 6 improve + 6 audit) — 26 doc files improved, 7 violations repaired in-flight (incl. README mermaid label granting variational_aggregate "rigorous" → corrected to "objective-backed"; dead test-path guidance; overstated study-count comment; British spellings); post-settle oracle battery `test_docs_contract + test_documentation + test_manuscript_claim_audit + test_xref_integrity` → "40 passed in 1.66s". Capability coverage per group reports: schema boundary (README, architecture.md, agent_instructions Rule 9, experiments-and-artifacts), release fingerprint (README, AGENTS, verification-commands, scripts docs), *_MATH tokens (tokens-and-labels), web theorem-box + slide aux-map rendering (rendering_pipeline).
- ISC-205: Read (raster) — regenerated 14_formalism slide deck frame 1 shows "Definitions 1 and 2 … Lemma 3 … Proposition 4 … Theorem 5 and Corollary 6" (combined-PDF numbers, zero "??"; pdftotext count of "??" = 0); template rendering suite "932 passed, 1 skipped".
- ISC-202: direct `codex exec --sandbox read-only` audit ran for hours with ZERO stdout and was stopped (TaskStop bep3yfaz7); failure mode captured: no structured verdict, active session rollouts throughout, and — R15 signal — output/reports were rewritten to smoke scale at 12:20:41 while codex was the only active actor (read-only sandbox evidently not preventing subprocess writes, or an over-eager probe ran the pipeline). Rule 2a cross-vendor gate remains UNSATISFIED (frontmatter blocker retained); third distinct failure mode of the codex audit path this iteration (agent truncation ×2, direct-exec hang ×1).
- 2026-07-18 (iteration 33w3 — EXECUTE, second smoke contamination + repair): the verdict-page raster caught hydration at smoke scale ("Across 6 matched trials", n_seeds=4) after the final render; reports mtime 12:20:41 postdates the run-vs-run parity verification. Repair: publication-profile `02_run_analysis.py` rerun, re-hydration, and full re-render scheduled; the definitive suite runs only after all co-actors are stopped. Learned: raster reads of RESULT pages (not just theorem pages) are the catch-all for silent scale regressions — the "6 trials" was invisible to every green validator because the tokens resolved consistently.
- ISC-200: Bash — definitive full suite as sole actor (prior red run diagnosed as concurrent-auditor .pytest_cache/.coverage contention; web-contract tests 6/6 green in isolation): "1058 passed in 901.61s (0:15:01)", "Total coverage: 93.55%", exit 0 (task b42gxzv5t). +178 tests vs session start.
- ISC-203: suite + diffs — no report key/filename/figure path/interface changes across waves 3-4: run-vs-run report parity (24/24 value-identical), figure-registry contract green, release round-trip green, token tables green inside the 1058-test run.
- Reflection Ledger v2 (waves 3-4): entry_id 07a3e55d-e0bc-4b0f-829d-ffa4cde797be, 2026-07-18T20:43:51.567Z (16/16/0/0/0, E4 classifier).

### Wave 5 verification (tri-lens E5, workflow wf_10fef518-800: 10 specialists, 9 returned + 1 solo re-run, adversarial synthesis)

- ISC-212: fp:deconstruct — chain map produced; 4 weak links found and confirmed by synthesis re-read: vacuous effective-weight test (SYN-2), tautological EFE identity (SYN-4), 3 branch-artifact falsification rows (SYN-3), c=0 identity tested only via early-return branch (SYN-10). Weakest link named: the axis-3 raw effective-weight bound with zero non-vacuous coverage.
- ISC-213: sci:falsifiers — all 5 headline-claim falsifiers demonstrated able to fire via temp-scope python with quoted numerics (c>0 breaks pool identity; beta>0 breaks Bayes; entropy-attack flips the sweep comparison; NLL diverges/rcce bounded; broken eta breaks KL monotonicity). Claim-3 alternatives (seed artifact / saturation ceiling / estimand choice) each addressed via cited report fields.
- ISC-214: sci:negcontrols — census across gallery/BMR/registry/tamper/schema controls; 3+ inversions run in temp copies; 2 weak controls confirmed: hardcoded-True negative_controls dict (SYN-7), floorless BMR delta_F assertion (SYN-9).
- ISC-216: rt:contamination — census of all output/ writers; TOP FINDING confirmed by a third live incident during this very wave (n_seeds 240→4 mid-review): no gate pinned committed reports to publication scale. Guard test (tests/test_report_scale_guard.py) commissioned in the same wave.
- ISC-217: rt:clone — R12 audit: 22 load-bearing untracked files. CRITICAL: fresh clone cannot import the package. Minimal tracking set recorded: `git add src/analysis/report_schemas.py src/fedference/_validation.py src/figures/_metadata.py src/figures/generative_model_schema.py src/figures/message_passing.py src/figures/pomdp_loop.py` + 11 docs (docs/research/ ×6, docs/todo/ ×5) + 10 test files; plus 2 tracked-but-deleted nested output figures safe to stage as deletions. Staging/committing deferred to Daniel (publish boundary; sandbox protects .git deliberately).
- ISC-215: rt:gates — 3/3 injections caught by their gates (hardcoded decimal → token-provenance FAIL; caption axis-strip → caption-completeness FAIL; SLICE LANDED → docs-contract FAIL), each with md5-verified byte-exact restore per protocol; independently re-verified by primary post-wave: no residual markers/numbers in the three files, token+caption gates "7 passed", docs contract "22 passed". Bypass-vector analysis produced SYN-11 (percent-word regex gap) — fixed same wave with the percent-word pattern.
- Wave-5 fix batch (SYN-1..5,7..12 + scale guard): manuscript sign-label inversion corrected (13_methods ×2); falsification-table honesty rewrite citing off-switch residual tokens (14_formalism); EFE identity reframed as definitional consistency with independent semantic pins (14_formalism); continuous-slice limitation corrected + divergences.py comment (23_discussion + src); raw effective-weight bound test added (recomputes code formula from returned consensus); heuristic negative_controls now computed from grid rows (was hardcoded True — R8 class); test-double scan extended to tests/; BMR floor −0.5 (real value −1.6353); iterative-path c→0 identity test; percent-word provenance pattern; MAJ-1 phrasing scoped at 3 call sites (R14 sweep incl. aggregation_descent.py). Gates: ruff clean, mypy clean, "53 passed +1 in-flight flag" → settled: scale guard "3 passed", aggregation+heuristic "28 passed". The scale guard caught the mid-regeneration state on its very first run — proof-of-detection by live fire.

### Wave 5 EXECUTE record (tri-lens fixes + root cause)

- rt:claims solo re-run (first attempt died on structured-output cap) — 3 MAJOR + 2 MINOR verified rendered-artifact findings, all fixed: (F1) conclusion grafted the worst-rate 0.9880 onto "verdict rate" — rewritten to state both rates with their own tokens; (F2) "headline robust method" carried AR's accuracy while the headline tag was RKL — relabeled "most accurate robust member"; (F3) S09 claimed "does not reject / statistically indistinguishable" while the report's paired p=1.75e-05 REJECTS with gap CI [-0.0079,-0.0035] all-negative — narrative corrected to "small but statistically reliable negative gap" and swept to 21/23 call-sites (R14: "matches the flat baseline" phrasings corrected in 3 files); (F4) fig-11 citation carried Table-3-only numbers — cite corrected with colony distinction; (F5) rounding-arithmetic note added to the language-KL caption. Post-fix oracle battery 42+6 green; fixes verified in the re-rendered PDF by text probes (both-rates present, no "does not reject", no naive-minus-robust in the accuracy sense — the surviving S17 hit verified CORRECT against its generator's `naive.agent_weight[-1] - robust.agent_weight[-1]`).
- ROOT CAUSE of all four smoke contaminations found and fixed: tests/test_scripts_smoke.py ran `02_run_analysis.py --profile smoke` + `z_generate_manuscript_variables.py` as subprocesses against the REAL tree (documented as policy in its own docstring) — every full-suite run clobbered output/reports (n_seeds 240→4), variables, and the resolved manuscript tree; TODO.md documented the post-suite regeneration as workflow. Fix at ingestion (new `src/project_paths.py` env override `ACTIVE_FEDFERENCE_PROJECT_ROOT`, validated fail-loud; 4 scripts wired; smoke tests write to a `_make_project` scaffold; new `test_scripts_never_write_into_the_real_project_tree`; override-validation unit test). Proof: mtime sweep of real output/ shows 0 files modified during the 19:36–19:39 test run; n_seeds=240 survived; targeted gates "33 passed in 169.13s"; ruff/mypy clean. Incident #4's suite (1064 passed) also proved the contaminating tests were GREEN while clobbering — a green suite was the delivery vehicle, the exact laundering shape R8 warns about.
- Disk-full incident (ENOSPC at 100%/3.5Gi free on the 460Gi volume) interrupted two edits mid-wave; freed session temp artifacts and completed. Machine-level disk pressure surfaced to Daniel.
- ISC-218: synthesis dispositions — 13 confirmed findings ALL fixed (SYN-1..5, 7..11 + 3 rendered-artifact majors F1-F3 + F4/F5 minors) or explicitly deferred with reason (SYN-6 variational-dispatch decision, SYN-12 directional-conservatism test, SYN-13 'reliable' wording, MED-1 reordering — all Daniel-arbitration items, recorded below); root cause of the contamination class fixed at ingestion (project_paths override) with full-scale proof. Final suite: "1 failed, 1065 passed in 837.83s" where the single failure was a wave-5 agent collision (SYN-8 scan's over-broad \bmonkeypatch\b flagging the root-cause fix's legitimate monkeypatch.setenv env-path tests); scan narrowed to monkeypatch.setattr (behavior doubles) with rationale comment — "3 passed" in isolation, ruff clean; regex-constant change in one test file cannot affect the other 1065 (recorded as a Rule-4 judgment call rather than a fifth 14-minute run). Post-suite probe: n_seeds=240 — the suite no longer contaminates.

Dispositions for Daniel (wave-5, not unilaterally decided):
1. SYN-6: wire `variational_aggregate` into share_round's protocol dispatch (currently only naive|robust) vs document the exclusion as deliberate — affects whether the objective-backed rule participates in federation rounds.
2. FP-3/SYN-12: whether the axis-3 "conservative / sacrifices peak accuracy" label gets a directional pinning test (variational < robust point accuracy in the relevant regime) — currently half-tested.
3. FP-2: whether MED-1 (attack-geometry generalization, 1-2wk) is elevated to run right after MAJ-1's vocabulary lands, ahead of MAJ-4/5 — empirical-leverage vs ordering-integrity tradeoff.
4. SYN-4 billing: whether the EFE decomposition keeps "machine-checkable identity" status post-tautology-reframe (manuscript now states it is definitional; the semantic tests are the correctness surface).
5. Clone-correctness: 22 load-bearing untracked files need `git add` before any push (exact set in ISC-217); 2 tracked-but-deleted nested output figures to stage as deletions.
6. Machine: the 460Gi volume hit ENOSPC mid-wave (3.5Gi free after temp cleanup) — disk pressure needs attention.
- 2026-07-18 (roadmap closeout): added MIN-2 scoped page docs/todo/release-and-verification-ladder.md (clone-correct tracking set, fresh-clone full verification ladder incl. post-suite scale guard + rendered-surface count invariants + raster reads + fingerprint verify, cross-vendor verdict closure, claim-hygiene decision queue) — marked Critical, blocks any push/release; TODO.md MIN table + gates pointer + docs/todo/README.md index updated; docs contract battery "26 passed in 1.29s".

### Iteration 34 verification (2026-07-27)

- ISC-201: Template web renderer tests → `45 passed`; regenerated project web
  output contains 22 `.theorem-box` elements across 7 HTML section files.
  A bounded rendered-body scan found no raw theorem-environment or `\\texttt`
  markers inside the boxes, while emitted inline math remains in the
  MathJax-renderable `class="math inline">\\(` form. The dedicated rendered
  surface gate passed for 43 HTML files, 52 web assets, 42 slide PDFs, 42 TeX
  sources, and 42 logs.
- Full project gate: `1085 passed`, `93.28%` coverage, above the 90% floor.
  Ruff, mypy (`91` source files), layer isolation, metadata/invariant checks,
  focused claim/statistics tests, package validation, and rendered-surface
  validation all passed.
- Publication closure: the regenerated combined PDF is 78 pages and passes
  `qpdf --check`; the output contains 26 registered figure stems and 52 PNG/PDF
  web assets. These are release-surface counts, not scientific sample sizes.
- Statistical and claim review: strict integer/resampling validation and
  explicit one-value uncertainty metadata landed; the claim, visual, and
  literature audits now record the current evidence and retain conditional
  estimands. No heuristic-server, cross-host, or universal robustness claim
  was promoted.
- Release provenance: manifest schema 2 records profile, generator, package
  version, timestamp, aggregate fingerprint, and individual input-file hashes;
  build plus stale-input and artifact-tamper verification passed.
- At this historical checkpoint, stage-order freshness and clean-checkout
  replication were still open; the receipt implementation is now recorded by
  ISC-219 and the current clean-checkout/cross-vendor boundary remains open
  below rather than being silently converted to a green verdict.

### Iteration 35 verification (2026-07-27)

- ISC-219: content-hashed analysis, hydration, and render receipts are now
  recorded by the real pipeline entry points; dependency changes and missing
  receipts fail closed. Final probe: `pipeline freshness: PASS (analysis,
  hydration, render)`.
- ISC-220: `validate_clean_checkout.py` performs a real subprocess Git/tracking
  and import probe. It correctly returns `FAIL` for this intentionally dirty
  checkout with the new implementation files untracked; no clean-clone claim
  is made without an authorized stage/commit.
- Definitive machine gate: external-permission `validate_all.py full` returned
  `1095 passed in 922.60s (0:15:22)` and `93.16%` coverage. The sandbox-only
  socket bind failures were isolated as `PermissionError`; the socket suite
  then passed `18/18` with local permission, and all socket tests passed in the
  definitive run.
- Render/package gates: 42 slide PDFs, 42 TeX sources, 42 logs, 43 web HTML
  files, and 52 web assets passed the rendered-surface validator; `qpdf --check`
  found no PDF syntax or stream errors; release build plus `--verify` passed.
- Claim/statistics/scholarship boundary: the updated claim, visual, extended
  statistical, and literature records remain source-bound and conditional. No
  heuristic-server, cross-host, universal-robustness, or fresh-clone claim was
  upgraded.

### Iteration 36 verification (2026-07-28)

- ISC-221: the new source-bound complexity catalog accounts for the actual
  dense NumPy paths: log-linear, iterative-robust, and objective-backed
  variational aggregation; naive and iterative-robust self-excluding sharing;
  one-step state inference; and the local server round. Naive and robust
  sharing are separate operations and separate measured rows. The report
  records 8 analytic specifications and 9 seeded timing measurements with
  median/min--max repeats, work units, input digests, and machine metadata.
- Complexity receipt: publication grids use agent sizes 4--64, state sizes
  256--4096, sharing agent sizes 4--32, modality sizes 1--8, five measured
  repeats after one warmup, seed `20260728`, and the local Apple arm64 Python /
  NumPy environment. Observed log--log slopes are descriptive finite-grid
  diagnostics: log-linear 0.89, robust 0.96, variational 0.71, naive sharing
  1.77, robust sharing 1.97, state-axis 0.42/0.37/0.44, and inference 0.66.
  These do not constitute universal asymptotic, FLOP, cross-host, or network
  performance claims; timing spans are min--max ranges, not confidence
  intervals.
- Full machine gate: `validate_all.py full` returned `1113 passed` and
  `92.85%` coverage, with Ruff, mypy, layer isolation, report schemas, figure
  contracts, claim/caption/xref gates, package validation, and release
  verification passing.
- Publication gate: 42 slide PDFs, 42 TeX sources, 42 logs, 43 web HTML files,
  and 54 web assets passed rendered-surface validation; the figure registry
  contains 27 entries; pipeline freshness passed for analysis, hydration, and
  render; and `build_release.py --verify` passed for 449 artifacts.

### Iteration 37 verification (2026-07-28)

- ISC-222: BNN robustness reports now derive every replicate seed from the
  public seed argument, so changing the base seed changes the executed data
  replicates rather than only the bootstrap resample. Regression coverage
  compares distinct base seeds on the real experiment path.
- ISC-223: parameter recovery now uses a deterministic declared likelihood grid
  with independent observation draws and labels its interval as an empirical
  percentile interval across independent trials, never as a bootstrap or
  Bayesian credible interval. The report schema, tokens, caption, and figure
  are source-bound to that disposition.
- ISC-224: contamination-gallery and robustness-onset method selection is
  pooled over configured seeds before the final summaries; seed-bootstrap
  intervals and selected-method annotations are emitted into the reports,
  figures, captions, and tables. The existing paired statistical contrast
  remains the inferential winner rule; the figures do not cherry-pick a method
  per replicate.
- ISC-225: prospective sample-size calculations now honor the direction of the
  declared alternative and return a fail-closed maximum for a wrong-direction
  effect instead of silently using its magnitude.
- ISC-226: the manuscript preamble resolves TeX Live Basic's bundled Latin
  Modern Mono files by filename, removing the machine-local JuliaMono/fontspec
  failure and the resulting non-finite TeX scale errors. A clean template
  render produced an 80-page PDF with no LaTeX error, missing-character, or
  non-finite-formatting findings.
- Definitive machine gate: `validate_all.py full` returned `1115 passed in
  1020.93s` with `92.76%` coverage. Ruff, mypy over 96 source files, layer
  isolation, invariants, report schemas, claim/caption/xref/statistics gates,
  package validation, pipeline freshness, output validation, metadata checks,
  and release-manifest verification all passed.
- Publication gate: 42 slide PDFs, 42 TeX sources, 42 logs, 43 web HTML files,
  and 54 web assets passed rendered-surface validation; 27 registered figures
  are present; all 43 publication PDFs passed `qpdf --check`; and the release
  bundle verified 449 artifacts. Targeted raster QA found no clipping in the
  changed gallery, onset, parameter-recovery, or complexity visuals.
- Boundary retained: the working tree is intentionally dirty, so
  `validate_clean_checkout.py --skip-imports` correctly fails. No clean-clone,
  author-signoff, DOI, push, or universal robustness claim is made by this
  machine-local candidate.

### Iteration 38 verification (2026-07-28)

- ISC-227/228: the new proper-score controls and 40-cell conditional-world
  grid are source-owned, deterministic at the seed level, schema-validated,
  and rendered. The controls pass; the conditional grid retains mixed signs,
  so no universal attack-geometry claim is promoted.
- ISC-229: the hybrid categorical/Gaussian representation passes exact
  zero-robustness recovery and finite positive-robustness validity tests; its
  objective and full active-inference task remain explicitly open under MAJ-3.
- ISC-230/231: report contracts, figure registry, manuscript tokens/captions,
  claim/statistical/visual audits, and the forward TODO index all reconcile the
  new evidence. Completed finite-grid, proper-score, and receipt slices were
  removed from the forward TODO pages under the project rule.
- ISC-232: the component verification ladder passed with 1,149 tests and
  92.48% source coverage; Ruff, mypy over 101 files, invariants, layer
  isolation, package/web/rendered-surface checks, pipeline freshness, PDF
  raster QA, `qpdf --check`, and release-manifest verification all passed.
  The final surface is an 80-page PDF, 42 manuscript sections, 42 slide decks,
  43 HTML surfaces, 29 registered figures, and 58 PNG/PDF figure assets.
- ISC-233: the clean-checkout required-tracking set now includes the new
  source modules, tests, reports, figures, and current audit surfaces. The
  post-commit `validate_clean_checkout.py` probe passes with 948 tracked files
  and a successful import probe. A genuinely fresh-clone full ladder and a
  structured cross-vendor verdict remain open; no push, DOI, or author sign-off
  is implied.

### Iteration 40 verification (2026-07-29)

- The authoritative changed-tree source gate completed **1,299 passed in
  1137.49 s (18:57)** with **90.44%** total `src/` coverage, above the 90%
  floor. The explicit `not slow`, integration, and publication profiles also
  passed before the final two surface-validator tests were added; those tests
  pass focused and are included in the definitive full run. The final marker
  collection selects 1,237, 17, and 108 tests respectively. Ruff, mypy over
  113 source and 113 test files, `uv lock --check`, layer isolation, and diff
  whitespace checks are green.
- Wheel and source-distribution builds installed into separate empty
  environments. Both installed CLIs listed the registry and loaded the
  packaged 150-row synthetic benchmark; the default import graph did not load
  Torch. A real server-theory smoke run wrote and verified a dirty-tree receipt
  without promoting it to confirmatory evidence.
- The rendered-surface validator now checks all 43 manuscript/slide PDFs with
  `qpdf --check` and `pdftotext`, requires complete slide PDF/TeX/log triplets,
  and scans every retained manuscript log in addition to the existing web
  gates. It correctly rejects the historical reviewer snapshot because
  `_xelatex_stdout.log` is an obsolete 83-page transcript containing two
  U+0002 missing-glyph warnings; the canonical 80-page July-28 log has no
  missing-glyph, undefined-reference, or material layout finding. All 80
  manuscript pages were rasterized and visually reviewed with no clipping,
  collision, missing figure, or blank-render defect. All 43 PDFs are
  structurally valid; the combined PDF is untagged.
- Publication freshness and release verification remain red by design:
  source/manuscript digests differ from the recorded analysis and hydration
  inputs, the release fingerprint names changed inputs, and the clean-checkout
  probe reports tracked modifications plus required untracked files. The
  system volume has approximately 21 GiB free, below the 40 GiB regeneration
  floor, so no analysis/hydration/render/release refresh or isolated-clone
  campaign was attempted. No active research data or caches were reclaimed.

### Iteration 41 verification (2026-07-29)

- ISC-248: the loopback API now rejects wildcard and non-loopback bind
  addresses and binds the checked numeric address without a second hostname
  lookup; the SQLite replay guard rejects reuse
  after reopen, resolves contended claims atomically across guard instances,
  and fails closed on symlink, directory, corrupt, and future-schema paths.
  A warning-strict coverage probe exposed and closed both the normal and
  failed-connect SQLite handle lifecycle; the socket suite now passes with all
  warnings promoted to errors.
  The MAJ-4A registry source bundle pins TLS 1.3, X.509, Python SSL, and Docker
  networking/security authorities, while the threat model preserves the
  shared-HMAC, poisoning, confidentiality, container-host, and physical-host
  no-claim boundaries.
- ISC-249: the web validator now rejects missing language/title, skip/main
  navigation, image alternatives, image-figure captions, full-size-link labels,
  and duplicate identifiers. The read-only current-snapshot probe passes all 43
  HTML pages and 58 assets. `pdfinfo` reports the combined 80-page PDF as
  `Tagged: no`; the new manuscript/documentation contract therefore names HTML
  as accessibility-enhanced and prohibits WCAG/PDF-UA promotion.
- The consolidated focused gate passed 105 security, registry, documentation,
  manuscript-cross-reference, caption, token-provenance, clean-checkout,
  web-package, surface, and socket tests in 4.58 seconds with warnings promoted
  to errors. Ruff, mypy over 113
  source files, layer isolation, and `git diff --check` are green. The live ISC
  parser reports 242 of 244 criteria passing.
- A broader manuscript-variable run was intentionally stopped after 10 passing
  tests because simultaneous RNA-seq and separate publication workloads made
  each real pipeline fixture unusually expensive. The last authoritative full
  changed-tree gate remains Iteration 40's 1,299-test result; every new branch
  in this iteration is covered by the focused gate above.
- No generated `output/` artifact changed. Free-space observations fell within
  approximately 21–23 GiB, so freshness, rendering, release, fresh-clone, DOI,
  and author-approval work remains correctly open under ISC-242.

### Iteration 42 verification (2026-07-30)

- The authoritative changed-tree gate completed **1,317 passed in 1021.81 s
  (17:01)** with **90.56%** total `src/` coverage. Ruff, mypy over 113 source
  files, numerical invariants, `uv lock --check`, the
  `src/fedference`/`infrastructure` layer boundary, and the post-suite
  publication-scale report guard pass. The full suite used the real report,
  figure, hydration, process, socket, package-resource, and publication
  producers; no mocks or competing producer were introduced.
- Wheel and source-distribution builds installed into separate empty
  environments on `/Volumes/blue`. Each installed `fedference` CLI enumerated
  the versioned registry and executed the packaged 150-row synthetic benchmark
  without importing Torch into the default NumPy/SciPy core.
- Official UCI archives for WDBC, Dry Bean, and Banknote Authentication were
  downloaded to an external read-only cache and matched the registered archive
  SHA-256 digests, member formats, row/feature/class counts, DOI, and CC BY 4.0
  metadata. A Dry Bean smoke receipt exercised the new solver-health fields and
  remained explicitly non-confirmatory.
- The source-bound analysis, hydration, three-surface render, and release bundle
  were regenerated in producer order after implementation. The final combined
  PDF has 80 pages; all 43 publication PDFs pass `qpdf --check` and text-layer
  validation, all manuscript pages and selected slide pages pass raster review,
  and the strengthened rendered-surface gate passes 43 HTML files and 58
  assets. A headed real-browser pass at desktop and 390-pixel mobile widths
  found no horizontal overflow, broken figure, failed request, or console
  error; keyboard skip navigation, deep links, and full-size figure links work.
  Four MathJax dynamic-font metadata warnings remain non-failing. The PDFs are
  still untagged and are not PDF/UA surfaces.
- Safe working headroom was provisioned without deleting active research data:
  the large Ollama model store and all review/build scratch data were moved to
  `/Volumes/blue`, with a verified external dirty-tree backup retained before
  the campaign. This workstation-local storage action does not prove a clean
  clone, a cross-vendor review, or a release.
- ISC-242 remains open. The uncommitted candidate must first be bound to an
  exact commit and reproduced from two isolated clones; cross-vendor verdict,
  confidentiality/license/author approval, DOI creation, and public release
  remain independent external gates.

### Iteration 43 verification protocol (2026-07-30)

- The first isolated-clone attempt exposed wall-clock drift in the release
  manifest rather than being counted as a pass. Release schema 3 now omits
  `generated_at` for unreleased builds; pipeline-freshness schema 2 validates
  nullable `recorded_at`; and manuscript hydration derives its build epoch only
  from `SOURCE_DATE_EPOCH`. Repeated release builds are byte-identical in the
  focused regression.
- One epoch (`1785205200`, `2026-07-28T02:20:00Z`) is pinned across analysis,
  hydration, PDF/web/slide rendering, and validation. The render receipt names
  template commit `03982d571d2acf41b6c69ecb9585b7b572a660da`, the reviewed
  dirty-overlay digest, and the epoch. This is explicit external-producer
  provenance, not a claim that this repository content-addresses that overlay.
- The refreshed chain contains 459 release artifacts, 43 structurally valid
  PDFs including the 80-page combined manuscript, 43 validated HTML pages, and
  58 checked web assets. The generated release manifest remains the sole
  authority for the byte total, rather than copying that volatile derived value
  into this source document. The combined and slide PDFs remain untagged,
  non-PDF/UA convenience surfaces.
- The tree currently collects 1,343 tests. Focused receipt/manuscript/script
  slices pass, together with Ruff and mypy over 113 source files. The final
  exact-commit full-suite and installation outcomes belong in external
  clean-clone receipts: embedding those post-commit results here would change
  the commit they certify.
- ISC-242 therefore remains open in this source snapshot. Two local isolated
  clone receipts can close its machine-local subgate, but cross-vendor review,
  confidentiality/license/author approval, DOI creation, and public release
  remain independent external decisions.

### Iteration 44 — scoped server theory and calibration integrity (2026-08-01)

- [x] ISC-254: MAJ-1's stated scoped-impossibility branch is complete for the
  declared separable class. The raw-log-pool construction proves that no
  continuously differentiable q-only term can make that q block the coordinate
  minimizer for every interior raw-weight/local-posterior input; the result
  explicitly excludes neither coupled, source-dependent, non-differentiable,
  nor fixed-point-only constructions. Probe: `test_server_theory.py`,
  Proposition `prop:raw-log-pool-no-go`, and typed report metadata.
- [x] ISC-255: The normalized-weight companion preserves the raw-update
  distinction: equal normalized reverse-KL weights with unequal forward-KL
  data-term differences reject the normalized reparameterization of the same
  natural class. Probe: exact divergence identities in
  `test_server_theory.py`.
- [x] ISC-256: The MAJ-1 proof is source-bound through the typed
  `heuristic_characterization.json` formal-no-go payload, manuscript
  proposition/table, notation registry, claim ledger, phase plan, and
  RedTeam boundary. The empirical grid remains separately labelled
  `open_no_global_objective`.
- [x] ISC-257: MAJ-8 calibration provenance now fails closed on a canonical
  payload containing episode identities/content/world families, full candidate
  declarations/scores, the selected configuration, and the estimand. Losing
  selections, digest/config/score/world-family tampering, nonconvergence, and
  fallback traces are rejected; canonical terminology retains a warned
  `beliefs` compatibility adapter. Probe: `test_calibration.py`.

### Iteration 47 — single-machine research-pilot hardening (2026-08-08)

- [x] ISC-258: Centralized YAML mapping validation is used by workflow readers;
  malformed top-level, `experiment`, and `bnn_torch` blocks fail closed. All
  temporary Git fixtures disable inherited `core.fsmonitor` and
  `core.untrackedcache` settings. Probes: `test_experiment_config.py`,
  `test_workflow.py`, `test_clean_checkout.py`, and `test_evidence.py`.
- [x] ISC-259: The portable BNN lane now exports/loads diagonal-Gaussian
  cavities and runs a cavity-conditioned synthetic CPU/MPS pilot with held-out
  log score, contamination control, device/fallback receipt, and exact
  checkpoint/resume equivalence. Source-dataset and CUDA parity remain open.
- [x] ISC-260: The calibration pilot records complete candidate scores, the
  frozen `AggregationConfig` fingerprint, a disjoint evaluation set, and a
  deliberate overlap rejection. Confirmatory budget/effect freezing remains
  open.
- [x] ISC-261: Hybrid tracking now retains naive, robust, discrete-only,
  continuous-only, and oracle-context controls and rejects singular covariance;
  its next-position predictive score remains pilot evidence.
- [x] ISC-262: Deterministic Four Rooms and Key-Door pilots exercise flat,
  oracle, learned, shuffled, and non-gating controls at matched horizons; no
  general hierarchy claim is promoted.
- [x] ISC-263: The Friston Eq. 2/Figures 5, 7, and 9 audit remains
  `paper-constrained reconstruction` and includes an analogue-relabeling
  negative control; exact source-protocol status is not asserted.
- [x] ISC-264: External benchmark reports now contain dataset-level nested-seed
  summaries and archive/split/recovery receipt controls. A three-dataset
  confirmatory result and manuscript hydration remain open.
