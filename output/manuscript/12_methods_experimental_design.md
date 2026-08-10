## Experimental design: studies, estimands, determinism, and power {#sec:methods-experimental-design}

The generative model of [@sec:methods-generative-model], the learning operators
of [@sec:methods-learning], and the corruption process of
[@sec:methods-contamination] are exercised by 9 studies, including
the contaminated-sentinel robustness sweep (Study 4). The shared configuration (seed budget, colony size,
contamination rates, divergences, trial counts, and the statistics settings) is
read from `experiment:` in [`manuscript/config.yaml`](config.yaml); the
remaining per-study parameters are tested code defaults in
`src/fedference/experiments/`. No value is hard-coded in the manuscript, and
each token below resolves to the same configuration the code executed.

## Determinism through fixed seeds and generated variables {#sec:methods-determinism}

All stochastic steps draw from explicitly seeded generators
(`np.random.default_rng`); the global `np.random` state is never touched. The
single-run studies use the first configured seed (0), and the
across-seed studies enumerate the deterministic seed list
$0,\ldots,n_{\text{seeds}}-1$. Re-running with the
same seed reproduces every number in the results bit-for-bit, so the bootstrap
confidence intervals and paired-test p-values of [@sec:methods-statistics] are
themselves deterministic functions of the seed.

The figure layer follows the same provenance rule. Captions are written to be
self-contained, with axes, resampling units, deterministic runs, truncated axes,
and error-band status disclosed in the caption rather than left to inference
[@rougier2014figures; @midway2020visualization]. This is why several results
figures state "single deterministic run" or "no error band" even when the
inferential evidence appears in an adjacent table.

## Study suite and contamination sweep {#sec:methods-studies}

Studies 1--3 implement reduced categorical protocols that are source-mechanism
analogues of the belief-sharing, language-acquisition, and model-reduction
mechanisms discussed by Friston et al. [-@friston2024federated] on the sentinel
POMDP; they are not exact source-protocol figure replications. The sweep adds the FedGVI robustness contribution
[@mildner2025fedgvi]. Unless a study specifies otherwise, the global defaults are
$n_{\text{seeds}} = 480$ independent seeds and
$n_{\text{trials}} = 960$ matched trials per condition.

| Study | What it measures | Key parameters |
|---|---|---|
| 1 — Belief sharing | Communicating vs. incommunicado colony free energy | `n_agents = 7`, `acuity = 0.55` |
| 2 — Language acquisition | Dirichlet-learning KL descent ([@eq:dirichlet-kl]) | `num_steps = 24` |
| 3 — Emergence / BMR | Reduced-vs-full model evidence ([@eq:bmr-deltaf]) | candidate states `n = 4` |
| 4 — Robustness sweep | Robust vs. naive consensus under contamination | `n_agents = 7`, `n_contaminated = 2` |
| 5 — Disjoint-FOV moving world | Communication necessity with non-overlapping fields of view | `n_agents = 3`, `fov_width = 2` (Supplement, [@sec:results-moving]) |

: Per-study configuration, read from `experiment:` in `config.yaml` and surfaced as manuscript tokens by `src/manuscript_variables.generate_variables`. The sample sizes carried into the statistics — the across-seed belief-sharing sample ($n = 480$ seeds), the language trajectory (25 ordered points summarized over $n = 480$ independent seeds), and the paired robustness trials ($n = 960$ per condition) — are reported with their respective results. {#tbl:study_params}

**Study 1 — belief sharing.** A colony of 7 sentinels
at the deliberately low acuity 0.55 each infer the creature
location ([@eq:state-inference]) and share beliefs through the log-linear pool
([@eq:log-linear-pool]). We compare a *communicating* colony against an
*incommunicado* one of the same size and seed, scoring each by the mean
variational free energy of [@eq:variational-free-energy]. Two protocol details:
state inference in this study substitutes a flat (uniform) prior for $D$, and
each belief's free energy is scored against the pooled evidence of all agents'
observations (the disclosure carried in [@sec:results-belief_sharing]). The
across-seed sample
is $n = 480$ seeds, one colony-mean free energy per seed
([@fig:free-energy], [@fig:belief-heatmap]).

**Study 2 — language acquisition.** Each configured seed runs one sentinel
trajectory using the conjugate Dirichlet update ([@eq:dirichlet-update]) over
24 steps. We record the KL descent of [@eq:dirichlet-kl] at
each ordered step, giving 25 points per trajectory and $n =
480$ independent seed trajectories for the pointwise interval
in [@fig:language-kl].

**Study 3 — emergence.** A redundant model reduction and a supported one are
scored by the BMR free energy of [@eq:bmr-deltaf] over $n = 4$
candidate states ([@fig:emergence-bmr]).

**Study 4 — robustness sweep.** The sweep varies two factors on a colony of
7 sentinels of which 2 are saboteurs
(`confident_wrong`, [@sec:methods-corruption]):

- **Contamination rate** over $\{0, 0.225, 0.45, 0.675, 0.9\}$ — the convex-mix weight of
  [@eq:contamination-mix] toward the confident-wrong delta.
- **Server robustness setting**, named with FedGVI client-loss/
  divergence vocabulary for cross-reference only
  $\{KLD, RKL, AR, beta, rcce\}$. `KLD` is the non-robust Friston / standard-Bayes
  baseline (server robustness 0, [@eq:robust-identity]) and serves as the design's
  negative control: the recovery identity guarantees it reproduces the naive
  log-linear pool exactly, so every robust-versus-naive contrast is scored against
  a comparator that is provably the un-robustified server rather than a separately
  tuned competitor. The remaining labels
  $\{RKL, AR, beta, rcce\}$ each select a fixed `robust_aggregate`
  down-weighting constant (the executed mapping is KLD (c=0.00), RKL (c=1.50), AR (c=1.30), beta (c=1.70), rcce (c=1.60);
  defined in `fedference.experiments._common`). None
  of these labels invoke the client-side `generalized_posterior` update of
  [@sec:method-genbayes] or the divergence family of [@sec:method-divergences];
  this sweep therefore exercises only the server-side heuristic axis of
  [@sec:robustness-axes] (`robust_aggregate`), never the per-agent rigorous
  axis. The executed per-divergence down-weighting strengths are fixed
  constants defined in `fedference.experiments._common` and recorded in the
  run reports.

The headline verdict pairs 960 independent trials at the fixed
contamination rate 0.800 — heavy contamination that degrades the
naive pool while staying below the pure-veto cliff
([@fig:robustness-sweep], [@fig:robust-weights]). Each trial contributes one
matched (naive, robust) accuracy pair, so the replication unit is the trial and
the estimand is the within-trial accuracy difference at that single rate. A complementary federated
logistic-regression baseline applies the same client-side robust loss to
flipped-label contamination, isolating the rigorous axis ([@fig:bnn-robustness]).

**Study 5 — disjoint-FOV moving world.** This extension ([@sec:results-moving])
places 3 agents on a 2-slot disjoint-FOV track to
test the necessity of communication when agents cannot observe the same positions.
Isolated agents are compared against EFE-guided communicating agents on accuracy
across the moving sentinel's trajectory. Five structural extension studies
(Studies 5--9, Supplementary sections) build on the same POMDP substrate and are
described there: the moving disjoint-FOV sentinel ([@sec:results-moving]), the
2-level hierarchical POMDP ([@sec:results-hierarchical]), the $N$-level extension
([@sec:results-3level]), the 2-D sensitivity sweep ([@sec:results-sensitivity]),
and parameter recovery ([@sec:results-parameter-recovery]).

## Sample size and prospective statistical power {#sec:methods-power}

The verdict design answers a deliberate question: pair *many* trials at *one*
high contamination rate rather than spread few trials across the rate curve. A
matched-pairs Wilcoxon test gains power from the number of matched pairs, so
concentrating $n = 960$ paired trials at the single rate
0.800 gives the test the resolution to detect the robustness
effect; scattering the same budget across 5 rates would dilute
every contrast. The across-seed studies are powered separately, with $n =
480$ seeds for belief sharing and $n =
480$ independent seed trajectories for language acquisition.
The 25 ordered learning points are repeated measures within
each seed, not additional independent replicates, so the language interval does
not count time points as samples.
The structural-extension and cross-study summary tier uses
$n = 128$ independent seeds and
$n = 40$ matched trials per contamination rate for its
robustness row. Trials and clients are nested within a seed and are reduced
before across-seed inference; they are not additional independent replicates.

The bounded red-team review grid is a separate source-bound analysis profile. It
uses 160 deterministic seed replicates, with
24 trials nested within each seed and scenario/rate cell,
and retains the registered rates 0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 across the finite attack
union clean, confident wrong, permutation, byzantine, drift, label noise, uniform. Its independent unit is the configured seed
within a declared cell; cells that share design structure are not treated as
independent worlds. Robust operating points, method settings, and rate profiles
are fixed before the review run. The grid reports selection-free contrasts and
keeps clean, uniform, label-noise, permutation, confident-wrong, Byzantine,
and drift controls visible. Completion of this finite review does not close
server-heuristic characterization, leakage-free calibration, external-data, or
protocol-reconstruction phases.

We do not merely assert adequacy — we compute the design power implied by the
observed effect. For the headline robust method (RKL) the
observed-effect design power of the paired Wilcoxon at the run's
$n = 960$, computed at $\alpha = 0.05$ against the
directional alternative `greater` (robust accuracy exceeds
naive), is 1.0000. The power approximation uses the deterministic
noncentral-normal approximation of [@sec:methods-statistics], deflated by the
Wilcoxon's Pitman asymptotic relative efficiency, so it is approximately
calibrated for the signed-rank test actually run (exact under a normal-shift
alternative; the power computation is one-sided while the reported p-values are
two-sided). To bound a confirmatory replication, the
prospective sample size needed to reach the target power
0.80 at the observed effect is
$n = 5$ matched trials — the explicit
sample-size budget a follow-up study should adopt. These power quantities
characterize the *server-side* aggregation contrast; per the honesty contract of
[@sec:robustness-axes] they do not certify the per-client $\beta$/rcce guarantees, which
are pinned by the locked core ([@sec:formalism]) rather than by these
aggregation-level statistics.

## Software environment and configuration fingerprint {#sec:methods-software}

The FedGVI core, the POMDP studies, and the logistic-regression baseline are
pure NumPy / SciPy — no GPU and no network. The deterministic-MLP neural complement
([@sec:results-baseline]) additionally uses PyTorch (CPU); the analysis pipeline
executes it and emits its numbers as tokens exactly like every other result when
the `torch` optional extra is installed, and otherwise records a skipped status
with unavailable-value sentinels instead of silently fabricating neural results.
Versions and platform — including the PyTorch version used for the MLP
complement — are recorded automatically in [@sec:reproducibility].
