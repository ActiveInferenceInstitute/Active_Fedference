# Contributions and evidence boundaries {#sec:contributions}

We make ten contributions, each paired with a theorem, figure, table, or
generated token and with an explicit boundary on what the evidence establishes.
They fall into four groups: the first two build the core and its recovery
contract; the next two report and statistically qualify the contaminated-consensus
result; the fifth and sixth make the robustness axes explicit and add the
objective-backed server rule; and the remaining four are scoped extensions —
parameter recovery, the tempered aggregation family, an aggregation-API transfer,
and a disjoint-observation communication test.

1. **A discrete-categorical FedGVI core.** A typed, deterministic, pure-NumPy/
   SciPy reimplementation of the FedGVI [@mildner2025fedgvi] generalized-Bayes
	   primitives — divergences, bounded robust losses, the generalized posterior,
	   the cavity/factor algebra, and robust aggregation — in the
	   discrete-categorical setting that active inference [@dacosta2020active] uses.
	   The objective ([@eq:gen-bayes]) and its closed-form tempered-softmax solution
	   are stated in Definition \ref{def:generalized-bayes} and tested by the recovery-limit probes. The whole core is
	   zero-mock and reproducible [@peng2011reproducible].

2. **A recovery-tested connection between categorical pooling and robust Bayes.**
   A recovery certificate showing that the client KL/negative-log-likelihood
   loss limits recover Bayes and the server's zero-robustness branch recovers
   the project log-linear pool. Under the explicit shared-support,
   posterior-log-potential, and fixed-weight assumptions in
   [@sec:method-aggregation], that pool specializes Friston et al.'s Eq. 7
   message-combination term [@friston2024federated], not the complete source
   protocol. Three recovery limits are pinned to bit-level residuals: the
   bounded losses recover NLL/Bayes
   (Corollary \ref{cor:closed-form-bayes} +
   Proposition \ref{prop:robust-loss-recovery}; $\le {{RECOVERY_BETA_MAXDIFF_MATH}}$ and $\le {{RECOVERY_RCCE_MAXDIFF_MATH}}$
   maximum residual, [@eq:standard-bayes]), the Rényi divergence recovers KL
   (Lemma \ref{lem:renyi-kl-limit}; $\le {{RECOVERY_RENYI_MAXDIFF_MATH}}$ residual, [@eq:renyi-limit]), and the server-side
   reweighting pool recovers the naive pool in its trusting limit
   (Theorem \ref{thm:belief-sharing-recovery};
   $\le {{RECOVERY_AGGREGATE_MAXDIFF_MATH}}$ residual, [@eq:robust-identity]).
   Robustness is thereby a tested recovery-limit extension, not a replacement.

3. **End-to-end evaluation of robust federated active inference.** Three worked
   categorical source-mechanism analogues — communicating colonies reaching lower
   free energy ([@sec:results-belief_sharing]), Dirichlet language acquisition
   ([@sec:results-language]), and structure emergence by Bayesian model reduction
   [@friston2011post] ([@sec:results-emergence]) — plus a contaminated-sentinel
   robustness sweep ([@sec:results-robustness]) in which the naive pool degrades
   and at least one server-side robust member clears the configured threshold at
   the most severe swept rate.

4. **A statistically qualified server-side contrast.** The "robust beats naive"
   conclusion for the declared contamination rate is produced only by a
   matched-pairs Wilcoxon signed-rank test
   [@wilcoxon1945individual] deflated across the divergence family with
   Benjamini–Hochberg FDR [@benjamini1995controlling], reported with bootstrap
   confidence intervals [@efron1993bootstrap] and observed-effect design-power
   planning.
   Across {{SWEEP_N_TRIALS}} paired trials the headline display method
   ({{SWEEP_HEADLINE_METHOD}}; tied set: {{SWEEP_HEADLINE_TIE_SET}}) reaches accuracy {{SWEEP_BEST_VERDICT_ACCURACY_MEAN}}
   against the naive pool's {{SWEEP_NAIVE_VERDICT_ACCURACY_MEAN}} at the verdict rate
   ($q = {{SWEEP_BEST_QVALUE_MATH}}$, rank-biserial-derived $d$-equivalent = {{SWEEP_BEST_D_EQUIVALENT}}). The predeclared selection rule is {{SWEEP_HEADLINE_SELECTION_RULE}}; the method with the largest paired mean difference is {{SWEEP_LARGEST_MEAN_DIFFERENCE_METHOD}}. Every
   headline number is a generated token.

5. **An explicit accounting of the robustness axes.** A clear separation
   ([@sec:robustness-axes]) between the client-side per-agent update, which
   inherits FedGVI's bounded-influence result under its stated source
   assumptions; the sharp server-side reweighting
   heuristic, whose positive formal property is its recovery limit and whose
   declared separable objective class has a scoped no-go result; and the
   conservative variational server rule, which is objective-backed and has a
   redescending effective-weight update but is not the accuracy-maximizer. Downstream users are told
   exactly which result is theoretically backed and which is a labeled heuristic.

6. **An objective-backed server aggregator with redescending weights.** We
   derive an aggregation free energy [@eq:agg-free-energy] whose exact block
   updates define the `variational_aggregate` rule ([@sec:method-variational],
   [@sec:supp-variational]): each exact block update monotonically decreases a
   stated objective, a converged fixed point is coordinatewise stationary, and
   the implementation keeps the lowest observed objective among converged
   configured starts (or reports the best unfinished trace as non-converged),
   recovers the standard
   log-linear pool in the trusting limit ([@eq:robust-identity]), and — unlike the
   sharp heuristic — carries a proven raw effective-weight bound
   ([@fig:aggregation-descent], [@fig:bounded-influence]). The honest cost, stated
   plainly, is conservatism: it is a maximum-entropy-biased consensus and trades peak
   point-accuracy for that control, so it complements rather than replaces the
   sharp heuristic of contribution 5.

7. **Executed finite-grid acuity-recovery experiment.** At each value in the
   {{PARAM_RECOVERY_ACUITY_GRID}} acuity grid, the study generates
   {{PARAM_RECOVERY_N_OBSERVATIONS}} synthetic observations in each of
   {{PARAM_RECOVERY_N_TRIALS}} trials and selects acuity by marginal-likelihood
   grid search over the declared finite grid. The observed mean absolute error is
   {{PARAM_RECOVERY_MEAN_ABS_ERROR}} with $R^2 =
   {{PARAM_RECOVERY_R_SQUARED}}$ ([@fig:parameter-recovery]). Acuity-by-colony-size
   behavior belongs to the separate sensitivity study; it is not a parameter-
   recovery result.

8. **The F$_\lambda$ tempered aggregation family.** A one-parameter
   $\lambda>0$ generalization of the variational aggregate
   ([@sec:supp-tempered]): $F_\lambda(q, a) = \sum_n a_n\cdot\mathrm{CE}(q,q_n)
   - \lambda H(q) + (1/c)\,\mathrm{KL}_{\mathrm{gen}}(a\|w)$. At $\lambda = {{TEMPERED_ENTROPY_WEIGHT_DEFAULT}}$ the temperature is unity
   and the objective reduces to the standard variational aggregate bit-for-bit.
   Lower $\lambda$ sharpens the variational $q$-block toward a maximizing state;
   it does not algebraically recover `robust_aggregate`. The raw
   effective-weight update and its bound are preserved for all $\lambda$. Full
   derivation in [@sec:supp-tempered].

9. **An aggregation-API transfer demonstration.** The same `robust_aggregate` API
   that governs the POMDP studies is exercised unchanged with one deterministic MLP
   trained with the density-power $\beta$-loss
   ({{BNN_HIDDEN_DIM}} hidden units, $\beta={{BNN_BETA}}$; generalized variational
   inference with a point-mass variational family) as the per-client model,
   supporting portability of the server API to this additional model class
   ([@sec:results-baseline]) when the optional `torch` extra is installed
   ([@sec:methods-software]); without it the MLP run is skipped and its tokens
   render accordingly.

10. **Communication benefit under disjoint observations.** A multi-agent extension
    ([@sec:results-disjoint-fov]) in which {{V4_N_AGENTS}} agents each observe a {{V4_FOV_WIDTH}}-slot
    disjoint window shows that belief sharing materially improves over isolated-agent
    accuracy in the declared configuration — isolated agents clear the {{V4_CHANCE_BASELINE}} chance baseline but stay
    far below the communicating consensus, which itself remains well short of full accuracy: across
    {{V4_N_SEEDS}} seeds isolated accuracy is {{V4_ISO_MEAN}} versus communicating {{V4_COMM_MEAN}},
    a reproducible margin under the declared matched-seed comparison (Wilcoxon
    $p = {{V4_WILCOX_PVALUE}}$); this is evidence for the configured
    disjoint-observation protocol, not a universal communication theorem.

The remainder of the paper proceeds as follows. [@sec:methods] develops the
FedGVI core and the recovery limits; [@sec:formalism] states the numbered
recovery theorems and the expected-free-energy identity; [@sec:methods-experimental-design]
fixes the configuration; [@sec:results] reports the {{N_STUDIES}} studies, beginning with
the recovery checks ([@sec:results-recovery]); [@sec:discussion] and
[@sec:conclusion] synthesize; and [@sec:reproducibility] and [@sec:limitations-scope] document
determinism, scope, limitations, and the standing of each robustness axis.
