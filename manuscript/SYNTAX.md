# Manuscript Syntax Reference (active_fedference)

This file is the **canonical cross-reference registry** for the standalone
*Active Fedference* manuscript. Section authors MUST use the exact label strings
in the registry tables below verbatim, so cross-references resolve and no two
display equations / sections / tables / figures collide.

> **Hard rule (this project).** Every number in prose is a `{{TOKEN}}` emitted by `src/manuscript_variables.generate_variables` — **never** hardcode a numeral. Cite with the bracket form (`[@key]`); cite `[@friston2024federated]` and `[@mildner2025fedgvi]` and keep every existing `references.bib` key. Display equations carry exactly one `{#eq:label}`; reference them as `[@eq:label]`. Figures, tables, and sections follow the pandoc-crossref forms below.

## Citation Syntax (Pandoc)

```markdown
[@friston2024federated]                 <!-- single -->
[@friston2024federated; @mildner2025fedgvi]   <!-- multiple -->
[@basu1998robust, p. 555]               <!-- with locator -->
Friston et al. [-@friston2024federated] <!-- author-suppressed narrative (year only) -->
```

All citation keys must exist in [`references.bib`](references.bib). Pandoc with `--natbib` converts `[@key]` to the right LaTeX cite command automatically; **never** write raw `\cite{}` in Markdown. The complete key set is fixed — do not invent keys:

`friston2024federated`, `mildner2025fedgvi`, `pascal1654probability`,
`huygens1657ratiociniis`, `montmort1708essay`, `bernoulli1713ars`,
`demoivre1718doctrine`, `bernoulli1738mensura`, `bayes1763essay`,
`laplace1774memoire`, `borda1784elections`, `condorcet1785essai`,
`bissiri2016general`, `knoblauch2022generalized`,
`futami2018robustvi`, `nguyen2026closedformgvi`, `huber2009robust`,
`basu1998robust`, `fujisawa2008robust`, `ghosh2015robust`,
`zhang2018generalized`, `grunwald2012safe`,
`jiang2008gibbs`, `miller2018coarsening`, `jewson2018divergence`,
`kleijn2012misspecification`, `friston2011post`, `genest1986combining`,
`genest1986externally`, `hinton2002products`, `carvalho2023logpooling`,
`abbas2009kullback`, `dietrich2021fully`, `tresp2000bayesian`,
`benjamini1995controlling`, `fay2010wilcoxon`, `wilcoxon1945individual`,
`nakagawa2007effect`, `wasserstein2016asa`, `peng2011reproducible`,
`rougier2014figures`, `midway2020visualization`, `friston2010free`,
`friston2017active`, `dacosta2020active`, `heins2022pymdp`,
`bagaev2023rxinfer`, `heins2023collective`, `albarracin2022epistemic`,
`kaufmann2021collective`, `smith2020active`, `mcmahan2017communication`,
`minsker2017median`, `ashman2022partitioned`, `bui2018partitioned`,
`li2022gammafl`, `blanchard2017krum`, `pillutla2022robust`,
`karakulev2025bayesian`, `efron1993bootstrap`, `koehler2009mcse`,
`mildner2025rates`, `morris2019simulation`, `loy2021lmeresampler`.

## Equation Environments

```markdown
<!-- Numbered display equation with its canonical label -->
$$
q^\ast(s) \;\propto\; \pi_0(s)\,\exp\!\big(-\tau \textstyle\sum_i L(s; o_i)\big)
$$ {#eq:tempered-softmax}

<!-- Reference in text -->
[@eq:tempered-softmax] is the minimiser; the recovery to Bayes is [@eq:standard-bayes].
```

**Never** use raw LaTeX `\ref` / `\eqref` in Markdown. Reference equations with `[@eq:label]` (parenthetical) or `@eq:label` (narrative). **One `{#eq:}` label per display equation, drawn verbatim from the registry below.**

### Canonical equation label registry

Each display equation in the manuscript gets exactly one of these labels. The "Owner section" column says which numbered file is the **canonical home** of the equation (where the `$$ … $$ {#eq:label}` definition lives); other sections reference it with `[@eq:label]` and never re-define it.

| Label string (use verbatim) | Equation / identity | Owner section file |
|---|---|---|
| `{#eq:gen-bayes}` | Generalised-(Gibbs)-Bayes objective: $q^\ast=\arg\min_q \mathbb{E}_q[\sum_i L(s;o_i)] + \tfrac1\tau D(q\,\|\,\pi_0)$ | `05_methods_generalized_bayes.md` |
| `{#eq:tempered-softmax}` | Tempered-softmax minimiser for $D=\mathrm{KL}$: $q^\ast(s)\propto\pi_0(s)\exp(-\tau\sum_i L)$ | `05_methods_generalized_bayes.md` |
| `{#eq:standard-bayes}` | KL/NLL recovery to standard Bayes: $q^\ast(s)\propto\pi(s)\prod_i p(o_i\mid s)$ (the corollary corner) | `07_methods_aggregation.md` |
| `{#eq:renyi-limit}` | Rényi $\alpha$-divergence KL limit: $D_\alpha(q\,\|\,p)\xrightarrow{\alpha\to1}\mathrm{KL}(q\,\|\,p)$ | `06_methods_divergences_and_losses.md` |
| `{#eq:beta-loss}` | Density-power $\beta$-loss (recentered) and its $\beta\to0\Rightarrow\mathrm{NLL}$ limit | `06_methods_divergences_and_losses.md` |
| `{#eq:rcce-loss}` | Robust categorical cross-entropy $L_{q_{\rm loss}}(p,o)=(1-p(o)^{q_{\rm loss}})/q_{\rm loss}$ and its $q_{\rm loss}\to0\Rightarrow\mathrm{NLL}$ limit | `06_methods_divergences_and_losses.md` |
| `{#eq:log-linear-pool}` | Log-linear pool $=\mathrm{softmax}(\sum_n w_n\log q_n)$ — categorical posterior-log-potential specialization of Eq. 7's message-combination term under explicit shared-support and fixed-weight assumptions (`friston_belief_share`) | `07_methods_aggregation.md` |
| `{#eq:robust-identity}` | Project-local identity: `robust_aggregate(·, robustness=0)` $\equiv$ `log_linear_pool(·)`; it does not recover the complete source protocol | `07_methods_aggregation.md` |
| `{#eq:cavity}` | Cavity operation: remove one site factor in natural-parameter space, $q_{-n}=\mathrm{normalize}(q/t_n)$ | `05_methods_generalized_bayes.md` |
| `{#eq:belief-round}` | Belief-sharing round: heard consensus excludes the agent's own message (sensory attenuation) | `08_methods_belief_sharing.md` |
| `{#eq:efe-decomposition}` | Expected-free-energy decomposition: $G(\pi)=\text{risk}+\text{ambiguity}=-(\text{pragmatic}+\text{epistemic})$ | `14_formalism.md` |
| `{#eq:efe-identity}` | EFE identity (machine-checkable): $\text{risk}+\text{ambiguity}\equiv-(\text{pragmatic}+\text{epistemic})$ | `14_formalism.md` |
| `{#eq:observation-likelihood}` | Generative likelihood $P(o\mid s)$ — the agent observation model | `09_methods_generative_model.md` |
| `{#eq:state-inference}` | State-inference update under variational Bayes | `09_methods_generative_model.md` |
| `{#eq:variational-free-energy}` | Variational free energy $F = \mathrm{KL}(q(s)\|p(s)) - \mathbb{E}_q[\log p(o\mid s)]$ | `09_methods_generative_model.md` |
| `{#eq:dirichlet-update}` | Dirichlet belief update (sufficient-statistic accumulation) | `10_methods_learning.md` |
| `{#eq:dirichlet-kl}` | Per-column categorical KL of the target likelihood $A^{\star}$ to the learned $\mathbb{E}[A]$, summed over hidden states (monotone descent to the standard-Bayes fixed point) | `10_methods_learning.md` |
| `{#eq:bmr-deltaf}` | BMR free-energy reduction $\Delta F$ for pruning a redundant factor | `10_methods_learning.md` |
| `{#eq:contamination-mix}` | Contamination mixture: $(1-\epsilon)\,p_{\text{honest}} + \epsilon\,p_{\text{adv}}$ | `11_methods_contamination.md` |
| `{#eq:agg-free-energy}` | Aggregation free energy $F(q,a)=\sum_n a_n \mathrm{CE}(q,q_n) - H(q) + \tfrac1c \mathrm{KL}(a\|\mathbf{w})$ | `07_methods_aggregation.md` |
| `{#eq:agg-updates}` | Block-coordinate closed-form minimizers: $q^*$ and $a^*$ of $F(q,a)$ | `07_methods_aggregation.md` |
| `{#eq:agg-q-min}` | $q$-minimizer of $F$: $q^*(s)=\mathrm{softmax}(\sum_n a_n \log q_n(s))$ | `27_supplement_aggregation_objective.md` |
| `{#eq:agg-a-min}` | $a$-minimizer of $F$: $a_n^* \propto w_n \exp(-c\cdot\mathrm{CE}(q,q_n))$ — the raw effective-weight form (bounded and redescending, not a bounded-influence-function guarantee) | `27_supplement_aggregation_objective.md` |
| `{#eq:raw-log-pool-block}` | Raw weighted log-pool block $Q(a;s)=\operatorname{softmax}(\sum_n a_n\log s_n)$ used by the sharp heuristic | `27_supplement_aggregation_objective.md` |
| `{#eq:separable-server-objective}` | Declared separable objective class ruled out by the scoped raw-log-pool proposition | `27_supplement_aggregation_objective.md` |
| `{#eq:raw-log-pool-witness-source}` | One-agent interior source construction used by the scoped raw-log-pool no-go proof | `27_supplement_aggregation_objective.md` |
| `{#eq:notation-variational-objective}` | Variational objective and coordinate updates for $c>0,\lambda>0$; the $\lambda\downarrow0$ tied-argmax endpoint is separately implemented | `30_supplement_notation.md` |
| `{#eq:notation-cavity}` | Normalized cavity $q_{-n}(s) \propto q(s)/t_n(s)$ | `30_supplement_notation.md` |
| `{#eq:notation-factor-replacement}` | Normalized site-factor replacement update | `30_supplement_notation.md` |
| `{#eq:gaussian-kl}` | Closed-form KL divergence between two Gaussians (supplement derivation) | `28_supplement_extended_methods.md` |
| `{#eq:gaussian-renyi}` | Closed-form Rényi $\alpha$-divergence between Gaussians | `28_supplement_extended_methods.md` |
| `{#eq:tempered-family}` | Tempered family for $c>0,\lambda>0$: $F_\lambda(q,a)=\sum_n a_n\mathrm{CE}(q,q_n) - \lambda H(q) + \tfrac1c \mathrm{KL}(a\|\mathbf{w})$ (V1) | `27_supplement_aggregation_objective.md` |
| `{#eq:tempered-updates}` | Tempered $q$-update for $\lambda>0$: $q^*(s)\propto\exp\!\bigl(\tfrac1\lambda\sum_n a_n \log q_n(s)\bigr)$; the $\lambda\downarrow0$ tied-argmax endpoint is separately implemented (V1) | `27_supplement_aggregation_objective.md` |
| `{#eq:l3-to-l2-message}` | L3→L2 empirical prior: $\tilde{D}_{\text{L2}}=\sum_k q_{\text{L3}}[k]\,p_{\text{L2|L3}}[k]$ (N-level top-down, V2 ext.) | `S11_results_3level_pomdp.md` |
| `{#eq:l2-to-l1-message}` | L2→L1 empirical prior: $\tilde{D}_{\text{L1}}=\sum_c q_{\text{L2}}[c]\,p_{\text{L1|L2}}[c]$ (N-level top-down, V2 ext.) | `S11_results_3level_pomdp.md` |
| `{#eq:sensitivity-seed}` | Deterministic per-cell seed formula: $\text{seed}_{\text{cell}} = \text{seed}_{\text{base}} + i \cdot 10^5 + j \cdot 10^3 + t$ (sensitivity grid, V8) | `S14_supp_sensitivity.md` |

> **Collision resolved (2026-06-26).** `{#eq:dirichlet-update}` and `{#eq:bmr-deltaf}` formerly had duplicate display-equation definitions in both `05_methods_generalized_bayes.md` and `10_methods_learning.md`. The `05_` display blocks have been removed and replaced with `[@eq:dirichlet-update]` and `[@eq:bmr-deltaf]` prose references. Canonical home remains `10_methods_learning.md`.

> **Collision rule.** `{#eq:gen-bayes}` lives only in its canonical owner,
> `05_methods_generalized_bayes.md`; the introduction and other sections refer to
> it with `[@eq:gen-bayes]` rather than restating an unlabelled display equation.
> The same rule applies to any equation an author is tempted to restate — define
> once in the owner file, reference everywhere else.

> **LaTeX label fix (2026-06-26).** `eq:tempered-softmax`, `eq:cavity`, and `eq:standard-bayes` each appear inside raw LaTeX theorem environments (`\begin{definition}`, `\begin{corollary}`). Because pandoc-crossref cannot process raw LaTeX blocks, the `{#eq:…}` tags inside those environments were replaced with explicit `\label{eq:…}` commands on the line immediately after the closing `$$`. XeLaTeX resolves these labels directly; `[@eq:…]` references in the Markdown prose still resolve via pandoc-crossref (which emits `\ref{eq:…}` in the output `.tex`). Both mechanisms point to the same `\label` — no duplicate definitions.

## Theorem / Definition Environments

The `amsthm` theorem environments are declared in [`preamble.md`](preamble.md).
All five kinds **share one counter** (`definition`/`lemma`/`proposition`/
`corollary` step the `theorem` counter), so numbering is a single monotone
sequence in render order. Every theorem-like environment carries a typed LaTeX
label on its opening line. Reference it as `Theorem \ref{thm:...}`,
`Lemma \ref{lem:...}`, `Proposition \ref{prop:...}`,
`Corollary \ref{cor:...}`, or `Definition \ref{def:...}`. Never hard-code the
printed counter: inserting or reordering a statement must update every reference
automatically. Because these environments are raw LaTeX, retain a nearby pandoc
section/equation reference for HTML navigation.

In Markdown, wrap a theorem-like statement in a raw LaTeX environment (pandoc passes it through):

```markdown
\begin{theorem}[Categorical message-combination specialization and local recovery]\label{thm:belief-sharing-recovery}
Under the stated finite shared-support, posterior-log-potential, and fixed-weight
assumptions, \texttt{log\_linear\_pool} specializes Eq. 7's
message-combination term; \texttt{robust\_aggregate(robustness=0)} equals that
pool by the project-local identity (equation~\ref{eq:robust-identity}).
\end{theorem}
```

### Canonical theorem/definition label registry

| Stable label | Kind | Statement (short) | Owner section file |
|---|---|---|---|
| `def:generalized-bayes` | `definition` | Generalised-(Gibbs)-Bayes posterior (objective [@eq:gen-bayes]) | `05_methods_generalized_bayes.md` |
| `def:cavity` | `definition` | Cavity / PVI factor update ([@eq:cavity]) | `05_methods_generalized_bayes.md` |
| `lem:renyi-kl-limit` | `lemma` | KL is the $\alpha\to1$ limit of the Rényi family ([@eq:renyi-limit]) | `06_methods_divergences_and_losses.md` |
| `prop:robust-loss-recovery` | `proposition` | $\beta$-loss and rcce recover NLL in the $\beta\to0$ / $q_{\rm loss}\to0$ limit ([@eq:beta-loss], [@eq:rcce-loss]) | `06_methods_divergences_and_losses.md` |
| `thm:belief-sharing-recovery` | `theorem` | Categorical posterior-log-potential specialization of Eq. 7's message-combination term plus the local $c=0$ identity ([@eq:log-linear-pool], [@eq:robust-identity]) | `07_methods_aggregation.md` |
| `cor:closed-form-bayes` | `corollary` | `generalized_posterior(KLD, NLL)` equals closed-form prior×likelihood Bayes ([@eq:standard-bayes]) | `07_methods_aggregation.md` |
| `prop:efe-decomposition` | `proposition` | Expected-free-energy decomposition identity ([@eq:efe-identity]) | `14_formalism.md` |
| `prop:tempered-aggregation` | `proposition` | Tempered aggregation free energy ([@eq:tempered-family]) | `14_formalism.md` |
| `def:aggregation-free-energy` | `definition` | Aggregation free energy ([@eq:agg-free-energy]) | `27_supplement_aggregation_objective.md` |
| `thm:variational-aggregation` | `theorem` | Variational aggregation: descent, recovery, effective-weight bound | `27_supplement_aggregation_objective.md` |
| `prop:raw-log-pool-no-go` | `proposition` | Scoped no-go for the declared separable raw-log-pool objective class | `27_supplement_aggregation_objective.md` |
| `prop:federation-bit-identity` | `proposition` | Federation transport is bit-identical to in-process result | `29_supplement_federation_protocol.md` |

> The three-robustness-axes honesty contract binds here: **the recovery theorem
> and its closed-form-Bayes corollary cover only the recovery identity and the
> source-conditional client-side axis.** No theorem, lemma, or proposition may grant the
> server-side `robust_aggregate` divergence-reweighting heuristic the FedGVI
> bounded-influence guarantee. Its positive formal property is the
> `robustness=0` recovery limit ([@eq:robust-identity]);
> `prop:raw-log-pool-no-go` is an explicitly scoped negative result, not an
> objective certificate or universal no-objective theorem. The objective-backed
> `variational_aggregate` server rule carries a raw effective-weight bound but is conservative.

**Raw-LaTeX formalism labels.** Theorem-like environments use typed `\label`
identifiers and automatic `\ref` references. `test_xref_integrity` verifies
uniqueness, presence, resolution, and the absence of hard-coded counters.

## Figure References

```markdown
![Full self-contained caption sentence.](../output/figures/NAME.png){#fig:label width=80%}

<!-- Reference in text -->
[@fig:robustness-sweep] shows the naive pool collapsing under contamination.
```

- PNGs must exist in `output/figures/` at render time (generated by `src/figures/` via `src/analysis/workflow.py`).
- Always set `width=80%` (the project default) to avoid float-too-large warnings.
- Captions are self-contained — they render in the PDF, provide the current
  HTML image alternative, and supply the long-form figure description. Keep
  them understandable without color alone; the release review checks whether
  the generated alternative plus adjacent caption causes confusing repetition.

### Canonical figure label registry

Current figure set, produced by generators in `src/figures/` (all wired through `src/analysis/workflow.run_analysis_pipeline`). Use the label string verbatim and the exact file path.

| Label string (use verbatim) | PNG path | Generator in `src/figures/` | What it shows |
|---|---|---|---|
| `{#fig:system-overview}` | `../output/figures/system_overview.png` | `system_overview.generate_system_overview` | Three-panel schematic: setup, naive equal-weight pooling, and heuristic robust aggregation — true-state consensus mass is derived from the schematic's pooled beliefs (`SYSTEM_OVERVIEW_METADATA`), not hand-typed |
| `{#fig:belief-heatmap}` | `../output/figures/belief_heatmap.png` | `belief_heatmap.generate_belief_heatmap` | Per-agent belief matrix before/after one share round |
| `{#fig:free-energy}` | `../output/figures/free_energy_comparison.png` | `free_energy_comparison.generate_free_energy_comparison` | Study 1 categorical source-mechanism analogue to the belief-sharing mechanism illustrated in Friston Fig. 5 |
| `{#fig:language-kl}` | `../output/figures/language_kl_decay.png` | `language_kl_decay.generate_language_kl_decay` | Study 2 categorical source-mechanism analogue to the language-acquisition estimand related to Friston Fig. 7; seed-level pointwise CI |
| `{#fig:emergence-bmr}` | `../output/figures/emergence_bmr.png` | `emergence_bmr.generate_emergence_bmr` | Study 3 BMR diagnostic related to the mechanism in Friston Fig. 9 |
| `{#fig:robustness-sweep}` | `../output/figures/robustness_sweep.png` | `robustness_sweep.generate_robustness_sweep` | Consensus accuracy vs contamination rate, naive vs robust members (Study 4), with matched-trial CI and max-rate robust-minus-naive separation annotated |
| `{#fig:efe-decomp}` | `../output/figures/efe_decomposition.png` | `efe_decomposition.generate_efe_decomposition` | Additive risk-plus-ambiguity cost view and signed pragmatic/epistemic waterfall terminating at $G(\pi)$; identity residual annotated ([@eq:efe-identity]) |
| `{#fig:robust-weights}` | `../output/figures/robust_influence_weights.png` | `robust_influence_weights.generate_robust_influence_weights` | Per-agent server-side influence weight; contaminated agents highlighted — **labelled heuristic axis** |
| `{#fig:bnn-robustness}` | `../output/figures/bnn_robustness.png` | `bnn_robustness.generate_bnn_robustness` | Held-out accuracy vs label contamination; standard (nll/KLD) vs robust per-client FedGVI (rcce/AR) |
| `{#fig:descent-comparison}` | `../output/figures/descent_comparison.png` | `descent_comparison.generate_descent_comparison` | Single-start log-linear-pool capture basin vs multi-start escape; $F$-gap annotated — **heuristic axis** (iteration 6) |
| `{#fig:moving-world}` | `../output/figures/moving_world.png` | `moving_world.generate_moving_world` | 3-condition bar chart (isolated / communicating / EFE-guided): accuracy, free-energy gap, steps to consensus (V4) |
| `{#fig:hierarchical-pomdp}` | `../output/figures/hierarchical_pomdp.png` | `hierarchical_pomdp.generate_hierarchical_pomdp` | 2×3 six-panel belief dynamics — top row: 2-level (L1 posteriors, L2 context evolution, colony consensus); bottom row: 3-level extension (L1 posteriors, L2+L3 posteriors, accuracy gap over n_trials) (V2, Studies 6–7) |
| `{#fig:hierarchical-bmr}` | `../output/figures/hierarchical_bmr.png` | `hierarchical_bmr.generate_hierarchical_bmr` | Per-level Bayesian surprise for the degenerate vs informative 3-level worlds — the meta-context is pruned when non-gating, kept when informative (V2 ext., structure learning, MAJ-7) |
| `{#fig:heuristic-breakdown}` | `../output/figures/heuristic_breakdown.png` | `heuristic_breakdown.generate_heuristic_breakdown` | Empirical characterization of robust_aggregate: numerical influence gap, measured finite breakdown point, and finite-search attack-grid coverage (MAJ-1) |
| `{#fig:sensitivity-heatmap}` | `../output/figures/sensitivity_heatmap.png` | `sensitivity_heatmap.generate_sensitivity_heatmap` | 2-panel heatmap of federation accuracy gain over acuity × colony size (Study 8) |
| `{#fig:cross-study-summary}` | `../output/figures/cross_study_summary.png` | `cross_study_summary.generate_cross_study_summary` | Horizontal bar chart of per-study federation benefit ± 95 % bootstrap CI with positive/near-zero/negative row counts (Studies 1–9) |
| `{#fig:parameter-recovery}` | `../output/figures/parameter_recovery.png` | `parameter_recovery.generate_parameter_recovery` | Two-panel parameter-recovery figure: recovered vs true acuity (identity-line scatter with 95 % empirical percentile interval across independent trials) and mean absolute error per acuity level (Study 9) |
| `{#fig:complexity-scaling}` | `../output/figures/complexity_scaling.png` | `complexity_scaling.generate_complexity_scaling` | Implementation-derived asymptotic orders and seeded machine-scaling diagnostics for aggregation, leave-one-out sharing, and state inference |
| `{#fig:graphical-abstract}` | `../output/figures/graphical_abstract.png` | `figures/graphical_abstract.generate_graphical_abstract` | Layered graphical abstract: recovery anchor, federated network, deterministic consensus cards, and the three non-transferable robustness axes |
| `{#fig:generative-model-schema}` | `../output/figures/generative_model_schema.png` | `generative_model_schema.generate_generative_model_schema` | Formal schematic of temporal, hierarchical, and factorial/categorical model depth; no empirical data |
| `{#fig:message-passing}` | `../output/figures/message_passing.png` | `message_passing.generate_message_passing` | Symbolic local-update, broadcast, server-fusion, and three-axis claim-ownership map; no empirical data |
| `{#fig:pomdp-loop}` | `../output/figures/pomdp_loop.png` | `pomdp_loop.generate_pomdp_loop` | Hidden-state, observation, action, transition, and federated-belief loop; selected paths are executed in different studies |
| `{#fig:aggregation-descent}` | `../output/figures/aggregation_descent.png` | `aggregation_descent.generate_aggregation_descent` | Variational free energy $F(q,a)$ vs block-coordinate iteration — monotone non-increasing convergence (server-side objective-backed axis) | `19_results_robustness.md` |
| `{#fig:bounded-influence}` | `../output/figures/bounded_influence.png` | `bounded_influence.generate_bounded_influence` | Probed-agent normalized influence weight under `variational_aggregate` vs naive pool — empirical redescending-weight diagnostic | `19_results_robustness.md` |
| `{#fig:contamination-gallery}` | `../output/figures/contamination_gallery.png` | `contamination_gallery.generate_contamination_gallery` | Descriptive pooled-display-member accuracy per contamination mechanism with conditional seed-bootstrap bars and win-fraction screen; not selection-free inference | `28_supplement_extended_methods.md` |
| `{#fig:robustness-onset}` | `../output/figures/robustness_onset.png` | `robustness_onset.generate_robustness_onset` | Descriptive pooled-display-member accuracy vs contamination rate per directional mechanism with conditional seed-bootstrap bands, onset marker, and final gap; not selection-free inference | `28_supplement_extended_methods.md` |
| `{#fig:conditional-world}` | `../output/figures/conditional_world.png` | `conditional_world.generate_conditional_world` | Finite conditional-world grid over hidden state, observability, attack mechanism, and adversarial weight geometry | `28_supplement_extended_methods.md` |
| `{#fig:robustness-review-grid}` | `../output/figures/robustness_review_grid.png` | `robustness_review_grid.generate_robustness_review_grid` | Expanded all-method, selection-free conditional/rate robustness review surface; seeds are the inferential unit and trials are nested | `28_supplement_extended_methods.md` |
| `{#fig:belief-quality}` | `../output/figures/belief_quality.png` | `belief_quality.generate_belief_quality` | Proper categorical log-score controls and reliability diagnostic with seed-level uncertainty | `28_supplement_extended_methods.md` |
| `{#fig:disjoint-fov-world}` | `../output/figures/disjoint_fov_world.png` | `disjoint_fov_world.generate_disjoint_fov_world` | 3-condition bar chart (isolated/communicating/EFE-guided) for disjoint-FOV extension of moving sentinel world (V4) | `S05_results_moving_world.md` |

> **Honesty contract for figures.** `{#fig:robust-weights}` visualises the
> **server-side heuristic** (`robust_aggregate` divergence-reweighting); its
> caption must say so and must NOT claim a bounded-influence guarantee.
> `{#fig:bnn-robustness}`'s robust curve exercises the **per-client FedGVI**
> mechanism (rcce/AR client losses); the cited source guarantee remains
> conditional on its matching assumptions. `{#fig:bounded-influence}` belongs
> to the conservative **variational server** axis and demonstrates only the
> tested redescending normalized-weight path. Captions must keep all three axes
> distinct.

## Table References

```markdown
| Contamination rate | Naive (KLD) accuracy | Best robust accuracy |
|---|---|---|
{{SWEEP_RATE_TABLE_ROWS}}

: Self-contained caption sentence. {#tbl:robustness-sweep}

<!-- Reference in text -->
[@tbl:robustness-sweep] shows the monotone naive decline.
```

Caption goes on the line **after** the table, starting with `: `, ending with the `{#tbl:label}` attribute. Reference with `[@tbl:label]`.

### Canonical table label registry

Existing tables keep their labels (note: the underscore forms below already appear in the live prose — keep them verbatim to avoid breaking `[@tbl:…]` references). New tables get the kebab labels.

| Label string (use verbatim) | Caption summary | Owner section file | Status |
|---|---|---|---|
| `{#tbl:robustness_sweep}` | Consensus accuracy $q(\text{true state})$ by contamination rate (`{{SWEEP_RATE_TABLE_ROWS}}`) | `19_results_robustness.md` | existing |
| `{#tbl:robustness_verdict}` | Per-method paired Wilcoxon + BH-FDR verdict vs naive pool (`{{SWEEP_VERDICT_TABLE_ROWS}}`) | `19_results_robustness.md` | existing |
| `{#tbl:study_params}` | Per-study configuration from `experiment:` (`config.yaml`) | `12_methods_experimental_design.md` | existing |
| `{#tbl:repro_env}` | Software + configuration fingerprint (Python/NumPy/SciPy/platform/config hash) | `26_reproducibility.md` | existing |
| `{#tbl:repro_artifacts}` | Generated-artifact inventory (figures/data/reports/total) | `26_reproducibility.md` | existing |
| `{#tbl:verdict-effects}` | Standardized-effect verdict (rank-biserial r, d-equivalent, label, acc-diff CI, raw p, q, reject) (`{{SWEEP_VERDICT_EFFECT_TABLE_ROWS}}`) | `19_results_robustness.md` | new |
| `{#tbl:accuracy-at-verdict}` | Per-method accuracy at the verdict rate with 95% bootstrap CI (`{{SWEEP_ACCURACY_AT_VERDICT_TABLE_ROWS}}`) | `19_results_robustness.md` | new |
| `{#tbl:paired-by-rate}` | Per-rate naive-vs-robust paired tests, BH-deflated per method (`{{SWEEP_PAIRED_BY_RATE_TABLE_ROWS}}`) | `19_results_robustness.md` | new |
| `{#tbl:hier-params}` | Study 6 hierarchical POMDP execution parameters, including agent/trial budget and L2/L1 state cardinalities | `S10_supp_hierarchical_pomdp.md` | new |
| `{#tbl:nlevel3-params}` | Study 7 three-level hierarchical POMDP execution parameters, including agent/trial budget and L3/L2/L1 state cardinalities | `S12_supp_3level_pomdp.md` | new |
| `{#tbl:contamination-gallery}` | Descriptive pooled-display robust-vs-naive accuracy, win fraction, and conditional CI under each contamination mechanism (`{{GALLERY_TABLE_ROWS}}`) | `28_supplement_extended_methods.md` | new |
| `{#tbl:robustness-onset}` | Descriptive pooled-display onset rate and worst-rate naive/robust accuracy (`{{ONSET_TABLE_ROWS}}`) | `28_supplement_extended_methods.md` | new |
| `{#tbl:server-theory-witness}` | Deterministic raw-log-pool and normalized-weight scoped no-go witness inventory | `27_supplement_aggregation_objective.md` | new |

## Section Labels

Every H1 carries a `{#sec:<name>}` label so cross-section references (`[@sec:methodology]`) survive reordering. The manuscript is **decomposed into more, smaller numbered files** (see "Modular section decomposition" below); each H1 keeps a stable `{#sec:}` label, and subsections that are cross-referenced carry their own `{#sec:}` labels too.

### Canonical section label registry

| Label string (use verbatim) | H1 / H2 title | Owner file |
|---|---|---|
| `{#sec:abstract}` | Abstract | `00_abstract.md` |
| `{#sec:introduction}` | Introduction: from belief sharing to robust generalized Bayes | `01_introduction.md` |
| `{#sec:intro-active-inference}` | Active inference supplies generative agents and shared beliefs (H2) | `01_introduction.md` |
| `{#sec:intro-robust-bayes}` | Robust and federated Bayes supplies bounded-influence updating (H2) | `01_introduction.md` |
| `{#sec:intro-questions}` | Questions, design, and evidence boundary (H2) | `01_introduction.md` |
| `{#sec:intro-visual-map}` | How to read the visual architecture (H3) | `01_introduction.md` |
| `{#sec:gap}` | Research gap and claim boundary (H1) | `02_gap.md` |
| `{#sec:gap-threads}` | Five reviewed threads and their open intersection (H2) | `02_gap.md` |
| `{#sec:gap-bridge}` | The belief-fusion bridge evaluated here (H2) | `02_gap.md` |
| `{#sec:robustness-axes}` | Guarantee map: three robustness axes (H2) | `02_gap.md` |
| `{#sec:robustness-axes-results}` | Three robustness axes remain distinct in the results (H3) | `16_results_belief_sharing.md` |
| `{#sec:contributions}` | Contributions and evidence boundaries (H2) | `03_contributions.md` |
| `{#sec:methods}` | Methods: the federated active-inference stack (H1 — IMRAD top-level) | `04_methods_overview.md` |
| `{#sec:methodology}` | Methodology alias for the methods overview | `04_methods_overview.md` |
| `{#sec:method-protocol}` | Federation protocol: local update, server fusion, broadcast (H2) | `04_methods_overview.md` |
| `{#sec:method-notation}` | Notation for beliefs, losses, and divergences (H2) | `04_methods_overview.md` |
| `{#sec:method-genbayes}` | Generalized Bayes: the route back to standard Bayes (H2) | `05_methods_generalized_bayes.md` |
| `{#sec:method-learning}` | Conjugate likelihood learning for the shared model (H2) | `05_methods_generalized_bayes.md` |
| `{#sec:method-bmr}` | Bayesian model reduction for structure comparison (H2) | `05_methods_generalized_bayes.md` |
| `{#sec:method-divergences}` | Divergences: robust objectives and the KL limit (H2) | `06_methods_divergences_and_losses.md` |
| `{#sec:method-losses}` | Robust losses: bounded influence at the Bayes corner (H2) | `06_methods_divergences_and_losses.md` |
| `{#sec:method-aggregation}` | Aggregation and message passing: standard pool, heuristic, and variational server (H2) | `07_methods_aggregation.md` |
| `{#sec:method-message-passing}` | Protocol map: local updates, broadcast, and server fusion (H3) | `07_methods_aggregation.md` |
| `{#sec:method-variational}` | Variational aggregation with objective-backed weight control (H3) | `07_methods_aggregation.md` |
| `{#sec:method-belief-sharing}` | Belief sharing: the standard aggregation corner (H2) | `08_methods_belief_sharing.md` |
| `{#sec:methods-generative-model}` | Generative model: categorical states, observations, actions, and hierarchy (H2, demoted from H1) | `09_methods_generative_model.md` |
| `{#sec:methods-state-space}` | State space: one shared latent factor (H2) | `09_methods_generative_model.md` |
| `{#sec:methods-abcd}` | Four categorical tensors: likelihood, transitions, preferences, priors (H2) | `09_methods_generative_model.md` |
| `{#sec:methods-state-inference}` | One-step variational state inference in the grid world (H2) | `09_methods_generative_model.md` |
| `{#sec:methods-pomdp-loop}` | Hidden-state to action loop: the POMDP substrate (H2) | `09_methods_generative_model.md` |
| `{#sec:methods-learning}` | Learning stack: EFE, Dirichlet updates, and BMR (H2, demoted from H1) | `10_methods_learning.md` |
| `{#sec:methods-dirichlet}` | Conjugate Dirichlet learning from co-occurrence counts (H2) | `10_methods_learning.md` |
| `{#sec:methods-efe}` | Expected free energy as the action-selection objective (H2) | `10_methods_learning.md` |
| `{#sec:methods-bmr}` | Bayesian model reduction for structure emergence (H2) | `10_methods_learning.md` |
| `{#sec:methods-contamination}` | Contamination models: declared failure modes for belief fusion (H2, demoted from H1) | `11_methods_contamination.md` |
| `{#sec:methods-corruption}` | Corruption process for adversarial belief broadcasts (H2) | `11_methods_contamination.md` |
| `{#sec:methods-contamination-axes}` | How contamination meets the three robustness axes (H2) | `11_methods_contamination.md` |
| `{#sec:methods-experimental-design}` | Experimental design: studies, estimands, determinism, and power (H2, demoted from H1) | `12_methods_experimental_design.md` |
| `{#sec:methods-determinism}` | Determinism through fixed seeds and generated variables (H2) | `12_methods_experimental_design.md` |
| `{#sec:methods-studies}` | Study suite and contamination sweep (H2) | `12_methods_experimental_design.md` |
| `{#sec:methods-power}` | Sample size and prospective statistical power (H2) | `12_methods_experimental_design.md` |
| `{#sec:methods-software}` | Software environment and configuration fingerprint (H2) | `12_methods_experimental_design.md` |
| `{#sec:methods-statistics}` | Statistical protocol: matched comparisons, intervals, and bounded claims (H2, demoted from H1) | `13_methods_statistics.md` |
| `{#sec:methods-paired}` | Paired comparison and standardized effect size (H2) | `13_methods_statistics.md` |
| `{#sec:methods-bootstrap}` | Bootstrap interval estimates (H2) | `13_methods_statistics.md` |
| `{#sec:methods-fdr}` | Multiple-testing deflation by BH-FDR (H2) | `13_methods_statistics.md` |
| `{#sec:methods-statistics-power}` | Prospective power analysis for the verdict rate (H2) | `13_methods_statistics.md` |
| `{#sec:methods-reporting}` | Reporting tables and the honesty boundary (H2) | `13_methods_statistics.md` |
| `{#sec:formalism}` | Formalism: recovery limits, EFE, and tempered aggregation | `14_formalism.md` |
| `{#sec:formalism-recovery}` | Recovery limits as the proof surface (H2) | `14_formalism.md` |
| `{#sec:formalism-efe}` | Expected-free-energy identity as an algebraic check (H2) | `14_formalism.md` |
| `{#sec:formalism-tempered}` | Tempered aggregation free energy and the accuracy-guarantee trade (H2, V1) | `14_formalism.md` |
| `{#sec:formalism-tempered-interpretation}` | What the entropy weight controls (H3, V1) | `14_formalism.md` |
| `{#sec:formalism-tempered-recovery}` | Recovery at the qualified log-linear-pool corner (H3, V1) | `14_formalism.md` |
| `{#sec:formalism-tempered-evidence}` | What the accuracy--guarantee trade can establish (H3, V1) | `14_formalism.md` |
| `{#sec:formalism-tempered-interpretation-summary}` | Publication-facing interpretation (H3, V1) | `14_formalism.md` |
| `{#sec:results}` | Results: recovery checks and study suite | `15_results_recovery.md` |
| `{#sec:results-belief_sharing}` | Belief sharing lowers free energy at the project-pool corner (H2) | `16_results_belief_sharing.md` |
| `{#sec:results-language}` | Language acquisition follows conjugate Dirichlet updating (H2) | `17_results_language.md` |
| `{#sec:results-emergence}` | Bayesian model reduction selects supported structure (H2) | `18_results_emergence.md` |
| `{#sec:results-robustness}` | Contamination sweep: regime-dependent server behavior under declared attacks (H2) | `19_results_robustness.md` |
| `{#sec:results-verdict}` | Earned robustness verdict at the decisive rate (H3, statistics) | `19_results_robustness.md` |
| `{#sec:results-variational}` | Variational aggregator: conservative objective-backed weight control (H3) | `19_results_robustness.md` |
| `{#sec:results-recovery}` | Recovery limits: standard-Bayes and project-pool corners are exact to machine precision (H2) | `15_results_recovery.md` |
| `{#sec:results-baseline}` | Client-side robustness complement: categorical FedGVI baseline (H2) | `20_results_baseline.md` |
| `{#sec:discussion}` | Discussion: what the evidence supports (H1) | `21_discussion_findings.md` |
| `{#sec:discussion-limit}` | The recovery limit is the formal anchor (H2) | `21_discussion_findings.md` |
| `{#sec:discussion-joint}` | What the study suite jointly shows (H2) | `21_discussion_findings.md` |
| `{#sec:discussion-identifiability}` | What this simulation identifies—and what it does not (H2) | `21_discussion_findings.md` |
| `{#sec:discussion-verdict}` | The robustness verdict is conditional and statistically qualified (H2) | `21_discussion_findings.md` |
| `{#sec:discussion-axes}` | Three robustness axes remain separate (H2) | `21_discussion_findings.md` |
| `{#sec:discussion-tempered}` | Accuracy and effective-weight control can be traded explicitly (H2) | `21_discussion_findings.md` |
| `{#sec:discussion-downstream}` | Why the boundary matters downstream (H2) | `21_discussion_findings.md` |
| `{#sec:related-work}` | Related work: active inference, federated Bayes, and the scoped bridge (H2, demoted from H1) | `22_discussion_related_work.md` |
| `{#sec:related-historical}` | Pre-modern probability, inverse probability, and collective judgment (H2) | `22_discussion_related_work.md` |
| `{#sec:related-aif}` | Active inference: generative agents, EFE, and colonies (H2) | `22_discussion_related_work.md` |
| `{#sec:related-fl}` | Robust and federated Bayes outside active inference (H2) | `22_discussion_related_work.md` |
| `{#sec:related-gap}` | The specific bridge added here (H2) | `22_discussion_related_work.md` |
| `{#sec:limitations}` | Limitations and claim boundaries (H2, demoted from H1) | `23_discussion_limitations.md` |
| `{#sec:limitations-axes}` | Three robustness axes: theorem, heuristic, and objective (H2) | `23_discussion_limitations.md` |
| `{#sec:limitations-scope}` | Scope boundaries that the evidence does not cross (H2) | `23_discussion_limitations.md` |
| `{#sec:limitations-stats}` | What the statistics can and cannot claim (H2) | `23_discussion_limitations.md` |
| `{#sec:future}` | Future work: testing the open boundaries (H2, demoted from H1) | `24_discussion_future.md` |
| `{#sec:future-server}` | Make the sharp server heuristic variational (H2) | `24_discussion_future.md` |
| `{#sec:future-scale}` | Promote the baseline to original FedGVI scale (H2) | `24_discussion_future.md` |
| `{#sec:future-hierarchical}` | Extend hierarchical federation beyond the current stack (H2) | `24_discussion_future.md` |
| `{#sec:future-transport}` | Move from process transport to true multi-machine federation (H2) | `24_discussion_future.md` |
| `{#sec:future-continuous}` | Move beyond categorical state spaces (H2) | `24_discussion_future.md` |
| `{#sec:conclusion}` | Conclusion: a recovery-tested bridge with bounded claims (H1) | `25_conclusion.md` |
| `{#sec:conclusion-recovery}` | The durable result is a recovery contract (H2) | `25_conclusion.md` |
| `{#sec:conclusion-evidence}` | What the evidence establishes away from the corner (H2) | `25_conclusion.md` |
| `{#sec:conclusion-significance}` | Why the bridge matters for active inference (H2) | `25_conclusion.md` |
| `{#sec:conclusion-boundaries}` | What remains unproved (H2) | `25_conclusion.md` |
| `{#sec:conclusion-program}` | A falsifiable research program (H2) | `25_conclusion.md` |
| `{#sec:conclusion-position}` | Final position (H2) | `25_conclusion.md` |
| `{#sec:reproducibility}` | Reproducibility: execution record and recovery checks | `26_reproducibility.md` |
| `{#sec:repro-determinism}` | Determinism contract for seeded scientific results (H2) | `26_reproducibility.md` |
| `{#sec:repro-environment}` | Environment fingerprint for the reported run (H2) | `26_reproducibility.md` |
| `{#sec:repro-accessibility}` | Reader-surface accessibility boundary (H2) | `26_reproducibility.md` |
| `{#sec:repro-tests}` | Test and coverage evidence for the claim surface (H2) | `26_reproducibility.md` |
| `{#sec:repro-artifacts}` | Artifact inventory for figures, data, and reports (H2) | `26_reproducibility.md` |
| `{#sec:repro-recovery}` | Recovery-limit certificate for the client and project-pool corners (H2) | `26_reproducibility.md` |
| `{#sec:references}` | References | `99_references.md` |

### Supplement section labels

Supplement files use numbers above `26` (aggregation-objective, extended-methods, federation protocol, and notation) and `S##` prefixes for study-specific supplements added in later iterations. Labels use the same pandoc-crossref `{#sec:}` convention and resolve identically in combined renders.

| Label string (use verbatim) | Title / level | Owner file |
|---|---|---|
| `{#sec:supp-variational}` | Supplement: variational aggregation objective and weight control (H1) | `27_supplement_aggregation_objective.md` |
| `{#sec:supp-why-heuristic}` | Why the sharp heuristic is not yet variational (H2) | `27_supplement_aggregation_objective.md` |
| `{#sec:supp-derivation}` | Aggregation free energy and its block minimizers (H2) | `27_supplement_aggregation_objective.md` |
| `{#sec:supp-theorem}` | Formal properties of the conservative server rule (H2) | `27_supplement_aggregation_objective.md` |
| `{#sec:supp-witnesses}` | Numerical witnesses for descent and influence bounds (H2) | `27_supplement_aggregation_objective.md` |
| `{#sec:supp-tempered}` | Tempered aggregation family for the accuracy-guarantee trade (H2, V1) | `27_supplement_aggregation_objective.md` |
| `{#sec:supp-extended}` | Supplement: extended methods for scoped generalization (H1) | `28_supplement_extended_methods.md` |
| `{#sec:supp-gaussian}` | Continuous-state divergence bridge for Gaussian beliefs (H2) | `28_supplement_extended_methods.md` |
| `{#sec:supp-contamination}` | Additional contamination models for the robustness surface (H2) | `28_supplement_extended_methods.md` |
| `{#sec:supp-gallery}` | Contamination gallery by corruption mechanism (H3) | `28_supplement_extended_methods.md` |
| `{#sec:supp-conditional-world}` | Conditional world and attack-geometry grid (H1, MED-1) | `28_supplement_extended_methods.md` |
| `{#sec:supp-belief-quality}` | Proper scores and calibration controls (H1, MED-2) | `28_supplement_extended_methods.md` |
| `{#sec:supp-onset}` | Robustness onset by corruption mechanism (H3) | `28_supplement_extended_methods.md` |
| `{#sec:supp-greedy-bmr}` | Greedy multi-hypothesis model reduction beyond the main BMR study (H2) | `28_supplement_extended_methods.md` |
| `{#sec:supp-federation}` | Federation transport protocol and bit-identity witness (H2, V3) | `29_supplement_federation_protocol.md` |
| `{#sec:supp-notation}` | Authoritative supplemental notation contract (H1) | `30_supplement_notation.md` |
| `{#sec:results-moving}` | Moving sentinel world: communication benefit depends on field of view (H2, V4) | `S05_results_moving_world.md` |
| `{#sec:results-disjoint-fov}` | Disjoint field-of-view extension (H3, V4) | `S05_results_moving_world.md` |
| `{#sec:supp-moving}` | Supplement: moving-world methods and condition definitions (H2, V4) | `S06_supp_moving_world.md` |
| `{#sec:results-hierarchical}` | Hierarchical POMDP: federated belief sharing across levels (H2, V2) | `S09_results_hierarchical_pomdp.md` |
| `{#sec:supp-hierarchical}` | Supplement: hierarchical POMDP methods and parameters (H2, V2) | `S10_supp_hierarchical_pomdp.md` |
| `{#sec:results-3level}` | Three-level hierarchical POMDP: an executed test of the N-level template (H2, V2 ext.) | `S11_results_3level_pomdp.md` |
| `{#sec:supp-3level}` | Supplement: N-level hierarchical POMDP methods (H2, V2 ext.) | `S12_supp_3level_pomdp.md` |
| `{#sec:results-hierarchical-bmr}` | Structure learning: does the hierarchy earn its depth? (H2, V2 ext.) | `S16_results_hierarchical_bmr.md` |
| `{#sec:results-heuristic-characterization}` | Sharp server heuristic: influence and finite-breakdown characterization (H2, MAJ-1) | `S17_results_heuristic_characterization.md` |
| `{#sec:results-sensitivity}` | Parameter sensitivity of federation benefit | `S13_results_sensitivity.md` |
| `{#sec:supp-sensitivity}` | Supplement: parameter-sensitivity methods | `S14_supp_sensitivity.md` |
| `{#sec:results-parameter-recovery}` | Parameter recovery: acuity selection on the tested grid | `S15_results_parameter_recovery.md` |

## Modular section decomposition

The manuscript is already decomposed into small section files. Render order is
sorted filename order: `00_abstract.md` opens the paper, `01_`–`29_` carry the
main methods/results/discussion/supplement spine, `S##` files carry
study-specific supplements, and `99_references.md` closes the manuscript.
The label tables above are therefore the current live owner map, not a future
section-authoring plan.

## Preamble Injection

[`preamble.md`](preamble.md) is parsed by `infrastructure/rendering/_pdf_latex_helpers.extract_preamble` (the ```latex fence) and injected by `inject_latex_preamble`. It now provides, in addition to the core math/layout packages:

- `amsthm` theorem environments — `theorem`, plus `definition`/`lemma`/`proposition`/`corollary` sharing the `theorem` counter (single monotone numbering).
- `caption` with `\captionsetup{font=footnotesize,labelfont=bf,skip=3pt}` (booktabs already present).
- `cleveref` and `titlesec` are intentionally **not** loaded (unavailable in this TeX tree and not required): cross-references resolve through pandoc-crossref's own `\ref`-style names — no `\cref` appears — and section heading styling falls back to the pandoc/LaTeX defaults.

Do **not** duplicate package imports already in the infrastructure renderer or re-declare these theorem environments inside section files.

## Prose Conventions

- **Every numeral is a `{{TOKEN}}`** from `src/manuscript_variables.generate_variables` — no hand-typed numbers, not even in captions or tables (table bodies are `{{…_TABLE_ROWS}}` tokens).
- Carry the **three-robustness-axes honesty**: the client-side $\beta$/rcce generalised-Bayes update is the FedGVI-faithful, claim-bearing axis; the server-side `robust_aggregate` divergence-reweighting is a complementary **heuristic** whose positive formal property is the `robustness=0` recovery limit and whose declared separable objective class has a scoped no-go result; and `variational_aggregate` is an objective-backed server rule with a raw effective-weight bound and empirical redescending behavior. No claim may transfer a guarantee or accuracy verdict across axes.
- No "In summary" / "In conclusion" at section ends (RASP standard).
- Active voice for methodology; name the module and the pinning test (`src/fedference/…`, ISC-N) when stating a claim.
- One idea per paragraph; define each display equation once in its owner file and reference it elsewhere with `[@eq:label]`.

## See Also

- [`AGENTS.md`](AGENTS.md) — RASP protocol and AI agent constraints
- [`preamble.md`](preamble.md) — Active LaTeX preamble (theorem envs, caption; cleveref/titlesec not loaded)
- [`config.yaml`](config.yaml) — paper metadata, keywords, `experiment:` parameters
