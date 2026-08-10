## Three-level hierarchical POMDP: an executed test of the N-level template {#sec:results-3level}

The 2-level hierarchical POMDP ([@sec:results-hierarchical]) couples location
inference to a single global context. The N-level architecture
(`fedference.pomdp.build_nlevel_world`) provides a parameterized stack
of levels; the canonical {{NLEVEL3_N_LEVELS}}-level example couples location (L1; {{NLEVEL3_N_LOCATIONS}}
states) to a context variable (L2; {{NLEVEL3_N_CONTEXTS}} states: ``quiet`` /
``alert``) and further to a meta-context variable (L3; {{NLEVEL3_N_META_CONTEXTS}}
states: ``low_threat`` / ``high_threat``) that gates the L2 prior.

$$
\tilde{D}_{\text{L2}} = \sum_k q_{\text{L3}}[k]\,p_{\text{L2|L3}}[k]
$$ {#eq:l3-to-l2-message}

$$
\tilde{D}_{\text{L1}} = \sum_c q_{\text{L2}}[c]\,p_{\text{L1|L2}}[c]
$$ {#eq:l2-to-l1-message}

The inference algorithm (`fedference.pomdp.nlevel_infer`) performs
{{NLEVEL3_N_ITERS}} passes of top-down / bottom-up alternating minimization:
the top-down pass propagates empirical priors from L3 → L2 → L1 via
[@eq:l3-to-l2-message] and [@eq:l2-to-l1-message]; the bottom-up pass updates
each level's belief from the marginal evidence contributed by the level below.

We compare two conditions over {{NLEVEL3_N_TRIALS}} seeded trials with
{{NLEVEL3_N_AGENTS}} agents at sensor acuity {{NLEVEL3_ACUITY}}:

* **Flat** — agents ignore all hierarchy and infer location under a uniform prior;
* **3-level** — agents run {{NLEVEL3_N_ITERS}} alternating-minimization iterations
  across all three levels before federating.

The measured location accuracies are {{NLEVEL3_LOC_ACC_FLAT}} (flat) and
{{NLEVEL3_LOC_ACC_3LEVEL}} (3-level), a gap of {{NLEVEL3_LOC_ACC_GAP}}. Across
{{NLEVEL3_N_SEEDS}} independent seeds the 3-level location accuracy is
{{NLEVEL3_LOC_ACC_3LEVEL_MEAN}} (SD {{NLEVEL3_LOC_ACC_3LEVEL_STD}}; {{CI_PERCENT}} % CI
{{NLEVEL3_LOC_ACC_3LEVEL_CI_LO}}–{{NLEVEL3_LOC_ACC_3LEVEL_CI_HI}}) versus flat
{{NLEVEL3_LOC_ACC_FLAT_MEAN}} ({{CI_PERCENT}} % CI
{{NLEVEL3_LOC_ACC_FLAT_CI_LO}}–{{NLEVEL3_LOC_ACC_FLAT_CI_HI}}), a mean accuracy gap of
{{NLEVEL3_LOC_ACC_GAP_MEAN}} ({{CI_PERCENT}} % CI
{{NLEVEL3_LOC_ACC_GAP_CI_LO}}–{{NLEVEL3_LOC_ACC_GAP_CI_HI}};
Wilcoxon signed-rank $p = {{NLEVEL3_WILCOX_PVALUE}}$, effect size
$r = {{NLEVEL3_EFFECT_SIZE}}$, {{NLEVEL3_EFFECT_LABEL}}; the location gap over the
flat baseline is not statistically significant at this seed count). The 3-level
condition additionally reports context accuracy {{NLEVEL3_CTX_ACC}} and
meta-context accuracy {{NLEVEL3_META_CTX_ACC}}. Against the two-state chance
baseline of $0.5$, location is recovered and the intermediate context latent is
resolved well above chance, but the meta-context latent is only marginally above
chance — the weakest of the three levels — and is therefore *not* convincingly
recovered here. The study thus demonstrates that the generic $N$-level
alternating-minimization runs and federates end-to-end and recovers the fastest
(location) and intermediate (context) latents; full recovery of the slowest
(meta-context) level is left open. The full figure
comparing 2-level and 3-level belief dynamics is [@fig:hierarchical-pomdp]. The
declarative layer specification used by the generic constructor is documented in
the supplement ([@sec:supp-3level]). For the effect of acuity and colony size on
these results, see [@sec:results-sensitivity].
