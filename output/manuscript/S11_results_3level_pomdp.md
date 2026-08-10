## Three-level hierarchical POMDP: an executed test of the N-level template {#sec:results-3level}

The 2-level hierarchical POMDP ([@sec:results-hierarchical]) couples location
inference to a single global context. The N-level architecture
(`fedference.pomdp.build_nlevel_world`) provides a parameterized stack
of levels; the canonical 3-level example couples location (L1; 9
states) to a context variable (L2; 2 states: ``quiet`` /
``alert``) and further to a meta-context variable (L3; 2
states: ``low_threat`` / ``high_threat``) that gates the L2 prior.

$$
\tilde{D}_{\text{L2}} = \sum_k q_{\text{L3}}[k]\,p_{\text{L2|L3}}[k]
$$ {#eq:l3-to-l2-message}

$$
\tilde{D}_{\text{L1}} = \sum_c q_{\text{L2}}[c]\,p_{\text{L1|L2}}[c]
$$ {#eq:l2-to-l1-message}

The inference algorithm (`fedference.pomdp.nlevel_infer`) performs
4 passes of top-down / bottom-up alternating minimization:
the top-down pass propagates empirical priors from L3 → L2 → L1 via
[@eq:l3-to-l2-message] and [@eq:l2-to-l1-message]; the bottom-up pass updates
each level's belief from the marginal evidence contributed by the level below.

We compare two conditions over 960 seeded trials with
4 agents at sensor acuity 0.85:

* **Flat** — agents ignore all hierarchy and infer location under a uniform prior;
* **3-level** — agents run 4 alternating-minimization iterations
  across all three levels before federating.

The measured location accuracies are 0.984 (flat) and
0.966 (3-level), a gap of -0.019. Across
128 independent seeds the 3-level location accuracy is
0.976 (SD 0.005; 95 % CI
0.976–0.977) versus flat
0.981 (95 % CI
0.980–0.981), a mean accuracy gap of
-0.004 (95 % CI
-0.005–-0.003;
Wilcoxon signed-rank $p = 0.0000$, effect size
$r = 0.724$, medium; the location gap over the
flat baseline is not statistically significant at this seed count). The 3-level
condition additionally reports context accuracy 0.697 and
meta-context accuracy 0.547. Against the two-state chance
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
