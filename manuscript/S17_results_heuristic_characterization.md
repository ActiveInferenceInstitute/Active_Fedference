## Sharp server heuristic: influence and finite-breakdown characterization {#sec:results-heuristic-characterization}

The server-side `robust_aggregate` rule is the sharp heuristic axis of the
three-axes design ([@sec:robustness-axes-results]). It has BH-rejected positive
contrasts in the configured accuracy verdict in [@sec:results-verdict] but has
declared reversals elsewhere.

Unlike the objective-backed `variational_aggregate`, no closed-form
free-energy derivation has been established for it in this repository. A
separate scoped proposition in the aggregation-objective supplement rules out the
declared continuously differentiable, separable forward-KL objective class for the
implementation's raw log-pool block; it does not rule out every broader coupled
or fixed-point-only construction.

The rule therefore remains a heuristic whose
positive formal property is bit-identical recovery of the log-linear pool at
`robustness = 0` ([@eq:robust-identity]). This section does not promote the
scoped negative result into an objective certificate; it *measures* the
heuristic empirically, and the measurement makes its honesty boundary concrete.

We measure two things ([@fig:heuristic-breakdown]). First, a **numerical
influence function**: we drag one agent's belief a growing fraction toward a
confident-wrong contamination point and read its converged pooling weight. At
`robustness = 0` the weight is a flat $1/n$ at every perturbation — the naive
pool never down-weights anyone — which anchors the instrument to the proven
recovery corner.

At positive robustness the dragged agent's influence falls (not
strictly monotonically — a tiny drag can briefly *raise* it before the
divergence penalty dominates, an honest non-monotonicity we report rather than
smooth away).

Second, and more consequentially, a **breakdown witness**. We add colluding
confident-wrong adversaries — all broadcasting the same false state — to a fixed
colony of {{HCHAR_N_HONEST}} honest sentinels until each aggregator's consensus
argmax is *captured* (flips to the adversaries' target). The sharp heuristic is
captured by {{HCHAR_ROBUST_BREAKDOWN_K}} colluders; the conservative
objective-backed variational rule withstands more, capitulating only at
{{HCHAR_VARIATIONAL_BREAKDOWN_K}}.

Both counts are **finite**
({{HCHAR_HAS_FINITE_BREAKDOWN}}): a colluding majority overwhelms either rule.
That finite breakdown point is the honest headline: neither rule has an
unconditional truth-recovery claim under coordinated collusion. The absence of
an objective theorem for `robust_aggregate` is a separate derivational
boundary.

The finite capture measurement neither establishes
estimator-level B-robustness nor refutes the variational rule's stated raw
effective-weight result.

The report also runs a declared diagnostic grid over state dimension, honest-agent
count, robustness, four simple attack mechanisms, and balanced versus
adversary-downweighted base weights. This is a coverage instrument for finding
counterexamples, not a random sample of worlds and not a theorem search over all
simplexes. A finite capture row is evidence against a universal guarantee; an
uncaptured row is only “not found within this search budget.”

![Three-panel empirical characterization of the server-side heuristic. Source relation: original project diagnostic of `robust_aggregate`; estimands: numerical influence, finite-search capture count, and declared-grid capture fraction. In Panel A, the x-axis is the perturbation fraction moving one agent toward a confident-wrong belief and the y-axis is its converged normalized pooling weight. Circle-solid and square-dashed paths distinguish the naive pool and robust heuristic; a dark dotted rule marks $1/n$, and a direct arrow label gives the terminal gap. In Panel B, the x-axis is aggregator and the y-axis is colluding-adversary count at consensus-argmax capture: $k={{HCHAR_ROBUST_BREAKDOWN_K}}$ for the heuristic and $k={{HCHAR_VARIATIONAL_BREAKDOWN_K}}$ for the objective-backed variational rule. Forward and dotted hatches, dark keylines, method names, and direct `captured at k` labels duplicate color. Panel C's x-axis is declared attack mechanism and its y-axis is the fraction of searched rows with finite capture inside the configured budget; direct percentages identify each bar. Units are normalized weight, adversary count, and finite-grid fraction. This is a deterministic finite grid with no population resampling; each declared seeded scenario row is a computational unit, not an exchangeable population replicate, so no confidence interval is shown. The grid fraction is neither a probability nor a global breakdown bound, and these settings do not establish bounded influence, Byzantine tolerance, or universal robustness.](../output/figures/heuristic_breakdown.png){#fig:heuristic-breakdown width=95%}
