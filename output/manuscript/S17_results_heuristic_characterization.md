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
or fixed-point-only construction. The rule therefore remains a heuristic whose
positive formal property is bit-identical recovery of the log-linear pool at
`robustness = 0` ([@eq:robust-identity]). This section does not promote the
scoped negative result into an objective certificate; it *measures* the
heuristic empirically, and the measurement makes its honesty boundary concrete.

We measure two things ([@fig:heuristic-breakdown]). First, a **numerical
influence function**: we drag one agent's belief a growing fraction toward a
confident-wrong contamination point and read its converged pooling weight. At
`robustness = 0` the weight is a flat $1/n$ at every perturbation — the naive
pool never down-weights anyone — which anchors the instrument to the proven
recovery corner. At positive robustness the dragged agent's influence falls (not
strictly monotonically — a tiny drag can briefly *raise* it before the
divergence penalty dominates, an honest non-monotonicity we report rather than
smooth away).

Second, and more consequentially, a **breakdown witness**. We add colluding
confident-wrong adversaries — all broadcasting the same false state — to a fixed
colony of 5 honest sentinels until each aggregator's consensus
argmax is *captured* (flips to the adversaries' target). The sharp heuristic is
captured by 2 colluders; the conservative
objective-backed variational rule withstands more, capitulating only at
4. Both counts are **finite**
(Yes): a colluding majority overwhelms either rule.
That finite breakdown point is the honest headline: neither rule has an
unconditional truth-recovery claim under coordinated collusion. The absence of
an objective theorem for `robust_aggregate` is a separate derivational
boundary, and the finite capture measurement neither establishes
estimator-level B-robustness nor refutes the variational rule's stated raw
effective-weight result.

The report also runs a declared diagnostic grid over state dimension, honest-agent
count, robustness, four simple attack mechanisms, and balanced versus
adversary-downweighted base weights. This is a coverage instrument for finding
counterexamples, not a random sample of worlds and not a theorem search over all
simplexes. A finite capture row is evidence against a universal guarantee; an
uncaptured row is only “not found within this search budget.”

![Source relation: original project diagnostic of the server-side heuristic;
estimand: numerical influence, finite-search breakdown count, and declared-grid
capture fraction; uncertainty: deterministic seeded colonies, so no resampling
interval is shown. Empirical characterization of the `robust_aggregate` heuristic (two panels plus
an optional attack-grid diagnostic).
Left panel (numerical influence): the x-axis is the perturbation fraction by
which one agent's belief is dragged toward a confident-wrong contamination
point; the y-axis is that agent's converged normalized pooling weight, plotted
for the naive pool (flat at $1/n$, dotted reference) and the robust heuristic
(down-weighting). The inset reports the final naive-minus-robust weight gap at
the end of the probed path. Labeled "empirical, at these settings — not a guarantee."
Right panel (measured breakdown point): the x-axis is the aggregator (robust
heuristic vs objective-backed variational); the y-axis is the number of
colluding confident-wrong adversaries that captures that aggregator's consensus
argmax — the robust heuristic at $k = 2$ and the
variational rule at $k = 4$. Both bars are
finite, so neither rule has an unconditional truth-recovery guarantee against
coordinated collusion; this does not negate the variational rule's per-agent
effective-weight theorem. Deterministic seeded colonies (no resampling), so no
error band is applicable. The optional third panel reports the fraction of
declared grid rows with finite capture within the configured adversary budget;
it is not a probability or a global breakdown bound.](../output/figures/heuristic_breakdown.png){#fig:heuristic-breakdown width=95%}
