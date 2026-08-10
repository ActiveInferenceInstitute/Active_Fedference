## Supplement: N-level hierarchical POMDP methods {#sec:supp-3level}

This supplement specifies the generic $N$-level architecture that
[@sec:results-3level] exercises at depth three: how the meta-context (L3),
context (L2), and location (L1) factors are chained through conditioned priors,
what the declarative `LayerSpec` interface fixes versus leaves free, and the
top-down/bottom-up passes the inference runs. It answers *what the executed
3-level result is a special case of* — the reason the same code runs at other
depths without new mathematics — while recording that only the declared 3-level
configuration is empirically evaluated here.

### Generative model for an N-level hierarchy

The 3-level POMDP implemented in
`fedference.pomdp.build_3level_world` extends the 2-level construction
([@sec:supp-hierarchical]) by adding a top-level meta-context factor:

* **L3 (meta-context)** — 2 states (``low_threat`` /
  ``high_threat``) with initial uniform prior, gating the L2 context prior;
* **L2 (context)** — 2 states (``quiet`` / ``alert``) with
  context-conditioned L1 location priors, gating the L1 prior;
* **L1 (location)** — the standard 3x3 grid with 9 states
  and sensor acuity 0.85.

The conditioned priors are (see [@eq:l3-to-l2-message] and [@eq:l2-to-l1-message]):

| L3 state | L2 prior (quiet, alert) |
|---|---|
| ``low_threat`` | (0.50, 0.50) — uniform context |
| ``high_threat`` | (0.20, 0.80) — peaked at alert |

| L2 state | L1 prior |
|---|---|
| ``quiet`` | uniform over all 9 location states |
| ``alert`` | mass 0.60 at center cell (flat index 4), residual uniform |

### Generic N-level architecture

`fedference.pomdp.LayerSpec` and `fedference.pomdp.build_nlevel_world`
implement the generic N-level version. The declarative layer specification is
stored at ``src/fedference/config/hierarchical_layers.yaml`` and mirrors the
canonical 3-level defaults (a standalone documentation artifact not read by any
code path, kept in sync with the ``build_3level_world`` defaults). The constructor
accepts depth ≥ 2; the executed empirical result in this manuscript is restricted
to the declared 3-level configuration, and the leaf layer must carry
``n_states == N_LOCATIONS``.

### Inference algorithm across hierarchy levels

`fedference.pomdp.nlevel_infer` performs 4 passes of
top-down / bottom-up alternating minimization over all N levels:

1. **Top-down pass** — compute the empirical prior for each level by marginalizing
   over the level above ([@eq:l3-to-l2-message], [@eq:l2-to-l1-message]).
2. **L1 update** — one-step variational posterior on the observation:
   $q_{\text{loc}} = \operatorname{softmax}(\log \widetilde{\pi}_{0,\mathrm{L1}} +
   \log A[\text{obs},\,\cdot])$.
3. **Bottom-up pass** — update each non-leaf level's belief from the marginal
   evidence contributed by the level below:
   $\ell_j = \log(\tilde{p}_{\text{child|parent=}j}^\top q_{\text{child}})$.

After 4 iterations the agent broadcasts all N level beliefs;
the colony federates each level independently via a log-linear pool
([@eq:log-linear-pool]).

### Study parameters for the three-level run

| Parameter | Value |
|---|---|
| Agents | 4 |
| Trials | 960 |
| Acuity | 0.85 |
| Alternating-min iterations | 4 |
| L3 meta-context states | 2 |
| L2 context states | 2 |
| L1 location states | 9 |
| Seed | 0 |

: Study 7 three-level hierarchical POMDP execution parameters: agent count,
seeded trial budget, observation acuity, alternating-minimization iterations,
and the L3/L2/L1 state cardinalities used by the three-level condition.
{#tbl:nlevel3-params}

The executed three-level configuration is summarized in [@tbl:nlevel3-params].
