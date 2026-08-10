## Supplement: hierarchical POMDP methods and parameters {#sec:supp-hierarchical}

This supplement makes the two-level construction of [@sec:results-hierarchical]
concrete: how location (L1) is coupled to context (L2) through
context-conditioned priors, what the alternating-minimization update actually
computes, and the exact parameters of the executed run. It answers the question
the main section brackets — *by what mechanism does a second latent level enter
the inference at all* — and thereby fixes why the hierarchy resolves context
above chance while leaving location accuracy statistically unchanged.

### Generative model for context-gated location inference

The two-level POMDP implemented in
`fedference.pomdp.build_hierarchical_world` couples the sentinel's
{{HIER_N_LOCATIONS}}-location L1 factor to a {{HIER_N_CONTEXTS}}-state L2 context
factor via **context-conditioned L1 priors**:

* **L1 (location)** — the standard 3x3 grid of `build_sentinel_world` with
  ``n_s = `` {{HIER_N_LOCATIONS}} states and sensor acuity {{HIER_ACUITY}};
* **L2 (context)** — a binary state (``quiet`` / ``alert``) with a symmetric
  transition matrix (persistence {{HIER_CTX_PERSIST}}) and an initial uniform
  prior;
* **L1 priors given context** — ``quiet``: uniform over all {{HIER_N_LOCATIONS}}
  cells; ``alert``: mass {{HIER_ALERT_CENTER_MASS}} at the center cell (the den),
  the residual spread uniformly.

### Inference algorithm for top-down empirical priors

`fedference.pomdp.hierarchical_infer` performs {{HIER_N_ITERS}} passes of
alternating minimization:

1. **L2 → L1 empirical prior**: $\widetilde{\pi}_{0,\mathrm{L1}} =
   \sum_c q_{\text{ctx}}[c]\,\pi_{0,\mathrm{L1}\mid c}$
   (a soft mixture of the two context-conditioned priors).
2. **L1 update**: one-step variational posterior
   $q_{\text{loc}} = \operatorname{softmax}(\log \widetilde{\pi}_{0,\mathrm{L1}} +
   \log A[\text{obs},\,\cdot])$.
3. **L1 → L2 marginal evidence**:
   $\ell_c = \log\bigl(\pi_{0,\mathrm{L1}\mid c}^{\top}
   A[\text{obs},\,\cdot]\bigr)$
   (evidence for context $c$ from the observed location likelihood).
4. **L2 update**: $q_{\text{ctx}} = \operatorname{softmax}(\log \pi_{0,\mathrm{L2}} + \ell)$.

After {{HIER_N_ITERS}} iterations the agent broadcasts both $q_{\text{loc}}$ and
$q_{\text{ctx}}$; the colony federates each level independently via a log-linear
pool ([@eq:log-linear-pool]).

### Study parameters for the hierarchical condition

| Parameter | Value |
|---|---|
| Agents | {{HIER_N_AGENTS}} |
| Trials | {{HIER_N_TRIALS}} |
| Acuity | {{HIER_ACUITY}} |
| Alternating-min iterations | {{HIER_N_ITERS}} |
| L2 context states | {{HIER_N_CONTEXTS}} |
| L1 location states | {{HIER_N_LOCATIONS}} |
| Seed | {{HIER_SEED}} |

: Study 6 hierarchical POMDP execution parameters: agent count, seeded trial
budget, observation acuity, alternating-minimization iterations, and the L2/L1
state cardinalities used by the two-level condition. {#tbl:hier-params}

The executed hierarchical configuration is summarized in [@tbl:hier-params].
