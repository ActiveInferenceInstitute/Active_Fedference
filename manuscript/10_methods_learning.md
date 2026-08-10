## Learning stack: EFE, Dirichlet updates, and BMR {#sec:methods-learning}

Active-inference agents do more than infer states under a fixed model: they
learn the parameters of the model, score policies by expected free energy, and
revise model structure. The active-inference community has standardized all
three operations [@dacosta2020active; @smith2020active], and Friston et al.
[-@friston2024federated] place them at the heart of the federated belief-sharing
scenario. We reimplement each in closed form so the language-acquisition,
expected-free-energy, and emergence studies of [@sec:methods-experimental-design]
rest on machine-checkable quantities rather than fitted curves.

## Conjugate Dirichlet learning from co-occurrence counts {#sec:methods-dirichlet}

A sentinel learns its observation model $A$ by placing a Dirichlet prior with
concentration $a$ on each column and updating it conjugately from
observation-state co-occurrence counts (Friston et al. [-@friston2024federated],
their equations 9--12). One learning step adds the expected sufficient statistics for that
step and reads off the column-normalized posterior mean,

$$
a \;\leftarrow\; a + \text{counts},
\qquad
\mathbb{E}[A]_{o s} \;=\; \frac{a_{o s}}{\sum_{o'} a_{o' s}},
$$ {#eq:dirichlet-update}

implemented in `dirichlet_learning.learn_likelihood`. Intuitively, each
concentration vector is a running tally of how often each outcome was seen while
the creature occupied a given state: the prior seeds that tally with
pseudo-counts, every step adds the co-occurrences it witnessed, and the posterior
mean is simply the tally renormalized into a categorical. Likelihood learning is
therefore bookkeeping — accumulate counts, then normalize — with no iterative
optimization loop. We drive
[@eq:dirichlet-update] with the expected sufficient statistics under the true
model — a fixed count batch $\text{count\_scale}\cdot A^{\star}$ per step,
optionally jittered by a seeded generator. As the concentrations accumulate, the
expected likelihood $\mathbb{E}[A]$ converges to the data-generating $A^{\star}$;
convergence is measured by the per-column KL divergence summed over hidden
states,

$$
\mathrm{KL}\big(A^{\star} \,\|\, \mathbb{E}[A]\big)
\;=\; \sum_{s}\sum_{o} A^{\star}_{o s}\,\ln\frac{A^{\star}_{o s}}{\mathbb{E}[A]_{o s}},
$$ {#eq:dirichlet-kl}

which decreases monotonically toward zero — the standard-Bayes / KL fixed point.
The learned likelihood always has full support (the Dirichlet prior is strictly
positive), so [@eq:dirichlet-kl] is finite throughout. ISC-17 pins the
monotone-decreasing KL trajectory, and the language-acquisition study of
[@sec:methods-experimental-design] reports the descent of [@eq:dirichlet-kl]
across {{LANGUAGE_NUM_STEPS}} steps.

The implementation also carries the $\eta$ forgetting hyperprior of Friston
et al. [-@friston2024federated], their equation 12: before each conjugate addition the running concentration is decayed so
the *total* concentration mass saturates at $\eta$ rather than growing without
bound, modeling an agent that stays adaptable instead of becoming infinitely
confident. With $\eta$ unset the classical unbounded accumulation of
[@eq:dirichlet-update] is recovered.

## Expected free energy as the action-selection objective {#sec:methods-efe}

A sentinel scores a candidate policy $\boldsymbol{\pi}$ by its expected free energy $G(\boldsymbol{\pi})$,
which the active-inference formulation decomposes two equivalent ways
(Friston et al. [-@friston2024federated], their equation 2): a cost view of risk plus
ambiguity, and a value view of pragmatic plus epistemic value. The two views are
the same scalar rearranged, stated as [@eq:efe-decomposition] and pinned to a
zero residual by the algebraic identity [@eq:efe-identity] in [@sec:formalism].
We compute every term in closed form from the categorical model $(A, B, C, D_0)$
in `expected_free_energy.decompose`:

- *Risk* is $\mathrm{KL}\big(q(o\mid\boldsymbol{\pi})\,\|\,p_C(o)\big)$, the deviation of the
  policy-predicted outcomes from the preferred-outcome pmf
  $p_C(o)=\mathrm{softmax}(C)[o]$; write
  $q_{\boldsymbol{\pi}}(o):=q(o\mid\boldsymbol{\pi})$ for this scored-policy
  outcome predictive, used in the
  remaining terms.
- *Ambiguity* is the expected likelihood entropy
  $\mathbb{E}_{q(s)}\!\big[H[p(o\mid s)]\big]$, the outcome uncertainty given the
  state.
- *Pragmatic value* is the expected log-preference
  $\mathbb{E}_{q_{\boldsymbol{\pi}}(o)}[\ln p_C(o)]$ — the utility, exploitation term.
- *Epistemic value* is the state-outcome mutual information
  $H[q_{\boldsymbol{\pi}}(o)] - \mathbb{E}_{q(s)}[H[p(o\mid s)]]$ — the expected information gain
  that drives exploration.

Because there is no sampling, the identity of [@eq:efe-identity] holds to
floating-point tolerance; ISC-19 (`expected_free_energy`) pins the residual of
the decomposition to zero and pins each term's semantics independently
(deterministic likelihoods give zero ambiguity; uninformative likelihoods give
zero epistemic value; preference-matched predictions lower risk).
[@fig:efe-decomp] visualizes the additive cost view and the signed
pragmatic/epistemic waterfall terminating at $G(\boldsymbol{\pi})$ — a deterministic identity
(Proposition \ref{prop:efe-decomposition}), not a fitted result.

## Bayesian model reduction for structure emergence {#sec:methods-bmr}

Sentinels also revise model *structure*. Bayesian model reduction (BMR) scores
whether a reduced model — for example one that prunes a redundant location column
by shrinking its concentration toward zero — has more evidence than the full
model, *without re-running inference* (Friston & Penny via the post-hoc model
optimization lineage [@friston2011post]; the same Beta-function identity is
their equation 13 (Friston et al. [-@friston2024federated])). Because the likelihood is
shared, the reduced posterior is available in closed form,
$\text{reduced\_post} = \text{post} + \text{reduced\_prior} - \text{prior}$, and
the change in (negative) variational free energy is a difference of log
multivariate Beta functions,

$$
\begin{aligned}
\Delta F
&\;=\; \ln B(\text{prior}) + \ln B(\text{reduced\_post})
  - \ln B(\text{post}) - \ln B(\text{reduced\_prior}),\\
\ln B(a)
&\;=\; \textstyle\sum_k \ln\Gamma(a_k)
  - \ln\Gamma\!\big(\sum_k a_k\big),
\end{aligned}
$$ {#eq:bmr-deltaf}

computed in `bayesian_model_reduction.reduce`. A positive $\Delta F$ in
[@eq:bmr-deltaf] means the reduced model carries more evidence — the pruned
structure was redundant and should be adopted; a negative $\Delta F$ means the
reduction destroyed support the data require. When the reduced prior equals the
prior the score is identically zero in exact algebra, a zero point the suite
pins to machine precision (ISC-20).

The emergence study of [@sec:methods-experimental-design] uses this operation
over $n = {{EMERGENCE_N}}$ candidate states. It contrasts a redundant reduction
($\Delta F = {{EMERGENCE_DELTA_F_REDUNDANT}}$ nats; adopt) with a supported one
($\Delta F = {{EMERGENCE_DELTA_F_SUPPORTED}}$ nats; reject) in
[@fig:emergence-bmr]. This fixed-candidate algebraic comparison is deterministic,
so it has no resampled sample or bootstrap interval.
