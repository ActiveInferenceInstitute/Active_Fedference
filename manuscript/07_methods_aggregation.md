## Aggregation and message passing: standard pool, heuristic, and variational server {#sec:method-aggregation}

The server step lives in `aggregation.py`, where a categorical
specialization of the active-inference belief-sharing relation
[@friston2024federated] and the FedGVI objective [@mildner2025fedgvi] meet.
Each agent $n$ broadcasts a categorical local posterior $q_n(s)$ over the
shared latent factor, optionally with a scalar base weight $w_n$.

Two fusion
rules act directly on these broadcasts, and a third — the objective-backed
`variational_aggregate` of [@sec:method-variational] — refines the
second into descent on a stated objective.

The first is the **log-linear pool**,
a project-local product-of-experts consensus. In the terminology of opinion
pooling it is the logarithmic pool, a weighted geometric aggregation rule whose
Bayesian-coherence assumptions have been studied independently of active
inference [@genest1986combining; @genest1986externally;
@carvalho2023logpooling]; in machine-learning terms it is the
product-of-experts normalization of local posteriors [@hinton2002products]:

$$
\begin{aligned}
&\operatorname{log\_linear\_pool}(\{q_n\})\\
&\qquad = \mathrm{softmax}\!\Big(\textstyle\sum_n w_n \log q_n\Big).
\end{aligned}
$$ {#eq:log-linear-pool}

In the floating-point implementation, $q_n$ denotes the admitted posterior
after input validation, lower flooring of masses, and row normalization.
The floor acts before normalization and applies to small positive masses as
well as exact zeros. The pooling formula uses those admitted posteriors;
rescaling sufficiently small raw masses can change them.

For the source bridge, fix one finite shared support $\mathcal S$ with
$q_n(s)>0$ for every agent and state.

Suppose the inputs to Eq. 7's softmax
message-combination term can be represented as posterior log potentials
$m_n(s)=\log q_n(s)+\kappa_n$, where $\kappa_n$ is constant in $s$, and use
declared fixed weights $w_n$ that do not depend on the emerging consensus (the
unweighted case sets each $w_n=1$). Additive constants then cancel under
softmax, giving exactly [@eq:log-linear-pool].

This is a categorical
posterior-log-potential specialization of the source equation's
message-combination term, not a reconstruction of source message construction,
self-exclusion/cavity policy, scheduling, generative factors, or the complete
protocol.

The code alias `friston_belief_share` names this qualified
specialization only.

The second rule is **`robust_aggregate`**, an iteratively-reweighted pool that
discounts each agent by $\exp(-c\,\mathrm{KL}(q_n \,\|\, q))$ against the
emerging consensus $q$. A confidently-wrong (contaminated) agent sits far from
the consensus, can earn a small effective weight and be suppressed in the
declared diagnostic regimes.

This independently motivated rule does not
transfer FedGVI's client theorem to the server side: it is the **heuristic
robustness axis** of [@sec:robustness-axes], distinct from the per-agent
rigorous axis of [@sec:method-losses].

It is also only an analogy to robust federated aggregation methods such as
divergence-weighted gamma-mean aggregation, geometric-median robust aggregation,
or Byzantine-tolerant gradient aggregation [@li2022gammafl;
@pillutla2022robust; @blanchard2017krum]. Those methods motivate the risk
surface, but they do not supply this rule's guarantee.

The defining identity is bit-level: at zero robustness the reweighted pool is the
log-linear pool unchanged.

$$
\operatorname{robust\_aggregate}(0)
\;\equiv\;
\operatorname{log\_linear\_pool}.
$$ {#eq:robust-identity}

This is an exact project-local code identity. Under the stated
posterior-log-potential assumptions, its right-hand side specializes the
message-combination term of Eq. 7; the identity itself neither recovers nor
certifies the complete source protocol [@friston2024federated].

### Protocol map: local updates, broadcast, and server fusion {#sec:method-message-passing}

The visual map in [@fig:message-passing] makes the protocol boundary explicit:
each client updates and broadcasts a categorical posterior; the server chooses
the standard pool, heuristic, or variational route. This is a mechanistic
schematic, not an additional benchmark: client-side FedGVI is source-
conditional, server-heuristic accuracy is conditional on declared contamination,
and the variational route owns objective/descent/raw-weight properties.

#### Visual protocol map (schematic)

![Message-passing schematic for Active Fedference. Source relation: source-inspired original schematic related to Friston et al. (2024), Eq. 7 and Fig. 5; estimand: protocol stages and claim ownership; uncertainty: none. Read the four numbered bands from top to bottom. The y-axis rows order execution from Stage 1 through Stage 4; the x-axis is not quantitative and separates the three parallel fusion routes within Stage 3. Stage 1 begins with private categorical observations, applies the local update, and keeps NLL/KLD recovery and robust client-loss claims at the client. Stage 2 transports categorical posteriors $q_n(s)$ in the unchanged protocol-v1 envelope rather than raw outcomes. Stage 3 routes the same broadcasts through the standard log-linear pool, the server-side `robust_aggregate` heuristic, or objective-backed `variational_aggregate`; distinct boxes, dashed arrows, and claim-owner badges identify each route without relying on color. Stage 4 returns a recipient-specific, cavity-excluded posterior, so an agent does not hear its own message. The standard route combines the client KL/NLL/$\beta=0$ recovery with only the qualified categorical Eq. 7 message-combination specialization, while the heuristic retains recovery-limit status and conditional evidence only; no client guarantee migrates to either server rule. This deterministic formal schematic has no empirical curve, interval, sample size, or replication unit and does not reconstruct the complete source protocol.](../output/figures/message_passing.png){#fig:message-passing width=95% data-slide-manifest="../output/figures/message_passing.slides.json"}

\begin{theorem}[Categorical combination and recovery]\label{thm:belief-sharing-recovery}
For finite $\mathcal S$ and $q_n(s)>0$, take Eq. 7 inputs
$m_n(s)=\log q_n(s)+\kappa_n$, with state-constant $\kappa_n$ and fixed $w_n$.
$\operatorname{softmax}(\sum_n w_n m_n)$ equals
(\ref{eq:log-linear-pool}). At $c=0$, the server returns it by
(\ref{eq:robust-identity}). This identifies categorical combination and
recovery only, not the protocol or $c>0$ behavior.
\end{theorem}

At $c=0$, every reweighting multiplier is $\exp(0)=1$, the iteration is
skipped, and the same pool code path returns.

\begin{corollary}[Closed-form Bayes recovery]\label{cor:closed-form-bayes}
With KL/NLL, the posterior of (\ref{eq:gen-bayes}) is the prior-times-likelihood
product in (\ref{eq:standard-bayes}). Thus
\texttt{generalized\_posterior(KLD, NLL)} reproduces Bayes. Under Theorem
\ref{thm:belief-sharing-recovery}, pooling those posteriors specializes Eq. 7's
categorical combination term, not its complete protocol.
\end{corollary}

$$
q^\ast(s) \;\propto\; \pi_0(s)\,\textstyle\prod_i p(o_i\mid s),
$$ {#eq:standard-bayes}

Pooling gives the project log-linear pool of [@eq:log-linear-pool].

The largest observed discrepancy between `robust_aggregate(robustness=0)` and
`log_linear_pool` is {{RECOVERY_AGGREGATE_MAXDIFF}} — bit-identical, since the
zero-robustness branch runs the same code path — and between
`generalized_posterior(KLD, NLL)` and the analytic Bayes posterior is
{{RECOVERY_POSTERIOR_MAXDIFF}}, exact to machine precision (about one ULP);
both are reported in [@sec:results-recovery], so [@eq:robust-identity] and
[@eq:standard-bayes] are verified identities rather than approximations.

The honesty contract binds at exactly this point. The recovery theorem and its
corollary cover only the recovery identity and the per-agent rigorous axis of
[@sec:method-losses]; no statement *about `robust_aggregate`* transfers a
bounded-influence guarantee to that divergence-reweighting, whose positive
property is the $\texttt{robustness}=0$ limit of [@eq:robust-identity].

A scoped
no-go rejects a declared separable objective class without supplying a broader
objective certificate. The
per-agent influence weights that the heuristic produces under contamination are
*illustrated*, not guaranteed, in [@fig:robust-weights]. The separate
[@fig:bnn-robustness] is an exploratory generalized-Bayes point-estimate
logistic-regression baseline comparing joint NLL/L2 and RCCE/L2
configurations. Because both loss and shrinkage differ, it does not isolate an
RCCE-only effect or test the genuine per-client FedGVI property, which remains
source-conditional in [@sec:method-losses].

The next subsection closes this exact gap
on the server side with a *different*, objective-backed aggregator.

### Variational aggregation with objective-backed weight control {#sec:method-variational}

The related server construction becomes a genuinely variational rule by
replacing the heuristic's reverse-KL weight update with a forward
cross-entropy update. For $c>0$ and $\lambda>0$, treat the consensus $q$ and a
vector of effective weights $a = (a_n)$ as joint variational parameters and
define the **aggregation free energy**

$$
\begin{aligned}
F_\lambda(q, a)
&= \sum_n a_n\,\mathrm{CE}(q, q_n) - \lambda H(q)\\
&\quad + \tfrac{1}{c}\,\mathrm{KL_{gen}}(a \,\|\, w).
\end{aligned}
$$ {#eq:agg-free-energy}

where $\mathrm{CE}(q, q_n) = -\sum_i q_i \log q_{n,i}$ is the cross-entropy of the
consensus relative to agent $n$, $H(q)$ is the consensus entropy, $c$ is the
robustness, and $\lambda>0$ is the `entropy_weight` coefficient
(default $\lambda={{TEMPERED_ENTROPY_WEIGHT_DEFAULT}}$);
$\mathrm{KL_{gen}}(a \,\|\, w) = \sum_n [a_n \log(a_n/w_n) - a_n + w_n]$
is the generalized KL between the effective and base weights.

Each block of $F_\lambda$
has a closed-form minimizer, so alternating

$$
\begin{aligned}
q
&\leftarrow \mathrm{softmax}\!\Big(
  \tfrac{1}{\lambda}\textstyle\sum_n a_n \log q_n\Big),\\
a_n
&\leftarrow w_n\,\exp\!\big(-c\,\mathrm{CE}(q, q_n)\big).
\end{aligned}
$$ {#eq:agg-updates}

is exact block-coordinate descent on [@eq:agg-free-energy] (`variational_aggregate`)
for $c>0$ and $\lambda>0$.

The implementation defines the
$\lambda\downarrow0$ endpoint separately as a deterministic tied-argmax rule;
$\lambda=0$ is not substituted into the objective or its $q$-update.

This substitution changes both orientation and scale:
$\mathrm{CE}(q,q_n)=\mathrm{KL}(q\|q_n)+H(q)$, and its common $H(q)$ term scales
all raw weights, which changes the entropy of the subsequent unnormalized
weighted log pool.

The paired $q$- and $a$-updates in [@eq:agg-updates] are exact
block minimizers of the stated objective; that fact does not derive the
reverse-KL heuristic.

Because $F$ is biconvex, we run the descent
**multi-start** (pool, uniform, and arithmetic-mean seeds, lowest observed $F$
among converged starts; otherwise the lowest unfinished trace is returned with
`converged=False`) so a near-one-hot adversary is not left at the product-of-experts seed in
the tested contamination regimes — the detail that supports the effective-weight
diagnostic
([@sec:supp-variational]).

The full derivation, the formal statement
(block descent, $c\to0$ recovery, and the raw effective-weight bound), and the numerical witnesses
are in [@sec:supp-variational]; the empirical descent and influence collapse are
shown in [@fig:aggregation-descent] and [@fig:bounded-influence] and reported in
[@sec:results-variational].

The authoritative notation supplement records the
complete objective contract in [@eq:notation-variational-objective].

This upgrades the server side from an untracked heuristic to a derived generalized-Bayes
aggregation with an explicit redescending raw-weight property: a single
confidently-wrong agent earns raw weight $a_n = w_n\exp(-c\,\mathrm{CE}(q,q_n)) \le w_n$
that vanishes as it diverges, whereas the naive pool grants every agent the fixed
weight $w_n$ however wrong it is.

The trade is conservatism — the $-H(q)$ term
makes the stationary point a maximum-entropy-biased consensus consistent with the weighted
cross-entropies, so `variational_aggregate` is deliberately flatter than the
product-of-experts and does *not* maximize peak point-accuracy. The two
server-side rules therefore play complementary, never-conflated roles, both
reported: the sharp `robust_aggregate` heuristic for accuracy under contamination
([@sec:results-verdict]) and the conservative `variational_aggregate` for a
server-side objective with stated weight control
([@sec:results-variational]).

A temperature parameter $\lambda>0$ (controlled
by `entropy_weight`, default
$\lambda = {{TEMPERED_ENTROPY_WEIGHT_DEFAULT}}$) generalizes the objective to
$F_\lambda$; lower $\lambda$ sharpens the variational $q$-block toward a
maximizing state for its current weighted log pool.

The tempered family
([@sec:supp-tempered]; objective [@eq:tempered-family]) recovers the
full-entropy variational aggregator at
$\lambda = {{TEMPERED_ENTROPY_WEIGHT_DEFAULT}}$ and has a separately
implemented deterministic tied-argmax endpoint as $\lambda\downarrow0$;
neither endpoint is guaranteed accurate and neither is an algebraic recovery of
`robust_aggregate`.

The effective-weight $a$-update is unchanged for
every $\lambda>0$, so the raw-weight bound holds over the objective-defined
family.
