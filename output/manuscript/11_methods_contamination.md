## Contamination models: declared failure modes for belief fusion {#sec:methods-contamination}

Robust belief fusion only earns its keep when some agents are wrong. The
active-inference community has built ensembles that coordinate by sharing beliefs
and observations [@friston2024federated; @heins2023collective;
@albarracin2022epistemic; @kaufmann2021collective], but it has assumed those
beliefs are trustworthy in the cited modeled protocols: fusion is treated as
exact-Bayes pooling of well-calibrated reports.

The robust-Bayes and
federated-learning literatures
[@mcmahan2017communication; @ashman2022partitioned; @mildner2025fedgvi] have, in
turn, studied robustness to corrupted clients under their declared settings, but outside the
generative-model-bearing POMDP setting.

This section defines the corruption
process that lets us test fusion robustness inside the active-inference colony —
the experimental complement of the robust aggregation rule of
[@sec:method-aggregation].

## Corruption process for adversarial belief broadcasts {#sec:methods-corruption}

In the sentinel world a healthy sentinel reports a well-calibrated categorical
over the creature's location; a contaminated one reports something corrupted.
`contamination.contaminate` manufactures the corrupted reports. Every corruption
is a convex mixture of the agent's belief $b$ with a corruption target $t$,
governed by a single rate $r \in [0, 1]$,

$$
\tilde b \;=\; (1 - r)\, b \;+\; r\, t,
$$ {#eq:contamination-mix}

so the experiments sweep exactly one knob. The convex form of
[@eq:contamination-mix] gives a clean limit and is the anchor of the suite
(ISC-26): at $r = 0$ every corruption kind returns the input belief unchanged, so
contamination is a strict, continuous departure from the uncorrupted Friston
belief-share — never a discontinuity.

This section defines the three core
corruption targets $t$, each capturing a distinct failure of a federated agent.
Geometrically the three are three landmarks of the probability simplex — a wrong
vertex (`confident_wrong`), the flat centroid (`uniform`), and a random interior
point (`label_noise`) — so the mixture of [@eq:contamination-mix] drags an honest
belief toward a qualitatively different destination in each case.

Two further
mechanisms (`byzantine` and `drift`) extend the same convex-mix contract and are
introduced in the extended-methods supplement ([@sec:supp-contamination]).

**`confident_wrong` — the adversarial sentinel.** This is the lookout that points
to one wrong cell and insists on it with total certainty. The target is a one-hot
spike on a wrong state, $t = \mathrm{onehot}(s_{\text{wrong}})$, so $\tilde b$ is
mixed toward a confident, mistaken delta.

Callers choose $s_{\text{wrong}}$ explicitly;
the verdict sweep of [@sec:methods-experimental-design] fixes it once per colony
as the state diametrically opposite the true state on the location grid, held
constant across the entire rate sweep, rather than deriving it from the agent's
current belief.

At $r = 1$ this is a pure delta on the wrong cell. This is the
saboteur that is *sure* and *mistaken*: exactly the agent that robust
aggregation must reject.

**`label_noise` — the miscalibrated sentinel.** This is the lookout with a
scrambled sensor: it is not lying toward any particular cell, only diluting every
honest report with the same fixed sprinkle of noise. The target is a fixed noisy
categorical drawn once from a $\mathrm{Dirichlet}(1)$ (a random but valid pmf),
modeling a sentinel whose report is partly random rather than adversarial.

Because the noisy target is drawn once and then held fixed across the rate sweep, the
corruption has no direction to exploit and no single cell to veto — the robust
pool meets diffuse degradation, not a targeted attack.

**`uniform` — the apathetic sentinel.** This is the lookout that shrugs: it has
lost track of the creature and calls every cell equally likely. The target is the
maximum-entropy uniform pmf $t = (1/n_s)\mathbf{1}$, modeling a saturated sentinel
that has lost all information. At $r = 1$ it reports uniform, contributing no
evidence to the pool rather than actively pulling it toward a wrong cell.

All three share the contract of [@eq:contamination-mix] and require an explicit
seeded generator — `label_noise` uses it to draw the noisy target — so every
contaminated report is reproducible. The grid of rates the sweep uses,
$\{0, 0.225, 0.45, 0.675, 0.9\}$, deliberately stops below the pure-veto limit $r = 1$,
where a fully-confident wrong delta forces *every* pooling rule's accuracy to
zero and the robust-versus-naive contrast vanishes.

## How contamination meets the three robustness axes {#sec:methods-contamination-axes}

The authoritative guarantee taxonomy is [@sec:robustness-axes]; this section
only records where the declared corruption enters each route.

### Client bounded-loss route {#sec:methods-contamination-client}

A corrupted local observation enters the per-agent generalized-Bayes update of
[@sec:method-losses]. FedGVI's bounded-influence result applies only under the
source theorem's matching loss, divergence, model, and regularity assumptions
[@mildner2025fedgvi].

The logistic-regression comparison in [@fig:bnn-robustness] is a narrower exploratory
proxy: it compares RCCE and NLL point-estimate gradients under two L2
coefficients. Its legacy `AR` argument selects the larger coefficient; it does
not evaluate Alpha-Rényi divergence or inherit the source theorem from the
observed curve.

### Heuristic server route {#sec:methods-contamination-heuristic-server}

A corrupted posterior broadcast enters `robust_aggregate` at fusion.
Divergence reweighting can assign a smaller effective weight to a report far
from the current consensus, as the finite diagnostics
[@fig:robustness-sweep; @fig:robust-weights] illustrate.

The rule's positive formal property is the exact $c=0$ recovery limit
[@eq:robust-identity], not a client-loss bound, Byzantine guarantee, or
universal robustness theorem.

### Objective-backed variational server route {#sec:methods-contamination-variational-server}

The same corrupted posterior can instead enter `variational_aggregate`, whose
declared free energy, block-descent property, and effective-weight bound are
given in [@sec:method-variational]. Those properties belong to that
forward-oriented objective and do not certify `robust_aggregate` or the source
client's bounded-loss theorem.

Contamination is therefore a common stressor, not a license to transfer a
guarantee or empirical result between client, heuristic-server, and
variational-server lanes.
