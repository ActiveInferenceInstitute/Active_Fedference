## Generative model: categorical states, observations, actions, and hierarchy {#sec:methods-generative-model}

The {{N_STUDIES}} studies — including the contaminated-sentinel robustness
sweep (Study 4) — run on one shared sentinel world (Studies 5–7 use its moving
and hierarchical variants): a discrete sentinel partially-observable Markov
decision process (POMDP) using the categorical world structure illustrated by
Friston et al. [-@friston2024federated], Figures 1 and 4. We adopt the
discrete-state active-inference formulation that the community has standardized
around — the categorical $A/B/C/D_0$ generative model of da Costa et al.
[-@dacosta2020active], the same object the `pymdp` [@heins2022pymdp] and
RxInfer [@bagaev2023rxinfer] toolboxes operate on — and reimplement it in pure
NumPy in `pomdp.py` so the colony, its sensors, and its dynamics are exactly the
ones the analysis executes. A colony of sentinels watches a single hidden
creature whose location is the shared latent factor they federate beliefs about.

The structural map in [@fig:generative-model-schema] follows the same
generative-model vocabulary while making the implementation boundary visible:
Panel A shows what one agent actually sees — one noisy categorical report over
the {{CONFIG_N_LOCATIONS}} possible locations — while Panel B identifies the $A/B/C/D_0$ factors
that turn that report into a local posterior. Temporal depth then describes
state, observation, posterior, and control order; hierarchical depth describes
conditioned priors. It is a formal schematic, not an assertion that every
displayed dependency is simultaneously estimated in every study.

![Formal categorical generative-model schema. Source relation: source-inspired original schematic related to Friston et al. (2024), Figs. 1 and 4; estimand: categorical dependency structure; uncertainty: none. The x-axis is the dependency or role order within each panel; the y-axis positions hidden states, observations, model factors, and optional context levels. Panel A shows a hidden {{CONFIG_N_LOCATIONS}}-cell location and the corresponding categorical likelihood row $A[o,s]$, making the private sensory report explicit. Panel B shows $A$, $B$, $C$, and $D_0$ feeding the local posterior $q(s)$; Panel C shows temporal state, observation, posterior, and action order; Panel D shows optional top-down conditioned priors. The equation ribbon records the implemented state-inference form and the zero-robustness recovery identity. This deterministic formal schematic contains no fitted values, empirical sample, error band, or confidence interval.](../output/figures/generative_model_schema.png){#fig:generative-model-schema width=95%}

## State space: one shared latent factor {#sec:methods-state-space}

The world holds one hidden factor: the creature's location on a square grid of
side $L$, giving $n_s = L^2$ location states. Our sentinel world uses the
$3\times 3$ cardinality illustrated in Friston et al. [-@friston2024federated], Fig. 1, so
$n_s = {{CONFIG_N_LOCATIONS}}$ —
the cardinality `pomdp.N_LOCATIONS` exposes and `experiment_config` carries as
`n_locations`. This single location factor is precisely the latent the colony
gossips about: it is the shared argument of the log-linear pool
([@eq:log-linear-pool]) and of every belief-sharing round ([@eq:belief-round]).
Fixing one hidden factor keeps the recovery limits of [@sec:formalism]
closed-form and exactly testable, rather than approximated.

## Four categorical tensors: likelihood, transitions, preferences, priors {#sec:methods-abcd}

The generative model is the tuple $(A, B, C, D_0)$ in the discrete active-inference
convention: a categorical probability mass function is a non-negative vector
summing to one, and a likelihood matrix is shape $(n_o, n_s)$ whose columns
(indexed by hidden state) are categorical.

**Observation likelihood $A = P(o\mid s)$.** Each sentinel observes the
creature's cell through a noisy sensor. With probability `acuity` the sensor
reports the true cell; the residual mass $1-\text{acuity}$ spreads uniformly
over the other $n_s-1$ cells. With outcome cardinality $n_o = n_s$, the single
location modality is one $(n_s, n_s)$ matrix:

$$
A_{o s} \;=\;
\begin{cases}
\text{acuity}, & o = s,\\
\dfrac{1-\text{acuity}}{n_s - 1}, & o \neq s,
\end{cases}
\qquad \textstyle\sum_{o} A_{o s} = 1 .
$$ {#eq:observation-likelihood}

The acuity in [@eq:observation-likelihood] tunes how peaked the sensor is: high
acuity gives a near-diagonal $A$ that pins the creature; the belief-sharing study
deliberately runs the colony at the low acuity
$\text{acuity} = {{BELIEF_SHARING_ACUITY}}$, where no single sentinel can resolve
the location alone and the colony must pool evidence to do so. When a seeded
generator is supplied, each sentinel's acuity is jittered by a small
non-negative perturbation, so a colony carries slightly heterogeneous
likelihoods while every column remains a proper pmf.

**Transition tensor $B = P(s'\mid s, u)$.** The creature moves on the grid under
three control paths — `still`, `left`, `right` — so $B$ has shape
$(n_s, n_s, n_u)$ with $n_u = {{CONFIG_N_ACTIONS}}$. `still` is the deterministic self-loop;
`left` and `right` decrement and increment the grid column, saturating at the
walls (a wall-adjacent move in the wall's direction is a self-loop). All three
controls act on the column index alone, so the creature's row is preserved and
its motion is confined to the horizontal axis of the grid — a deliberate
one-dimensional control over the two-dimensional location factor. Every slice
$B_{\cdot\,\cdot\,u}$ is column-normalized by construction, so the transition is
a valid categorical for each action.

**Log-preference $C$.** The sentinel prefers to *see* the creature near the den
(the center cell), encoded as a log-preference vector of shape $(n_o, 1)$ with a
positive bump on the center outcome and zero elsewhere. The preferred-outcome
distribution that the expected-free-energy decomposition of [@sec:methods-learning]
uses is $p_C(o) = \mathrm{softmax}(C)[o]$.

**Initial prior $D_0$.** The creature is believed to start at the grid center, so
$D_0$ of shape $(n_s, 1)$ places unit mass on the center cell. $D_0$ enters state
inference as the log-prior of the one-step variational update
([@eq:state-inference] below).

The columns-are-pmfs invariant is not assumed — it is pinned by ISC-15, which
checks that every column of $A$ and of each $B_{\cdot\,\cdot\,u}$ sums to one.

## One-step variational state inference in the grid world {#sec:methods-state-inference}

Given an observation $o$, a sentinel forms a posterior over the creature's
location by a single softmax step (Friston et al. [-@friston2024federated],
Eq. 4): the log-prior plus the additive log-likelihood message, summed over any
conditionally independent modalities $m$,

$$
q(s) \;=\; \mathrm{softmax}\!\Big(\ln D_0(s) \;+\; \textstyle\sum_m \ln A_m[o_m, s]\Big).
$$ {#eq:state-inference}

The message $\ln A_m[o_m, \cdot]$ is the row of $A_m$ that the observed outcome
selects; summing messages over modalities makes each modality an additive
evidence term — the categorical product-of-experts. The companion variational
free energy, the scalar [@eq:state-inference] minimizes, is

$$
F[q] \;=\; \mathbb{E}_q\!\big[\ln q(s) - \ln D_0(s) - \textstyle\sum_m \ln A_m[o_m, s]\big]
\;=\; \mathrm{KL}\big(q \,\|\, D_0\big) \;-\; \mathbb{E}_q\!\big[\textstyle\sum_m \ln A_m[o_m, s]\big],
$$ {#eq:variational-free-energy}

reported in nats. The one-step posterior of [@eq:state-inference] is its unique
minimizer, where $F$ equals the negative log model evidence. Both live in
`belief_updating.infer_states` and `belief_updating.vfe`, and the free energy of
[@eq:variational-free-energy] is the quantity the communicating-versus-incommunicado
colony comparison of [@sec:methods-experimental-design] scores.

This inference step is not a separate mechanism bolted onto the colony: it is the
$L=\mathrm{NLL}$, learning-rate-1 special case of the generalized-Bayes posterior
[@eq:gen-bayes], reusing the same locked softmax. That client identity recovers
the stated categorical Bayes substrate at its trusting limits. The separate
server identity in [@sec:method-aggregation] then yields the project log-linear
pool under its qualified Eq. 7 message-combination bridge; together these do not
recover the complete Friston protocol ([@sec:formalism]).

## Hidden-state to action loop: the POMDP substrate {#sec:methods-pomdp-loop}

The categorical POMDP loop in [@fig:pomdp-loop] separates the common latent-state
substrate from the federation transport. In the sentinel interpretation, an
agent is a location-sensitive observer: the hidden state is one of {{CONFIG_N_LOCATIONS}} cells,
the private outcome is a noisy categorical report of that location, and the
agent sends its posterior over the location rather than the report itself. The
flat belief-sharing studies use the observation, posterior, and communication
subset; the moving-world extension also executes transition and EFE-guided
action paths. The diagram therefore gives readers the active-inference context
without turning a conceptual loop into a claim that every study estimates every
latent or policy quantity.

![Sentinel-world and active-inference loop. Source relation: source-inspired original schematic related to Friston et al. (2024), Figs. 1 and 4; estimand: POMDP message-and-action sequence; uncertainty: none. The x-axis is the POMDP cycle in Panel C from hidden state through observation, posterior, action, and next state; the y-axis separates the shared-world, belief-sharing, and temporal-loop panels. Panel A shows three agents viewing the same {{CONFIG_N_LOCATIONS}}-cell hidden world through private, noisy categorical observations; raw observations remain local. Panel B shows those local posteriors entering a log-linear-pool or robust server and returning as a cavity-excluded consensus. Panel C gives the POMDP cycle from hidden state $s_t$ through observation $o_t$, posterior $q_t(s)$, action $u_t$, and transition to $s_{t+1}$, with $A$, $B$, and $C$ marking the likelihood, transition, and preference factors. The flat studies execute the inference-sharing branch; the moving-world extension also executes transitions and EFE-guided actions. This is a deterministic model schematic, not an uncertainty-bearing empirical result.](../output/figures/pomdp_loop.png){#fig:pomdp-loop width=95%}
