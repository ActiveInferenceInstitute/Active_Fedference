## Divergences: robust objectives and the KL limit {#sec:method-divergences}

```{=latex}
\ifcsname proposition\endcsname
\else
\newtheorem{proposition}{Proposition}
\fi
```

The generalized-Bayes objective [@eq:gen-bayes] has exactly two tunable
ingredients: the divergence $D$ that regularizes the update toward the prior or
cavity, and the loss $L$ that measures data fidelity. This section develops both
— the divergence family first, the robust loss family in [@sec:method-losses] —
and shows that each carries a limit in which it collapses to its standard
counterpart, KL for the divergence and NLL for the loss. Those client-side
limits establish recovery to standard Bayes. The distinct categorical
server bridge in [@sec:method-aggregation] then identifies a qualified
log-linear message-combination specialization; it does not recover the
complete source belief-sharing protocol.

The regularizing divergence $D$ decides how far a client's updated belief may
move from its cavity, so choosing $D$ is a modeling decision rather than a
numerical detail. The family lives in `divergences.py`. We implement the forward
KL (the standard-Bayes case), the reverse KL (FedGVI's `RKL` client divergence),
the standard $\alpha$-Rényi diagnostic, FedGVI's Alpha-Rényi normalization
(AR), and total variation (a bounded distance in $[0,1]$). The single most
important recovery property is that the robust members recover the KL
divergence in a limit:

$$
D_\alpha(q \,\|\, p) \;\xrightarrow[\alpha\to 1]{}\; \mathrm{KL}(q \,\|\, p).
$$ {#eq:renyi-limit}

\begin{lemma}[KL is the \(\alpha\to1\) limit of the Rényi family]\label{lem:renyi-kl-limit}
For categorical pmfs \(q, p\) on a finite support, the \(\alpha\)-Rényi divergence
\(D_\alpha(q\,\|\,p) = (\alpha-1)^{-1}\log\sum_k q_k^\alpha p_k^{1-\alpha}\) tends
to \(\mathrm{KL}(q\,\|\,p)\) as \(\alpha\to1\), the limit (\ref{eq:renyi-limit}). The
\texttt{divergences.py} implementation switches to the KL closed form inside a small
band around \(\alpha=1\), so on that band the equality is exact rather than merely
asymptotic.
\end{lemma}

KL is the divergence that makes generalized Bayes collapse to standard Bayes.
When local posteriors are then combined by the separately specified categorical
message-combination specialization in [@sec:method-aggregation], the project
recovers its log-linear-pool corner; neither step reconstructs the complete
belief-sharing protocol of Friston et al. [@friston2024federated]. Everything
robust is a controlled departure from that fixed point; Lemma
\ref{lem:renyi-kl-limit} is the formal hinge, and the largest
observed Rényi-versus-KL discrepancy in the recovery band is
0 (reported in [@sec:results-recovery]).

The standard Rényi diagnostic is `renyi_divergence`; FedGVI's `AR` regularizer
is `alpha_renyi_divergence`, equal to the standard form divided by $\alpha$.
For the finite categorical support, `generalized_posterior` solves the named
Alpha-Rényi objective through its scalar normalization condition rather than
using a generic power-softmax shortcut. This distinction keeps the reported
limit and the implemented objective aligned. The `AR` regularizer is not merely
a diagnostic: it is exercised as a client divergence in the categorical FedGVI
baseline of [@sec:results-baseline], where it pairs with the rcce loss of
[@sec:method-losses] to constitute the genuine per-client robustness axis.

## Robust losses: bounded influence at the Bayes corner {#sec:method-losses}

The data-fidelity term of [@eq:gen-bayes] lives in `losses.py`. Standard Bayes
uses the negative log-likelihood, $\mathrm{NLL}(p, o) = -\log p(o)$, which is
*unbounded*: a single contaminated observation with $p(o)\to 0$ dominates the
posterior. This is precisely the fragility the robust-Bayes literature was built
to remove [@basu1998robust; @fujisawa2008robust; @ghosh2015robust;
@zhang2018generalized], extended into robust-divergence variational inference
[@futami2018robustvi], and the property FedGVI imports into federated inference
[@mildner2025fedgvi]. The robust-statistics vocabulary here is the usual
influence-function one [@huber2009robust]: bounded losses reduce the leverage of
extreme observations, while NLL does not. We implement two categorical robust
losses, each of which recovers NLL in a limit.

The density-power ($\beta$) loss [@basu1998robust; @fujisawa2008robust;
@ghosh2015robust; @futami2018robustvi] is recentered so that the scalar limit is
exact:

$$
L_\beta(p, o) \;=\; -\frac{p(o)^\beta - 1}{\beta}
\;+\; \frac{\sum_k p_k^{\,\beta+1} - 1}{\beta+1},
\qquad L_\beta \xrightarrow[\beta\to 0]{} \mathrm{NLL}.
$$ {#eq:beta-loss}

The robust categorical cross-entropy (generalized cross-entropy)
[@zhang2018generalized] is

$$
L_{q_{\rm loss}}(p, o) \;=\;
\frac{1 - p(o)^{q_{\rm loss}}}{q_{\rm loss}},
\qquad L_{q_{\rm loss}} \xrightarrow[q_{\rm loss}\to 0]{} \mathrm{NLL},
$$ {#eq:rcce-loss}

which by l'Hôpital recovers NLL as $q_{\rm loss}\to 0$ and at
$q_{\rm loss}=1$ is the bounded
mean-absolute-error loss $1 - p(o)$, finite exactly where NLL diverges.

\begin{proposition}[\(\beta\)-loss and rcce recover NLL]\label{prop:robust-loss-recovery}
The recentered density-power loss \(L_\beta\) of (\ref{eq:beta-loss}) tends to the
negative log-likelihood as \(\beta\to0\), and the robust categorical
cross-entropy \(L_{q_{\rm loss}}\) of (\ref{eq:rcce-loss}) tends to the negative
log-likelihood as \(q_{\rm loss}\to0\). Both limits are exact in the
implementation; the largest observed
\(\beta\to0\) discrepancy from the NLL closed form is 0
and the largest \(q_{\rm loss}\to0\) discrepancy is 0
(Section~\ref{sec:results-recovery}). At the bounded end the loss stays finite where NLL
diverges, the source of the robustness validated in Section~\ref{sec:results-robustness}.
\end{proposition}

Taking the loss-parameter limits ($\beta\to0$ or
$q_{\rm loss}\to0$) reproduces standard Bayes. Combining those local posteriors
through the qualified categorical specialization in
[@sec:method-aggregation] is a separate server step, not a recovery claim for
the complete belief-sharing protocol of Friston et al.
[-@friston2024federated]. This is the
**per-agent rigorous robustness axis**: it is derived from [@eq:gen-bayes] and
provably limits to Bayes through Proposition \ref{prop:robust-loss-recovery} and
Lemma \ref{lem:renyi-kl-limit}, and it is the axis
that carries FedGVI's bounded-influence guarantee under the cited matching assumptions. The complementary
server-side divergence-reweighting heuristic of [@sec:method-aggregation] is a
distinct device and is never granted this guarantee ([@sec:robustness-axes]).
