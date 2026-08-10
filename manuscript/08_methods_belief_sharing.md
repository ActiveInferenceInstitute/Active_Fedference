## Belief sharing: the standard aggregation corner {#sec:method-belief-sharing}

`belief_sharing.share_round` lifts the aggregation rule of
[@sec:method-aggregation] to a colony of categorical sentinel agents. Each
agent has a private sensory outcome and a local posterior over the same shared
latent location; it broadcasts the posterior, not its raw observation. Following
the sensory attenuation that the active-inference formulation of belief sharing
imposes [@friston2024federated] — "agents do not hear themselves" — an agent's
heard consensus excludes its own message:

$$
q_{-n} \;=\; \mathrm{normalize}\!\left(q/t_n\right),
$$ {#eq:belief-round}

so the round in [@eq:belief-round] implements the declared categorical
colony-hive-mind mechanism. With the naive fusion rule of
[@eq:log-linear-pool], [@eq:belief-round] is the project's standard
log-linear-pool consensus. Under the explicit shared-support,
posterior-log-potential, and fixed-weight assumptions of
[@sec:method-aggregation], it specializes Eq. 7's message-combination term;
it does not reconstruct the complete source protocol. With the server-side
robust rule it yields a hive-mind that can down-weight a contaminated sentinel —
an effect the contamination sweep of [@sec:results-robustness] measures rather
than assumes, and one that carries no guarantee beyond the recovery limit. The
per-round diagnostics — the post-sharing belief matrix, the global consensus,
and the mean surprise and accuracy against a known ground-truth state — are
returned by `share_round` and drive Studies 1 and 4.

[@fig:message-passing] makes this concrete: three sentinel cards begin with
different private categorical views of the nine-cell world, convert those
views into local posteriors, and send only those posteriors to a fusion route.
The return path is a cavity message, so the consensus heard by agent $n$
excludes the local posterior $q_n$. The figure remains a protocol map rather than a new result; the
empirical belief matrix and free-energy comparison remain the evidence surfaces
in [@fig:belief-heatmap] and [@fig:free-energy].

Because [@eq:belief-round] calls the aggregation rule of [@eq:log-linear-pool] or
its robust generalization, the recovery identity of [@eq:robust-identity]
propagates upward: a colony running `share_round` at zero server robustness is
bit-identical to a colony running the project's standard log-linear-pool round.
Under the qualified categorical bridge, the pool realizes only the source
message-combination specialization; the robust round is a project extension,
not a reconstruction of the active-inference ensemble literature
[@friston2024federated; @heins2023collective]. [@sec:results-belief_sharing]
reports that communicating colonies reach a mean variational free energy of
{{BELIEF_SHARING_MEAN_F_COMMUNICATE}} nats against
{{BELIEF_SHARING_MEAN_F_INCOMMUNICADO}} nats for incommunicado colonies across
{{BELIEF_SHARING_N_SEEDS}} seeds, with the per-agent belief matrix before and
after a round shown in [@fig:belief-heatmap] and the colony comparison in
[@fig:free-energy].

The honesty boundary of [@sec:robustness-axes] carries through the lift unchanged. The
robustness that [@eq:belief-round] inherits when the colony fuses with the
server-side heuristic is the divergence-reweighting device of
[@sec:method-aggregation], whose positive property is the naive-recovery
limit and whose declared separable objective class has a scoped no-go result;
the per-agent FedGVI bounded-influence result enters the colony only
through the rcce/AR client losses of [@sec:method-losses], under the source
theorem's matching assumptions, applied inside each agent's local
generalized-Bayes update. The robustness sweep in
[@sec:results-robustness] and the variational supplement in
[@sec:supp-variational] keep the three axes distinct. The federation transport
([@sec:supp-federation]) realizes this sharing
over queue-backed worker channels; by
Proposition \ref{prop:federation-bit-identity}, the federation
bit-identity result, the consensus is bit-identical to the in-process call, so
the channel adds no precision loss while leaving multi-machine network
transport as future work.
