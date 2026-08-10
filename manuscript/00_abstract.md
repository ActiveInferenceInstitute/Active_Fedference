# Abstract {#sec:abstract}

Multi-agent active inference gives a natural account of belief sharing: agents
hold local posteriors over a shared latent state, communicate those beliefs, and
pool them into a colony-level consensus. The same mechanism is fragile when a
member is miscalibrated, corrupted, or strategically wrong. Because the standard
pool multiplies the reports together, a single confident-but-wrong broadcast that
puts near-zero mass on the true state can pull the whole consensus off it,
outweighing many honest members. The colony therefore needs a way to preserve the
useful structure of belief sharing while limiting the influence of contaminated
beliefs.

This paper presents Active Fedference, a discrete-categorical framework that
connects robust federated generalized variational inference with active
inference belief sharing. The main bridge is structural: standard belief
sharing appears as the non-robust corner of a broader generalized-Bayes family,
while robust losses, conservative server fusion, and explicit aggregation
diagnostics describe how the system moves away from that corner under declared
contamination mechanisms. The result is not a replacement for belief sharing,
but a containment result: ordinary belief sharing is recovered when robustness
is turned off. Bounded-loss theory applies on the client axis, while the
variational-server axis supplies an objective-backed redescending weight update.

The manuscript separates three robustness axes that are often conflated. First,
client-side generalized-Bayes updates change how each agent absorbs evidence;
this is the rigorous axis, carrying FedGVI's bounded-influence result only under
the source theorem's loss, model, and contamination assumptions. Second, a
sharp server-side reweighting heuristic suppresses beliefs that pull away from
the emerging consensus, while carrying only its recovery-limit guarantee — no
proven objective and no bounded-influence bound. Third, a variational
aggregation rule supplies a more conservative objective-backed server
alternative, with a raw effective-weight bound but not an estimator-level
bounded-influence proof for the normalized consensus. Keeping these axes separate
lets the paper state exactly which claims are proven, which are empirical, and
which remain engineering extensions.

The study suite then exercises the framework as an end-to-end research system:
recovery checks anchor the standard-Bayes limit, belief-sharing studies verify
the communication baseline, contamination experiments test robust consensus,
and extension studies probe moving agents, hierarchical latent structure,
sensitivity to acuity and colony size, parameter recovery, and single-host
socket-backed federation traces. All reported quantities are generated from deterministic
analysis artifacts and injected into the manuscript by token, so the paper,
figures, release package, and validation reports remain tied to the same
execution record.

The open-source repository is ActiveInferenceInstitute/Active_Fedference.
The production Zenodo release DOI is [{{PUBLICATION_DOI}}]({{PUBLICATION_DOI_URL}}),
and the repository and deposited PDF point to each other through this DOI and
the repository URL.

**Keywords:** {{CONFIG_KEYWORDS}}

The complete system schematic is shown in [@fig:graphical-abstract].

\newpage
![Graphical abstract. Source relation: original project schematic; estimand: component relationships and recovery boundaries; uncertainty: none. **Recovery ribbon:** the zero-robustness identity anchors the construction at the standard log-linear belief-sharing pool. **Network panel:** the federated diagram shows {{SYSTEM_OVERVIEW_N_AGENTS}} agents ({{SYSTEM_OVERVIEW_N_HONEST}} honest, {{SYSTEM_OVERVIEW_N_ADVERSARIAL}} adversarial) transmitting categorical beliefs to a central server. x-axis is agent position in the ring layout (left to right); y-axis/rows: each per-agent mini-bar glyph indexes posterior probability mass over hidden states. **Consensus panel:** deterministic outcome cards under {{SYSTEM_OVERVIEW_CONTAMINATION_PCT}}% adversarial contamination compare the naive pool with canonical `robust_aggregate` heuristic reweighting; the displayed {{SYSTEM_OVERVIEW_NAIVE_ACC_PCT}}% and {{SYSTEM_OVERVIEW_ROBUST_ACC_PCT}}% are computed from the schematic beliefs. **Axis strip:** client-side FedGVI, server-side heuristic, and variational-server claims are shown as separate routes with non-transferable guarantees. This deterministic formal/mechanistic schematic has no CI, error band, or significance marker; it does not assign the variational server's objective-backed property to `robust_aggregate`.](../output/figures/graphical_abstract.png){#fig:graphical-abstract width=100%}
