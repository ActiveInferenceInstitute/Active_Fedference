# Abstract {#sec:abstract}

Multi-agent active inference gives a natural account of belief sharing: agents
hold local posteriors over a shared latent state, communicate those beliefs, and
pool them into a colony-level consensus. The same mechanism is fragile when a
member is miscalibrated, corrupted, or strategically wrong.

Because the standard
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
contamination mechanisms.

The result is not a replacement for belief sharing,
but a containment result: ordinary belief sharing is recovered when robustness
is turned off. Bounded-loss theory applies on the client axis, while the
variational-server axis supplies an objective-backed redescending weight update.

Following the manuscript's authoritative three-axis guarantee map, the paper
keeps source-conditional client updates, a recovery-only heuristic server rule,
and an objective-backed variational server rule as separate claim owners, with
no robustness guarantee transferred between them.

The study suite then exercises the framework as an end-to-end research system:
recovery checks anchor the standard-Bayes limit, belief-sharing studies verify
the communication baseline, contamination experiments test robust consensus,
and extension studies probe moving agents, hierarchical latent structure,
sensitivity to acuity and colony size, parameter recovery, and single-host
socket-backed federation traces.

All reported quantities are generated from deterministic
analysis artifacts and injected into the manuscript by token, so the paper,
figures, release package, and validation reports remain tied to the same
execution record.

The open-source repository is ActiveInferenceInstitute/Active_Fedference.
This development manuscript has no assigned version DOI; published versions bind their reserved DOI, deposited PDF, and repository URL only after the release gates close.

**Keywords:** active inference, federated learning, generalised variational inference, belief sharing, robustness, FedGVI

The complete system schematic is shown in [@fig:graphical-abstract].

\newpage
![Graphical abstract. Source relation: original project formal/mechanistic schematic with configured colony and outcome summaries; estimand: component relationships, recovery boundaries, and the displayed true-state probability mass under two server rules. Read from the federated network through the consensus cards to the claim-owner strip. The x-axis is agent position in the ring or the left-to-right comparison order; y-axis/rows encode each agent's categorical posterior mass and the three distinct robustness lanes. White mini posterior bars with solid inbound arrows identify honest agents, whereas white crosses with dashed inbound arrows identify adversarial broadcasts; direct method labels, percent badges, keylines, and numbered reading order duplicate color. The project-local identity `robust_aggregate(c=0) ≡ log_linear_pool` is stated separately from the categorical specialization of the Eq. 7 message-combination term, which requires shared support, admitted posterior-log potentials, and fixed weights. Under 40% configured contamination, the deterministic cards report 39% versus 50% true-state mass from the beliefs shown. Units are probability mass, agent count, or conceptual route; one configured explanatory colony supplies no replication unit, CI, error band, or significance test. The client theorem, heuristic server evidence, and variational effective-weight property are non-transferable, and the schematic does not reconstruct the complete source protocol or establish universal robustness.](../output/figures/graphical_abstract.png){#fig:graphical-abstract width=100% data-slide-manifest="../output/figures/graphical_abstract.slides.json"}
