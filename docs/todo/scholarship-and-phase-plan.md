# Scholarship-indexed phase plan and evidence ladder

[Back to roadmap](../../TODO.md)

## Status

- Priority class: Major
- State: Open
- Evidence baseline: [`extended-statistical-audit-2026-07-14.md`](../research/extended-statistical-audit-2026-07-14.md)
- Owner surface: `TODO.md`, `docs/research/literature-audit.md`, manuscript future-work sections, phase-specific acceptance gates

**Historical audit note — 2026-08-01:** the then-bounded
`robustness_review_grid` recorded a 64-seed × nested-trial evidence slice with
explicit controls, selection-free payloads, and disjoint component seed ranges.
The current source increases that budget and requires a fresh run before it can
support a source-current claim. Neither record completes or reclassifies any
MAJ phase; sharp server-heuristic
characterization, calibration, portable/external BNN, external data,
source-protocol reconstruction, continuous/hybrid, richer hierarchy,
authenticated federation, and clean-clone release gates remain open.

## Rationale

The remaining roadmap items are not one homogeneous engineering backlog. They
span three different kinds of evidence: mathematical characterization of a
server rule, conditional empirical scaling of a client method, and systems
transport/security. Treating them as a single list invites category errors—for
example, interpreting a secure transport result as Byzantine robustness, or
interpreting a larger neural network as evidence for a theorem about the
categorical server heuristic. This page makes the dependencies, scholarship,
estimands, and negative controls explicit before the next expansion.

The source bridge is [Friston et al. (2024), *Federated inference and belief
sharing*](https://pmc.ncbi.nlm.nih.gov/articles/PMC11139662/). The generalized
Bayesian bridge is [Mildner et al. (2025), *Federated Generalised Variational
Inference*](https://proceedings.mlr.press/v267/mildner25a.html). The aggregation
genealogy is anchored by [Genest and Zidek
(1986)](https://doi.org/10.1214/ss/1177013825) and [Rufo et al.
(2012)](https://doi.org/10.1214/12-BA714). These sources motivate the
questions; they do not certify the repository's extensions.

## Scope

Maintain a phase-indexed roadmap for the completed scoped MAJ-1 result, the
remaining MAJ-2 through MAJ-8 lanes, the MAJ-2A/2B and MAJ-4A/4B execution
splits, and the parked privacy, multimodal, streaming, and language tracks.
Every phase must specify:

1. a source bundle and the assumptions that are imported versus re-proved;
2. one primary estimand and its independent unit;
3. a smallest effect of interest, MCSE target, maximum budget, comparison
   family, minimal falsifiable experiment, and at least one negative control;
4. separate smoke, pilot, and confirmatory profiles;
5. the required report, figure, caption, and provenance artifacts; and
6. a claim boundary that says what a successful or null result establishes.

This is planning and evidence control. It does not promise that any open phase
will produce a positive result.

Current capability: report-level evidence and negative controls exist in
[`extended-statistical-audit-2026-07-14.md`](../research/extended-statistical-audit-2026-07-14.md);
residual scope is selecting and executing the next phase against that baseline.
The present evidence supports conditional claims only — not a universal server
guarantee, a universal attack taxonomy, or a general hierarchy advantage.

## Implementation Notes

### Scholarly source bundles

| Bundle | Verified sources | Design consequence |
| --- | --- | --- |
| Belief sharing and active inference | [Friston et al. (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11139662/); [Friston et al. (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5461873/); [de Vries and Friston (2017)](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2017.00095/full) | Keep $A/B/C/D$, hidden states, observations, and action-context notation explicit; distinguish the conceptual POMDP loop from the executed moving-world path. |
| Generalized Bayes and robust losses | [Bissiri et al. (2016)](https://doi.org/10.1111/rssb.12158); [Mildner et al. (2025)](https://proceedings.mlr.press/v267/mildner25a.html); [Futami et al. (2018)](https://proceedings.mlr.press/v80/futami18a.html) | Re-check the exact loss, prior, cavity, and boundedness assumptions before transferring any client-side result to a new model family or server rule. |
| Opinion pools and aggregation | [Genest and Zidek (1986)](https://doi.org/10.1214/ss/1177013825); [Rufo et al. (2012)](https://doi.org/10.1214/12-BA714); [Pillutla et al. (2022)](https://arxiv.org/abs/1912.13445) | Treat the ordinary log pool as a coherent aggregation corner, not as a contamination-resistant estimator; characterize the heuristic independently. |
| POMDP and message-passing structure | [Kaelbling et al. (1998)](https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf); [Heins et al. (2022)](https://github.com/infer-actively/pymdp); [Bagaev et al. (2023)](https://baggepinnen.github.io/RxInfer.jl/stable/) | Verify state/action semantics and inference schedules before claiming a deeper hierarchy or a richer task family. |
| Federated systems and privacy | [McMahan et al. (2017)](https://proceedings.mlr.press/v54/mcmahan17a.html); [Bonawitz et al. (2017)](https://arxiv.org/abs/1611.04482); [Abadi et al. (2016)](https://arxiv.org/abs/1607.00133) | Transport integrity, secure aggregation, differential privacy, and statistical robustness are separate axes with separate threat models and tests. |
| Byzantine and robust aggregation | [Blanchard et al. (2017)](https://papers.nips.cc/paper_files/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html); [Pillutla et al. (2022)](https://arxiv.org/abs/1912.13445) | A server heuristic must be tested against adaptive and non-i.i.d. attacks; one declared contamination gallery cannot support universal Byzantine language. |
| Bayesian neural-network scaling | [Blundell et al. (2015)](https://proceedings.mlr.press/v37/blundell15.html); [Izmailov et al. (2021)](https://proceedings.mlr.press/v139/izmailov21a.html); [Mildner et al. (2025)](https://proceedings.mlr.press/v267/mildner25a.html) | A point-mass MLP is a complement, not a mean-field posterior or a faithful source-scale FedGVI reproduction; architecture and objective parity are preregistered. |
| Protocol source and parity | [FedGVI public implementation](https://github.com/Terje-M/FedGVI), pinned in `research_registry.py`; [Friston et al. (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11139662/) | Describe the FedGVI server as site factors, cavities, and factor replacement—not moment matching. Use exact-replication language only when every required parity row matches. |
| Emerging design inputs | [Closed-form GVI preprint](https://arxiv.org/abs/2606.25492); [hierarchical active-inference preprint](https://arxiv.org/abs/2604.15679) | These 2026 preprints guide theorem and task design; they are not proofs or evidence for this repository. |
| Simulation precision and reproducibility | [Morris et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6492164/); [Koehler et al. (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3337209/); [Loy and Korobova (2021)](https://arxiv.org/abs/2106.06568); [Wilkinson et al. (2016)](https://doi.org/10.1038/sdata.2016.18) | Choose seed budgets from pilot variance and target MCSE; preserve clustering; publish machine-readable provenance and distinguish conditional simulation precision from external validity. |

### Ordered phases

| Wave / target | Primary question | Prerequisite | Primary estimand / unit | Required falsifier or boundary |
| --- | --- | --- | --- | --- |
| Foundation / v0.1 | Can a clean installation enumerate and execute registered research without mutating the reviewer snapshot? | Safe headroom and release authority | Exact schema/receipt/artifact verification; unit is one isolated repository build | Stale source chain, untracked load-bearing file, dirty output overwrite, failed wheel/sdist install, or clone divergence blocks release |
| Server theory / v0.2 (MAJ-1, scoped closure) | Does the declared separable objective class produce `robust_aggregate`'s raw log-pool block? | Interior-simplex domain and positive raw weights | Completed: numbered scoped no-go proposition, executable raw and normalized witnesses, typed report metadata | An in-class objective with the stated all-interior q-block would refute the proposition; broader coupled or fixed-point-only constructions remain outside its scope |
| Calibration / v0.2 (MAJ-8) | Which robustness settings maximize calibration-world proper score without evaluation leakage? | Versioned configuration and receipts | Mean held-out log score per independent calibration world | Episode identifier/content overlap, changed frozen config, or reversed confirmatory effect blocks the intended claim |
| Portable FedGVI / v0.3 (MAJ-2A) | Does the source-protocol-matched mean-field BNN show a proper-score effect under a locked M4 budget? | Source-revision parity matrix and pilot | Paired held-out log-score difference per seeded end-to-end run | Null/reversed proper-score result remains visible and prohibits the uncertainty-robustness claim |
| Source-scale FedGVI / external (MAJ-2B) | Does the exact CUDA configuration reproduce the source-scale result? | MAJ-2A plus external CUDA | Same source-defined estimand and seed unit | Declarative configuration is never presented as executed evidence |
| External data / v0.3 (MAJ-6) | Does the effect replicate across the three pinned UCI datasets? | Hash/license/schema/split contracts and calibration | Per-dataset paired held-out log-score difference; dataset is higher-level unit | Source, leakage, corrupted-cache, permutation, null, and reversal controls |
| Friston reconstruction / v0.4 (MAJ-7) | Can Eq. 2 and Figures 5, 7, and 9 be reconstructed with complete parameter and estimand parity? | Source-protocol extraction | Source-defined quantity in native units and source-defined unit | Any unresolved parity row forces “paper-constrained reconstruction” |
| Hybrid tracking / v0.4 (MAJ-3) | Does robust hybrid fusion improve held-out prediction under explicit context/outlier controls? | Categorical, Gaussian, zero-robustness, and covariance recovery gates | Posterior-predictive log score per seeded tracking world | Failed recovery gate or null/reversed locked effect blocks broad language |
| Hierarchy tasks / v0.4 (MAJ-5) | Does learned hierarchy improve fixed-horizon success across Four Rooms and Key-Door? | Hybrid recovery gates | Episode success; task is higher-level unit, seeds nested | Flat, oracle, shuffled, and non-gating controls; one-task effects stay task-specific |
| Local emulator / v1.0 (MAJ-4A) | Does an mTLS-default Docker federation preserve consensus and fail closed under declared faults? | Stable versioned envelope | Consensus identity and fault disposition per container round | Drop, duplicate, delay, replay, tamper, timeout, restart, out-of-order, or wrong-key mismatch |
| Physical hosts / external v1.x (MAJ-4B) | Do physically distinct hosts reproduce the emulator contract? | MAJ-4A and external hosts | One distinct-host deployment run | Local containers cannot satisfy this unit |

### Release evidence contract

- **v0.1:** two isolated fresh-clone passes, wheel and source-distribution
  installation, installed CLI smoke, confidentiality/attribution/license review,
  and author approval.
- **Each v0.x:** one new source-bound evidence pack. A negative scientific
  result may ship; malformed reports, failed controls, stale receipts, or
  unresolved source discrepancies may not.
- **v1.0:** three independently verifiable evidence packs, stable public
  schemas, both reader documentation paths, MAJ-4A, clean-clone reproduction,
  and external release approval.
- **v1.x:** prioritize streaming/nonstationary belief sharing, then multimodal
  missingness. Privacy or secure-aggregation claims require a threat model and
  leakage-measurement protocol first.

### Parked tracks

These tracks are deliberately below the evidence-gated release waves. They are included so
future contributors have an evidence contract before adding attractive but
claim-expanding capabilities.

| Track | Source bridge and prerequisite | Primary estimand / independent unit | Minimum falsifier | Boundary after a positive result |
| --- | --- | --- | --- | --- |
| Multimodal beliefs | Friston et al. (2024) and de Vries & Friston (2017); requires an explicit product/factor structure for modalities and a missing-modality policy | Cross-modal consensus loss and calibration by seed × modality-missingness condition; independent unit is seed/world | Permuted modality alignment, single-modality ablation, and adversarially corrupted modality must fail to improve the declared target | Does not establish human-like multimodal cognition or robustness to every sensor-fusion failure |
| Privacy-preserving federation | Bonawitz et al. (2017) for secure aggregation and Abadi et al. (2016) for differential privacy; requires a threat model and a privacy accountant | Utility/privacy frontier at a declared $(\varepsilon,\delta)$ and communication budget; independent unit is seeded deployment/round | Wrong-key, colluding-server, dropout, reconstruction, and privacy-budget accounting controls must fail closed | Secure aggregation or DP does not imply statistical robustness, Byzantine tolerance, or confidentiality against every side channel |
| Online/streaming inference | Friston et al. (2017) plus the sequential-update semantics of the categorical model; requires a time-indexed estimand and reset policy | Prequential loss, calibration drift, and recovery latency per stream; independent unit is stream/world, not observation | Concept drift, delayed/out-of-order messages, reset-vs-no-reset, and shuffled-time controls | A streaming result does not validate stationary episodic claims or long-horizon deployment safety |
| Language summaries | Friston et al. (2024) for mechanical language emergence; requires a separately specified generation/evaluation task and an explicit non-LLM baseline | Semantic fidelity and calibration of summaries per held-out world; independent unit is world/task, with a human or blinded rubric only if claimed | State-label permutation, template baseline, hallucination probe, and held-out vocabulary/task | Better summaries do not establish language emergence, grounded communication, or an improvement in inference accuracy |

For all four tracks, implementation is blocked until the source bundle is
expanded in `docs/research/literature-audit.md`, the privacy/threat or
evaluation protocol is written, and the manuscript claim ledger names the
new estimand. A diagram or demo alone is not evidence that the track has
landed.

### Statistical planning rules for future phases

- Pilot the variance of the primary estimand, then select a target MCSE using
  $n \approx (s / \mathrm{MCSE}_{target})^2$; do not select a large $n$ merely
  because it makes a p-value small.
- Declare whether the independent unit is a seed, dataset, deployment round,
  or participant. Clients, observations, and repeated episodes remain nested
  unless the design explicitly makes them independent.
- Use matched comparisons when the data-generating seed/world is shared, and
  use cluster-aware bootstrap or hierarchical summaries when observations are
  nested. Declare the multiple-testing family before inspecting outcomes.
- Report point estimate, interval, MCSE, sample/unit count, and the conditional
  data-generating mechanism together. A confidence interval is not a guarantee,
  and a power calculation is not evidence that the observed effect is real.
- Pilot data select budgets and calibration settings only. Smoke and pilot rows
  never enter confirmatory intervals, headline values, or manuscript claims.
- Every confirmatory registry entry records its source bundle, primary estimand,
  independent unit, smallest effect of interest, MCSE target, maximum budget,
  comparison family, falsifier, and no-claim outcome before execution.

### Required visual and documentation additions per phase

Every phase that lands code must add a source-owned report, a generator or
table with an explicit status (`formal`, `mechanistic`, `deterministic`, or
`data-bearing`), a caption naming axes/estimand/unit/uncertainty, and an alt
text that does not imply execution outside the phase. Registry entries must
resolve to the actual generator default and both PNG/PDF archival surfaces when
applicable. The RedTeam record must include a negative control and the final
render-surface result.

## Acceptance Criteria

- Primary estimand: the completed MAJ-1 result records theorem-or-counterexample
  status for its declared objective class; each remaining phase uses the
  estimand column of the ordered-phase table above.
- Independent replication unit: the MAJ-1 construction is an executable
  derivation reproducible from source; empirical phases use the declared unit
  in the phase table (seed, deployment round, or dataset), with clients,
  observations, and episodes treated as nested.
- Explicit falsifier: an in-class objective with the stated all-interior
  q-block would refute the MAJ-1 no-go; each remaining phase carries the
  falsifier column of its table row and must run it before accepting a positive
  result.
- Required artifacts and tests: any phase that lands code adds a source-owned
  report, a status-tagged generator or table, registry-resolved figures, and
  tests that extend the existing identity, docs-contract, caption, x-ref, and
  token-provenance gates; the probes below must pass in the same change.
- Documentation changes: the claim ledger, `docs/research/literature-audit.md`,
  the figure registry, caption gate, and this page's phase tables are updated in
  the same change that opens or closes a phase.
- This page is linked from `TODO.md` and `docs/todo/README.md`.
- The literature audit records each source bundle and the claim boundary it
  imposes; no discovery-only source is treated as a primary citation.
- Each MAJ phase has one primary estimand, one declared independent unit, one
  falsifier, and one explicit no-claim boundary before implementation begins.
- Any phase that changes the manuscript updates the claim ledger, caption gate,
  figure registry, artifact documentation, and RedTeam iteration in the same
  change.
- Phase completion requires a fresh source test run and a fresh template render;
  a green source suite alone cannot close a visual or semantic finding.

## Verification Probes

- `uv run pytest tests/test_docs_contract.py tests/test_caption_completeness.py tests/test_xref_integrity.py -q`
- `uv run ruff check src tests`
- `uv run python scripts/validate_all.py full`
- Inspect `output/reports/validation_report.json`, `output/reports/artifact_manifest.json`, and `output/data/stage_timings.json`.
- For each phase, run the documented negative control before accepting a positive result.

## Claim-Boundary Constraints

This roadmap must not be summarized as evidence that any open phase is already
solved. The current paper remains categorical and simulation-conditional. A
successful BNN, transport, continuous, hierarchical, or external-benchmark
phase would extend only the tested object and threat/model/data regime.
Prohibited claims (no-claim boundary):

- the `robust_aggregate` recovery identity does not become a bounded-influence
  theorem by adding a larger neural model or a secure socket;
- authenticated transport does not provide differential privacy or Byzantine
  robustness;
- a source-paper analogue does not prove a theorem for this repository;
- a dataset benchmark does not establish deployment validity or universality;
- a figure that depicts action selection does not establish that the action
  loop was executed in every belief-sharing study.

## Dependencies

The current identity tests, public configuration, registry, receipt verifier,
claim ledger, figure registry, token provenance checks, and template-rendering
pipeline are prerequisites for every phase. The MAJ-1 scoped no-go controls the
server-rule vocabulary; MAJ-8 controls tuning/evaluation separation. MAJ-2A controls the
portable/source-scale label, while MAJ-2B remains external. MAJ-3 recovery gates
block the large MAJ-5 task family. MAJ-4A may proceed after the envelope schema
stabilizes; only MAJ-4B can support a physical multi-host statement. MAJ-6 has
pinned source metadata but remains open until its pilot, confirmatory inference,
manuscript artifacts, and release receipt exist.
