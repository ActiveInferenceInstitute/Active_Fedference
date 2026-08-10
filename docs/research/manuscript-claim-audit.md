# Manuscript-wide claim audit

Updated 2026-08-10. This ledger records the evidence status of the manuscript's
load-bearing claims after the Research and RedTeam passes. It is intentionally
claim-level rather than sentence-level: local wording should remain consistent
with the row that governs it. The v0.1.0 GitHub/Zenodo publication completed on
2026-08-10 is a release-state fact only; it does not promote any scientific
claim or close the open research lanes below.

## Evidence statuses

- **Formal / executable:** an algebraic statement is present in the manuscript
  and the implementation has a corresponding invariant or negative-control test.
- **Conditional empirical:** supported by the declared seeded simulation and
  its estimand, not generalized beyond that data-generating mechanism.
- **Source-conditional:** inherited from a cited result only under that source's
  assumptions; not automatically a theorem of this repository.
- **Scoped implementation fact:** true of the executed code path, but not a
  universal property of the method family.
- **Open:** a motivating question or limitation that remains unestablished.

## Historical source-bound ledger — iteration 36

The historical review separates source equations and protocols, executable
identities, implementation analogues, conditional simulation evidence, and
unresolved research claims. A source citation does not turn a repository
extension into a source theorem, and a green finite simulation does not close a
major research phase.

| Evidence lane | Source or contract | Executable/project surface | Permitted claim and red-team boundary |
|---|---|---|---|
| Source equation and protocol | [Friston et al. (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11139662/) | `log_linear_pool`, `belief_sharing`, and source-analogue figures | The categorical log pool is a source-grounded bridge and tested specialization; reduced studies are not exact source-figure or source-protocol replications. |
| Source theorem and protocol | [Mildner et al. (2025)](https://proceedings.mlr.press/v267/mildner25a.html) and the pinned FedGVI implementation | `generalized_bayes`, site/cavity utilities, factor replacement, and BNN protocol primitives | FedGVI claims transfer only under the cited loss, divergence, regularity, and protocol assumptions; they do not certify `robust_aggregate` or this categorical active-inference architecture. |
| Generalized-Bayes formalism | [Bissiri et al. (2016)](https://doi.org/10.1111/rssb.12158) | `generalized_posterior(tau=...)`, NLL/KLD recovery tests, and the notation supplement | The finite categorical formula is an implementation analogue with executable limits, not a new theorem about every loss, divergence, or model family. |
| Opinion-pooling genealogy | [Genest and Zidek (1986)](https://doi.org/10.1214/ss/1177013825) | Log-linear pooling and comparator controls | The ordinary pool has a documented aggregation interpretation; no contamination, Byzantine, causal, or deployment guarantee is imported from the pooling literature. |
| Server heuristic | Repository contract and `src/fedference/aggregation.py` | `robust_aggregate` recovery identity, finite influence diagnostics, and scoped raw-log-pool/normalized-weight no-go witnesses | This is a server-side heuristic with conditional behavior. A formal proposition excludes the declared separable objective class; no client-side FedGVI theorem, universal robustness, unique winner, or universal no-objective claim is made. |
| Objective-backed server rule | `27_supplement_aggregation_objective.md` | `variational_aggregate`, raw/normalized effective weights, and descent/bound tests | The raw bound and objective descent are executable properties of the declared finite implementation; they do not establish normalized-estimator B-robustness or truth recovery. |
| Conditional simulation evidence | Declared configuration, seed schedule, and typed report schemas | Sweep, gallery/onset, conditional-world, and `robustness_review_grid` reports | Results are conditional on finite mechanisms and estimands. Seeds are the inferential unit where declared; trials remain nested. Controls and null/reversed cells are retained. |
| Unresolved research claim | Open phase pages in `docs/todo/` | Leakage-free calibration, portable/external BNN, external data, Friston reconstruction, continuous/hybrid, richer hierarchy, authenticated federation, independent reproducibility, and future release authority | A bounded review grid or green source test does not close a major research phase, an independent-verification gate, or author signoff; the published v0.1.0 DOI is not scientific evidence. |

## Current source contract — iteration 45

The current candidate raises the source-bound review-grid design to 160 seeds
with 24 nested trials per seed/cell and requires maximum directional-cell MCSE
at or below 0.01. It also binds publication analysis to the canonical declared
configuration and a pre-run input snapshot; a provisional hydration cannot
record a publication stage, and bundle operations require current generated
metadata plus live PDF/slide/web validation. These additions strengthen
integrity, provenance, and precision. They do not enlarge the finite,
conditional empirical claims or close any external-authority, replication, or
open-science gate.

## Current source contract — iteration 46

The publication fallback in `src/experiment_config.py` now matches the shipped
`manuscript/config.yaml` budgets and rejects malformed mapping blocks instead of
silently treating lists or scalars as empty configuration. The review-grid
producer also consumes the configured FDR level, power level, and planning
alternative; the nested report records those executed values and its schema
checks them. Diagnostic smoke grids explicitly report that the MCSE target was
not evaluated, while publication grids fail closed unless the registered
target is met. These are implementation-integrity controls, not new scientific
evidence or a relaxation of the finite claim boundary.

## Current source contract — iteration 47

The single-machine research lanes now have executable bounded pilots and a
common typed CLI report write boundary. The calibration pilot separates
calibration from evaluation and rejects intentional overlap; the synthetic BNN
pilot exercises cavity-conditioned site updates, CPU/MPS receipts, and
checkpoint/resume equivalence; the hybrid pilot retains matched controls and a
singular-covariance negative control; and the Four Rooms/Key-Door pilot retains
flat, oracle, learned, shuffled, and non-gating controls. These are pilot
implementation facts only. They do not close MAJ-2A/3/5, establish universal
robustness, or convert a next-position predictive score into a general control
result.

The Friston audit remains a separately named, paper-constrained reconstruction
lane. Its analogue-relabeling negative control prevents the reduced categorical
matrices from being reported as exact source-protocol replication. The external
benchmark producer now exposes dataset-level nested-seed summaries and archive,
split, and recovery controls, but no pinned three-dataset confirmatory result or
manuscript token is promoted. Source-scale CUDA execution, physical multi-host
federation, cross-vendor verification, and external scientific release authority
remain outside this local evidence package. The v0.1.0 GitHub/Zenodo publication
is complete, but it does not close those scientific or governance boundaries.

## Load-bearing claims

| Claim family | Status after audit | Evidence / required wording |
|---|---|---|
| Standard belief sharing is recovered at the KL/NLL/$\beta=0$ corner | Formal / executable | Theorems and recovery residuals in `14_formalism.md`, `15_results_recovery.md`, and the locked tests. Call it a recovery identity, not a universal equivalence of literatures. |
| The generalized posterior recovers closed-form Bayes | Formal / executable | `generalized_posterior(KLD, NLL)` and the closed-form residual token. |
| The client-side robust update has bounded influence | Source-conditional | FedGVI and robust-loss assumptions govern the transfer; do not imply that every generalized posterior or every data-generating process is bounded. |
| `robust_aggregate` is robust in the theorem-backed sense | Open / explicitly denied | Its proven property here is the zero-robustness recovery limit. Its accuracy is a server-side conditional empirical result. |
| `variational_aggregate` has objective-backed weight control | Formal / executable | Stated free energy, block updates, raw effective-weight bound, and numerical descent witnesses; do not convert this into truth-recovery or peak-accuracy claims. |
| Robust server methods dominate the standard pool | Conditional empirical | False at low contamination in the current sweep; the strongest supported claim is regime-dependent benefit under severe declared contamination. |
| The robustness verdict is statistically supported | Conditional empirical | Matched trials, Wilcoxon test, BH-FDR within the declared family, percentile-bootstrap intervals, and observed-effect planning. The fixed hidden state and attack target remain part of the estimand. |
| The robustness figure caption reports the plotted estimand | Scoped implementation fact | The rendered caption now hydrates matched-trial profile means from `per_rate_summary`; a regression test rejects the deterministic single-colony scalar when it differs. |
| The manuscript renders across publication surfaces | Source-current scoped implementation fact only when matching receipts verify | The 2026-08-01 second-pass template stages produced 43/43 section decks, 43 TeX sources, 44 HTML surfaces including the index, an 84-page combined PDF, and 30 registered figure stems with 60 PNG/PDF assets; those are historical local build facts. The current surface must instead be traced through a matching analysis → test receipt → final hydration → render → web/PDF/slide validation chain. The HTML accessibility enhancement and `Tagged: no` PDF status remain build-surface facts, not scientific evidence. |
| The reported trial count is an independent sample size | Scoped implementation fact | Trials are conditionally independent within the declared world; seeds are the independent Monte Carlo unit for cross-study summaries. Do not count clients/episodes as extra replicates. |
| Bootstrap intervals describe deployment uncertainty | Open / explicitly denied | They quantify variation over the declared resampling unit, not alternate worlds, hidden states, attacks, or real deployments. |
| Communication is necessary in the moving-world study | Conditional empirical and split by control | The binary-complement control is a null/cost result; the larger disjoint-FOV extension shows a substantial conditional benefit over isolation. |
| The hierarchy improves location accuracy | Conditional empirical and not supported in the tested setting | The two-level result matches rather than beats the flat location baseline; context is additionally inferred. |
| The N-level extension runs at arbitrary depth | Scoped implementation fact | The constructor accepts a parameterized stack; the empirical result is the declared three-level run. Avoid “arbitrary depth” in results or conclusion. |
| Acuity is generically identifiable | Open / explicitly narrowed | The parameter-recovery study selects acuity on a finite tested grid; it is not a generic identifiability theorem. |
| The aggregation API is model-class-agnostic | Open / explicitly narrowed | One deterministic MLP provides an API-transfer demonstration; it does not establish model-class universality. |
| The cited literatures have never made this connection | Review-bounded negative claim | Use “in the reviewed/cited sources we did not find…” rather than a field-wide universal negative. |
| The three Friston studies are exact reproductions | Not supported; source-mechanism analogue only | Studies 1--3 use reduced categorical protocols. Exact recovery identities are the claims that receive machine-precision certification; exact source-protocol replication remains future work. |
| The visual bridge documents an executable result | Explicitly denied / schematic | The graphical abstract, generative-model, message-passing, and POMDP figures are deterministic explanatory surfaces. Their captions state their formal/mechanistic/model role and no-CI disposition; only the named data-bearing figures carry empirical estimands. |
| The dense categorical implementation has the reported computational complexity orders | Scoped implementation fact | `src/fedference/complexity.py`, the public-path experiment in `src/fedference/experiments/complexity.py`, and the generated complexity report/figure. State the representation, retained histories, fan-out, and excluded network wait; do not generalize to alternate implementations or hosts. |
| The measured scaling slopes establish the asymptotic orders | Open / explicitly narrowed | Repeated seeded timings are a machine-local diagnostic with min–max spans and descriptive log--log fits. Finite-grid overhead can make slopes sublinear; the symbolic source accounting, not slope equality, carries the implementation claim. |
| Conditional-world attack-geometry extension generalizes the robustness mechanism | Conditional empirical; mixed and falsifying for universality | `output/reports/conditional_world.json` covers 40 pre-registered cells, 16 independent seeds, and 12 nested trials per cell. Controls pass, but label-noise cells favor robust aggregation while permutation cells favor naive pooling; the result is a finite-grid mechanism diagnostic, not a universal attack result. |
| Robust aggregation improves reported belief quality under proper scores | Not supported in the tested subset | `output/reports/belief_quality.json` uses categorical log score as the primary seed-level estimand with Brier/ECE diagnostics. Oracle/uniform/confident-wrong controls pass, while all six tested robust-minus-naive log-score contrasts are negative; no belief-quality advantage is claimed. |
| The hybrid discrete/Gaussian API completes the continuous active-inference extension | Scoped implementation slice only | `src/fedference/hybrid.py` provides a typed Gaussian precision pool and a finite robust fixed-point diagnostic with exact zero-robustness recovery tests. It is not a full continuous/hybrid active-inference task, a variational objective, or evidence for MAJ-3 closure. |
| The public configuration, registry, CLI, and receipts establish a scientific result | Explicitly denied | `AggregationConfig`, `ExperimentSpec`, `DatasetSpec`, `RunReceipt`, and the installed CLI provide stable execution/provenance contracts. They can make a result auditable; they cannot make its effect positive, independent, or theoretically justified. |
| The scoped no-go gives `robust_aggregate` an objective certificate | Open / explicitly denied | `server_theory.py` proves only that the declared continuously differentiable separable class cannot have the raw log pool as its q-coordinate minimizer for every interior input. It does not provide an objective, rule out all objectives, or transfer a FedGVI theorem. |
| The FedGVI BNN source protocol has been replicated | Scoped protocol primitive only | `bnn_fedgvi.py` implements site factors, cavities, factor replacement, and checkpoints; the parity matrix still marks the source loss/divergence and cavity-conditioned client optimizer unresolved. CPU/MPS execution does not make the portable profile exact source-scale CUDA evidence. |
| The registered UCI path establishes an external robustness effect | Open / explicitly denied | Three archives have DOI/license/URL/hash/schema/preprocessing/split contracts and an executable benchmark/receipt path. No confirmatory dataset-level evidence pack or manuscript result has been produced; smoke scores carry no scientific claim. |
| The tracking fixture closes MAJ-3 | Scoped implementation slice only | `hybrid_tracking.py` executes a seeded context-gated closed-loop tracking fixture. Its log score is on-policy, its known-context component is not an oracle controller, and its risk-plus-variance quantity is a predictive-risk surrogate rather than expected free energy. The held-out, discrete-only, continuous-only, oracle, covariance, calibration, and confirmatory comparisons remain open. |
| The Friston source experiments are now exact replications | Not supported | Machine-readable parity matrices make unresolved parameters visible and force the label “paper-constrained reconstruction.” No source-protocol reconstruction has yet earned exact-replication status. |
| The versioned socket envelope establishes secure multi-host deployment | Explicitly denied | Protocol version, round, worker, configuration hash, payload digest, authentication mode, HMAC compatibility, digest replay validation, enforced loopback binding, and caller-selected in-memory or restart-durable local round-ID guards are single-machine transport facts. The SQLite guard is not a shared multi-host replay domain. Docker/mTLS emulation and physically distinct-host receipts remain separate open lanes; integrity is not confidentiality, privacy, identity-bound authentication, or Byzantine tolerance. |

## Structural refactor decisions

The title changes use claim-bearing language where prior titles implied a stronger
result than the evidence supports:

- introduction → robust generalized Bayes;
- gap → research gap and claim boundary;
- results → contamination sweep under severe attacks;
- moving-world study → sharing helps / is needed, reflecting its null binary control;
- hierarchical study → executed test of the N-level template;
- discussion → what the evidence supports;
- limitations → limitations and claim boundaries;
- future work → testing the open boundaries;
- conclusion → recovery-tested bridge with bounded claims.

The introduction now states the research questions and the conditional estimand.
The discussion now separates formal identities, conditional simulation behavior,
Monte Carlo precision, and deployment generalization. The conclusion retains the
three-axis guarantee map and removes claims of universal dominance, generic
identifiability, model-class universality, and arbitrary-depth empirical scaling.

## Source map

The scholarship behind the statistical wording is recorded in
[`literature-audit.md`](literature-audit.md). In particular, Morris, White &
Crowther motivate explicit estimands and Monte Carlo-error reporting; Koehler,
Brown & Haneuse motivate simulation-precision reporting; Loy & Korobova motivate
respecting nested/clustered resampling units; Genest & Zidek anchor the
log-linear-pool genealogy; and the Friston and Mildner papers anchor the two
source formalisms. These sources inform the boundaries; none is treated as proof
of this repository's empirical outcomes.

The dated numerical reconciliation is recorded in
[`extended-statistical-audit-2026-07-14.md`](extended-statistical-audit-2026-07-14.md).
That audit is the companion to this claim ledger: it records the dated report
values, replication units, negative controls, and no-claim boundaries without
turning a generated snapshot into a theorem.

## MAJ-1 characterization update

The formal result now resolves the roadmap's scoped-impossibility branch:
`server_theory.py` constructs the raw-log-pool contradiction and a
normalized-weight companion, and the typed characterization report binds both
witness summaries to the source report. The proof covers the declared
separable class only; it is not evidence that every possible coupled,
source-dependent, non-differentiable, or fixed-point-only construction fails.

The current `heuristic_characterization.json` report adds a deterministic grid
over state dimension, honest-agent count, robustness, attack mechanism, and
base-weight imbalance. Each row is labelled as a scoped implementation fact and
records whether a finite capture witness was found within the configured search
budget. The grid is deliberately not treated as a random-world sample, a
breakdown probability, a global bound, or a proof of an objective.

## Prior committed closure record — 2026-07-28

The corrected machine-local pass kept the scientific claim boundaries above
unchanged. The full source suite completed **1,149 passed** with **92.48%**
coverage on `src/` (floor: 90%); Ruff, mypy over 101 source files, the
`src/fedference` layer-isolation check, invariants, report schemas, focused
claim/figure/statistics/release tests, pipeline freshness, and the explicit
package and rendered-surface gates also passed. The publication surfaces
contain 42 manuscript sections, 42 slide decks, 43 HTML surfaces including the
index, an 80-page combined PDF, 29 registered figure stems, and 58 PNG/PDF
figure assets. `qpdf --check` passed for all 43 publication PDFs. These are
reproducibility and presentation facts only; they do not enlarge any empirical
or theorem-backed claim in this ledger.

## Historical development candidate — 2026-07-30

The public API/registry/receipt, packaged-data, transport, BNN-protocol, and
roadmap additions remained an uncommitted development candidate, not release
evidence at that date. Its analysis, hydration, three-surface render, and release
bundle were regenerated in producer order and their content-bound receipts
verified against the then-current source inputs. The obsolete retained XeLaTeX stdout
log was removed because the current renderer no longer produces it.

The combined manuscript and slide PDFs pass structural and raster inspection;
the HTML manuscript additionally passes the strengthened accessibility,
responsive-layout, keyboard-navigation, deep-link, image, and asset checks in
a real browser. The PDFs remain untagged convenience surfaces and are not
PDF/UA artifacts. Package, source, rendering, or accessibility validation does
not change any theorem or empirical claim above. This dated candidate record
predates the published v0.1.0 snapshot and is not a current release-status
statement.
