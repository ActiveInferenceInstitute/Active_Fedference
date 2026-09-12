"""Machine-readable provenance contracts for every project figure.

The registry is deliberately data-only: figure modules remain pure plotting
functions, while this module records what a rendered artifact means, how it is
related to the literature, and what replication unit supports any interval.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Final

from analysis.visual_contracts import COMPLEX_FIGURE_GENERATORS

FIGURE_SUPPORT_MODULES: Final[frozenset[str]] = frozenset({
    "__init__", "_common", "_metadata", "_presentation",
    "_presentation_flows", "_presentation_robustness", "_presentation_diagnostics",
    "_presentation_estimates",
    "_presentation_schematics",
    "_presentation_studies",
    "_presentation_worlds",
})

_GENERATORS: Final[tuple[str, ...]] = (
    "aggregation_descent",
    "application_integrity_flow",
    "belief_heatmap",
    "belief_quality",
    "bnn_robustness",
    "bounded_influence",
    "complexity_scaling",
    "conditional_world",
    "contamination_gallery",
    "cross_study_summary",
    "descent_comparison",
    "disjoint_fov_world",
    "efe_decomposition",
    "emergence_bmr",
    "evidence_replication_map",
    "free_energy_comparison",
    "generative_model_schema",
    "graphical_abstract",
    "heuristic_breakdown",
    "hierarchical_bmr",
    "hierarchical_pomdp",
    "language_kl_decay",
    "message_passing",
    "moving_world",
    "parameter_recovery",
    "pomdp_loop",
    "robust_influence_weights",
    "robustness_onset",
    "robustness_review_grid",
    "robustness_sweep",
    "sensitivity_heatmap",
    "source_render_provenance",
    "system_overview",
)

_SOURCE_CITATION = "Friston et al. (2024), Federated inference and belief sharing"

# Concise alternatives for the tagged PDF figure structure and other readers
# that cannot inspect the raster.  These describe the visible encodings and
# the claim boundary; the neighbouring manuscript caption remains the long
# description with source-bound numeric tokens.
_ALT_TEXT: Final[dict[str, str]] = {
    "aggregation_descent": (
        "Line plot of variational free energy versus block-coordinate iteration. "
        "The trace drops steeply at first, then remains flat through convergence."
    ),
    "application_integrity_flow": (
        "Three vertical panels show request validation and canonical hashing, a two-by-two key for the "
        "four exact solver_status codes, and atomic request/result/receipt writing with a separate "
        "three-level verification lane. Red dashed exits reject invalid input or report mismatches; a "
        "neutral dashed exit retains non-nominal results."
    ),
    "belief_heatmap": (
        "Heatmap of posterior probability over nine location cells for seven agent "
        "rows and a consensus row. Agent mass is dispersed, while the consensus "
        "row concentrates near one cell."
    ),
    "belief_quality": (
        "Two-panel control figure. Open circles, open diamonds, and filled crosses compare oracle, "
        "uniform, and confidently-wrong categorical log scores with seed intervals; the right panel "
        "uses the same marker-and-dash identities for binned reliability coordinates against a dotted "
        "perfect-calibration line. Brier score and ECE remain report-only diagnostics."
    ),
    "bnn_robustness": (
        "Held-out accuracy curves compare joint NLL/L2=0.05 and RCCE/L2=0.10 configurations in an "
        "exploratory point-estimate logistic-regression proxy. Because loss and shrinkage change "
        "together, the contrast cannot isolate an RCCE-only effect. Seed intervals separate most at "
        "the within-grid selected peak contamination; both decline at the highest rate."
    ),
    "bounded_influence": (
        "An open-triangle dash-dot path shows a variational server's normalized influence weight falling as "
        "one agent drifts toward a confident-wrong belief. A circle-solid naive reference stays flat, direct "
        "endpoint labels identify both rules, and an X marks the first displayed half-weight crossing."
    ),
    "complexity_scaling": (
        "Four-panel log-log timing figure showing median wall-clock scaling with "
        "agents, states, and modalities for aggregation, self-excluding sharing, "
        "and state inference. Distinct markers, dashes, open fills, and endpoint labels identify operations; "
        "error bars show timing spans and dotted lines are normalized order guides."
    ),
    "conditional_world": (
        "Two-panel finite-grid robustness diagnostic. The left heatmap shows the "
        "robust-minus-naive true-state-mass contrast across attack mechanisms and "
        "world or acuity cells; the right panel shows finite-grid means with "
        "min/max spans by attack mechanism."
    ),
    "contamination_gallery": (
        "Grouped bars compare naive pooling with the pooled display robust member "
        "for five contamination mechanisms. Bars show mean true-state consensus "
        "accuracy with seed-bootstrap intervals; annotations identify the selected "
        "method and its across-seed win fraction."
    ),
    "cross_study_summary": (
        "A page-compatible two-column composition keeps native units separate: six signed accuracy rows "
        "span the left column, while information or free-energy rows in nats and one parameter-recovery "
        "R-squared row occupy the upper and lower right. Labels, whiskers, hatches, and zero rules duplicate "
        "color; intervals belong to a separate harmonized seed-level rerun."
    ),
    "descent_comparison": (
        "A filled-circle solid path and an open-diamond dotted path compare single-start and multistart "
        "variational free-energy descent as neutral initialization conditions. The single start remains in a "
        "higher observed basin; a dark dotted rule marks the lower multistart terminal level."
    ),
    "disjoint_fov_world": (
        "Two-panel disjoint-field-of-view diagnostic. The left bars compare isolated "
        "and communicating consensus accuracy; the right bars compare EFE-guided "
        "and random movement, where both policies are near ceiling."
    ),
    "efe_decomposition": (
        "Two views of a categorical expected-free-energy identity in nats. The left "
        "stack shows risk plus ambiguity; the right signed waterfall shows pragmatic "
        "value and a negative epistemic correction ending at the same terminal value."
    ),
    "emergence_bmr": (
        "Dotted- and cross-hatched bars show Bayesian model-reduction free-energy differences for pruning a "
        "redundant versus supported likelihood column. Direct signed labels and a dotted zero rule show that "
        "the configured redundant-pruning control is positive and supported-column pruning is negative."
    ),
    "free_energy_comparison": (
        "A paired estimation plot joins each seed's filled-circle incommunicado and open-diamond "
        "communicating "
        "free energy, then shows open-diamond differences, a dotted zero rule, and the percentile-bootstrap "
        "interval of incommunicado minus communicating values. Positive differences mean lower free energy "
        "with communication."
    ),
    "generative_model_schema": (
        "Four-panel categorical generative-model schematic: private observation, "
        "A/B/C/D-zero factors feeding a local posterior, temporal state-observation-"
        "action order, and optional hierarchical context."
    ),
    "graphical_abstract": (
        "Numbered schematic linking private categorical beliefs, honest and contaminated broadcasts, a "
        "fusion server, and three non-transferable claim lanes. Mini posterior bars with solid inbound "
        "arrows distinguish honest agents from white crosses with dashed inbound arrows. The project "
        "identity "
        "at zero server robustness is separate from the qualified categorical Eq. 7 specialization."
    ),
    "heuristic_breakdown": (
        "Three-panel server-heuristic diagnostic: circle-solid versus square-dashed numerical influence, "
        "forward- versus dotted-hatch finite colluder counts for two aggregators, and directly labeled "
        "fractions "
        "of a declared attack grid with capture."
    ),
    "hierarchical_bmr": (
        "Horizontal bars show Bayesian surprise at two non-leaf levels in informative and degenerate "
        "configured worlds. The threshold classifies whether the upper information channel is retained; "
        "this diagnostic is not a Bayesian-model-reduction emergence result."
    ),
    "hierarchical_pomdp": (
        "Six-panel hierarchical POMDP diagnostic. The top row shows two-level "
        "location, context, and colony beliefs; the bottom row shows three-level "
        "beliefs and measured hierarchical-minus-flat accuracy gaps. Plain and forward-hatched bars, "
        "distinct "
        "trajectory markers, peak labels, and direct signed values duplicate color."
    ),
    "language_kl_decay": (
        "Seed-mean KL divergence from the true categorical likelihood to the learned "
        "likelihood across ordered learning batches. The curve decreases rapidly and "
        "then approaches zero, with a seed-bootstrap interval band."
    ),
    "message_passing": (
        "Four-band protocol schematic: agents update from private observations, broadcast local categorical "
        "posteriors, enter one of three standard, heuristic, or variational fusion routes, and receive "
        "cavity-excluded posteriors. Route-specific badges keep client and server claim owners separate."
    ),
    "moving_world": (
        "Three-panel moving-world comparison of isolated, communicating, and "
        "expected-free-energy-guided conditions, showing accuracy, signed free-energy "
        "gap, and steps to consensus."
    ),
    "parameter_recovery": (
        "Two-panel sensor-acuity recovery diagnostic. Open estimate markers with capped empirical intervals "
        "follow a dotted identity line; open dotted-hatch bars show absolute error at each tested acuity, "
        "with "
        "a labeled dotted rule for the global mean."
    ),
    "pomdp_loop": (
        "Three-panel active-inference schematic showing a shared hidden world, one "
        "cavity-excluded belief-sharing round, and the temporal hidden-state, "
        "observation, posterior, action, and transition loop."
    ),
    "robust_influence_weights": (
        "Bar chart of server weights for two contaminated and five honest agents. "
        "The heuristic assigns near-zero weight to contaminated broadcasts and "
        "larger, near-equal weights to honest agents; the dotted line is equal weight."
    ),
    "robustness_onset": (
        "Three mechanism-specific panels plot pooled robust-member and naive consensus "
        "accuracy against contamination rate. Vertical lines mark descriptive onset "
        "rates, and byzantine contamination shows a transient contrast before collapse."
    ),
    "robustness_review_grid": (
        "Portrait robustness review with a full-width top heatmap and three rate profiles stacked below "
        "for confident-wrong, Byzantine, and drift attacks. Every predeclared server preset uses a "
        "distinct marker and dash; elbow leaders route endpoints into a dedicated right label lane. "
        "Heatmap second lines are finite-grid spans, not confidence intervals."
    ),
    "robustness_sweep": (
        "Consensus true-state-mass curves compare the standard log-linear pool with four actual "
        "robust-aggregate server presets. Distinct markers and dashes show matched-trial means with "
        "95-percent percentile-bootstrap intervals; the horizontal rule is the declared accuracy floor."
    ),
    "sensitivity_heatmap": (
        "Two heatmaps of accuracy gaps over sensor acuity and colony size: "
        "communicating minus isolated above, and hierarchical minus flat below. "
        "A symmetric color scale is centered at zero and hatched cells "
        "mark the declared display band using unrounded gaps. Printed values are rounded, "
        "so equal printed values can have different hatching; the band is not a confidence interval."
    ),
    "source_render_provenance": (
        "Three vertical panels show a five-input dependency matrix beside a crossing-free producer path, "
        "fourteen reverse-invalidation cards, and cycle and authorization boundaries. Letter codes and "
        "arrows duplicate color; GitHub and Zenodo remain separately authorized terminal states."
    ),
    "evidence_replication_map": (
        "Large-type evidence ledger with fourteen numbered result-family rows and six directly labeled "
        "columns. Each row names its class, estimand and unit, replication unit, nesting, permitted support, "
        "and prohibited generalization. A five-step nesting strip and three no-transfer boxes keep client, "
        "server, study, transport, application, and open confirmation evidence separate."
    ),
    "system_overview": (
        "Three-part configured diagnostic of a five-agent colony with two A-labeled adversarial and three "
        "H-labeled honest agents, naive equal-weight pooling, and heuristic reweighting. Cross-hatched A "
        "roles and plain-filled H roles have distinct keylines; direct state, argmax, true-state-mass, "
        "agent-role, and influence-weight labels show the off-target naive outcome and displayed heuristic "
        "recovery without color."
    ),
}

# Complex figures receive a structured description distinct from their concise
# image alternative and self-contained caption. Paragraph one follows the
# visible panel order and names non-colour encodings; paragraph two identifies
# the estimand/units, uncertainty/replication structure, and no-claim boundary.
_LONG_DESCRIPTIONS: Final[dict[str, str]] = {
    "application_integrity_flow": (
        "Read from panel A downward. Panel A moves a labeled JSON or Python request through strict "
        "schema and one-dimensional shape validation, a canonical semantic request and SHA-256, and "
        "a separate destination-and-provenance safety gate before one delegation to aggregate_result. "
        "One dashed red branch ends at exit 2 for an invalid request; another ends at exit 2 for an "
        "unsafe destination or unavailable required provenance. Both terminate before directory creation. "
        "Panel B carries the rich numerical result through "
        "a convergence-by-fallback classifier and then a two-by-two exact solver_status key: nominal, "
        "converged_with_fallback, not_converged, and not_converged_with_fallback. Its neutral dashed branch "
        "retains valid artifacts at exit 1 when health is non-nominal. Panel C carries the canonical request "
        "and rich result into request.json and result.json, binds both into receipt.json, and then enters "
        "verification. Solid arrows encode these data dependencies; a separate dotted sequence "
        "above the artifact boxes encodes the atomic request.json, result.json, receipt.json write order. "
        "The lower "
        "lane separates artifact integrity, source equivalence, and optional nominal solver health, while a "
        "dashed red branch reports an exact mismatch without reinterpreting the result."
        "\n\nThis is a source-owned implementation map, not an experiment. Nodes and arrows carry "
        "categorical "
        "software states rather than a numerical estimand, so there is no sampling interval or "
        "replication unit. Shape, line style, direct labels, and ordered numbering duplicate the colour "
        "roles. Artifact integrity, requested source equivalence, and nominal solver health do not "
        "establish calibration, domain suitability, scientific validity, a downstream decision, or "
        "acceptance."
    ),
    "evidence_replication_map": (
        "Read the full-width ledger from row 1 through row 14 and across its six labeled columns. Every row "
        "aligns one headline result family or public boundary with its evidence-class code, estimand and "
        "unit, independent unit, nested structure, permitted support, and prohibited support. The compact "
        "class "
        "key distinguishes formal or executable, source-conditional, conditional empirical, scoped "
        "implementation, and open evidence by symbols and keylines as well as color. Separate rows retain "
        "the project identity, client theorem, paired free energy, likelihood learning, BMR control, "
        "exploratory client baseline, heuristic and variational server rules, moving/disjoint-field worlds, "
        "hierarchical/sensitivity studies, acuity recovery, protocol, application provenance, and external "
        "confirmation. The top strip supplies a generic nesting grammar from configured study or run "
        "through "
        "seed when present, matched trial or condition, agent or contamination role, and ordered step or "
        "posterior state. Each level applies only where the corresponding lane's report declares it. Three "
        "numbered no-transfer boxes close the ledger."
        "\n\nThe estimand and units differ by lane; the independent unit is therefore read from the "
        "same row, "
        "not inferred from the number of agents, trials, or time steps. Intervals, where present in the "
        "underlying result, resample the declared seed or matched-trial unit with lower levels nested. "
        "The no-transfer boxes state that guarantees cannot migrate between client, server, transport, "
        "and application evidence, and formal or software evidence does not imply independent external "
        "confirmation. Exact numerical estimates and intervals remain in their typed reports and the "
        "native-unit cross-study summary."
    ),
    "graphical_abstract": (
        "Follow the numbered panels from private categorical beliefs to the fusion server and then to "
        "bounded outputs. Honest agents contain white mini posterior bars and use solid inbound arrows, "
        "whereas adversarial agents contain white crosses and use dashed inbound arrows, so role remains "
        "visible without colour. The outcome cards directly label both fusion rules and their displayed "
        "true-state masses. The central server separates the exact project "
        "identity robust_aggregate(c=0) equivalent to log_linear_pool from the qualified relation to the "
        "categorical Eq. 7 message-combination term. The final panel keeps three claim owners distinct: "
        "generalized-Bayes client updates under stated source assumptions, a divergence-reweighted server "
        "heuristic with conditional finite evidence, and an objective-backed variational server rule with "
        "an effective-weight property."
        "\n\nThis deterministic formal/mechanistic schematic displays configured true-state-mass summaries "
        "for one explanatory colony but has no sampled quantitative axis, uncertainty interval, or "
        "independent replication unit. "
        "Numbering, inner glyphs, line patterns, keylines, and direct labels repeat every semantic colour. "
        "The "
        "identity is project-local, and the Friston relation requires shared support, admitted posterior-"
        "log potentials, and fixed weights. The diagram does not reconstruct the complete source "
        "protocol and does not transfer client theorems to either server rule."
    ),
    "message_passing": (
        "Read the vertically ordered bands from client update to transport, fusion, and recipient return. "
        "The client band separates NLL/KLD recovery and generalized-Bayes client losses from all server "
        "operations. The transport band carries only categorical posteriors in the unchanged protocol-v1 "
        "envelope. Three parallel fusion routes use distinct shapes and dash patterns for the log-linear "
        "pool, divergence-reweighted robust_aggregate heuristic, and objective-backed "
        "variational_aggregate rule. Claim-owner badges identify a source theorem under assumptions, "
        "conditional heuristic evidence, an effective-weight property, and a qualified Eq. 7 "
        "specialization. The return band sends a cavity-excluded posterior to each recipient."
        "\n\nThe figure maps categorical messages and ownership, not a measured effect; it therefore has no "
        "interval or replication unit. Posterior-only broadcast and self-exclusion are implementation "
        "properties. The source relation is limited to the categorical message-combination term under "
        "shared-support, posterior-log-potential, and fixed-weight assumptions. It is not a complete "
        "Friston-protocol reconstruction, a physical multi-host demonstration, or a theorem transfer from "
        "robust client losses to server fusion."
    ),
    "generative_model_schema": (
        "Read panels A through D. Panel A relates a private observation to the local categorical hidden "
        "state. Panel B shows A, B, C, and initial-state D factors feeding the local posterior. Panel C "
        "orders hidden "
        "state, observation, posterior update, action, and transition through time. Panel D adds an "
        "optional higher-level context that modulates the lower categorical process while retaining local "
        "observations. Solid arrows encode the implemented categorical dependencies used by the applicable "
        "studies; the dashed top-level context edge identifies a qualified optional extension. Every node "
        "and "
        "factor is directly labeled."
        "\n\nThis is a conceptual source-inspired schema with categorical states, outcomes, factors, and "
        "messages as its units. It contains no estimated quantity, resampling interval, or independent "
        "replication unit. The diagram identifies the implemented factorization and optional hierarchy; "
        "it does not claim that every source-model component, continuous state, learning rule, or "
        "multi-agent protocol has been reconstructed."
    ),
    "pomdp_loop": (
        "Read the panels from shared world to one federation round and then through time. The first panel "
        "places agents around a shared categorical hidden world while keeping observations private. The "
        "second panel sends local posteriors to fusion and returns one cavity-excluded posterior per "
        "recipient; solid arrows and direct labels distinguish posterior broadcast and return, while the "
        "dashed recipient connector marks the cavity-exclusion qualification. The third panel orders "
        "hidden state, observation, posterior, policy or action, and transition, with the federation step "
        "inserted at the posterior boundary. Solid arrows trace the implemented inference, exchange, action, "
        "and transition sequence; dashed sensing connectors preserve the distinction between a shared hidden "
        "world and private observations."
        "\n\nThe units are conceptual categorical states, outcomes, messages, actions, and ordered "
        "steps. There "
        "is no uncertainty interval or replication unit. The schematic describes the project POMDP and "
        "moving-world extension; it does not demonstrate a complete source protocol, physical multi-host "
        "network, Byzantine tolerance, privacy mechanism, or universally beneficial communication policy."
    ),
    "complexity_scaling": (
        "Read the four panels in order. The first three log-log panels vary agents, categorical states, "
        "and modalities for aggregation, self-excluding sharing, and local state inference. Circle, "
        "square, triangle, dash, and direct endpoint labels identify operations independently of colour. "
        "Vertical spans show the recorded minimum and maximum wall-clock times around the median; dark "
        "dotted rules are normalized analytic-order guides rather than fitted confidence bands. The final "
        "panel aligns measured log-log slopes with the implementation-derived order expected for each "
        "operation."
        "\n\nThe estimands are median elapsed seconds and dimensionless log-log scaling slopes over the "
        "fixed "
        "benchmark grid. The repetition unit is a timing repeat inside one locked software and hardware "
        "environment; min-max spans are not confidence intervals and do not quantify machine-to-machine "
        "variation. Exact plotted values are available in the registered fallback table. The figure "
        "supports implementation scaling within the measured range, not an asymptotic lower bound, "
        "distributed-throughput claim, or universal hardware benchmark."
    ),
    "hierarchical_pomdp": (
        "Read the top row for the two-level condition and the bottom row for the three-level condition. "
        "Within each row, the panels progress from lower-level location beliefs through context beliefs to "
        "the colony-level comparison. In the location and consensus panels, plain left-offset bars denote "
        "the "
        "flat reference and forward-hatched right-offset bars denote hierarchical inference; marker shapes, "
        "dash patterns, peak labels, and direct signed values supply the corresponding non-colour "
        "distinctions "
        "in the trajectory and accuracy panels. Probability axes retain their native zero-to-one "
        "scale, while the final panel reports the measured hierarchical-minus-flat location-accuracy gap "
        "for the corresponding configured protocol."
        "\n\nThe estimands are categorical posterior probability and final location-accuracy difference, in "
        "probability or fractional units. The visible trajectories are deterministic seeded diagnostics; "
        "the separate harmonized cross-study rerun owns any seed-bootstrap interval. Ordered states and "
        "steps are nested inside the configured run and are not independent replications. The figure "
        "supports behavior of these finite hierarchical worlds, not universal hierarchical advantage or "
        "source-protocol parity."
    ),
    "robustness_review_grid": (
        "Read the portrait figure from top to bottom. Panel A is a full-width conditional-world heatmap; "
        "each cell prints a signed robust-minus-standard true-state-mass mean and, on its second line, half "
        "the finite-grid min-max span. Panels B, C, and D then stack confident-wrong, Byzantine, and drift "
        "rate profiles. Every predeclared server preset has a distinct marker and dash. Elbow leaders route "
        "curve endpoints into a reserved right-side label lane, while a dark dotted rule marks zero. The "
        "standard pool is the circle-solid reference; no result-dependent method selection is applied."
        "\n\nThe estimand is the signed difference in true-state probability mass, measured in probability "
        "units. Rate-profile points are seed means with 95-percent percentile-bootstrap intervals over "
        "independent configured seeds; trials and agents remain nested within each declared cell. The "
        "conditional summary uses its declared finite-grid span. The exact-value fallback repeats every "
        "displayed grouped mean and half min-max span from the renderer's shared helper. Mixed signs are "
        "retained: the map does not establish a universal winner, bounded "
        "influence, Byzantine tolerance, or generalization beyond the reviewed grid."
    ),
    "cross_study_summary": (
        "Read Panel A down the full left column, then Panel B at upper right and Panel C at lower right. "
        "Panel A shows six signed accuracy contrasts in fractions; Panel B shows two signed information or "
        "free-energy contrasts in nats; Panel C shows one parameter-recovery fit in unitless R-squared. "
        "Horizontal bars carry interval whiskers and direct signed mean labels. Forward hatching marks "
        "positive estimands, cross-hatching negative estimands, and dotted hatching near-zero estimands; "
        "dark dotted rules mark zero in every native-unit panel. Each study appears only in its applicable "
        "panel."
        "\n\nEach displayed interval is produced by the separate harmonized seed-level rerun, even where a "
        "corresponding primary figure is a deterministic single-posterior diagnostic. Seeds are the "
        "resampling and independent unit within each configured study; observations, agents, trials, and "
        "ordered steps remain nested. The Study 4 row is the within-run display-selected maximum across "
        "non-reference server presets at the declared worst rate; it is not a preselected method or an "
        "inferential winner. Exact values are listed in the fallback table. Native units prevent "
        "a pooled ranking: this synthesis does not imply common effect scales, universal communication "
        "benefit, or generalization across unmodeled datasets and deployments."
    ),
    "source_render_provenance": (
        "Read Panel A first across the dependency matrix and then through the two-band process path. Matrix "
        "rows name five source-owned inputs and three upstream producers; columns name seven downstream "
        "targets. Each populated cell carries P, G, or R to identify a producer, gate, or receipt dependency "
        "without relying on color. The adjacent crossing-free path orders provisional hydration, tests and "
        "coverage, final hydration, exact clean Template rendering, reader surfaces, surface checks, "
        "provenance, manifests, and the separately authorized GitHub and Zenodo terminals. Solid arrows are "
        "producer or receipt dependencies and dashed forward arrows are gates. Panel B groups fourteen "
        "changed owners into dashed reverse-dependency cards; each arrow points from immediate stale "
        "targets on the right back to the changed owner on the left. Panel C records the non-circular source "
        "contract, "
        "authorization separation, and four numbered no-claim boundaries."
        "\n\nThe estimand is categorical dependency and authorization state, so there is no statistical "
        "uncertainty, sample size, or replication unit. Codes, keylines, border styles, direct labels, and "
        "ordered numbering duplicate color. The figure is source-owned and never reads the downstream "
        "manifest it helps document. A green build does not authorize publication or establish scientific "
        "validity, PDF/UA or WCAG conformance, or a larger scientific claim surface."
    ),
}

_EXACT_VALUE_FALLBACKS: Final[dict[str, str]] = {
    "belief_quality": "fig-values:belief-quality",
    "bnn_robustness": "fig-values:bnn-robustness",
    "complexity_scaling": "fig-values:complexity-scaling",
    "conditional_world": "fig-values:conditional-world",
    "cross_study_summary": "fig-values:cross-study-summary",
    "robustness_review_grid": "fig-values:robustness-review-grid",
    "sensitivity_heatmap": "fig-values:sensitivity-heatmap",
}


def _default_metadata() -> dict[str, str]:
    return {
        "status": "diagnostic",
        "source_relation": "original project diagnostic",
        "source_figure": "",
        "source_equation": "",
        "source_citation": "",
        "estimand": "project-specific diagnostic quantity",
        "unit": "declared in the embedded caption",
        "uncertainty": "caption declares the interval or deterministic status",
        "replication_unit": "caption declares the replication unit",
    }


def _build_metadata() -> dict[str, dict[str, str]]:
    if set(_ALT_TEXT) != set(_GENERATORS):
        missing = sorted(set(_GENERATORS) - set(_ALT_TEXT))
        extra = sorted(set(_ALT_TEXT) - set(_GENERATORS))
        raise ValueError(f"figure alternative registry mismatch: missing={missing}, extra={extra}")
    metadata = {name: _default_metadata() for name in _GENERATORS}
    for name, alt_text in _ALT_TEXT.items():
        metadata[name]["alt_text"] = alt_text
    for name, long_description in _LONG_DESCRIPTIONS.items():
        metadata[name]["long_description"] = long_description
    for name, fallback in _EXACT_VALUE_FALLBACKS.items():
        metadata[name]["exact_value_fallback"] = fallback
    for name, estimand, unit in (
        ("free_energy_comparison", "incommunicado minus communicating colony mean free energy", "nats"),
        ("language_kl_decay", "seed-mean KL(true likelihood || learned likelihood) by learning step", "nats"),
        ("emergence_bmr", "BMR free-energy difference", "nats"),
    ):
        metadata[name].update(
            status="source-mechanism analogue",
            source_relation="source-mechanism analogue",
            source_citation=_SOURCE_CITATION,
            estimand=estimand,
            unit=unit,
            uncertainty=(
                "pointwise percentile bootstrap across independent seeds"
                if name == "language_kl_decay"
                else (
                    "95-percent percentile-bootstrap interval of paired seed differences"
                    if name == "free_energy_comparison"
                    else "none; deterministic closed-form comparison on a single posterior"
                )
            ),
            replication_unit=("not applicable" if name == "emergence_bmr" else "independent configured seed"),
        )
    metadata["free_energy_comparison"]["source_figure"] = "Fig. 5"
    metadata["language_kl_decay"]["source_figure"] = "Fig. 7"
    metadata["emergence_bmr"]["source_figure"] = "Fig. 9"

    metadata["efe_decomposition"].update(
        status="formal specialization",
        source_relation="formal specialization",
        source_equation="Eq. 2",
        source_citation=_SOURCE_CITATION,
        estimand="categorical expected-free-energy decomposition identity",
        unit="nats",
        uncertainty="none; deterministic algebraic identity",
        replication_unit="not applicable",
    )

    metadata["cross_study_summary"].update(
        status="native-unit diagnostic",
        source_relation="separate harmonized seed-level rerun across project diagnostics",
        estimand="study-level mean metric grouped by declared native unit",
        unit="fraction, nats, or R-sq by facet",
        uncertainty="95-percent percentile-bootstrap interval across independent configured seeds",
        replication_unit="independent configured seed",
    )
    for name, relation, estimand, unit, uncertainty, replication_unit in (
        (
            "aggregation_descent",
            "original project objective-descent diagnostic",
            "variational free energy by block-coordinate iteration",
            "nats",
            "none; deterministic seeded run",
            "not applicable",
        ),
        (
            "belief_heatmap",
            "original project diagnostic supporting the Study 1 analogue",
            "posterior probability mass by hidden state",
            "probability",
            "none; deterministic single-seed display",
            "not applicable",
        ),
        (
            "bounded_influence",
            "original project variational-server diagnostic",
            "normalized server influence weight along a declared outlier path",
            "normalized weight",
            "none; deterministic seeded sweep",
            "not applicable",
        ),
        (
            "descent_comparison",
            "original project objective-descent diagnostic",
            "variational free energy by iteration and initialization",
            "nats",
            "none; deterministic seeded runs",
            "not applicable",
        ),
        (
            "disjoint_fov_world",
            "original project extension of the source-inspired moving-world mechanism",
            "condition-level consensus accuracy in two declared protocols",
            "fraction",
            "across-seed standard-deviation error bars",
            "independent configured seed",
        ),
        (
            "hierarchical_bmr",
            "original project hierarchical information-gating diagnostic",
            "per-level Bayesian surprise and threshold keep/prune decision",
            "nats",
            "none; deterministic schematic worlds",
            "not applicable",
        ),
        (
            "hierarchical_pomdp",
            "source-inspired original diagnostic",
            "categorical posterior probabilities and final location-accuracy gap",
            "probability or fraction",
            "none; deterministic seeded protocol",
            "not applicable",
        ),
        (
            "heuristic_breakdown",
            "original project server-side heuristic diagnostic",
            "numerical influence, finite-search capture count, and grid capture fraction",
            "normalized weight, agents, or fraction by panel",
            "none; finite declared grid with no population resampling",
            "declared seeded scenario row; no population resampling",
        ),
        (
            "moving_world",
            "original project extension of the moving-world protocol",
            "condition-level accuracy, signed free-energy gap, and steps-to-consensus proxy",
            "fraction, nats, or steps by panel",
            "none; deterministic seeded run",
            "not applicable",
        ),
        (
            "parameter_recovery",
            "original project parameter-recovery diagnostic",
            "recovered acuity, absolute acuity error, and mean-recovery coefficient of determination",
            "probability units or unitless R-sq",
            "percentile interval across independent trials",
            "independent trial",
        ),
        (
            "robust_influence_weights",
            "original project server-side heuristic diagnostic",
            "normalized pooling weight by agent",
            "normalized weight",
            "none; deterministic single-run display",
            "not applicable",
        ),
        (
            "sensitivity_heatmap",
            "original project sensitivity diagnostic",
            "per-cell signed accuracy contrasts over acuity and colony size",
            "fraction",
            "none displayed; each cell is a mean over nested trials",
            "synthetic trial nested within configured cell",
        ),
        (
            "complexity_scaling",
            "original project computational-complexity diagnostic",
            "implementation-derived computational complexity and measured scaling slope",
            "seconds and dimensionless log-log slope",
            "repeated wall-clock min--max span; not a confidence interval",
            "fixed seeded benchmark grid and timing repeat",
        ),
        (
            "conditional_world",
            "original project finite conditional-world diagnostic",
            "conditional robust-minus-naive true-state mass across a finite world/attack grid",
            "true-state probability mass",
            "seed-level percentile bootstrap interval within each declared cell",
            "seeded world/scenario row; trials nested within row",
        ),
        (
            "robustness_review_grid",
            "original project expanded finite source-bound review diagnostic",
            (
                "seed-level robust-minus-naive true-state mass across the existing "
                "conditional cells and pooled rate profiles"
            ),
            "true-state probability mass",
            (
                "seed-level percentile bootstrap intervals in rate profiles; "
                "finite-grid min/max span in conditional summary"
            ),
            "independent configured seed within a declared cell; trials nested",
        ),
        (
            "belief_quality",
            "original project proper-scoring and calibration diagnostic",
            (
                "displayed categorical log score and binned reliability coordinates; "
                "Brier score and expected calibration error are report-only secondary diagnostics"
            ),
            "nats for log score; confidence and accuracy fractions for reliability coordinates",
            "seed-level percentile-bootstrap interval for displayed control log scores",
            "independent configured seed",
        ),
    ):
        metadata[name].update(
            status="diagnostic",
            source_relation=relation,
            estimand=estimand,
            unit=unit,
            uncertainty=uncertainty,
            replication_unit=replication_unit,
        )
    for name, estimand, unit, uncertainty, replication_unit in (
        (
            "belief_heatmap",
            "posterior probability by categorical state",
            "probability",
            "deterministic seeded run",
            "not applicable",
        ),
        (
            "bnn_robustness",
            "held-out accuracy by contamination rate",
            "fraction",
            "seed-level interval",
            "independent configured seed",
        ),
        (
            "contamination_gallery",
            "pooled display-member accuracy by contamination mechanism",
            "fraction",
            "method-specific seed bootstrap interval plus paired difference interval",
            "independent configured seed",
        ),
        (
            "disjoint_fov_world",
            "consensus accuracy by movement or communication condition",
            "fraction",
            "across-seed spread",
            "independent configured seed",
        ),
        (
            "hierarchical_pomdp",
            "posterior probability and location-accuracy gap",
            "probability or fraction",
            "deterministic seeded run",
            "not applicable",
        ),
        (
            "moving_world",
            "condition-level accuracy, free-energy gap, and steps proxy",
            "fraction, nats, or steps",
            "deterministic seeded run",
            "not applicable",
        ),
        (
            "parameter_recovery",
            "acuity parameter recovery fit, absolute error, and mean-recovery coefficient of determination",
            "probability units or unitless R-sq",
            "empirical percentile interval across independent trials",
            "independent synthetic trial within true-acuity grid point",
        ),
        (
            "robustness_onset",
            "pooled display accuracy and robust-minus-naive onset gap",
            "fraction",
            "method-pooled seed bootstrap interval at each rate",
            "independent configured seed",
        ),
        (
            "robustness_sweep",
            "matched-trial accuracy by contamination rate",
            "fraction",
            "95-percent percentile-bootstrap interval over matched trials",
            "matched synthetic trial within the fixed configured world",
        ),
        (
            "sensitivity_heatmap",
            "per-cell signed accuracy contrast",
            "fraction",
            "none displayed; each cell is a mean over nested trials",
            "synthetic trial nested within configured cell",
        ),
    ):
        metadata[name].update(
            estimand=estimand,
            unit=unit,
            uncertainty=uncertainty,
            replication_unit=replication_unit,
        )

    metadata["parameter_recovery"]["seed_role"] = (
        "base RNG stream; seed is not the independent analysis unit"
    )

    for name, source_figure, source_equation in (
        ("generative_model_schema", "Figs. 1 and 4", ""),
        ("message_passing", "Fig. 5", "Eq. 7"),
        ("pomdp_loop", "Figs. 1 and 4", ""),
    ):
        metadata[name].update(
            status="schematic",
            source_relation="source-inspired original schematic",
            source_figure=source_figure,
            source_equation=source_equation,
            source_citation=_SOURCE_CITATION,
            estimand="conceptual mechanism and claim-ownership map",
            unit="categorical states, outcomes, and messages",
            uncertainty="none; conceptual schematic",
            replication_unit="not applicable",
        )
    metadata["graphical_abstract"].update(
        status="deterministic formal/mechanistic schematic",
        source_relation=(
            "original project schematic with configured colony and outcome summaries"
        ),
        estimand=(
            "component relationships and recovery boundaries plus displayed naive and "
            "heuristic true-state probability mass"
        ),
        unit="conceptual route, agent count, or probability mass",
        uncertainty="none; deterministic configured beliefs",
        replication_unit="not applicable; one configured explanatory colony",
    )
    metadata["system_overview"].update(
        status="deterministic configured diagnostic schematic",
        source_relation="original project failure-and-repair diagnostic for one configured colony",
        estimand="displayed posterior-mass and normalized influence-weight contrasts",
        unit="probability mass or normalized server weight",
        uncertainty="none; one deterministic configured colony",
        replication_unit="not applicable; one configured explanatory colony",
    )
    metadata["bnn_robustness"].update(
        status="exploratory conditional baseline",
        source_relation=(
            "original exploratory point-estimate logistic-regression proxy comparing two joint "
            "loss/L2 configurations; legacy AR selects stronger L2 rather than a weight-space divergence"
        ),
        estimand=(
            "held-out classification accuracy for joint NLL/L2=0.05 and RCCE/L2=0.10 "
            "configurations by contamination rate"
        ),
        unit="fraction",
        uncertainty="95-percent percentile-bootstrap interval across synthetic-data seeds",
        replication_unit="independent synthetic-data seed",
    )
    metadata["application_integrity_flow"].update(
        status="explanatory implementation map",
        source_relation="source-owned labeled-application and receipt contract",
        estimand="implemented validation, solver-health, artifact, and verification states",
        unit="categorical software state",
        uncertainty="none; deterministic source-owned contract",
        replication_unit="not applicable",
    )
    metadata["evidence_replication_map"].update(
        status="explanatory evidence map",
        source_relation="source-owned claim-class and replication-unit contract",
        estimand="evidence ownership and permitted interpretation by claim lane",
        unit="evidence class and declared replication unit",
        uncertainty="none; deterministic source-owned contract",
        replication_unit="not applicable",
    )
    metadata["source_render_provenance"].update(
        status="explanatory provenance map",
        source_relation="source-owned producer-order and invalidation contract",
        estimand="producer dependency, stale-invalidation, and authorization state",
        unit="pipeline stage or publication state",
        uncertainty="none; deterministic source-owned contract",
        replication_unit="not applicable",
    )
    return metadata


FIGURE_METADATA: Final[dict[str, dict[str, str]]] = _build_metadata()


def figure_metadata(generator: str) -> dict[str, str]:
    """Return a defensive copy of the metadata contract for *generator*."""
    if generator not in FIGURE_METADATA:
        raise KeyError(f"no figure metadata contract for generator {generator!r}")
    return deepcopy(FIGURE_METADATA[generator])


__all__ = ["COMPLEX_FIGURE_GENERATORS", "FIGURE_METADATA", "figure_metadata"]
