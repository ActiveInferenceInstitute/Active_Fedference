"""Machine-readable provenance contracts for every project figure.

The registry is deliberately data-only: figure modules remain pure plotting
functions, while this module records what a rendered artifact means, how it is
related to the literature, and what replication unit supports any interval.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Final

_GENERATORS: Final[tuple[str, ...]] = (
    "aggregation_descent", "belief_heatmap", "belief_quality", "bnn_robustness", "complexity_scaling",
    "bounded_influence", "contamination_gallery", "cross_study_summary",
    "conditional_world",
    "descent_comparison", "disjoint_fov_world", "efe_decomposition",
    "emergence_bmr", "free_energy_comparison", "generative_model_schema",
    "graphical_abstract", "heuristic_breakdown", "hierarchical_bmr",
    "hierarchical_pomdp", "language_kl_decay", "message_passing",
    "moving_world", "parameter_recovery", "pomdp_loop",
    "robust_influence_weights", "robustness_onset", "robustness_review_grid", "robustness_sweep",
    "sensitivity_heatmap", "system_overview",
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
    "belief_heatmap": (
        "Heatmap of posterior probability over nine location cells for seven agent "
        "rows and a consensus row. Agent mass is dispersed, while the consensus "
        "row concentrates near one cell."
    ),
    "belief_quality": (
        "Two-panel control figure. The left panel compares oracle, uniform, and "
        "confidently-wrong mean categorical log scores; the right panel plots "
        "empirical accuracy against mean confidence with a perfect-calibration line."
    ),
    "bnn_robustness": (
        "Held-out accuracy curves for standard NLL/KLD and robust RCCE/AR client "
        "losses across per-client label-contamination fractions, with seed-level "
        "interval bands. The curves are similar at low contamination and separate "
        "at moderate contamination before both fall at the highest rate."
    ),
    "bounded_influence": (
        "Line plot of a variational server's normalized influence weight as one "
        "agent drifts toward a confident-wrong belief. The variational weight falls "
        "below the flat naive-pool reference; the plot is a tested-path diagnostic."
    ),
    "complexity_scaling": (
        "Four-panel log-log timing figure showing median wall-clock scaling with "
        "agents, states, and modalities for aggregation, self-excluding sharing, "
        "and state inference. Error bars show timing spans and dotted lines are "
        "normalized order guides."
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
        "Three vertically stacked horizontal native-unit summaries: accuracy gaps, "
        "information or free-energy changes, and parameter-recovery R-squared. "
        "Each study is shown with a seed-bootstrap interval; facets are not ranked "
        "across units."
    ),
    "descent_comparison": (
        "Line plot comparing single-start and multi-start variational free-energy "
        "descent. The single start remains in a higher local basin, while the "
        "multi-start run reaches a lower basin after fewer iterations."
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
        "Two bars show Bayesian model-reduction free-energy differences for pruning "
        "a redundant versus a supported likelihood column. The redundant-pruning "
        "control is positive and the supported-column control is negative."
    ),
    "free_energy_comparison": (
        "Per-seed free-energy points over communicating and incommunicado colony "
        "conditions, with mean bars and across-seed spread. The communicating mean "
        "is lower than the incommunicado mean."
    ),
    "generative_model_schema": (
        "Four-panel categorical generative-model schematic: private observation, "
        "A/B/C/D-zero factors feeding a local posterior, temporal state-observation-"
        "action order, and optional hierarchical context."
    ),
    "graphical_abstract": (
        "Wide schematic linking private categorical beliefs, contaminated broadcasts, "
        "a robust aggregation server, naive versus reweighted consensus, and three "
        "separate claim axes: client FedGVI, server heuristic, and variational server."
    ),
    "heuristic_breakdown": (
        "Three-panel server-heuristic diagnostic: numerical influence along a probed "
        "contamination path, finite colluder counts needed to capture consensus for "
        "two aggregators, and the fraction of a declared attack grid with capture."
    ),
    "hierarchical_bmr": (
        "Horizontal bars show Bayesian surprise at two non-leaf hierarchy levels for "
        "informative and degenerate worlds. The informative upper level is kept, "
        "whereas the degenerate upper level falls below the prune threshold."
    ),
    "hierarchical_pomdp": (
        "Six-panel hierarchical POMDP diagnostic. The top row shows two-level "
        "location, context, and colony beliefs; the bottom row shows three-level "
        "beliefs and the measured hierarchical-minus-flat accuracy gaps."
    ),
    "language_kl_decay": (
        "Seed-mean KL divergence from the true categorical likelihood to the learned "
        "likelihood across ordered learning batches. The curve decreases rapidly and "
        "then approaches zero, with a seed-bootstrap interval band."
    ),
    "message_passing": (
        "Three-lane protocol schematic: agents keep private observations, broadcast "
        "local categorical posteriors, and enter standard, heuristic, or variational "
        "server fusion. The bottom lane separates the ownership of each claim."
    ),
    "moving_world": (
        "Three-panel moving-world comparison of isolated, communicating, and "
        "expected-free-energy-guided conditions, showing accuracy, signed free-energy "
        "gap, and steps to consensus."
    ),
    "parameter_recovery": (
        "Two-panel sensor-acuity recovery diagnostic. The left scatter with empirical "
        "intervals follows the identity line; the right bars show absolute recovery "
        "error at each tested acuity."
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
        "Expanded finite robustness review. The left heatmap summarizes conditional "
        "robust-minus-naive contrasts for two adversarial weights; the three right "
        "panels show signed rate profiles for all predeclared methods under confident-"
        "wrong, byzantine, and drift attacks."
    ),
    "robustness_sweep": (
        "Consensus accuracy curves for naive KLD and four robust client operating "
        "points across contamination rates. Lines and markers show matched-trial means "
        "with 95-percent seed-bootstrap intervals; a horizontal line marks the "
        "predeclared accuracy floor."
    ),
    "sensitivity_heatmap": (
        "Two heatmaps of accuracy gaps over sensor acuity and colony size: "
        "communicating minus isolated on the left, and hierarchical minus flat on "
        "the right. A symmetric color scale is centered at zero and hatched cells "
        "mark near-zero gaps."
    ),
    "system_overview": (
        "Three-part schematic of a five-agent colony with two adversarial agents, "
        "naive equal-weight pooling, and heuristic reweighting. The naive argmax is "
        "off target, while the reweighted consensus recovers the displayed true state."
    ),
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
    for name, estimand, unit in (
        ("free_energy_comparison", "communicating minus incommunicado colony mean free energy", "nats"),
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
                    "across-seed standard-deviation spread; not a confidence interval"
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
        source_relation="original project native-unit summary",
        estimand="study-level mean metric grouped by declared native unit",
        unit="fraction, nats, or R-sq by facet",
        uncertainty="pointwise bootstrap confidence interval across independent seeds",
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
            "original project BMR structure-learning diagnostic",
            "per-level Bayesian surprise and prune/keep decision",
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
            "none; deterministic seeded colonies",
            "not applicable",
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
            "recovered acuity and absolute acuity error",
            "probability units",
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
            "per-cell federation accuracy gap over acuity and colony size",
            "fraction",
            "none; deterministic per-cell mean",
            "not applicable",
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
            "independent configured seed",
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
            "categorical log score, with Brier and reliability diagnostics as secondary measures",
            "nats, squared probability error, or fraction",
            "seed-level percentile bootstrap interval for controls",
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
            "acuity parameter recovery fit and absolute error",
            "R-sq or parameter units",
            "empirical percentile interval across independent trials",
            "independent configured seed",
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
            "pointwise bootstrap confidence interval",
            "independent configured seed",
        ),
        (
            "sensitivity_heatmap",
            "per-cell federation accuracy gap",
            "fraction",
            "deterministic per-cell mean",
            "not applicable",
        ),
    ):
        metadata[name].update(
            estimand=estimand,
            unit=unit,
            uncertainty=uncertainty,
            replication_unit=replication_unit,
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
    for name in ("graphical_abstract", "system_overview"):
        metadata[name].update(
            status="schematic",
            source_relation="original project schematic",
            estimand="project scope and component relationships",
            unit="conceptual components",
            uncertainty="none; conceptual schematic",
            replication_unit="not applicable",
        )
    return metadata


FIGURE_METADATA: Final[dict[str, dict[str, str]]] = _build_metadata()


def figure_metadata(generator: str) -> dict[str, str]:
    """Return a defensive copy of the metadata contract for *generator*."""
    if generator not in FIGURE_METADATA:
        raise KeyError(f"no figure metadata contract for generator {generator!r}")
    return deepcopy(FIGURE_METADATA[generator])


__all__ = ["FIGURE_METADATA", "figure_metadata"]
