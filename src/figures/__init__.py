"""Figure generators for the Active Fedference project (re-export barrel).

Headless (Agg) matplotlib figures backing the manuscript, all sharing one
visual language and provenance contract via :mod:`figures._common` and
:mod:`figures._metadata`:

* :func:`generate_belief_heatmap` — colony posteriors over the shared grid (Fig. 1/4);
* :func:`generate_free_energy_comparison` — communicating vs incommunicado F (source-mechanism analogue);
* :func:`generate_robustness_sweep` — consensus accuracy vs contamination (ISC-27/30);
* :func:`generate_language_kl_decay` — seed-aggregated language-acquisition KL
  curve (source-mechanism analogue);
* :func:`generate_emergence_bmr` — model-reduction emergence contrast (source-mechanism analogue);
* :func:`generate_efe_decomposition` — EFE risk/ambiguity == -(pragmatic+epistemic) identity (Eq. 2);
* :func:`generate_robust_influence_weights` — per-agent robust pooling weights (heuristic);
* :func:`generate_bnn_robustness` — logistic-regression accuracy vs label contamination;
* :func:`generate_aggregation_descent` — variational free-energy descent (axis-3 rigor);
* :func:`generate_bounded_influence` — outlier weight vs divergence diagnostic;
* :func:`generate_contamination_gallery` — robust vs naive across attack mechanisms;
* :func:`generate_descent_comparison` — single-start capture vs multi-start escape;
* :func:`generate_robustness_onset` — naive vs robust accuracy vs rate, per mechanism;
* :func:`generate_sensitivity_heatmap` — 2D sweep of federation benefit over acuity × colony size (Study 8);
* :func:`generate_complexity_scaling` — implementation-derived asymptotic
  orders and machine scaling diagnostics;
* :func:`generate_cross_study_summary` — cross-study federation benefit overview with 95 % CI;
* :func:`generate_hierarchical_pomdp` — 2x3 six-panel belief dynamics for
  2-level and 3-level hierarchical POMDP (Studies 6-7);
* :func:`generate_parameter_recovery` — recovered vs true acuity (Study 9);
* :func:`generate_disjoint_fov_figure` — disjoint-FOV communication necessity and EFE navigation benefit (V4).
* :func:`generate_conditional_world` — finite conditional-world and attack-geometry grid.
* :func:`generate_belief_quality` — proper-score controls and reliability diagnostic.
* :func:`generate_generative_model_schema` — temporal, hierarchical, and
  factorial categorical model schematic.
* :func:`generate_message_passing` — belief-sharing message path and claim-ownership map.
* :func:`generate_pomdp_loop` — hidden-state, observation, action, and federation-loop schematic.
"""

from __future__ import annotations

from ._common import apply_style, figures_dir, robust_color
from ._metadata import FIGURE_METADATA, figure_metadata
from .aggregation_descent import generate_aggregation_descent
from .belief_heatmap import generate_belief_heatmap
from .belief_quality import generate_belief_quality
from .bnn_robustness import generate_bnn_robustness
from .bounded_influence import generate_bounded_influence
from .complexity_scaling import generate_complexity_scaling
from .conditional_world import generate_conditional_world
from .contamination_gallery import generate_contamination_gallery
from .cross_study_summary import generate_cross_study_summary
from .descent_comparison import generate_descent_comparison
from .disjoint_fov_world import generate_disjoint_fov_figure
from .efe_decomposition import generate_efe_decomposition
from .emergence_bmr import generate_emergence_bmr
from .free_energy_comparison import generate_free_energy_comparison
from .generative_model_schema import generate_generative_model_schema
from .graphical_abstract import generate_graphical_abstract
from .heuristic_breakdown import generate_heuristic_breakdown
from .hierarchical_bmr import generate_hierarchical_bmr
from .hierarchical_pomdp import generate_hierarchical_pomdp
from .language_kl_decay import generate_language_kl_decay
from .message_passing import generate_message_passing
from .moving_world import generate_moving_world
from .parameter_recovery import generate_parameter_recovery
from .pomdp_loop import generate_pomdp_loop
from .robust_influence_weights import generate_robust_influence_weights
from .robustness_onset import generate_robustness_onset
from .robustness_review_grid import generate_robustness_review_grid
from .robustness_sweep import generate_robustness_sweep
from .sensitivity_heatmap import generate_sensitivity_heatmap

__all__ = [
    "apply_style",
    "FIGURE_METADATA",
    "figures_dir",
    "figure_metadata",
    "generate_aggregation_descent",
    "generate_disjoint_fov_figure",
    "generate_belief_heatmap",
    "generate_belief_quality",
    "generate_bnn_robustness",
    "generate_bounded_influence",
    "generate_contamination_gallery",
    "generate_complexity_scaling",
    "generate_conditional_world",
    "generate_cross_study_summary",
    "generate_descent_comparison",
    "generate_efe_decomposition",
    "generate_hierarchical_pomdp",
    "generate_emergence_bmr",
    "generate_heuristic_breakdown",
    "generate_hierarchical_bmr",
    "generate_free_energy_comparison",
    "generate_graphical_abstract",
    "generate_generative_model_schema",
    "generate_language_kl_decay",
    "generate_moving_world",
    "generate_message_passing",
    "generate_parameter_recovery",
    "generate_pomdp_loop",
    "generate_robust_influence_weights",
    "generate_robustness_onset",
    "generate_robustness_review_grid",
    "generate_robustness_sweep",
    "generate_sensitivity_heatmap",
    "robust_color",
]
