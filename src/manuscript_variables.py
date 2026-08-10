"""Manuscript variable generation — public entry point.

Implementation lives in :mod:`manuscript_vars` (split for maintainability).
"""

from manuscript_vars import generate_variables, render_manuscript_tree, save_variables
from manuscript_vars.loaders import (
    _bnn_robustness_variables,
    _bnn_torch_variables,
    _count_tests,
    _disjoint_fov_variables,
    _format_residual,
    _format_residual_math,
    _recovery_residuals,
)
from manuscript_vars.tokens import _format_cohens_d, _format_d_equivalent

__all__ = [
    "_bnn_robustness_variables",
    "_bnn_torch_variables",
    "_count_tests",
    "_disjoint_fov_variables",
    "_format_cohens_d",
    "_format_d_equivalent",
    "_format_residual",
    "_format_residual_math",
    "_recovery_residuals",
    "generate_variables",
    "render_manuscript_tree",
    "save_variables",
]
