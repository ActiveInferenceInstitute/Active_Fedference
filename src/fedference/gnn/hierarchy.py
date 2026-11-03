"""Native hierarchy mapping, explicit overrides and empirical-prior messages.

Friston et al. (2024) motivates belief sharing; no full protocol equivalence.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from numpy.typing import NDArray

from .admission import GNNAdmissionError
from .native import NativeWorld, native_world
from .structured_admission import StructuredProgram, _array, _native
from .types import CategoricalModel, HierarchicalModel, Tensor

if TYPE_CHECKING:
    from ..pomdp import LayerSpec


class HierarchyWorld(TypedDict):
    L1: NativeWorld
    layers: list[LayerSpec]
    n_levels: int
    level_priors: list[NDArray[np.float64]]
    level_transitions: list[NDArray[np.float64]]
    conditioned_priors: list[list[NDArray[np.float64]]]
    model_digest: str



@dataclass(frozen=True, slots=True)
class HierarchyParameters:
    """Explicit source-preserving hierarchy overrides; no PMF normalization."""

    depth: int = 4
    acuity: float | None = None
    goal_bonus: float | None = None
    priors: tuple[tuple[float, ...] | None, ...] | None = None
    conditioned_priors: tuple[tuple[tuple[float, ...], ...] | None, ...] | None = None



def hierarchy_leaf(program: StructuredProgram) -> NativeWorld:
    """Expose the source leaf channel through the shared categorical adapter."""
    _native(program)
    if not isinstance(program.model, HierarchicalModel):
        raise GNNAdmissionError("GNN_KIND", "hierarchy")
    metadata = replace(program.model.metadata, capabilities=("fedference.gnn/categorical/1",))
    return native_world(CategoricalModel(metadata, program.model.layers[-1].categorical))



def hierarchy_depth(program: StructuredProgram, depth: int) -> StructuredProgram:
    """Select a source-defined suffix; its new root keeps its declared default prior."""
    _native(program)
    model = program.model
    if (
        not isinstance(model, HierarchicalModel)
        or type(depth) is not int
        or not 2 <= depth <= len(model.layers)
    ):
        raise GNNAdmissionError("GNN_LAYERS", "depth")
    layers = model.layers[-depth:]
    model = replace(model, layers=(replace(layers[0], parent=None, conditioned_prior=None), *layers[1:]))
    result = replace(program, model=model)
    _native(result)
    return result



def hierarchy_world(program: StructuredProgram) -> HierarchyWorld:
    """Map source levels/conditional edges to the existing native inference contract."""
    from ..pomdp import LayerSpec

    _native(program)
    model = program.model
    if not isinstance(model, HierarchicalModel):
        raise GNNAdmissionError("GNN_KIND", "hierarchy")
    conditioned = []
    for layer in model.layers[1:]:
        assert layer.conditioned_prior is not None
        # Native factories expose owned, contiguous parent-PMF vectors.
        # Retaining transposed views selects a different strided dot-product
        # path on some NumPy/BLAS builds, breaking exact native parity.
        conditioned.append([prior.copy() for prior in _array(layer.conditioned_prior).T])
    priors = [_array(layer.categorical.D) for layer in model.layers]
    layers = [
        LayerSpec(
            n_states=len(layer.categorical.states),
            labels=layer.categorical.states,
            default_prior=priors[i],
            conditioned_priors=conditioned[i] if i < len(conditioned) else None,
        )
        for i, layer in enumerate(model.layers)
    ]
    return {
        "L1": hierarchy_leaf(program),
        "layers": layers,
        "n_levels": len(layers),
        "level_priors": priors,
        "level_transitions": [_array(layer.categorical.B) for layer in model.layers],
        "conditioned_priors": conditioned,
        "model_digest": program.digest,
    }



def infer_hierarchy(
    program: StructuredProgram, observation: int, *, n_iters: int = 4
) -> tuple[NDArray[np.float64], ...]:
    """Execute native message passing with the admitted source likelihood and gates."""
    from ..pomdp import nlevel_infer

    world = hierarchy_world(program)
    return tuple(nlevel_infer(world["L1"]["A"][0], observation, dict(world), n_iters=n_iters)["q_levels"])



def parameterize_hierarchy(program: StructuredProgram, parameters: HierarchyParameters) -> StructuredProgram:
    """Bind all declared overrides into the admitted IR and its digest."""
    from .native import SentinelParameters, parameterize_sentinel

    selected = hierarchy_depth(program, parameters.depth)
    model = selected.model
    assert isinstance(model, HierarchicalModel)
    layers = list(model.layers)
    if parameters.priors is not None:
        if len(parameters.priors) != len(layers):
            raise GNNAdmissionError("GNN_LAYERS", "prior overrides")
        for i, prior in enumerate(parameters.priors):
            if prior is not None:
                c = layers[i].categorical
                layers[i] = replace(layers[i], categorical=replace(c, D=Tensor(c.D.axes, c.D.shape, prior)))
    if parameters.conditioned_priors is not None:
        if len(parameters.conditioned_priors) != len(layers) - 1:
            raise GNNAdmissionError("GNN_LAYERS", "conditioned overrides")
        for i, conditioned_prior in enumerate(parameters.conditioned_priors, 1):
            if conditioned_prior is not None:
                cp = layers[i].conditioned_prior
                assert cp is not None
                array = np.asarray(conditioned_prior, dtype=np.float64).T
                layers[i] = replace(
                    layers[i], conditioned_prior=Tensor(cp.axes, array.shape, tuple(array.ravel()))
                )
    leaf = parameterize_sentinel(
        CategoricalModel(
            replace(model.metadata, capabilities=("fedference.gnn/categorical/1",)), layers[-1].categorical
        ),
        SentinelParameters(parameters.acuity, parameters.goal_bonus),
    )
    layers[-1] = replace(layers[-1], categorical=leaf.categorical)
    result = replace(selected, model=replace(model, layers=tuple(layers)))
    _native(result)
    return result
