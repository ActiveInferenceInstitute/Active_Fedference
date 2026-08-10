# Runtime-surface and composability review

Review date: 2026-07-17. This follow-up attacks executable residue and module
composition after the 2026-07-16 FirstPrinciples/RedTeam review. It preserves
the distinction between a real implementation that is intentionally scoped and
an incomplete adapter.

## Verdict

The active runtime surface contains no retired placeholder or compatibility
markers after this pass, and no test-double APIs are used by runtime Python.
The one actual no-op was an empty `TYPE_CHECKING` block in the
variational Torch module. The federation boundary test was stale: it described
the package as a single-machine scaffold while a tested loopback-TCP adapter was
already exported. Both defects are fixed and guarded by tests.

## First-principles decomposition

The function of the reviewed surfaces is narrow:

1. Runtime modules must execute real domain, transport, figure, publication, or
   orchestration behavior.
2. Optional Torch models must compose with standard PyTorch tooling: parameter
   registration, `state_dict`, optimizers, and module nesting.
3. Transport documentation must describe every landed adapter while preserving
   the cross-host boundary.
4. A verifier must fail when a retired marker or forbidden test-double API enters
   the executable surface.

These are functional constraints. Naming conventions and the location of a
helper are soft constraints; they were changed where they obscured the actual
behavior.

## Findings and fixes

| Surface | Finding | Fix and negative control |
| --- | --- | --- |
| `bnn_variational_torch.py` | Empty `if TYPE_CHECKING: pass` was dead residue; the class manually managed parameters instead of being a composable module | Removed the no-op, made `VariationalMLP` an `nn.Module`, registered standard state, and added invalid-control tests |
| `bnn_baseline_torch.py` | Point-mass model had the same manual-parameter composition gap | Made `PointMassMLP` an `nn.Module`, used `self.parameters()` for Adam, and added state-dict/parameter tests |
| `federation` boundary test | Test name and assertion described a scaffold while `run_socket_round` was already exported | Test now asserts queue/process/loopback-TCP visibility and only rejects unlanded cross-host server symbols |
| Figure/release fixtures | Placeholder result and byte labels obscured that these are concrete deterministic fixtures | Renamed to fixture/minimal-byte terminology without changing the real computations |
| Runtime markers | No executable marker was guarding against retired placeholder terms or test-double APIs | Added `tests/test_runtime_surface.py` over `src/` and `scripts/` runtime Python |
| Optional token test | A report-availability skip could silently remove a source-of-truth check | Removed the skip; the committed reviewer reports are now a required contract |
| Hierarchical figure fallback | A no-report call could silently substitute an illustrative simulation for the measured accuracy-gap panel | Publication calls now require both executed reports; the seeded fallback requires the explicit `allow_illustrative_fallback=True` opt-in and is covered by a negative-control test |

## RedTeam oracle result

The verifier was incomplete before the patch: Ruff and mypy could certify the
manual Torch classes, and the old federation test could certify a stale boundary
because it did not assert the exported socket adapter. The new negative controls
are:

- `test_point_mass_mlp_is_composable_module`;
- `test_variational_mlp_is_composable_module`;
- `test_variational_mlp_rejects_invalid_configuration`;
- `test_hierarchical_pomdp_requires_executed_reports_by_default`; and
- the complete `src/` and `scripts/` marker/API scans in `test_runtime_surface`.
- `test_federation_transport_boundary_is_explicit_not_cross_host`; and
- `test_runtime_surface` marker/API scans.

## Remaining boundary

This pass does not claim a paper-faithful FedGVI server, GPU-scale experiment,
cross-host deployment, TLS operations, or fault-tolerant federation. Those remain
open in MAJ-2 and MAJ-4 under [`TODO.md`](../../TODO.md). The Torch classes are
now standard composable modules, but that API improvement does not change the
scientific scope of their executed experiments.
