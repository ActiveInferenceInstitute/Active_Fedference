# AGENTS.md — `src/fedference/`

Read [README.md](README.md) first. Parent rules: [../AGENTS.md](../AGENTS.md).
Architecture: [`../../docs/core/architecture.md`](../../docs/core/architecture.md).

- Keep mathematical primitives side-effect free. Only the named boundary
  adapters may perform caller-authorized I/O, always with an explicit path.
- Never import `infrastructure` anywhere under this package. The layer check is
  `! grep -rn "import infrastructure" src/fedference/`.
- Never make Torch a default import. `bnn_baseline_torch.py` and
  `bnn_variational_torch.py` stay behind the `bnn` extra.
- Keep `__init__.py` and `__all__` synchronized with the stable public surface.
  A new export needs a test that uses it through the public name.
- New domain math goes in a module here, with a mirrored
  `tests/fedference/test_<module>.py`. Zero mocks; seeded NumPy; coverage stays
  at or above 90%.
- Keep the claim boundary: `robust_aggregate(robustness=0)` is bit-identical to
  `log_linear_pool`, and that identity is project-local. Do not describe the
  code as a reconstruction of the full Friston et al. protocol.
- Subpackage rules live in [gnn/AGENTS.md](gnn/AGENTS.md),
  [evidence_contracts/AGENTS.md](evidence_contracts/AGENTS.md),
  [federation/AGENTS.md](federation/AGENTS.md) and
  [experiments/AGENTS.md](experiments/AGENTS.md).
- Do not add prose under [models/](models/) or [data/](data/) beyond their
  existing README: the model files are exact-name validator inputs.
