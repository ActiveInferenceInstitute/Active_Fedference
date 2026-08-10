# Source Code Style Guide

Source conventions for Active Fedference are intentionally strict because the
project mixes numerical invariants, statistical evidence, and manuscript-gated
reproducibility.

## Design priorities

- Reproducibility over convenience: every stochastic path is threaded through
  explicit seeds.
- Small pure functions in `src/fedference/` with explicit parameter names.
- Deterministic testability for every core behavior.
- Typed public interfaces and explicit validation boundaries.

## Core style

- Use `from __future__ import annotations` and explicit type annotations for new
  public functions.
- Prefer composition and small helper functions over long scripts.
- Keep project invariants close to implementation and validate assumptions at
  function boundaries.

## Domain-specific rules

- Keep all active-inference mechanics in `src/fedference/` modules.
- No infrastructure imports inside `src/fedference/`.
- Use `np.random.default_rng(seed)` for all pseudo-random generation.
- Preserve numeric semantics when refactoring: if a value is a derived constant,
  keep stable tolerance bands explicit.

## Error handling

- Raise `ValueError` for invalid input combinations.
- Keep raised messages concrete and actionable.
- Avoid broad `except Exception` in core numerical pathways.

## Docstrings

- Public API functions in `src/fedference/` should include argument and return
  expectations.
- Keep tests and docs as the behavioral contract for subtle mathematical
  semantics.

## Testing discipline

- Zero-mock policy remains in force for all behavior that drives results.
- Prefer deterministic fixtures and explicit numeric probes over fuzzy property
  checks where claims are numerically grounded.

## See also

- [AGENTS.md](AGENTS.md)
- [`../tests/PATTERNS.md`](../tests/PATTERNS.md)
- [`../docs/development/style_guide.md`](../docs/development/style_guide.md)
