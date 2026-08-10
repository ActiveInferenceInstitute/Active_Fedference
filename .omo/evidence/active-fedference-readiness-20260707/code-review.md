# ULW Code Review Gate

Scope: final publication-readiness diff across manuscript prose, bibliography,
TODO scope, CI/package validation, output generation logic, generated artifacts,
and ULW evidence.

Finding: approve. No publication-blocking code, documentation, provenance, or
claim-boundary defects remain after the final pass.

Evidence:

- `src/analysis/workflow.py` now resolves figure captions from generated
  manuscript output when available, preventing raw manuscript tokens from
  leaking into figure registries.
- `tests/analysis/test_workflow.py` covers resolved captions and rejects raw
  token leakage in the validator-compatible registry.
- `tests/test_docs_contract.py` adds forward-only TODO and CI package/release
  guardrails; the focused contract suite passed after the final TODO count edit.
- The bibliography change is claim-relevant and cited in live prose; extra
  Perplexity leads were rejected as padding.
- The remaining TODO scope is intentionally limited to six Major scientific
  upgrades.

Residual risk: one existing PyTorch scalar conversion warning appears in the
full test run. It is not a publication blocker.
