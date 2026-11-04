# `src/fedference/evidence_contracts/` — typed receipt boundary

Versioned schemas, verification, custody and strict JSON for run and application
receipts. This package decides whether a declared artifact set and its
provenance are internally consistent; it does not decide whether a scientific
claim is true.

## Modules

| Area | Modules |
| --- | --- |
| Schema and strict parsing | `schema.py`, `run_schema.py`, `application_schema.py`, `strict_json.py` |
| Verification | `run_verification.py`, `application_verification.py`, `source_verification.py` |
| Custody and hashing | `custody.py`, `hashing.py` |
| Declarations | `specifications.py` |

## Contract

- Receipts bind declared inputs, outputs, source revision and provenance. A
  receipt proves integrity of the declared bytes, not scientific validity or a
  downstream decision.
- Strict JSON rejects duplicate keys and non-finite constants. Schema validation
  rejects unexpected fields. Parsing is fail-closed.
- Custody writes are atomic and no-clobber; a failed write leaves the prior
  complete artifact in place.
- No API tokens, credentials or machine-local paths belong in a receipt.

## Check it

```bash
uv run --locked --extra dev pytest tests/fedference/test_evidence.py \
  tests/fedference/test_application.py \
  tests/fedference/test_application_provenance_edges.py -q
```

Parent contract: [../AGENTS.md](../AGENTS.md).
