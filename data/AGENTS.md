# Data Directory — Agent Guide

Versioned project **inputs** only. Pipeline outputs must not be committed here.

## Input inventory

| File | Contract |
| --- | --- |
| `claim_ledger.yaml` | Source-tiered structural/display facts used by evidence validation |
| `synthetic_tabular.csv` | Deterministic compatibility fixture; synthetic smoke data, never external-domain evidence |

`synthetic_tabular.csv` must remain byte-identical to
`../src/fedference/data/synthetic_tabular.csv`, the copy packaged in wheels and
source distributions. External UCI archives never belong here: their
source/license/hash declarations live in the research registry, while acquired
bytes remain in a caller-owned cache.

## `claim_ledger.yaml`

Evidence-registry for manuscript claims that are intentionally sourced from
manuscript structure, figure display conventions, code, examples, or generated
reports rather than `{{VARIABLE}}` injection.

### Schema (preserve when adding rows)

| Field | Purpose |
| --- | --- |
| `claim_id` | Stable identifier |
| `kind` | Claim category |
| `value` | Declared numeric or textual value |
| `source` | Provenance (module, manuscript section, artifact) |
| `source_tier` | Trust tier for validation |
| `freshness` | Staleness policy |
| `artifact_path` | Optional path to backing file |

## Edit protocol

1. Edit only when manuscript claims, figure defaults, or source-backed numeric facts change.
2. Re-run evidence validation / pipeline stages that consume the ledger.
3. Do not store generated CSV/JSON/PNG or downloaded external archives under
   `data/`.
4. If the synthetic fixture changes, update both copies intentionally and run
   the package plus benchmark smoke tests.

Quick orientation: [`README.md`](README.md).
