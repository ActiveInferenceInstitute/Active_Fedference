# `src/manuscript_vars/` — manuscript token generation

Deterministic `{{TOKEN}}` production and Markdown hydration for the manuscript.
Every numeric value in prose resolves through this package; nothing is
hardcoded in a section.

## Modules

| Module | Role |
| --- | --- |
| `generate.py` | Builds the token map from analysis reports and configuration. |
| `loaders.py` | Reads the validated report/config inputs. |
| `tokens.py` | Token groups and formatting rules. |
| `render.py` | Hydrates a manuscript tree from the token map. |

## Pipeline

`generate_variables()` reads the analysis JSON reports and
`manuscript/config.yaml`, then writes `output/data/manuscript_variables.json`.
Final non-draft hydration derives `ISC_TOTAL`/`ISC_PASSED` from `ISA.md` and
loads `TEST_COUNT`, `COVERAGE_PERCENT` and core environment tokens from the
fresh test-coverage receipt. Hydration never re-collects the test suite while
rendering.

`manuscript_vars.render.render_manuscript_tree()` substitutes the tokens into
the Markdown sections. The label registry is
[`../../manuscript/SYNTAX.md`](../../manuscript/SYNTAX.md); the rules are in
[`../../docs/manuscript/tokens-and-labels.md`](../../docs/manuscript/tokens-and-labels.md).

## Check it

```bash
uv run --locked --extra dev pytest tests/test_manuscript_variables.py \
  tests/test_token_provenance.py tests/test_token_tables.py \
  tests/test_token_sourcing.py -q
```

Parent contract: [../AGENTS.md](../AGENTS.md).
