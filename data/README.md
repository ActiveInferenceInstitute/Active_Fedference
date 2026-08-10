# active_fedference/data

Project-maintained **inputs** for the code exemplar (not pipeline outputs).

## Quick reference

| File | Role |
| --- | --- |
| `claim_ledger.yaml` | Active Fedference structural/display facts for evidence validation |
| `synthetic_tabular.csv` | Deterministic **synthetic** 3-class/4-feature tabular dataset (150 rows), a reproducible stand-in for a real external dataset used by the tabular-benchmark harness (`src/fedference/benchmark.py`) |

The installable copy lives at
`src/fedference/data/synthetic_tabular.csv` and is declared as package data.
The two byte streams must remain identical; wheel/sdist smoke tests execute the
default benchmark path so a missing packaged fixture fails CI.

External UCI archives are deliberately not committed here. Their DOI, CC BY
4.0 license, source URL, archive member, schema, preprocessing contract, and
SHA-256 live in `src/fedference/research_registry.py`. The installed benchmark
command downloads only into a caller-supplied cache, verifies the archive, and
records archive/member/split hashes in its report and receipt:

```bash
uv run --locked fedference benchmark \
  --dataset-id uci-banknote --profile smoke --seed 42 \
  --cache-dir .tmp/uci-cache --output-dir .tmp/banknote-smoke
```

### `synthetic_tabular.csv`

Generated deterministically from three well-separated 4-D Gaussian clusters
under a fixed seed (`np.random.default_rng(20240601)`); columns `f0,f1,f2,f3,label`
with `label` in `{0,1,2}`. It is **not** real-world data and carries no domain
claim — it exercises the compatibility harness offline. For a user-owned CSV
with numeric features and an integer `label` column, pass its path directly to
`fedference.benchmark.run_tabular_benchmark`; do not mix it into the registered
three-dataset evidence pack without a new source/license/hash declaration.

Generated analysis outputs belong under `output/` during pipeline runs, not here.

Schema and edit protocol: [`AGENTS.md`](AGENTS.md).
