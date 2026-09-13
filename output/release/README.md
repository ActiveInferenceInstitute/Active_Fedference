# Active Fedference — release bundle provenance

Generated at: 2026-09-13T01:42:53Z by `src/publication/release_manifest.py`
(invoked via `uv run --locked python scripts/build_release.py`).

Pipeline profile: `publication`; generator version: `6`.

Artifact scope: `publication-payload-v1`.

Artifacts: 2723 files, 149884519 bytes, over:

- `.zenodo.json`: 1 file(s)
- `CITATION.cff`: 1 file(s)
- `LICENSE`: 1 file(s)
- `codemeta.json`: 1 file(s)
- `output/data`: 5 file(s)
- `output/docs`: 2 file(s)
- `output/figures`: 1226 file(s)
- `output/manuscript`: 47 file(s)
- `output/pdf`: 4 file(s)
- `output/reports`: 29 file(s)
- `output/slides`: 133 file(s)
- `output/web`: 1273 file(s)

Verify integrity from the project root:

```bash
shasum -a 256 -c output/release/sha256sums.txt
# or: uv run --locked python scripts/build_release.py --verify
```

Scientific reports are regenerated under the seeds in
`manuscript/config.yaml`; rendered containers are rebuilt by the pinned
publication toolchain. The manifest is re-derived on each build and
records the actual bytes; it is never hand-edited.

Downstream artifact-manifest, evidence/statistics, validation, and
rendered-provenance controls are intentionally outside this payload
manifest so they can bind these release bytes without a checksum cycle.
They remain mandatory and are verified by their dedicated gates.
