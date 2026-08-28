# Active Fedference — release bundle provenance

Generated at: omitted for a byte-reproducible unreleased build.
(invoked via `uv run --locked python scripts/build_release.py`).

Pipeline profile: `publication`; generator version: `6`.

Artifact scope: `publication-payload-v1`.

Artifacts: 342 files, 46537684 bytes, over:

- `.zenodo.json`: 1 file(s)
- `CITATION.cff`: 1 file(s)
- `LICENSE`: 1 file(s)
- `codemeta.json`: 1 file(s)
- `output/data`: 6 file(s)
- `output/docs`: 2 file(s)
- `output/figures`: 61 file(s)
- `output/manuscript`: 46 file(s)
- `output/pdf`: 4 file(s)
- `output/reports`: 25 file(s)
- `output/slides`: 87 file(s)
- `output/web`: 107 file(s)

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
