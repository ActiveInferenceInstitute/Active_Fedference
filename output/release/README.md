# Active Fedference — release bundle provenance

Generated at: 2026-08-09T04:56:18Z by `src/publication/release_manifest.py`
(invoked via `uv run python scripts/build_release.py`).

Pipeline profile: `publication`; generator version: `4`.

Artifacts: 475 files, 44350310 bytes, over:

- `.zenodo.json`: 1 file(s)
- `CITATION.cff`: 1 file(s)
- `codemeta.json`: 1 file(s)
- `output/data`: 6 file(s)
- `output/docs`: 2 file(s)
- `output/figures`: 61 file(s)
- `output/manuscript`: 46 file(s)
- `output/pdf`: 4 file(s)
- `output/reports`: 35 file(s)
- `output/slides`: 211 file(s)
- `output/web`: 107 file(s)

Verify integrity from the project root:

```bash
shasum -a 256 -c output/release/sha256sums.txt
# or: uv run python scripts/build_release.py --verify
```

Scientific reports are regenerated under the seeds in
`manuscript/config.yaml`; rendered containers are rebuilt by the pinned
publication toolchain. The manifest is re-derived on each build and
records the actual bytes; it is never hand-edited.
