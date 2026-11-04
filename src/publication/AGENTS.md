# AGENTS.md — `src/publication/`

- Never publish implicitly. Importing a module, validating a surface or
  building a manifest must not create, update or publish a Zenodo deposition.
- Keep DOI normalization and the version-bound PDF filename in
  `identifiers.py`; do not add a second formatting rule in a script or test.
- The release manifest is derived from actual bytes and is never hand-edited.
  Regenerate it with the owning tool and verify with
  `uv run --locked python scripts/build_release.py --verify`.
- Surface validation is structural. Do not describe a pass as WCAG or PDF/UA
  conformance.
- Keep credentials out of artifacts, manifests and logs. Read tokens from the
  environment or an explicit caller-owned path.
- Keep web validation deterministic and offline; no browser download or network
  fetch at this boundary.
- A new release action needs an explicit operation, a typed error path and a
  test that proves the no-op path does not write.
- Parent rules: [../AGENTS.md](../AGENTS.md).
