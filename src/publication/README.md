# `src/publication/` — release and publication boundary

Packaging, identifiers, freshness, release assets, rendered-surface validation
and the Zenodo boundary. Publishing is an explicit reviewed action, never a
side effect of validation.

## Modules

| Area | Modules |
| --- | --- |
| Identity | `identifiers.py`, `metadata.py` |
| Freshness and validation | `pipeline_freshness.py`, `validation_receipt.py`, `clean_checkout.py` |
| Release assets | `release_assets.py`, `release_manifest.py` |
| Surfaces | `surface_validation.py`, `template_renderer.py`, `web_package.py` |
| Zenodo | `zenodo.py`, `zenodo_admission.py`, `zenodo_auth.py`, `zenodo_edits.py`, `zenodo_errors.py`, `zenodo_files.py`, `zenodo_lineage.py`, `zenodo_metadata.py`, `zenodo_models.py`, `zenodo_operations.py`, `zenodo_publish.py`, `zenodo_response.py` |
| Web helpers | `web/` |

## Boundaries

- DOI normalization and the version-bound manuscript filename live in
  `identifiers.py`. Do not duplicate either rule.
- `release_manifest.py` derives the manifest from actual bytes; the tracked
  release README and checksums are generated, never hand-edited.
- `surface_validation.py` checks rendered PDF, slide and web invariants. Passing
  it is not a WCAG or PDF/UA conformance claim.
- The Zenodo modules are typed boundary adapters. They never publish implicitly;
  a deposition update or publish is a separate reviewed operation.
- Tokens come from the caller's environment or an explicit caller-owned file.
  Artifacts and manifests carry no credentials.

## Check it

```bash
uv run --locked --extra dev pytest tests/test_publication_identifiers.py \
  tests/test_publication_metadata.py tests/test_release_assets.py \
  tests/test_release_manifest.py tests/test_surface_validation.py \
  tests/test_web_publication_contract.py tests/test_zenodo.py -q
```

Release rules: [`../../docs/reference/zenodo-release.md`](../../docs/reference/zenodo-release.md).
Parent contract: [../AGENTS.md](../AGENTS.md).
