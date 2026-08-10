# Zenodo release boundary

Active Fedference has one production Zenodo deposition:

- deposition: `21864004`
- DOI: [`10.5281/zenodo.21864004`](https://doi.org/10.5281/zenodo.21864004)
- public record: <https://zenodo.org/records/21864004>

The DOI is the permanent identifier for the public release. The deposited PDF
and the public GitHub repository
([`ActiveInferenceInstitute/Active_Fedference`](https://github.com/ActiveInferenceInstitute/Active_Fedference))
cross-reference each other through this release metadata. The official API
boundary is documented in the [Zenodo REST API documentation](https://developers.zenodo.org/).

## Source of truth

`manuscript/config.yaml` owns the DOI. The metadata emitter propagates it to
`CITATION.cff`, `.zenodo.json`, `codemeta.json`, and the manuscript token
`{{PUBLICATION_DOI}}`. `src/publication/identifiers.py` provides the shared
normalization contract; `src/publication/zenodo.py` provides the typed,
standard-library client; and `scripts/zenodo_release.py` is the thin CLI
boundary. The token is read from an ignored dotenv file or process environment
and is never committed, printed, or included in the release manifest.

## Metadata and upload verification

Run these commands from the repository root after the source-current analysis,
test, hydration, render, web, and release gates have passed:

```bash
ENV_FILE=/Users/mini/Documents/GitHub/template/.env

uv run python scripts/emit_metadata.py --check
uv run python scripts/zenodo_release.py \
  --env-file "$ENV_FILE" \
  --deposition-id 21864004 \
  --update-metadata
uv run python scripts/zenodo_release.py \
  --env-file "$ENV_FILE" \
  --deposition-id 21864004 \
  --upload output/pdf/active_fedference_combined.pdf
uv run python scripts/zenodo_release.py \
  --env-file "$ENV_FILE" \
  --deposition-id 21864004 \
  --verify output/pdf/active_fedference_combined.pdf
```

The upload command is intended to run once for the final PDF filename. If the
deposition already contains that filename, inspect its checksum before any
replacement. `--reserve` creates a new draft and must not be used for this
release.

## Publication gate

Publishing is the only irreversible operation in this adapter:

```bash
uv run python scripts/zenodo_release.py \
  --env-file "$ENV_FILE" \
  --deposition-id 21864004 \
  --publish --confirm-publish
```

Run it only after final PDF review, metadata review, licence/author approval,
and the GitHub release decision. After publication, verify both the DOI
redirect and the public record metadata, including the GitHub related
identifier and uploaded-PDF checksum.

## Invariants

- The DOI in `manuscript/config.yaml`, generated metadata, manuscript token,
  rendered PDF, and Zenodo record must agree.
- The uploaded PDF must be generated after the final source and test gates;
  checksum verification does not substitute for manuscript or scientific
  review.
- The HTML surface remains the accessibility-enhanced canonical reader. The
  current untagged PDF is structurally and visually checked, but no PDF/UA
  conformance claim is made.
- Zenodo receipts do not replace clean-clone evidence, the public GitHub push,
  licence/confidentiality/author approval, or any DOI/publisher policy review.
- A DOI reservation, file upload, or publication never promotes null, reversed, failed, or
  underpowered research outcomes into claims.
