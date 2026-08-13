# Zenodo release boundary

Active Fedference v1.0.1 is the current public release:

- prior deposition: `21864004` (v0.1.0)
- current deposition: `21919307` (v1.0.1)
- DOI: [`10.5281/zenodo.21919307`](https://doi.org/10.5281/zenodo.21919307)
- public record: <https://zenodo.org/records/21919307>
- state: **published** (`done`)
- released PDF: `active_fedference_combined.pdf` (the same bytes as the
  top-level GitHub manuscript PDF)

The v1.0.1 DOI is the permanent identifier for the current release. Its PDF
and the public GitHub repository
([`ActiveInferenceInstitute/Active_Fedference`](https://github.com/ActiveInferenceInstitute/Active_Fedference))
cross-reference each other through the release metadata. The v0.1.0 record
remains available as the prior version. The official API boundary is
documented in the [Zenodo REST API documentation](https://developers.zenodo.org/).

## Source of truth

`manuscript/config.yaml` owns the DOI. The metadata emitter propagates it to
`CITATION.cff`, `.zenodo.json`, `codemeta.json`, and the manuscript token
`{{PUBLICATION_DOI}}`. `src/publication/identifiers.py` provides the shared
normalization contract; `src/publication/zenodo.py` provides the typed,
standard-library client; and `scripts/zenodo_release.py` is the thin CLI
boundary. The token is read from an ignored dotenv file or process environment
and is never committed, printed, or included in the release manifest.

## Creating the next version

Never update or upload against a published deposition. Zenodo's `newversion`
action creates a linked unpublished draft, preserves the concept record, and
inherits the prior metadata and files. The CLI resolves Zenodo's
`latest_draft` link and exposes inherited-file replacement explicitly:

```bash
ENV_FILE="/path/to/ignored/zenodo.env"
SOURCE_ID="21919307"  # latest published deposition, not the global concept id

uv run --locked python scripts/zenodo_release.py \
  --env-file "$ENV_FILE" \
  --new-version-of "$SOURCE_ID"
```

Record the returned draft id and reserved DOI before changing
`manuscript/config.yaml`. Then emit metadata, regenerate the complete
source-bound analysis/hydration/render chain, and run the release checks.

## Metadata and upload verification

Run these commands only after the source-current test, analysis, hydration,
render, web, and release gates have passed:

```bash
DRAFT_ID="<unsubmitted-draft-id>"

uv run --locked python scripts/emit_metadata.py --check
uv run --locked python scripts/zenodo_release.py \
  --env-file "$ENV_FILE" \
  --deposition-id "$DRAFT_ID" \
  --update-metadata
uv run --locked python scripts/zenodo_release.py \
  --env-file "$ENV_FILE" \
  --deposition-id "$DRAFT_ID" \
  --upload output/pdf/active_fedference_combined.pdf \
  --replace-existing
uv run --locked python scripts/zenodo_release.py \
  --env-file "$ENV_FILE" \
  --deposition-id "$DRAFT_ID" \
  --verify output/pdf/active_fedference_combined.pdf
```

`--replace-existing` is required only when a new-version draft inherited a
same-named prior file whose checksum differs. The adapter refuses that case by
default, and it never deletes or replaces a file on a published deposition.

## Publication gate

Publishing is the only irreversible operation in this adapter:

```bash
uv run --locked python scripts/zenodo_release.py \
  --env-file "$ENV_FILE" \
  --deposition-id "$DRAFT_ID" \
  --verify output/pdf/active_fedference_combined.pdf \
  --publish --confirm-publish
```

Run it only after final PDF review, metadata review, licence/author approval,
and the GitHub release decision. The publish command requires the checksum
verification flag and the draft-state guard. After publication, verify the DOI
redirect and public record metadata, including the GitHub related identifier
and uploaded-PDF checksum.

For the current public record, the no-token checks are:

```bash
curl -fsSIL https://doi.org/10.5281/zenodo.21919307
curl -fsSL https://zenodo.org/api/records/21919307 | jq \
  '{doi, version: .metadata.version, files: [.files[] | {key, size, checksum}], related_identifiers: .metadata.related_identifiers}'
```

The prior v0.1.0 record remains independently checkable:

```bash
curl -fsSL https://zenodo.org/api/records/21864004 | jq \
  '{doi, version: .metadata.version, files: [.files[] | {key, size, checksum}]}'
```

## Invariants

- The v1.0.1 DOI in `manuscript/config.yaml`, generated metadata, manuscript
  token, rendered PDF, README, and Zenodo record must agree.
- The uploaded PDF must be generated after the final source and test gates;
  checksum verification does not substitute for manuscript or scientific
  review.
- The HTML surface remains the accessibility-enhanced canonical reader. The
  current untagged PDF is structurally and visually checked, but no PDF/UA
  conformance claim is made.
- Zenodo receipts do not replace clean-clone evidence, the public GitHub push,
  licence/confidentiality/author approval, or any DOI/publisher policy review
  for a future version.
- A DOI reservation, file upload, or publication never promotes null, reversed,
  failed, or underpowered research outcomes into claims.
