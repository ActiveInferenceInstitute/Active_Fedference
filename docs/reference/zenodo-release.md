# Zenodo release boundary

Active Fedference has one production Zenodo deposition:

- deposition: `21864004`
- DOI: [`10.5281/zenodo.21864004`](https://doi.org/10.5281/zenodo.21864004)
- public record: <https://zenodo.org/records/21864004>
- state: **published** (`done`)
- released PDF: `active_fedference_combined.pdf` (the same bytes as the
  top-level GitHub manuscript PDF)

The DOI is the permanent identifier for the public release. The deposited PDF
and the public GitHub repository
([`ActiveInferenceInstitute/Active_Fedference`](https://github.com/ActiveInferenceInstitute/Active_Fedference))
cross-reference each other through this release metadata. The official API
boundary is documented in the [Zenodo REST API documentation](https://developers.zenodo.org/).

The commands below describe the guarded workflow for a future unsubmitted
deposition. First inspect the deposition state; do not call `--update-metadata`
or `--upload` against the published v0.1.0 record. A changed manuscript needs
a separately reviewed Zenodo version/deposition, and publication remains
irreversible for that new record.

## Source of truth

`manuscript/config.yaml` owns the DOI. The metadata emitter propagates it to
`CITATION.cff`, `.zenodo.json`, `codemeta.json`, and the manuscript token
`{{PUBLICATION_DOI}}`. `src/publication/identifiers.py` provides the shared
normalization contract; `src/publication/zenodo.py` provides the typed,
standard-library client; and `scripts/zenodo_release.py` is the thin CLI
boundary. The token is read from an ignored dotenv file or process environment
and is never committed, printed, or included in the release manifest.

## Metadata and upload verification for a new draft

Run these commands from the repository root after the source-current analysis,
test, hydration, render, web, and release gates have passed:

```bash
# Point this at an ignored dotenv file on the operator's machine.
ENV_FILE="/path/to/ignored/zenodo.env"
# Replace the quoted value with an existing unsubmitted deposition ID.
DRAFT_ID="<unsubmitted-deposition-id>"

uv run --locked python scripts/emit_metadata.py --check
uv run --locked python scripts/zenodo_release.py \
  --env-file "$ENV_FILE" \
  --deposition-id "$DRAFT_ID" \
  --update-metadata
uv run --locked python scripts/zenodo_release.py \
  --env-file "$ENV_FILE" \
  --deposition-id "$DRAFT_ID" \
  --upload output/pdf/active_fedference_combined.pdf
uv run --locked python scripts/zenodo_release.py \
  --env-file "$ENV_FILE" \
  --deposition-id "$DRAFT_ID" \
  --verify output/pdf/active_fedference_combined.pdf
```

The upload command is intended to run once for the final PDF filename on an
`unsubmitted` deposition. If a draft already contains that filename, inspect
its checksum before any replacement. `--reserve` creates a new draft; it is
not needed for the published v0.1.0 record.

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
verification flag and the draft-state guard. After publication, verify both the DOI
redirect and the public record metadata, including the GitHub related
identifier and uploaded-PDF checksum.

For the current published record, the no-token public checks are:

```bash
curl -fsSIL https://doi.org/10.5281/zenodo.21864004
curl -fsSL https://zenodo.org/api/records/21864004 | jq \
  '{doi, files: [.files[] | {key, size, checksum}], related_identifiers: .metadata.related_identifiers}'
```

The authenticated checksum check for the already published v0.1.0 file is
also safe and read-only:

```bash
uv run --locked python scripts/zenodo_release.py \
  --env-file "$ENV_FILE" \
  --deposition-id 21864004 \
  --verify Active_Fedference_Research_Manuscript_Zenodo_10.5281-zenodo.21864004.pdf \
  --remote-filename active_fedference_combined.pdf
```

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
  licence/confidentiality/author approval, or any DOI/publisher policy review
  for a future version. The v0.1.0 public GitHub push and DOI publication are
  already complete.
- A DOI reservation, file upload, or publication never promotes null, reversed, failed, or
  underpowered research outcomes into claims.
