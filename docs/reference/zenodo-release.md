# Zenodo release boundary

Active Fedference v1.0.4 is the current public release. Its source-bound
reviewer snapshot and release metadata are published:

- public deposition: `21972644` (v1.0.4)
- public DOI: [`10.5281/zenodo.21972644`](https://doi.org/10.5281/zenodo.21972644)
- public GitHub release: [`v1.0.4`](https://github.com/ActiveInferenceInstitute/Active_Fedference/releases/tag/v1.0.4)
- public PDF: `active_fedference_combined.pdf` (the same bytes as the local
  top-level released manuscript PDF)
- prior public deposition: `21969756` (v1.0.3)

The v1.0.4 DOI is now publicly resolvable. Its PDF and the public GitHub repository
([`ActiveInferenceInstitute/Active_Fedference`](https://github.com/ActiveInferenceInstitute/Active_Fedference))
cross-reference each other through the release metadata. The v1.0.3 and older
records remain available as prior versions. The official API boundary is
documented in the [Zenodo REST API documentation](https://developers.zenodo.org/).

## Source of truth

`manuscript/config.yaml` owns the DOI, manuscript title, subtitle, and the
reader-facing paper abstract. That abstract must remain synchronized with
`manuscript/00_abstract.md`; a short package description is not an acceptable
Zenodo abstract. The metadata emitter propagates the DOI to
`CITATION.cff`, `.zenodo.json`, `codemeta.json`, and the manuscript token
`{{PUBLICATION_DOI}}`. It emits the software name to the citation surfaces,
but emits the complete paper title (`paper.title` plus `paper.subtitle`) to
Zenodo. Zenodo's API calls the record's abstract field `description`, so the
Zenodo `description` must equal the normalized paper abstract after DOI
hydration and removal of source-only Markdown link/code delimiters, not the
shorter software `publication.description`. `codemeta.json` carries both
fields explicitly. `src/publication/identifiers.py` provides the shared
normalization contract; `src/publication/zenodo.py` provides the typed,
standard-library client; and `scripts/zenodo_release.py` is the thin CLI
boundary. The token is read from an ignored dotenv file or process environment
and is never committed, printed, or included in the release manifest.

## Correcting published metadata on the current DOI

Zenodo permits metadata-only corrections to a published record without
changing its DOI. The explicit project adapter supports this path for a
description/abstract correction and never uploads, deletes, or replaces a
published file. The operation creates an editable metadata draft; inspect the
returned state and publish it only after the corrected description has been
reviewed:

```bash
ENV_FILE="/path/to/ignored/zenodo.env"
PUBLISHED_ID="21972644"

uv run --locked python scripts/zenodo_release.py \
  --env-file "$ENV_FILE" \
  --deposition-id "$PUBLISHED_ID" \
  --edit-published-metadata

uv run --locked python scripts/zenodo_release.py \
  --env-file "$ENV_FILE" \
  --deposition-id "$PUBLISHED_ID" \
  --edit-published-metadata \
  --verify output/pdf/active_fedference_combined.pdf \
  --publish --confirm-publish
```

The second command is idempotent with respect to an already-open metadata edit
draft: it updates the draft from the generated `.zenodo.json`, verifies that
the unchanged PDF still matches the published file, and then publishes the
metadata correction. The DOI remains the same. See [Zenodo's published-record
editing guidance](https://help.zenodo.org/docs/deposit/manage-records/#edit)
for the repository's upstream metadata-edit contract.

## Creating the next version

Never upload or replace files against a published deposition. When the paper
or released files change, Zenodo's `newversion` action creates a linked
unpublished draft, preserves the concept record, and inherits the prior
metadata and files. The CLI resolves Zenodo's
`latest_draft` link and exposes inherited-file replacement explicitly:

```bash
ENV_FILE="/path/to/ignored/zenodo.env"
SOURCE_ID="21969756"  # latest published deposition, not the global concept id

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
verification flag and either the ordinary draft-state guard or the explicit
published-metadata-edit path. After publication, verify the DOI redirect and
public record metadata, including the GitHub related identifier and
uploaded-PDF checksum.

For the current public v1.0.4 record, the no-token checks are:

```bash
curl -fsSIL https://doi.org/10.5281/zenodo.21972644
curl -fsSL https://zenodo.org/api/records/21972644 | jq \
  '{doi, version: .metadata.version, files: [.files[] | {key, size, checksum}], related_identifiers: .metadata.related_identifiers}'
```

The prior v1.0.3 record remains independently checkable:

```bash
curl -fsSL https://zenodo.org/api/records/21969756 | jq \
  '{doi, version: .metadata.version, files: [.files[] | {key, size, checksum}]}'
```

The prior v0.1.0 record remains independently checkable:

```bash
curl -fsSL https://zenodo.org/api/records/21864004 | jq \
  '{doi, version: .metadata.version, files: [.files[] | {key, size, checksum}]}'
```

## Invariants

- The published v1.0.4 DOI in `manuscript/config.yaml`, generated metadata,
  manuscript token, rendered PDF, README, live Zenodo record, and public
  GitHub release must agree.
- The Zenodo record title must be the complete paper title plus subtitle, and
  its `description` field must be the full source-controlled abstract. A
  short package description is not an acceptable Zenodo abstract.
- The uploaded PDF must be generated after the final source and test gates;
  checksum verification does not substitute for manuscript or scientific
  review.
- The HTML surface remains the accessibility-enhanced canonical reader. The
  source-current combined manuscript PDF carries the repository's tagged-PDF
  structure gate (`Tagged: yes`, qpdf-visible `/Lang`, language, and
  `StructTreeRoot`); the validator accepts catalog language when Poppler omits
  its optional `Language:` line. Slide PDFs are separate outputs. Tagged
  structure is not PDF/UA conformance, and no such claim is made without a
  retained conformance report and manual review. Older
  Zenodo records may preserve their historical surface properties.
- Zenodo receipts do not replace clean-clone evidence, the public GitHub push,
  licence/confidentiality/author approval, or any DOI/publisher policy review
  for a future version.
- A DOI reservation, file upload, or publication never promotes null, reversed,
  failed, or underpowered research outcomes into claims.
