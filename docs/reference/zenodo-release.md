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

`manuscript/config.yaml` owns the current development/release identity,
manuscript title, subtitle, and reader-facing paper abstract. A development
version such as `1.1.0.dev0` uses an empty `publication.doi`, the exact plain-text
status `publication.doi_status: "(forthcoming)"`, and a null release date. This
prevents a renderer from fabricating a resolver link for an unassigned
identifier and generates no version DOI. A final version requires its reserved
DOI and approved date and removes the development-only status. That abstract
must remain synchronized with
`manuscript/00_abstract.md`; a short package description is not an acceptable
Zenodo abstract. The abstract terminates with
`{{PUBLICATION_IDENTITY_SENTENCE}}`, which hydrates to neutral no-DOI prose in
development and to the assigned version DOI in the final state. In a final
release, the metadata emitter also propagates the DOI to `CITATION.cff`,
`.zenodo.json`, and `codemeta.json`; in development, those generated surfaces
omit DOI/date fields entirely. It emits the software name to the citation surfaces,
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

## Immutable v1.0.4 record

Deposition `21972644`, DOI `10.5281/zenodo.21972644`, its metadata, and its PDF
are immutable inputs to the v1.1 workflow. Do not run published-record edit,
file replacement, or publish commands against that deposition. The repository
adapter retains a separately authorizable metadata-repair capability for an
exceptional correction, but that capability is not part of the v1.1 release
plan and must not be inferred from this guide. Any future repair to the
published v1.0.4 record requires its own explicit owner authorization and
verification record.

## Creating the next version

Never upload or replace files against a published deposition. When the paper
or released files change, Zenodo's `newversion` action creates a linked
unpublished draft, preserves the concept record, and inherits the prior
metadata and files. The CLI resolves Zenodo's
`latest_draft` link and exposes inherited-file replacement explicitly:

Creating the v1.1 draft begins only after the development PR is merged, public
`main` is green, and separate release-start approval fixes the authors,
title/abstract, license, attribution, confidentiality disposition, repository
destination, target date, and authority to create the linked Zenodo draft.

```bash
ENV_FILE="/path/to/ignored/zenodo.env"
SOURCE_ID="21972644"  # latest published deposition, not the global concept id

uv run --locked python scripts/zenodo_release.py \
  --env-file "$ENV_FILE" \
  --new-version-of "$SOURCE_ID"
```

Record the returned draft id and reserved DOI before changing
`manuscript/config.yaml` from the empty-DOI/forthcoming-status development state
to the assigned DOI/date final release identity and removing `doi_status`. Then
emit metadata, regenerate the complete source-bound analysis/hydration/render
chain, and run the release checks.

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

- The immutable v1.0.4 DOI, released PDF, README latest-published-release
  section, live Zenodo record, and public GitHub release must agree. The current
  post-v1.0.4 development config and generated metadata must not claim that DOI.
- Development identity requires matching PEP 440 development package/manuscript
  versions, no assigned DOI, and no release date. Final identity requires a
  matching final version, assigned DOI/date, and a new exact version/DOI-named
  top-level PDF. Clean-checkout validation always requires all five immutable
  top-level PDFs from v0.1.0 through v1.0.4 to remain tracked; a development
  revision does not require a new v1.1 PDF, while a final v1.1.0 revision adds
  its exact new PDF without replacing any historical file. The checked-in
  [`historical-release-pdfs.json`](historical-release-pdfs.json) ledger binds
  each historical filename to its SHA-256; both clean-checkout validation and
  release-bundle preflight reject deleted, substituted, or modified bytes.
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
