# Release state reconciliation — 2026-10-27

The published v1.1.0 GitHub assets and Zenodo PDF agree byte for byte with
their declared digests. This dated observation does **not** close MED-4, MED-5,
MIN-2, MIN-3, ISC-89, or ISC-242. The source backlog retains obligations whose
historical execution cannot be inferred from hosting.

## Observed release identity

The machine-readable observation below is consumed by the offline ledger test.
It records this observation date, not a promise of future remote freshness.

```json
{
  "observed_on": "2026-10-27",
  "version": "1.1.0",
  "github_release_id": 392502751,
  "tag": "v1.1.0",
  "tag_object": "c1f1a331817773414a4ef8722444979f0ecbd1bf",
  "tag_commit": "c75954181184545acc31cda17c1013d4dfb34d31",
  "zenodo_record_id": 22149133,
  "zenodo_concept_id": 21864003,
  "doi": "10.5281/zenodo.22149133",
  "concept_doi": "10.5281/zenodo.21864003",
  "pdf": {
    "name": "Active_Fedference_Research_Manuscript_v1.1.0_Zenodo_10.5281-zenodo.22149133.pdf",
    "bytes": 12604278,
    "sha256": "8ca4d13e9316d4640184b8fdeb9300a000bf9e8203946b21aef1fe5c23caf729"
  },
  "zenodo_pdf_sha256": "8ca4d13e9316d4640184b8fdeb9300a000bf9e8203946b21aef1fe5c23caf729"
}
```

[GitHub release metadata](https://api.github.com/repos/ActiveInferenceInstitute/Active_Fedference/releases/tags/v1.1.0)
records a non-draft, non-prerelease publication at 2026-09-20T16:53:37Z.
The annotated tag object and peeled commit agree with the public API read and
local object database. The release target is that commit, not the current
working branch. [The exact-commit CI run](https://github.com/ActiveInferenceInstitute/Active_Fedference/actions/runs/35488624905)
is completed with conclusion `success`; this single run does not establish
every historical PR-head/merge-ref gate or a two-clone campaign.

[Zenodo record 22149133](https://zenodo.org/api/records/22149133) records version
1.1.0, concept record 21864003, and version relation index 5 with `is_last=true`
at capture time. Its related identifiers link the canonical repository as
`isSupplementTo` and cite the two source papers. This confirms the concept
chain, not the authorization or exact draft-creation sequence from predecessor
record 21972644. Its one file, `active_fedference_combined.pdf`, has declared
MD5 `7e504dbef6e0e3d6572116580c719bce`; recomputed MD5, SHA-256, size, and
byte equality agree with the GitHub manuscript.

## Downloaded bytes

All five GitHub asset sizes and SHA-256 digests were independently recomputed.
The four payload entries in downloaded `SHA256SUMS.txt` agree; that file's
own digest is checked against the API rather than a self-reference.

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| `active_fedference-1.1.0-py3-none-any.whl` | 664758 | `dc5f5ce58414ad31a4e875d76007e031f45d702c4c3d6c0ad048a70969057eb6` |
| `active_fedference-1.1.0.tar.gz` | 2007619 | `5cdc09af7e38db2c3a11ff6869d9be45336263e7550f5f76fbb42e2bad971793` |
| Versioned manuscript PDF named above | 12604278 | `8ca4d13e9316d4640184b8fdeb9300a000bf9e8203946b21aef1fe5c23caf729` |
| `manifest.json` | 587359 | `fa7fb04a3e2db900f3fa7eda0a74a5a0c1141bc977ba5ba31ed08aebfd3b89e4` |
| `SHA256SUMS.txt` | 430 | `9fa27bf8728286cf6fec9ff37b686d8e5ed2002c081271b2c2824c007fc49f3c` |

The [immutable PDF ledger](../reference/historical-release-pdfs.json) binds all
six historical PDFs. The production validator reads every local original;
no historical PDF was regenerated or replaced. The hosted manifest equals
the checked-in `output/release/manifest.json` at capture. Its combined output
PDF entry is a different artifact (12604277 bytes; SHA-256
`7af79aded489a198bca5753f4df034385f11fc681dcb28895c66cb419b93a169`).
It must not be substituted for the versioned released PDF.

## Local receipts and renderer

The configured renderer and the recorded render-stage receipt both identify
`docxology/template` commit `60ba9159124a1a1c91a7bfd51fba4f3b2da64ca6`,
with recorded tree state `clean`. The receipt timestamp is
2026-09-19T00:00:00Z. This is the receipt's declaration, not a fresh checkout
or visual inspection of that renderer. ISA iteration 50's older
`cea2904d` narrative does not automatically transfer to this identity.

| Inspected local receipt | SHA-256 at capture | Evidentiary limit |
| --- | --- | --- |
| `output/data/pipeline_provenance.json` | `5f9daf1c45a6ed3d413dddd60c9db51ea79412efbf4df77a460057d777d21ac7` | Producer input/output declarations, not exact-final-commit clone certification |
| `output/data/test_coverage_receipt.json` | `82a479efd44ec4ef58d1b01daa295bae1fdf62ef6cbb5e4284eba2bb97922fb6` | Retained source-bound test receipt, not a new full-suite result |
| `output/reports/validation_report.json` | `3bc83cadf255c35be6b8223cca933b05b307593182fd5bee601d239ed5e73724` | Its recorded passing summary is not independently rerun visual acceptance |
| `output/reports/artifact_manifest.json` | `9d61151371d3ac2b2d8e86bafba6582028beb32b5da83d3fc10caaec704a2d84` | Artifact declarations, not an eligible human verdict |

A fresh read-only verification against the working checkout reports one
release provenance-fingerprint mismatch and four pipeline freshness findings:
changed analysis inputs and digest, plus changed hydration inputs and digest.
The complete diagnostics are retained in `local-receipt-verification.json`.
Consequently these retained outputs are **not source-current for this working
tree**. The immutable hosted release was not rebuilt to conceal this mismatch.

The public working checkout at inspection had HEAD
`dcaf38d71fe3fc5cffaea5a9cba6a980288ba1d8` and tree
`be32b75b022006990dce3b3328eadd86375c03de`, with concurrent uncommitted work.
Bounded private-mirror metadata reads returned HEAD
`e22a9870844da464db317a47c41c9473482be93a` and tree
`d7356a71e90572647d28d38d24bba10b478d58d7`.
Those trees differ. A tracked-status read emitted an IPC warning; no clean
private-state or synchronization certification is inferred. Private content
was not copied into this public ledger.

## Remaining obligations

Missing means not established by the bounded records inspected here; it does
not assert that no external record exists. Original owner pages remain the
acceptance authority.

| Criterion | Established here | Exact missing evidence and disposition |
| --- | --- | --- |
| [MED-4](../todo/visual-scholarship-integration.md) | Renderer identity is declared in config and receipt; exact release-commit CI exists | Exact renderer merge/review and CI, confidentiality history audit, source/generated PR-head and merge-ref checks, complete figure/HTML/reveal/Beamer/PDF and zoom/keyboard evidence bound to that renderer/source. **Open.** |
| [MED-5](../todo/v1-1-release-publication.md) | Annotated tag identity, five downloaded assets, checksums, published Zenodo record and matching PDF | Release-only normal-merge record and all required checks; exact-SHA clone/verdict bridge; linked-draft inspection and fresh publication approval; final MIN-3 closure; one normal private merge with equal final-public tree and unchanged MAJ-8 branch point. **Open; do not republish to reconstruct history.** |
| [MIN-2](../todo/release-and-verification-ladder.md) | Local immutable-PDF validation and hosted CI are bounded observations | Sanitized-history confidentiality evidence, two sequential isolated full-ladder receipts for the exact final-main SHA, initial 40 GiB headroom, exact renderer binding and eligible identified structured verdict. **Open.** |
| [MIN-3](../todo/release-documentation-handoff.md) | Live identities and downloaded package hashes agree | Complete final-main-bound link/metadata/package-membership checks; installed own-data examples from both downloaded wheel and sdist outside checkout; normal documentation closure merge and backlog disposition. **Open.** |
| [ISC-89](../../ISA.md) | Existing deferred row remains authoritative | Both eligible independent-human/Advisor review and genuinely different-vendor final-bundle execution, with exact artifact set, environment, commands, findings and dispositions. Owner-author or same-family review is insufficient. **Deferred/open.** |
| [ISC-242](../../ISA.md) | Existing unchecked row remains authoritative | Complete history/confidentiality record, exact-final-main two-clone ladder and identified human or different-vendor structured pass, plus explicit external dispositions. Hosting is insufficient. **Open.** |

## Evidence and reproducibility

Local evidence is retained under
`.omo/evidence/ulw/01a17de6-a86e-7501-b9da-357eeffeaabf/task-4-release-reconciliation/`.
It includes copied public download bytes, original metadata, fresh response
headers/body/status, computed digest checks, source/dirty-content custody,
local-receipt comparisons, negative controls, and cleanup disposition.

The first guarded unauthenticated GitHub release/tag/CI REST reads returned
**200**, with raw headers and bodies retained at the evidence root. A separate
subsequent read returned **403**; those bodies, headers, curl exit statuses,
and failures remain under the evidence `http/` directory. The supported `gh-axi`
read path supplied tag/CI records and the earlier release metadata/assets.
An attempted `--include` flag was rejected by that wrapper and is retained.
No HTTP 200 or headers are invented for wrapper reads. Fresh unauthenticated
Zenodo metadata returned **200**. Raw `*.json` filenames from `gh-axi --full`
contain its formatted response representation, not necessarily JSON.

Run the offline checks with the project's locked development environment:

```bash
uv run --locked --extra dev pytest tests/test_release_state_ledger.py -q -s
```

The happy node reads the structured observation and actual immutable PDFs.
The error node constructs real altered files, mismatching digests, malformed
and duplicate-key receipts, invokes the production validator, requires
rejection, and checks that failed inputs remain unchanged. Neither node
claims current hosted freshness. The separate captured download probe verifies
all hosted bytes against metadata and checksums. No remote state, tags,
publication artifacts, or private mirror were modified.
