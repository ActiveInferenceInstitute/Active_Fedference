# Release Documentation And Repository-Discovery Handoff

[Back to roadmap](../../TODO.md)

## Status

- Priority class: Minor
- State: Open
- Queue position: immediately after verified v1.1 GitHub and Zenodo publication
  and before the one-time tree-identical private-main synchronization
- Owner surface: README and documentation entry points, release links,
  installation examples, application discovery, and support boundaries

## Rationale

Release publication changes the canonical version, DOI, downloadable artifacts,
and installation instructions. A narrow post-publication handoff ensures that
readers consistently reach the official repository and verified release assets
without mistaking the private evidence remote, development identity, or source
checkout instructions for the current public installation surface.

## Scope

Verify the top-level README, documentation hub, application guide, standalone
boundary, package metadata, citation surfaces, release notes, example index,
and verification guide against the final public record and the downloaded
tagged wheel/sdist/source assets. Keep the official repository at
`https://github.com/ActiveInferenceInstitute/Active_Fedference`; describe
`docxology/active_fedference` only as the private/interim evidence remote where
that distinction is operationally necessary.

MED-5A must already have written the deterministic v1.1.0 tag/release links and
reserved version DOI into every tracked release surface before tagging. MIN-3
resolves those now-live endpoints, compares downloaded bytes and installed
behavior, and verifies that historical v1.0.4 links remain clearly historical,
the labeled Python API and `fedference aggregate` remain the primary own-data
routes, the smallest numeric API is retained, and receipt/scientific-validity
and HTML/PDF accessibility boundaries remain explicit. MIN-3 is a verification
and discovery handoff, not a post-publication rewrite of the immutable tagged
release payload.

## Implementation Notes

- Derive version and DOI text from the release identity rather than copying
  development placeholders or obsolete URLs.
- Test commands from an installed GitHub wheel and source distribution outside
  the checkout. Do not imply PyPI availability.
- Use contextual link names, text equivalents for diagrams, and exact expected
  output shapes without embedding volatile pass counts.
- Preserve source checkout, direct API, labeled API, CLI, sharing, process,
  socket, replay, research-command, and optional Torch/BNN routing guidance.
- If any live endpoint, packaged document, checksum, or installed example does
  not match, leave MIN-3 open and route the correction through a new reviewed
  public PR and, where immutable release bytes are affected, a separately
  authorized corrective version. Never patch the existing tag, GitHub asset,
  or published Zenodo bytes in place.

## Acceptance Criteria

- Primary estimand: not applicable; this is a documentation-discovery and
  installed-usage consistency check.
- Independent replication unit: not applicable; the audit/review unit is one
  documented entry point or command tested against one verified installed
  artifact and the corresponding public URL.
- Every current-release surface agrees on repository, version, DOI, tag,
  release URL, installation source, and supported command boundary.
- Historical v1.0.4 identity remains immutable and unambiguously historical.
- The installed application example runs from both wheel and source
  distribution without importing Torch in the default graph.
- Required acceptance evidence: link-check report, metadata consistency tests,
  installed-example transcripts, package-membership checks, and docs-contract
  results bound to final public `main`. The closure commit records that evidence
  and removes MIN-3's row/page; it does not change the versioned release payload.

## Verification Probes

- Search all public documentation and generated metadata for stale development
  identity, obsolete current-release links, and accidental public use of the
  private remote.
- Run lifecycle metadata validation, documentation contracts, example tests,
  package membership checks, and installed wheel/sdist application smoke.
- Resolve the GitHub tag/release and Zenodo DOI and compare their identifiers
  and hashes with the release record.

## Claim-Boundary Constraints

- Falsifier: a stale version/DOI, broken public link, private-remote confusion,
  undocumented artifact source, non-running installed example, or mismatched
  checksum prevents handoff closure.
- Prohibited claims: clear documentation and a working installed example do
  not establish scientific validity, application suitability, a downstream
  decision, universal robustness, PyPI availability, WCAG, or PDF/UA.

## Dependencies

Depends on MED-5 reaching verified GitHub and Zenodo publication. It must not
guess the final DOI, release date, asset hashes, or URLs before those records
exist. Close this item through a reviewed normal-merge public PR, including
removal of its row and scoped page from the forward-only queue, before the
single private-main synchronization. This handoff is a required post-publication
gate before the MAJ-8 branch begins, so scientific work cannot leave current-
release discovery surfaces in a stale development state.
