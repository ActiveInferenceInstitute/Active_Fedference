# Independent And Cross-Vendor Reproduction

[Back to roadmap](../../TODO.md)

## Status

- Priority class: Medium
- State: Open
- Queue position: parallel external evidence lane; not replaced by the v1.1
  owner-author release verdict
- Owner surface: ISC-89, reviewer eligibility, reproducible execution packet,
  findings, and dispositions

## Rationale

An author-run clone campaign and owner-author release verdict provide valuable
release certification but are not independent reproduction. ISC-89's recorded
criterion is conjunctive: the exact final artifact set needs both an identified
independent-human/Advisor review and a genuinely different-vendor execution.
Either lane alone is informative but cannot close ISC-89.

## Scope

Package one exact final public commit, renderer commit, environment lock,
command ladder, source-bound reports, claim ledger, no-claim boundaries,
distributions, and rendered surfaces for both evidence lanes:

1. an identified independent human or Advisor reviews and, where available,
   executes the final bundle; and
2. a genuinely different-vendor model or executor runs the declared packet
   against that same final bundle.

Each lane returns a structured `pass`, `concerns`, or `fail` record with
command/output evidence, not an untraceable summary. Every concern is
dispositioned in a reviewed change or retained as an explicit open limitation.
Earlier Advisor or vendor observations are useful history, but they must be
rerun or explicitly rebound with verified digests to the exact final SHA and
artifact set before they can contribute to closure.

A local subagent, a same-family restatement, and Daniel Ari Friedman's
owner-author verdict are ineligible for closing this lane. The work may proceed
after a fixed public evidence bundle exists, and it need not block v1.1 tagging
when the separate release-verdict requirement is met.

## Implementation Notes

- Record reviewer identity or vendor, model/tool version where applicable,
  date, exact source/tree/renderer/manifest digests, environment, commands,
  available outputs, and access limitations.
- Ask the reviewer to examine numerical contracts, claim-class separation,
  source/report/caption agreement, application provenance boundaries,
  accessibility assertions, and release reproducibility.
- Preserve null findings, unavailable probes, disagreements, and failed runs.
  Do not convert lack of access into a pass.
- Repair repository defects through normal reviewed branches and rerun the
  affected external review against the new exact SHA.

## Acceptance Criteria

- Primary estimand: not applicable; the target is independent error discovery
  and reproducibility of declared artifacts, not a scientific effect size.
- Independent replication units: one identified independent-human/Advisor lane
  and one genuinely different-vendor execution lane, both bound to the same
  exact final evidence bundle and operating outside the author-run local agent
  family.
- Both records are source-addressable, structured, complete about unavailable
  probes, and bind each finding to an exact file, artifact, or command.
- Every `concerns` item has an explicit fix, accepted limitation, or release-
  blocking disposition; a `fail` is never rewritten as a pass.
- Required acceptance evidence: separate eligibility statements, exact final
  digests, command transcripts or equivalent receipts, both structured
  verdicts, all finding dispositions, and a corresponding ISC-89 update in
  `ISA.md` only when the full conjunctive criterion is actually met.

## Verification Probes

- Compare both external records' source, tree, renderer, manifest, and
  distribution digests with the exact final public evidence bundle.
- Re-run any deterministic finding locally without changing the external
  record.
- Validate that no owner-author or local-subagent evidence is labeled
  independent or cross-vendor.
- Run the docs and claim-contract tests after dispositions are integrated.

## Claim-Boundary Constraints

- Falsifier: a missing lane, ineligible reviewer/executor, unfixed digest
  mismatch, stale pre-final evidence without explicit rebinding, missing command
  evidence, undispositioned concern, or `fail` prevents ISC-89 closure.
- Prohibited claims: either external lane, or both together, is not a scientific replication
  across datasets, hosts, populations, or implementations; it does not confer
  universal robustness, WCAG, PDF/UA, security, or deployment validity.
- The v1.1 owner-author `pass` may satisfy its release checkpoint while this
  broader lane remains open.

## Dependencies

Requires the exact final public commit and complete release evidence packet.
It is logically separate from ISC-242's exact-history, two-clone, and selected
owner-author release gate, although the same immutable digests anchor all three
records. Prior Advisor or vendor evidence must be rerun or digest-rebound to the
final packet; historical proximity is not equivalence.
