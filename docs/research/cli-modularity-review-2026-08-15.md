# CLI, orchestration, and documentation modularity review

Review date: 2026-08-15. This focused follow-up reviews the installed CLI,
pipeline-facing scripts, publication identity helpers, package documentation,
and the documentation test harness. It complements the broader
[runtime-surface review](runtime-surface-composability-review-2026-07-17.md)
and the normative [modularity contract](../development/modularity.md).

## Review ledger

| ID | Area | Severity | Evidence | Remediation | Status |
| --- | --- | --- | --- | --- | --- |
| MOD-01 | Installed CLI | High | `src/fedference_cli/__init__.py` combined parser, runner dispatch, output safety, JSON writes, and receipts in 638 lines | Split the stable facade, parser, command handlers, and support/receipt helpers; preserve `main` and `_report_fallbacks` imports | Fixed and covered by `tests/test_fedference_cli.py` |
| MOD-02 | Release identity | Medium | `clean_checkout.py` embedded the v1.0.2 manuscript filename, so a new DOI/version required a code edit | Derive the informative top-level PDF name from `manuscript/config.yaml` through `publication.identifiers.manuscript_pdf_filename` | Fixed and covered by identifier and clean-checkout tests |
| MOD-03 | Package documentation | Medium | CLI responsibility boundaries were described only in broad architecture prose | Add package-local `README.md`/`AGENTS.md`, include them in the source distribution, and route contributors through the cross-layer modularity guide | Fixed and checked by package/build/documentation contracts |
| MOD-04 | Documentation test runtime | Medium | Retired-name test traversed `.venv`, `.tmp`, and other ignored trees before filtering them, causing metadata-heavy stalls | Prune ignored directories during `os.walk` while retaining checks for untracked source-owned text | Fixed; focused documentation contract passes |

## Modularity decision

The installed command remains a process adapter. Research operations stay in
importable `fedference`/`analysis` modules; reports cross the typed write
boundary; figures consume declared report fields; manuscript tokens consume
producer-owned outputs; and scripts sequence these operations without
reimplementing them. The CLI's private modules are responsibility boundaries,
not additional public domain APIs.

The package-local contract is intentionally explicit:

- `fedference_cli.__init__` preserves the public facade;
- `_parser.py` owns argument grammar and process error mapping;
- `_commands.py` owns registry-backed command dispatch; and
- `_support.py` owns atomic writes, output isolation, validation, and receipts.

The same rule applies to release identity: the configured version and DOI are
the source, the metadata emitter owns generated citation surfaces, the
identifier helper owns the top-level PDF filename, and clean-clone validation
checks the resulting tracked artifact rather than a historical literal.

## Verification evidence

- `ruff check .` passes.
- Full `mypy src/` passes under the locked environment.
- CLI behavior tests pass with real seeded runners and no mocks.
- Documentation and package metadata contract tests pass.
- The reproducible wheel/sdist build passes twice under `SOURCE_DATE_EPOCH`;
  the installed wheel imports the core and runs `fedference list --json`.
- Static Mermaid validation passes for all README/docs blocks; the renderer
  probe remains a release-surface check and must be recaptured after the final
  source-current PDF and web package are generated.

## Residual boundaries

This review does not turn the CLI into a scientific authority, a smoke run into
confirmatory evidence, or loopback federation into physical multi-host
deployment. A final release still requires a clean immutable commit, the
source-current two-pass manuscript/render pipeline, final surface checks,
GitHub publication, and a separately authenticated Zenodo reservation/upload/
verification/publication sequence.
