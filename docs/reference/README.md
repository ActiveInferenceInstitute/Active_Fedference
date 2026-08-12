# Reference documentation

Copy-paste verification probes and command cheat sheet.

| Document | Purpose |
| --- | --- |
| [verification-commands.md](verification-commands.md) | ISA probes, coverage, identities, receipts, packaging, freshness, and release gates |
| [api-stability.md](api-stability.md) | Public API, schema versioning, additive v0.x evolution, and deprecation policy |
| [../development/quickstart.md](../development/quickstart.md) | Fresh-checkout setup, package installation, tests, analysis, and rendering order |
| [zenodo-release.md](zenodo-release.md) | Published DOI record, metadata binding, future draft upload, verification, and explicit publication boundary |

The package metadata source of truth is `pyproject.toml`; the publication
metadata emitter separately keeps `CITATION.cff`, `.zenodo.json`, and
`codemeta.json` aligned with `manuscript/config.yaml`. The MIT license and
source-distribution manifest are part of the archived package contract, not
just repository presentation files.
