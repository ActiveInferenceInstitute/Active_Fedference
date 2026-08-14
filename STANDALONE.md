# Standalone Notes — Active Fedference

## Purpose

Active Fedference is a self-contained research project: a discrete-categorical
reimplementation of FedGVI (Mildner et al., 2025, PMLR 267; arXiv:2502.00846) connected to
the federated belief-sharing of Friston et al. (2024, Neurosci. Biobehav. Rev.
156:105500). It demonstrates robust federated active inference, anchored on
separate tested project-local client and server recovery limits. Under the
documented categorical posterior-log-potential bridge assumptions, the project
log-linear pool specializes Friston Eq. 7; it is not a reconstruction of the
complete source message-passing protocol.

The public release target is
[ActiveInferenceInstitute/Active_Fedference](https://github.com/ActiveInferenceInstitute/Active_Fedference).
The v1.0.2 release is now published there and has the permanent Zenodo DOI
[`10.5281/zenodo.21934992`](https://doi.org/10.5281/zenodo.21934992), with the
public record at <https://zenodo.org/records/21934992>. The v1.0.1 and v0.1.0
records remain available as prior versions. This checkout remains a
standalone development/review source with the interim `docxology` remote; it
is not the public repository and future changes remain unreleased until an
explicit review and publication action.

## The standalone core

The genuinely standalone part is `src/fedference/` — pure NumPy/SciPy, typed,
deterministic, with **no `infrastructure.*` imports** (the layer contract,
ISC-21). It can be exercised in isolation:

```bash
uv run --locked --extra dev pytest tests/ \
  --cov=src --cov-fail-under=90
```

The central identity is checkable in three lines:

```python
import numpy as np
from fedference.aggregation import robust_aggregate, log_linear_pool
local_posteriors = [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.5, 0.4, 0.1]]
assert np.allclose(robust_aggregate(local_posteriors, robustness=0.0).consensus,
                   log_linear_pool(local_posteriors))  # exact project identity
```

This check does not reconstruct Friston et al.'s complete protocol; it checks
only the project-local server recovery identity used by the categorical bridge.

## Repository boundary

This checkout is a standalone repository, separate from both the sibling
template repository and the public GitHub target. Do not merge it into the
unrelated template remote. Credentials, uncached external data, and local
review scratch remain private and are excluded from the release; the reviewed
v1.0.2 source snapshot itself is public. Verify the repository root before
running render or release commands:

```bash
git rev-parse --show-toplevel
git remote -v
```

Zenodo publication is a composable release boundary: `src/publication/zenodo.py`
contains the typed REST client, while `scripts/zenodo_release.py` is the thin
new-version/update/upload/verify/publish adapter. The v1.0.2 deposition is
published; future metadata or PDF changes require a separately reviewed
Zenodo version/deposition. Credentials are read from the process environment
or a local ignored dotenv file and are never written to the repository,
generated manifest, or PDF.

## What is and is not infrastructure-free

`src/fedference/` is infrastructure-free by contract. The surrounding project
scaffolding (analysis scripts, figure generation, manuscript-variable injection)
is source-owned here and calls into `src/`. The sibling template checkout is used
only for rendering and validation commands that live outside the domain core. If
you fork the *core* out into its own package, take only `src/fedference/` and the
matching tests; rendering and publication packaging are a separate surface.

## Required edits if you fork

- `manuscript/config.yaml` — title, authors, keywords, and the `experiment:`
  block (keys mirror the `fedference.experiments` study-function keyword arguments).
- `manuscript/references.bib` — keep Friston et al. (2024) and Mildner et al.
  (2025) as the anchor citations.
- Regenerate analysis outputs and manuscript variables before editing any prose
  result claim — every number in the manuscript is a `{{TOKEN}}`, never a
  literal.

## What not to claim

- Do not attribute FedGVI's bounded-influence guarantee to the **server-side**
  `robust_aggregate` heuristic. The client-side bounded-loss update carries the
  cited source result only under its matching loss, divergence, and regularity
  assumptions. Carry the three-axis distinction: source-conditional
  client-side result, server-side heuristic, and conservative objective-backed
  variational server rule (see [`AGENTS.md`](AGENTS.md) and
  [`manuscript/02_gap.md`](manuscript/02_gap.md)).
- Do not report a "robust beats naive" result that did not survive the paired
  Wilcoxon test + Benjamini–Hochberg FDR deflation.
- Do not claim a reproduction of a Friston figure that the corresponding
  `fedference.experiments` study function did not actually produce under the configured seed.
- Do not present the loopback/HMAC adapter as confidential, identity-bound,
  Byzantine-tolerant, or cross-host. Its complete threat boundary is in
  [`docs/security/active_fedference-threat-model.md`](docs/security/active_fedference-threat-model.md).
- Do not present structural PDF validation as tagged PDF or PDF/UA
  conformance. The canonical accessibility-enhanced surface and promotion
  requirements are in
  [`docs/manuscript/accessibility.md`](docs/manuscript/accessibility.md).
