# Active Fedference Readiness Research Synthesis

Date: 2026-07-07
Mode: ULW research pass, Perplexity-led source discovery with primary-source verification.

## Scope

Publication-readiness scholarship pass, not a Major scientific-upgrade pass. The
review focused on whether the manuscript's active-inference / FedGVI / robust
federated aggregation bridge was missing claim-critical scholarship.

## Primary Sources Checked

- FedGVI: Mildner, Hamelijnck, Giampouras, and Damoulas, ICML/PMLR 267, 2025.
  Primary page: https://proceedings.mlr.press/v267/mildner25a.html
- Friston belief sharing: Friston et al., Neuroscience and Biobehavioral
  Reviews 156:105500, DOI `10.1016/j.neubiorev.2023.105500`.
  PubMed page was discoverable but browser-gated; DOI and bibliographic details
  were verified against the existing BibTeX and metadata drift check.
- GVI: Knoblauch, Jewson, and Damoulas, JMLR 23(132):1-109, 2022.
  Primary page: https://jmlr.org/papers/v23/19-1047.html
- Robust federated aggregation: Pillutla, Kakade, and Harchaoui, Robust
  Aggregation for Federated Learning, IEEE TSP 70:1142-1154, 2022; arXiv
  1912.13445. Primary arXiv page: https://arxiv.org/abs/1912.13445

## Citation Decision

Accepted:

- `pillutla2022robust` was added and cited in the aggregation/related-work
  bridge. Rationale: the existing manuscript cited Byzantine-tolerant Krum and
  gamma-mean robust aggregation but skipped the canonical geometric-median robust
  aggregation line. This is directly claim-relevant negative space for the
  paper's belief-fusion distinction.

Retained without new bibliography:

- FedGVI/PMLR and GVI/JMLR were already present. I updated surface prose and
  BibTeX metadata to recognize the PMLR FedGVI publication while retaining the
  arXiv identifier.
- Friston's DOI was already correct in `references.bib`; metadata had drifted to
  `2024.105500`, so `manuscript/config.yaml` and regenerated `.zenodo.json` were
  corrected to `2023.105500`.

Rejected as padding for this pass:

- Additional robust FL survey, Bulyan, signSGD, pFedBayes, DSVGD, and Bayesian
  robust aggregation leads from Perplexity were not added. They are useful
  broader context, but the manuscript does not currently make live claims that
  require them, and adding them would dilute the bibliography rather than repair
  a publication blocker.

## Prose Claim-Boundary Changes

- Absolute novelty claims were softened to "to our knowledge" or "connect
  explicitly" where a universal negative would require a systematic review.
- The server-side `robust_aggregate` heuristic remains bounded to its recovery
  limit; no external robust aggregation theorem is imported into it.
- The PyTorch complement is described as a point-mass deterministic MLP, not as a
  full mean-field BNN FedGVI reproduction.

## Publication-Readiness Implication

Scholarship is adequate for a readiness submission after the completed
validation, render, package, and browser-smoke gates. Remaining Major TODOs are
research extensions, not literature blockers.
