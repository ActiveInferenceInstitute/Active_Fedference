# First-principles and RedTeam review

Review date: 2026-07-16. Scope: the discrete FedGVI core, federation
transport boundary, thin orchestration, tests, verifier scripts, publication
artifacts, visual claims, and the forward roadmap. The review preserves the
existing dirty worktree and treats the acceptance contract in [`ISA.md`](../../ISA.md)
as authoritative.

## Executive verdict

The project has a coherent scientific spine, but its most important risks were
at the boundaries between a mathematical object and its implementation:
invalid probability mass was repaired by clipping, terminal influence
diagnostics could describe the previous iterate, the `AR` name mixed two
different Rényi normalizations, and release verification checked listed files
without rejecting unlisted files. These are now addressed with shared
validation, terminal-state recomputation, an exact finite-support Alpha-Rényi
solver, and exact-set manifest verification.

The central claim boundary remains unchanged:

| Surface | What is established | What remains deliberately unclaimed |
| --- | --- | --- |
| Client generalized Bayes | stated loss/divergence objective and KLD/NLL recovery | a result outside the declared finite categorical/client protocol |
| `robust_aggregate` | Friston recovery at zero robustness and conditional empirical behavior | a global objective, theorem, or estimator-level B-robustness guarantee |
| `variational_aggregate` | stated free energy, block-coordinate updates, descent diagnostics, raw effective-weight bound | a global optimum or estimator-level B-robustness theorem |
| Transport | queue/process and tested loopback-TCP/HMAC/replay slices | cross-host deployment, TLS operations, fault tolerance, privacy, or Byzantine security |

## Deconstruction

The project reduces to four contracts.

1. A categorical belief is a finite, finite-valued, non-negative pmf over a
   shared state space.
2. A recovery identity is stronger than a similarity claim: the zero-robustness
   server routes must agree with the log-linear pool, and KLD/NLL generalized
   Bayes must agree with ordinary Bayes.
3. A robustness claim needs an explicit objective or an explicit empirical
   boundary. A client loss objective cannot silently become a server estimator
   theorem.
4. A publication verifier must reject both stale listed artifacts and new
   unreviewed artifacts.

This decomposition explains the source organization: `_validation.py` owns
simplex boundaries; `divergences.py`, `losses.py`, and
`generalized_bayes.py` own client mathematics; `aggregation.py` owns the three
server rules; `belief_sharing.py` owns the self-exclusion protocol; scripts
only connect tests, invariants, reports, and release checks.

## RedTeam findings and dispositions

| Vector | Finding | Disposition |
| --- | --- | --- |
| Boundary inputs | NaN, infinity, negative mass, ragged rows, and invalid categorical indices could be silently clipped or mis-indexed | Fixed with shared finite-simplex and index validation; negative controls added |
| Iterative diagnostics | Robust and variational effective weights could be returned from the prior iterate rather than the returned consensus | Fixed; weights and final free energy are recomputed at the terminal state |
| Solver controls | Negative iteration counts and non-finite tolerances/robustness values were not rejected consistently | Fixed with shared control validation |
| Divergence naming | `renyi_divergence` was documented as FedGVI AR even though FedGVI uses the additional `1/alpha` normalization | Fixed by separating standard Rényi from `alpha_renyi_divergence` and updating dispatch |
| AR posterior | A power-softmax shortcut was presented as the AR minimizer | Fixed with a scalar normalization solve and active-set handling for alpha greater than one |
| Release integrity | A new file under a release root could pass because only manifest-listed files were checked | Fixed with exact-set, duplicate-path, count, byte, and digest checks |
| Verification completeness | The `full` profile omitted source lint, typing, layer, invariant, and release checks | Fixed with a `source` profile included by `full` |
| Documentation drift | README described transport as only queue/process based after a loopback-TCP slice had landed | Fixed; README and architecture now state the same transport boundary |
| Figure hygiene | Ruff failures in source-owned figure generators obscured the source gate; one renderer test also reran the full cross-study experiment twice | Fixed; the renderer now forwards its seed and tests consume a valid report payload; the complete figure suite is green and Ruff is clean for `src/`, `tests/`, and `scripts/` |

The Alpha-Rényi correction follows the definition used by the FedGVI paper,
not a stylistic rename. See [Mildner et al., FedGVI](https://arxiv.org/abs/2502.00846)
and the implementation docstrings in `src/fedference/divergences.py`.

## Verifier-first evidence map

The method/evidence diagram is maintained in
[`docs/core/architecture.md`](../core/architecture.md#method-and-evidence-boundary).
The executable map is:

| Question | Test or gate |
| --- | --- |
| Are inputs valid finite pmfs? | `tests/fedference/test_validation.py` and core edge tests |
| Do recovery identities hold? | `tests/fedference/test_core_identities.py` |
| Does the terminal diagnostic describe the terminal state? | robust aggregation edge test |
| Does AR minimize its stated finite objective? | generalized-Bayes AR edge test |
| Can release drift be detected? | `tests/test_release_manifest.py` tamper, missing, and extra-file controls |
| Is the verifier profile complete? | `tests/test_scripts_smoke.py` dry-run profile assertion |
| Is the domain layer isolated? | `scripts/validate_all.py source` layer command |
| Do figures remain source-owned and lint-clean? | figure registry contract plus Ruff |

## Reconstruction principles

The reconstruction keeps each method small and makes its guarantee visible at
the call site. Shared validation is intentionally narrow: valid zero entries
receive a positive numerical floor for log-domain formulas, while negative and
non-finite mass raises instead of being repaired. The robust heuristic and the
variational rule share no hidden objective. The variational rule reports an
objective history, while the heuristic reports only its convergence and
influence diagnostics.

The exact AR update is finite-support specific. For alpha below one it solves
the interior normalization equation; for alpha above one it enumerates the
lowest-loss active prefix so a boundary posterior is a valid solution. This is
the appropriate categorical implementation boundary, not a claim that the
same code solves continuous or hybrid active inference.

## Parallel-analysis synthesis

The internal specialist panel converged on the same five conclusions.

| Lens | Steelman | Adversarial check | Result |
| --- | --- | --- | --- |
| Mathematical objective | A named AR divergence should induce a named posterior update | Compare the returned posterior with the objective, including alpha above one where the optimum can be on a simplex face | Exact scalar solve with active-set support; objective negative control passes |
| Numerical stability | Small categorical supports should admit deterministic, reproducible updates | Probe NaN, infinity, negative mass, zero sums, extreme beliefs, and underflowed influence weights | Inputs reject invalid mass; valid extreme beliefs remain finite; fallback paths stay explicit |
| API and state | A diagnostic is useful only if it describes the result returned | Recompute effective weights from the returned consensus rather than trusting the previous iterate | Terminal robust and variational diagnostics are recomputed |
| Verification and release | A manifest is a review boundary, not merely a checksum list | Add an unlisted file, alter metadata, and test missing/tampered files | Exact-set and metadata checks fail closed |
| Claims, visuals, and roadmap | A visual method map can make proof boundaries legible | Look for README contradictions, guarantee transfer, or completed work relisted as TODO | Architecture Mermaid map, README reconciliation, review record, and existing roadmap preserve the boundaries |

The panel's counterargument is retained: the exact categorical AR solver does
not establish a continuous-state result, and an objective-backed variational
server rule does not upgrade the separate reverse-KL heuristic. Those are
roadmap questions, not implementation details to be inferred from a green test.

## Remaining roadmap

No completed engineering fix was promoted into the forward TODO. The open
scientific work remains scoped in [`TODO.md`](../../TODO.md) and its linked
pages: MAJ-1 server-rule characterization, MAJ-2 protocol-matched FedGVI BNN,
MAJ-3 continuous or hybrid recovery, MAJ-4 cross-host federation, MAJ-5 richer
hierarchical tasks, MAJ-6 independent benchmarks, and the MED/MIN evidence
extensions. In particular, the new exact categorical AR update does not close
the continuous-state or faithful server-side FedGVI items.

The next safe publication sequence is: pass the source profile, run the full
coverage gate, regenerate the publication snapshot from the declared config,
then run the exact-set release verifier. A green source gate is evidence of
implementation integrity; it is not evidence for any stronger scientific
claim than the table above.
