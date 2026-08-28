# Active Fedference API Reference

This reference routes callers to the supported public boundaries. It describes
the current categorical implementation; it is not evidence that an experiment
ran, a robustness claim generalized, or a physical multi-host deployment was
validated.

For installation, caller-owned data, method selection, end-to-end recipes, and
diagnostic policy, start with the source-owned
[`application guide`](../../docs/application-guide.md).

## Choose an interface

| Need | Public boundary |
| --- | --- |
| Deterministic categorical aggregation | `AggregationConfig` + `aggregate_result` |
| Per-recipient in-process belief sharing | `share_round` |
| One-machine process isolation | `run_multiprocess_round` |
| Loopback transport and replay testing | `run_socket_round` / `validate_socket_replay` |
| Isolated research runs and receipts | installed `fedference` CLI |

Runnable programs for every row are in [`examples/`](../../examples/README.md).

## Aggregation

### `fedference.AggregationConfig`

Validated configuration shared by direct, process, and socket adapters. Its
fields are `method`, `robustness`, `entropy_weight`, `max_iter`, `tol`, and
`multistart`; `fingerprint` is the canonical SHA-256 transport/evidence binding.

### `fedference.aggregate_result`

`aggregate_result(local_posteriors, config=None, base_weights=None)` is the
canonical rich-result entry point. `AggregationResult` exposes `consensus`,
`raw_effective_weights`, `normalized_effective_weights`, `iterations`,
`converged`, histories, and explicit `fallback_events`.

Raw `base_weights` are log-pooling coefficients, not probabilities normalized
before aggregation. Scaling every raw coefficient can therefore change the
consensus concentration. Use `normalized_effective_weights` only as the
normalized influence diagnostic.

### Aggregation methods

- `log_linear_pool(local_posteriors, base_weights=None)` implements the project
  product-of-experts. Under shared support, admitted posterior-log-potential,
  and fixed-weight assumptions, it is a categorical specialization related to
  Friston et al. Eq. 7—not a reconstruction of the complete source protocol.
- `robust_aggregate(..., robustness=..., base_weights=None)` is the project
  reverse-KL-reweighted server heuristic. Its established formal property is
  exact recovery of the project pool at zero robustness; do not transfer the
  FedGVI client theorem to this server rule.
- `variational_aggregate(..., robustness=..., entropy_weight=1.0, ...)` is the
  objective-backed conservative server rule. Its raw effective-weight bound and
  block-descent certificate are not estimator-level B-robustness.
- `aggregate(...)` remains the array-returning compatibility dispatcher.

The low-level robust solvers default to zero robustness, whereas an explicitly
robust `AggregationConfig` retains the compatibility default of one. New code
should always set robustness explicitly.

## Belief sharing and federation

- `share_round(..., config=..., exclude_self=True)` performs one in-process
  round. `consensus` is the global aggregation; `shared_posteriors` are
  recipient-specific leave-one-out values when self-exclusion is enabled.
- `run_multiprocess_round(..., config=...)` runs the uniform-base-weight global
  aggregation through spawned OS processes on one machine. Call it from a real
  guarded script (`if __name__ == "__main__":`) so spawn can import safely.
- `run_socket_round(..., config=..., auth_key=..., replay_path=...)` exercises
  the same uniform-base-weight rule over loopback TCP. HMAC protects frames
  under a caller-supplied key; it is not TLS, key exchange, or cross-host
  deployment security.
- `validate_socket_replay(replay, local_posteriors, consensus, config=...)`
  recomputes declared payload, envelope, configuration, and consensus digests.
  Replay files omit raw beliefs and are not self-sufficient data archives.

The process and socket convenience adapters currently expose uniform base
weights; use direct aggregation or sharing when non-uniform coefficients are
required.

Construct and retain one explicit `AggregationConfig` across direct, sharing,
process, socket, and replay boundaries. Adapter-specific compatibility defaults
can recover the same numeric pool while carrying different method declarations,
iteration budgets, and configuration fingerprints.

## Generalized Bayes and client losses

### `fedference.divergences`

- `kl_divergence(q, p)` and `reverse_kl(q, p)` provide categorical KL variants.
- `renyi_divergence(q, p, alpha)` is conventional Rényi divergence.
- `alpha_renyi_divergence(q, p, alpha)` is the FedGVI-normalized variant.
- `total_variation(q, p)` provides total variation distance.

### `fedference.losses`

- `nll(p, outcome)` is negative log likelihood.
- `beta_loss(p, outcome, beta)` recovers NLL as beta approaches zero.
- `rcce(p, outcome, q_loss)` is robust categorical cross-entropy; `q_loss`
  avoids collision with the conventional posterior symbol `q`.

### `fedference.generalized_bayes`

`generalized_posterior`, `cavity`, and `update_factor` implement the finite
categorical client-update boundary. The KLD/NLL client recovery is separate
from both server-side zero-robustness recovery statements.

## Experiments and evidence

`fedference.experiments` contains deterministic study functions such as
`run_belief_sharing`, `run_language_acquisition`, `run_emergence`, and
`run_robustness_sweep`. Their returned values are computation results, not
automatically citable evidence. The research registry declares estimands,
units, falsifiers, budgets, profiles, and no-claim outcomes.

The installed CLI exposes `list`, `run`, `benchmark`, `verify`, and `replay`.
Write commands require an explicit empty directory outside committed `output/`.
Smoke and pilot profiles test mechanics; they do not become confirmatory or
manuscript evidence merely because a receipt verifies.

## Optional and out-of-scope surfaces

Torch BNN modules require the `bnn` extra and stay outside the default import
graph. The project does not claim a complete Friston message-passing protocol,
universal robustness, estimator-level B-robustness for the variational server
rule, production cross-host federation, mTLS/key management, or paper-scale
external-service/GPU validation.

See module docstrings for full parameter contracts, the
[`examples/`](../../examples/README.md) ladder for executable usage, and the
[API stability policy](../../docs/reference/api-stability.md) for compatibility
rules.
