# Conceptual foundations

Active Fedference connects two federated inference traditions that, to our
knowledge, had not been formally linked before this project.

## Primary sources

| Paper | Role in this project |
| --- | --- |
| Friston et al. (2024), *Federated inference and belief sharing* | Belief-sharing message-passing rule (Eq. 7); sentinel POMDP world; reduced categorical source-mechanism analogues of Figs. 5, 7, and 9 |
| Mildner et al. (2025), FedGVI (PMLR 267; arXiv:2502.00846) | Generalised variational inference with robust losses and divergence-regularised aggregation |

BibTeX keys: `friston2024federated`, `mildner2025fedgvi` in
[`manuscript/references.bib`](../../manuscript/references.bib).

## Project-local identity and categorical bridge

The exact executable statement is local to this project: on the same
categorical inputs, zero-robustness server pooling equals the project's
log-linear pool. In code:

```python
from fedference.aggregation import robust_aggregate, log_linear_pool

assert (robust_aggregate(local_posteriors, robustness=0.0).consensus
        == log_linear_pool(local_posteriors)).all()
```

The comparison with Friston et al. (2024) Eq. 7 is a **categorical
posterior-log-potential specialization**, not a reconstruction of the complete
message-passing protocol. The bridge has three explicit assumptions:

1. all local posteriors share one finite categorical support;
2. each project posterior has the form $q_n=\operatorname{softmax}(m_n)$ for an
   admitted local message potential $m_n$; and
3. the project uses a fixed non-negative weight mapping without reconstructing
   the source factor graph, cavity structure, or message schedule.

Under those assumptions,
$\operatorname{softmax}(\sum_n w_n\log q_n)=\operatorname{softmax}(\sum_n w_n m_n)$:
the omitted local normalizers are state-independent. This establishes the
categorical correspondence only; it does not establish source-protocol
identity.

Robustness is a strict **generalisation** of the project pool, not a competing
method or full source reconstruction. At robustness zero the project-local
identity above is exact. Above zero, the server heuristic can resist a
confidently-wrong sentinel in declared honest-majority diagnostic regimes, but
it also has known reversals and a finite breakdown witness.

A second recovery limit pins generalised Bayes to ordinary Bayes:

`generalized_posterior(KLD, NLL)` equals closed-form prior×likelihood Bayes
(ISC-7 in [`ISA.md`](../../ISA.md)).

The `AR` label is reserved for FedGVI's Alpha-Rényi regularizer, whose finite
categorical update is solved from its scalar normalization condition. The
standard Rényi divergence remains available separately as
`renyi_divergence`; dividing it by $α$ gives `alpha_renyi_divergence`. This
distinction prevents a conventional Rényi diagnostic from being silently
presented as the FedGVI objective.

## The problem being solved

The project naive log-linear pool is a product-of-experts: every agent holds a
multiplicative veto. Under the bridge assumptions above, this exposes the
categorical mechanism related to Friston's belief-sharing. One miscalibrated
sentinel broadcasting a confident-but-wrong belief can drag the colony off the
truth, and the source mechanism itself is not a contamination experiment.
FedGVI solved robustness for federated **learning**. In the reviewed source
papers we did not find this exact categorical bridge; Active Fedference
therefore evaluates a scoped connection rather than making a literature-wide
absence claim.

## Three robustness axes (carry this distinction)

These axes are **not interchangeable**. The manuscript and sweep report label
them explicitly.

### 1. Client-side, rigorous (FedGVI contribution)

Robust per-agent `generalized_posterior` with a bounded loss (`rcce`, $\beta$-loss).
Derived from a stated objective; provably limits to NLL/Bayes; inherits FedGVI's
bounded-influence result only under the source theorem's stated loss, model, and
contamination assumptions. Exemplified by the logistic-regression baseline
(`fedference.bnn_baseline.fed_gvi_logreg` with `rcce`/`AR`).

### 2. Server-side, heuristic (complementary, sharp)

`robust_aggregate` divergence-reweighting (reverse KL) during belief-sharing
rounds. Only the **project-local recovery limit** (robustness 0 =
`log_linear_pool`) is proven. Under the stated bridge it is a categorical Eq. 7
specialization, not a full source-protocol identity. Do not attribute FedGVI's
per-client bounded-influence bound to this pooling heuristic. It is the
empirically sharp rule that wins the configured robustness verdict.

### 3. Server-side, objective-backed (complementary, conservative)

`variational_aggregate` is the objective-backed sibling of axis 2. Replacing
the reverse-KL update with forward cross-entropy
$\mathrm{CE}(q,s_n)=\mathrm{KL}(q\|s_n)+H(q)$ changes both orientation and the
common raw-weight scale. The resulting paired updates are exact
block-coordinate descent on the stated free energy
`aggregation_free_energy`; this does not derive the reverse-KL heuristic.
Each block update is non-increasing; a converged fixed point is coordinatewise
stationary. With the default `entropy_weight=1`, it recovers the log-linear
pool at robustness 0 (the same project-local corner as axis 2), and carries a
proven raw effective-weight bound: a diverging agent's weight
$a_n = w_n e^{-c\,\mathrm{CE}(q,s_n)} \le w_n$ vanishes, whereas the naive pool
holds it at the fixed $1/n$ on the diagnostic path. This is not, by itself, a
proof that the normalized consensus estimator is B-robust. The honest cost is
conservatism — the $-H(q)$ term
makes it a maximum-entropy-biased consensus, so it does *not* maximize peak accuracy.
It complements, never replaces, axis 2.

The triangle: axis 1 is source-theorem-backed under stated assumptions; axis 2
has conditional empirical wins and reversals *without* a server objective;
axis 3 is objective-backed *but* conservative. The
influence-weights and logistic-regression robustness figures exercise axes 2 and
1; the descent and legacy-named `bounded_influence` redescending-weight figure
exercise axis 3 without asserting estimator-level B-robustness. See
[`experiments-and-artifacts.md`](experiments-and-artifacts.md).

## Hierarchical multi-level federation (V2)

The flat sentinel world couples all agents at a single latent level (location).
V2 extends federation to arbitrary depth, reusing `log_linear_pool` at every
level independently — no new aggregation mechanism is introduced.

### 2-level model (Study 6)

`build_hierarchical_world` constructs a world where a **Level-2 context variable**
(quiet / alert) modulates the **Level-1 location prior** via conditioned priors
$D_{1|k}$. At inference time, `hierarchical_infer` computes:

$$\bar{q}_1 = \sum_k q_2[k] \cdot D_{1|k}$$

and updates the location posterior with this empirical prior. Federation is
applied at both levels separately by `share_round` / `log_linear_pool`. In the
current sentinel task, hierarchical inference matches the flat location
baseline while additionally resolving the context latent above chance; it does
not establish a location-accuracy advantage.

### 3-level model (Study 7)

`build_3level_world` stacks a **Level-3 meta-context** (low_threat / high_threat)
above the 2-level model. The conditioned-prior cascade is L3 → L2 → L1:

$$\bar{q}_2 = \sum_j q_3[j] \cdot D_{2|j}, \quad \bar{q}_1 = \sum_k q_2[k] \cdot D_{1|k}$$

`nlevel_infer` applies this cascade for any depth.

### N-level generic API

`LayerSpec` (a `@dataclass`) describes a single level: `name`, `n_states`,
`labels`, `default_prior`, and `conditioned_priors`. Passing a `list[LayerSpec]`
to `build_nlevel_world` constructs a world of arbitrary depth N≥2.
`nlevel_infer` handles any depth automatically. The canonical 3-level stack is
also specified declaratively in
`src/fedference/config/hierarchical_layers.yaml`.

**Key invariant.** Federation at every level reuses `log_linear_pool` unchanged.
Adding a new level requires only a new `LayerSpec`; no new aggregation logic is
needed.

## Scope boundaries

In scope: discrete categorical POMDP; queue-backed federation transport contract
(V3 — `src/fedference/federation/`; bit-identical to in-process, including a
single-machine OS-process helper but not true multi-machine deployment);
disjoint-FOV moving sentinel world with EFE-guided policy (V4 —
`pomdp.build_moving_world`,
`experiments.run_moving_world`); 2-level and N-level hierarchical POMDP federation
(V2 — `pomdp.build_hierarchical_world`, `pomdp.build_nlevel_world`,
`experiments.run_hierarchical_world`, `experiments.run_3level_world`); tempered
aggregation family `F_λ` (V1 — `aggregation_free_energy` with `entropy_weight`);
NumPy logistic-regression baseline and point-mass MLP complement (not a paper-scale GPU FedGVI result).

Three further items are implementation slices, not closed scientific scope: a
mean-field variational BNN plus diagonal-Gaussian site/cavity/factor-replacement
server and explicit CPU/MPS receipts (MAJ-2A; the cavity-conditioned client
optimizer and confirmatory sweep remain open); Gaussian and hybrid recovery
paths plus a minimal tracking fixture (MAJ-3; the full controlled benchmark
remains open); and a hash-, license-, schema-, split-, and receipt-bound
three-dataset UCI execution path (MAJ-6; the pilot, confirmatory inference, and
manuscript evidence pack remain open). Out-of-scope discussion and related
work: [`../../manuscript/23_discussion_limitations.md`](../../manuscript/23_discussion_limitations.md)
and [`../../manuscript/22_discussion_related_work.md`](../../manuscript/22_discussion_related_work.md).

**V3 boundary.** The original ISA listed "real multi-machine federation" as
out-of-scope. The current `FederationServer` + `FederationWorker` protocol over
`queue.Queue` transport serialises beliefs losslessly, `run_multiprocess_round`
runs the same protocol with single-machine OS worker processes, and
`run_socket_round` exercises real loopback TCP with a versioned configuration-
bound envelope, optional HMAC frame integrity, and persisted digest-verified
replay validation. This retires the
in-process serialization, unauthenticated-loopback, and in-memory-only replay
caveats. A caller-shared `ReplayGuard` can reject round-id reuse within one
running process; `PersistentReplayGuard` uses caller-owned SQLite state to
retain that protection across local process restarts. Durable multi-host
replay-domain design and the broader cross-host deployment boundary remain
open. The security assumptions and abuse paths are explicit in the
[repository threat model](../security/active_fedference-threat-model.md).

## See also

- Module map: [`architecture.md`](architecture.md)
- Experiment outputs: [`experiments-and-artifacts.md`](experiments-and-artifacts.md)
- Acceptance contract: [`../../ISA.md`](../../ISA.md)
