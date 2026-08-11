# Results: recovery checks and study suite {#sec:results}

Every quantitative assertion in this and the following results sections is a
generated token, hydrated by the manuscript-variable generator from analysis
outputs produced by `src/analysis/workflow.py` and
`src/fedference/experiments/` — no number is transcribed by hand. The studies implement categorical source-mechanism
analogues of the colony belief-sharing scenario [@friston2024federated] and add the contaminated-sentinel
robustness sweep and the federated neural-network baseline that are this
paper's robust-federated-learning contribution. All runs are deterministic under
seed 0.

We lead with the recovery limits, not with a study, because they are what makes
the studies a single coherent system rather than a collection of unrelated
experiments. The generalized-Bayes machinery of [@sec:methods] contains
the standard-Bayes client corner, while the server has the exact project-local
zero-robustness log-linear-pool identity. Under the explicit bridge in
[@sec:method-aggregation], that pool is a categorical specialization of Eq. 7's
message-combination term rather than the complete source protocol. We verify
those limited identities to machine precision before reporting anything built on
top of them.

## Recovery limits: standard-Bayes and project-pool corners are exact to machine precision {#sec:results-recovery}

The identities that anchor every result are the client recovery of standard
Bayes at the KL/NLL/$\beta\to0$ and $q_{\rm loss}\to0$ limits plus the
project-local server recovery to the log-linear pool at $c=0$ — the scoped
claims of [@sec:formalism] (Corollary \ref{cor:closed-form-bayes}, Lemma
\ref{lem:renyi-kl-limit}, Theorem \ref{thm:belief-sharing-recovery}). These
are not figures but exact equalities, pinned by the locked core test suite.
Under the theorem's shared-support, posterior-log-potential, and fixed-weight
assumptions, the server pool specializes Eq. 7's message-combination term; it
does not reproduce the source construction in full. Robustness is a tested
extension that vanishes at the stated recovery limits.

The five residuals below are the maximum absolute deviations between each
generalized-Bayes object and the standard object it must reproduce in the
trusting limit. Each is a deterministic constant of the mathematics, not a
per-run sample: it is reported as the maximum absolute deviation over the
recovery band, which is exactly $0$ where the implementation evaluates the
closed form at the limit (the Rényi/loss switch) and otherwise a tiny
floating-point residual:

- The server-side aggregator at zero robustness equals the log-linear pool
  ([@eq:robust-identity], Theorem \ref{thm:belief-sharing-recovery}): maximum absolute deviation
  0. This is the *naive-recovery* limit of the
  server-side heuristic — the only property proven for that axis (see
  [@sec:robustness-axes]).
- The generalized posterior under the KL divergence and the NLL loss equals the
  closed-form prior$\times$likelihood Bayes posterior
  ([@eq:standard-bayes], Corollary \ref{cor:closed-form-bayes}): maximum absolute deviation
  5.55e-17.
- The Rényi divergence recovers KL as $\alpha\to1$
  ([@eq:renyi-limit], Lemma \ref{lem:renyi-kl-limit}): residual 0.
- The $\beta$-loss recovers the NLL as $\beta\to 0$
  ([@eq:beta-loss], Proposition \ref{prop:robust-loss-recovery}): residual 0; and the
  robust categorical cross-entropy recovers the NLL as $q_{\rm loss}\to 0$
  ([@eq:rcce-loss], Proposition \ref{prop:robust-loss-recovery}): residual 0.

Because the Rényi divergence and the two categorical losses switch to their
exact closed form inside narrow numerical-stability bands around the limit
point (the Rényi switch band for $\alpha$ and the categorical-loss switch band
for $q_{\rm loss}$ and $\beta$), the three zero residuals above confirm that branch equals
the standard object — not, by themselves, that the *general* formula converges
there. As a genuine (non-branch) convergence witness, evaluating each general
formula strictly *outside* its switch band — $q_{\rm loss} = \beta =
1.00 \times 10^{-6}$ for the
categorical losses and $\alpha = 1.00001$ for the Rényi
divergence — gives residuals 1.12e-05 (rcce),
1.24e-05 ($\beta$-loss), and
1.66e-05 (Rényi): nonzero (a small multiple of the
input offset itself, as the first-order Taylor behavior near the limit
predicts) yet still several orders of magnitude below the $O(1)$ scale of the
loss/divergence values being compared, and shrinking monotonically as the
offset shrinks toward the switch band (verified in
`tests/fedference/test_core_identities.py`). This is evidence that the
general formula itself converges to the standard-Bayes limit, not merely that
the implementation switches to it exactly at the corner.

The first residual is the naive-aggregate limit of the *server-side* heuristic
(Theorem \ref{thm:belief-sharing-recovery}); the latter four are the per-agent generalized-Bayes recoveries
(Corollary \ref{cor:closed-form-bayes} +
Proposition \ref{prop:robust-loss-recovery}) and the divergence-family recovery
in the Rényi limit (Lemma \ref{lem:renyi-kl-limit},
[@eq:renyi-limit]) that define the theorem-bearing FedGVI axis under matching
assumptions. Keeping the three axes distinct
at the level of the recovery limits is what lets the robustness claims of
[@sec:results-robustness] and [@sec:results-baseline] rest on the per-agent axis
without leaning on the heuristic.

257 of 259 acceptance criteria are verified. The
pure-NumPy/SciPy core carries project test coverage of
90.10% (gate $\ge 90\%$), with every stochastic step threaded
through a single seeded `np.random.default_rng(0)`.
[@sec:reproducibility] records the full environment fingerprint, and the
expected-free-energy identity that underwrites the active-inference substrate is
proven and visualized in [@sec:formalism-efe] ([@fig:efe-decomp]).
