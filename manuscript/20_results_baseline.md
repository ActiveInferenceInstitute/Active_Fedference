## Exploratory generalized-Bayes logistic-regression baseline {#sec:results-baseline}

The sweep of [@sec:results-robustness] characterizes the heuristic server axis.
This exploratory baseline instead probes the client-side axis in a synthetic
point-estimate logistic-regression setting. The cited robust-Bayes and federated-
learning results provide the source context [@mcmahan2017communication;
@ashman2022partitioned; @bui2018partitioned]; this finite experiment does not
reproduce their source protocol or inherit their conclusions unconditionally.

An exploratory federated logistic-regression point-estimate proxy is trained
under client label contamination. Standard clients use the NLL gradient with
L2 coefficient $0.05$; the comparison uses the RCCE gradient at
$q={{BNN_ROBUSTNESS_LOSS_PARAM}}$ with L2 coefficient $0.10$. The legacy
`divergence="KLD"` and `divergence="AR"` arguments select those coefficients
for compatibility; this module does not evaluate KL or Alpha-Rényi divergence
between weight distributions, represent posterior covariance, or implement a
FedGVI generalized-posterior objective. The estimand is therefore a composite
contrast between two joint loss-and-shrinkage configurations, not an isolated
RCCE effect. RCCE's $q\to0$ gradient limit recovers the NLL gradient, while the
separate categorical recovery identities remain in [@sec:results-recovery].

The corresponding bounded-influence claim remains source-conditional on
FedGVI's assumptions [@mildner2025fedgvi]; this finite curve does not test that
theorem. The losses connect the density-power line
[@basu1998robust], generalized cross-entropy [@zhang2018generalized], and the
generalized-Bayes objective [@bissiri2016general; @knoblauch2022generalized].

The displayed configuration uses $q = {{BNN_ROBUSTNESS_LOSS_PARAM}}$ and
{{BNN_ROBUSTNESS_N_PER}} points per class per client. Those are configured
inputs, not parameters selected by this report. Only the peak-margin
contamination level is selected within the displayed contamination sweep, by
the maximum composite-configuration accuracy difference. The neighboring-$q$
sensitivity check is descriptive stability evidence, not leakage-free
calibration or confirmatory model selection.

As contamination rises, the two configuration curves remain close at lower
rates and separate over part of the moderate-to-high range. The largest
displayed composite-configuration mean difference occurs at
{{BNN_ROBUSTNESS_PEAK_CONTAM}} contamination, with margin
{{BNN_ROBUSTNESS_PEAK_GAP}}. Neighboring tested loss parameters retain a
thresholded separation at more than one contamination level, and the plotted
bands are seed-level {{CI_PERCENT}}% bootstrap intervals.

At the highest swept rate, {{BNN_ROBUSTNESS_MAX_CONTAM}}, both configurations
decline and no reliable ordering remains. Retaining that endpoint is necessary
for the finite-sweep interpretation. Together with the within-sweep
peak-contamination selection, it makes this an exploratory conditional pattern
rather than a precalibrated or confirmatory robustness result.

The separation in this small logistic-regression setting is modest. The recovery
identities establish implementation compatibility at the named limit
([@sec:results-recovery]); the bounded-influence result comes from the cited
FedGVI theorem under its assumptions [@mildner2025fedgvi], not from this curve's
gap. Source-dataset parity, leakage-free calibration, and posterior-uncertainty
experiments remain open ([@sec:future-scale]).

![Client-loss comparison under label contamination. Source relation: exploratory original-project point-estimate logistic-regression proxy; estimand: clean held-out accuracy for two joint loss-and-L2 configurations; uncertainty: seed-level percentile-bootstrap interval. The model uses one point-estimate weight vector for each of {{BNN_N_CLIENTS}} clients with {{BNN_ROBUSTNESS_N_PER}} points per class per client; no posterior covariance is computed. The x-axis is each client's label-contamination fraction. The y-axis is held-out accuracy on a clean synthetic test set, averaged over {{BNN_ROBUSTNESS_N_SEEDS}} independent seeds. Distinct markers and line styles identify NLL with L2 coefficient $0.05$ and RCCE with L2 coefficient $0.10$ at configured $q={{BNN_ROBUSTNESS_LOSS_PARAM}}$; bands show {{CI_PERCENT}}% seed-bootstrap intervals. Only the peak contamination is selected within the displayed grid. Because loss and shrinkage change together, the contrast cannot identify an RCCE-only effect. It does not establish leakage-free calibration, universal robustness, posterior uncertainty, an Alpha-Rényi result, or source-protocol replication.](../output/figures/bnn_robustness.png){#fig:bnn-robustness width=80%}

[@fig:bnn-robustness] is exploratory per-client proxy evidence. Its loss-limit
check, finite synthetic contrast, and the separate source-conditional theorem
have different roles; none comes from the server comparison of
[@sec:results-verdict]. The
authoritative three-axis boundary is [@sec:robustness-axes].

**PyTorch deterministic-MLP complement (executed).** The optional pipeline also
instantiates generalized variational inference with a point-mass MLP family:
Linear→ReLU→Linear→softmax with {{BNN_HIDDEN_DIM}} hidden units. Clients use the
density-power $\beta$-loss at $\beta = {{BNN_BETA}}$ for
{{BNN_N_STEPS}} Adam steps, and per-test-point predictions are fused with
`robust_aggregate` at `robustness = {{BNN_ROBUSTNESS}}` under PyTorch
{{PYTORCH_VERSION}}.

The executed consensus is a valid probability simplex, with maximum unit-mass
deviation {{BNN_CONSENSUS_SUM}}, and repeated seeded runs are bit-identical
({{BNN_DETERMINISTIC}}). At contamination {{BNN_TORCH_CONTAM}}, held-out
consensus accuracy is {{BNN_TORCH_STD_ACC}} for the $\beta\to 0$ client and
{{BNN_TORCH_ROBUST_ACC}} for the $\beta={{BNN_BETA}}$ client.

This single-seed endpoint demonstrates API transfer and deterministic execution,
not posterior-uncertainty inference, model-class universality, or a robust-loss
advantage at neural-network scale. It is separate from the
{{BNN_ROBUSTNESS_N_SEEDS}}-seed exploratory logistic-regression curve above and
from the source-conditional theorem. When PyTorch is absent, the pipeline records
unavailable-value sentinels; builds exercising this optional lane install the
`torch` extra ([@sec:reproducibility]).
