## Moving sentinel world: communication benefit depends on field of view {#sec:results-moving}

The hidden-state/action relation for this extension is summarized in the
categorical loop schematic [@fig:pomdp-loop]; the results below remain the
executed moving-world comparisons, not a claim that the schematic's full loop is
present in every flat belief-sharing study.

The static sentinel world lets every agent observe the same shared latent, so
belief sharing is a refinement rather than a requirement. To stress the
*necessity* of communication we add movement and **disjoint** fields of view.
The world is a linear grid of {{MOVING_N_POSITIONS}} cells holding a single
binary threat — left half (state 0) or right half (state 1). The
{{MOVING_N_AGENTS}} sentinels start at evenly tiled positions and each observe a
half-open window of cells, so in the default setup agent 0 watches the left half
and agent 1 the right half: their views do not overlap. Each agent's likelihood
is a confident, signed presence reading for the half it can see, and three
control paths (stay / left / right) let it reposition. The expected-free-energy
policy scores each candidate move by the expected posterior entropy after one
observation and takes the most information-seeking step.

We run {{MOVING_N_TRIALS}} trials of {{MOVING_N_STEPS}} steps each under three
conditions: *isolated* (random moves, no sharing), *communicating* (random moves
plus a log-linear-pool consensus each step), and *EFE-guided* (information-seeking
moves plus the same sharing). The measured consensus accuracies are
{{MOVING_ACC_ISOLATED}} (isolated), {{MOVING_ACC_COMMUNICATING}}
(communicating), and {{MOVING_ACC_EFE}} (EFE-guided), with a communicating
free-energy gap of {{MOVING_FE_GAP_COMMUNICATING}} nats relative to the isolated
baseline (negative: no free-energy advantage over isolated in this
binary-complement regime of logically complete half-views) ([@fig:moving-world]).

Across {{MOVING_N_SEEDS}} independent seeds the EFE-guided accuracy is
{{MOVING_ACC_EFE_MEAN}} ({{CI_PERCENT}} % CI {{MOVING_ACC_EFE_CI_LO}}–{{MOVING_ACC_EFE_CI_HI}}),
the communicating (random-moves + sharing) accuracy is {{MOVING_ACC_COMM_MEAN}} ({{CI_PERCENT}} % CI
{{MOVING_ACC_COMM_CI_LO}}–{{MOVING_ACC_COMM_CI_HI}}), and the isolated accuracy is
{{MOVING_ACC_ISO_MEAN}} ({{CI_PERCENT}} % CI {{MOVING_ACC_ISO_CI_LO}}–{{MOVING_ACC_ISO_CI_HI}}).
In this binary-complement regime the isolated condition is in fact
significantly *higher* on accuracy than the EFE-guided sharing condition —
their {{CI_PERCENT}} % intervals do not overlap — and the EFE-vs-isolated
accuracy contrast yields Wilcoxon signed-rank $p = {{MOVING_WILCOX_PVALUE}}$
({{MOVING_SIGNIFICANCE_VERDICT}}; isolated higher), effect size
$r = {{MOVING_EFFECT_SIZE}}$ ({{MOVING_EFFECT_LABEL}}). Sharing is therefore
not merely unnecessary in this regime; it costs a small but reliable amount
of accuracy. Nor does sharing
lower free energy here: the EFE free-energy gap (isolated surprise minus the
EFE-guided condition's surprise) is {{MOVING_FE_GAP_EFE_MEAN}} nats
({{CI_PERCENT}} % CI {{MOVING_FE_GAP_EFE_CI_LO}}–{{MOVING_FE_GAP_EFE_CI_HI}}) — negative,
so the pooled consensus is slightly *more* surprised by the true state than the
isolated baseline, because a single agent's view already suffices. The
accuracy case for *necessity* is therefore made only in the larger-state-space
disjoint-FOV extension below, not by these binary-complement numbers.

We report these numbers as measured, not assumed. The binary world carries a
logical complement: ruling out one's own half implies the other, so a single
agent's "not detected" still carries information about the global state, and an
isolated agent is not strictly blind. By design, the intended
*cannot-decide-alone* regime is the high-noise, few-step corner where one
sensor's evidence cannot overcome the flat prior; there belief sharing is
meant to fuse the two complementary views into a decisive consensus. That
regime is not separately measured here — the construction, the three actions, the EFE rule, and the
exact condition protocol are detailed in the supplement ([@sec:supp-moving]).

![Source relation: original project schematic for the moving-world protocol;
estimand: condition-level consensus accuracy, signed free-energy gap, and
steps-to-consensus proxy in the stated native units; uncertainty: deterministic
seeded run, so no resampling interval is shown. Moving sentinel world across the three conditions (x-axis is condition:
isolated, communicating, EFE-guided). Left panel: y-axis shows consensus
accuracy (fraction of {{MOVING_N_TRIALS}} trials whose pooled argmax matches
the truth). Center panel: y-axis shows the signed free-energy gap in nats
(isolated surprise minus the condition's surprise on the true state, so a
positive value would mean lower free energy than isolated; the measured gaps
are negative, plotted against a zero reference line and annotated per bar).
Right panel: y-axis shows a coarse steps-to-consensus proxy, with per-bar
value annotations showing the three conditions are essentially tied. Each
colony runs {{MOVING_N_STEPS}} steps over a {{MOVING_N_POSITIONS}}-cell linear
grid with {{MOVING_N_AGENTS}} disjoint-FOV agents. Deterministic seeded run,
so the bars carry no error band.](../output/figures/moving_world.png){#fig:moving-world width=80%}

### Disjoint field-of-view extension {#sec:results-disjoint-fov}

To test whether communication is necessary (not merely beneficial) when
observations are non-overlapping, we extend Study 5 to {{V4_N_AGENTS}} agents
each observing a {{V4_FOV_WIDTH}}-position disjoint window of a
{{V4_N_POSITIONS}}-position state space (chance-level accuracy
{{V4_CHANCE_BASELINE}}). Isolated agents achieve mean accuracy
{{V4_ISOLATED_ACCURACY}} — above chance but far from decisive, since no single
agent can infer the global state from a partial window alone. Communicating
agents pool complementary beliefs to reach {{V4_COMMUNICATING_ACCURACY}} (gap
{{V4_ACCURACY_GAP}}).

This is now a powered result, not an illustrative point estimate. Across
{{V4_N_SEEDS}} independent seeds the isolated accuracy is {{V4_ISO_MEAN}}
({{CI_PERCENT}}% CI {{V4_ISO_CI_LO}}–{{V4_ISO_CI_HI}}) and the communicating accuracy is
{{V4_COMM_MEAN}} ({{CI_PERCENT}}% CI {{V4_COMM_CI_LO}}–{{V4_COMM_CI_HI}}), both clearing
the {{V4_CHANCE_BASELINE}} chance baseline — isolated agents are *not* at
chance, since a partial FOV plus majority voting still carries some signal.
The paired Wilcoxon signed-rank test (communicating vs. isolated, matched by
seed) gives $p = {{V4_WILCOX_PVALUE}}$, effect size $r = {{V4_EFFECT_SIZE}}$
({{V4_EFFECT_LABEL}}): communicating beats isolated on every one of the
{{V4_N_SEEDS}} seeds, which is also why the p-value is at the smallest a
{{V4_N_SEEDS}}-seed paired sign test can report — it should be read as "every
seed agreed," not as a precise magnitude of evidence beyond that floor. Given
isolated performance is above chance, the precise claim is not that
communication is *logically* necessary for any signal at all, but that it is
necessary to approach the communicating-level accuracy under fully disjoint
observations: the gap between the two conditions is significant, large, and
reproducible, unlike the binary-complement contrast above.

We separately quantify EFE-guided navigation rather than asserting an
unquantified "widens the gap" effect. In a matched but smaller-scale
disjoint-FOV movement-policy comparison ({{V4_EFE_N_AGENTS}} agents,
{{V4_EFE_N_POSITIONS}}-position binary-state grid, belief sharing active in
both arms), EFE-guided accuracy is {{V4_EFE_ACC_MEAN}} versus
{{V4_RANDOM_ACC_MEAN}} for random movement ($p = {{V4_EFE_WILCOX_PVALUE}}$,
{{V4_EFE_EFFECT_LABEL}} effect): the two movement policies are not
significantly different, because both are already near ceiling once belief
sharing is active. We report this as the null result it is rather than
claiming an unmeasured EFE benefit. [@fig:disjoint-fov-world] summarizes the
necessity result.

![Source relation: source-inspired original project extension of the moving-world
mechanism; estimand: condition-level consensus accuracy in the two declared
disjoint-FOV protocols; uncertainty: across-seed standard-deviation error bars.
Disjoint-FOV extension of the moving sentinel world, as a two-panel figure
whose panels come from two separately configured experiments. Left panel
(communication necessity, {{V4_N_AGENTS}} agents each observing a
{{V4_FOV_WIDTH}}-position non-overlapping window of the
{{V4_N_POSITIONS}}-position world): the x-axis is the condition (isolated
vs.\ communicating); the y-axis is consensus accuracy — drawn as accuracy,
the fraction of trials whose pooled argmax matches the true state. The accuracy gap between communicating and isolated conditions quantifies
the necessity of belief sharing under fully disjoint fields of view, now backed
by the paired Wilcoxon test in the text ($p = {{V4_WILCOX_PVALUE}}$) rather than
a single point estimate. Right panel (EFE vs random navigation, a smaller
{{V4_EFE_N_AGENTS}}-agent, {{V4_EFE_N_POSITIONS}}-position configuration):
the x-axis is the movement policy (EFE-guided vs.\ random); the y-axis is
final consensus accuracy — the panel is titled as a null result because that
is what it shows. Both policies sit near ceiling once belief sharing is active, so the
EFE-guided vs.\ random contrast is the null result reported in the text
($p = {{V4_EFE_WILCOX_PVALUE}}$). In both panels, bars show accuracy averaged
across seeds; error bars
show the across-seed standard deviation.](../output/figures/disjoint_fov_world.png){#fig:disjoint-fov-world width=80%}
