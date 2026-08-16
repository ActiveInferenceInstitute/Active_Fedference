## Moving sentinel world: communication benefit depends on field of view {#sec:results-moving}

The hidden-state/action relation for this extension is summarized in the
categorical loop schematic [@fig:pomdp-loop]; the results below remain the
executed moving-world comparisons, not a claim that the schematic's full loop is
present in every flat belief-sharing study.

The static sentinel world lets every agent observe the same shared latent, so
belief sharing is a refinement rather than a requirement. To stress the
*necessity* of communication we add movement and **disjoint** fields of view.
The world is a linear grid of 4 cells holding a single
binary threat — left half (state 0) or right half (state 1). The
2 sentinels start at evenly tiled positions and each observe a
half-open window of cells, so in the default setup agent 0 watches the left half
and agent 1 the right half: their views do not overlap. Each agent's likelihood
is a confident, signed presence reading for the half it can see, and three
control paths (stay / left / right) let it reposition. The expected-free-energy
policy scores each candidate move by the expected posterior entropy after one
observation and takes the most information-seeking step.

We run 960 trials of 6 steps each under three
conditions: *isolated* (random moves, no sharing), *communicating* (random moves
plus a log-linear-pool consensus each step), and *EFE-guided* (information-seeking
moves plus the same sharing). The measured consensus accuracies are
0.999 (isolated), 0.977
(communicating), and 0.978 (EFE-guided), with a communicating
free-energy gap of -0.528 nats relative to the isolated
baseline (negative: no free-energy advantage over isolated in this
binary-complement regime of logically complete half-views) ([@fig:moving-world]).

Across 128 independent seeds the EFE-guided accuracy is
0.983 (95 % CI 0.982–0.984),
the communicating (random-moves + sharing) accuracy is 0.982 (95 % CI
0.982–0.983), and the isolated accuracy is
0.999 (95 % CI 0.999–0.999).
In this binary-complement regime the isolated condition is in fact
significantly *higher* on accuracy than the EFE-guided sharing condition —
their 95 % intervals do not overlap — and the EFE-vs-isolated
accuracy contrast yields Wilcoxon signed-rank $p = 0.0000$
(significant; isolated higher), effect size
$r = 1.000$ (large). Sharing is therefore
not merely unnecessary in this regime; it costs a small but reliable amount
of accuracy. Nor does sharing
lower free energy here: the EFE free-energy gap (isolated surprise minus the
EFE-guided condition's surprise) is -0.368 nats
(95 % CI -0.384–-0.353) — negative,
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

![Moving sentinel world across isolated, communicating, and EFE-guided conditions.
Source relation: original project schematic for the moving-world protocol;
estimand: condition-level consensus accuracy, signed free-energy gap, and
steps-to-consensus proxy in the stated native units; uncertainty: deterministic
seeded run, so no resampling interval is shown. Moving sentinel world across the three conditions (x-axis is condition:
isolated, communicating, EFE-guided). Left panel: y-axis shows consensus
accuracy (fraction of 960 trials whose pooled argmax matches
the truth). Center panel: y-axis shows the signed free-energy gap in nats
(isolated surprise minus the condition's surprise on the true state, so a
positive value would mean lower free energy than isolated; the measured gaps
are negative, plotted against a zero reference line and annotated per bar).
Right panel: y-axis shows a coarse steps-to-consensus proxy, with per-bar
value annotations showing the three conditions are essentially tied. Each
colony runs 6 steps over a 4-cell linear
grid with 2 disjoint-FOV agents. Deterministic seeded run,
so the bars carry no error band.](../output/figures/moving_world.png){#fig:moving-world width=80%}

### Disjoint field-of-view extension {#sec:results-disjoint-fov}

To test whether communication is necessary (not merely beneficial) when
observations are non-overlapping, we extend Study 5 to 3 agents
each observing a 2-position disjoint window of a
6-position state space (chance-level accuracy
0.167). Isolated agents achieve mean accuracy
0.35 — above chance but far from decisive, since no single
agent can infer the global state from a partial window alone. Communicating
agents pool complementary beliefs to reach 0.55 (gap
0.19).

This is now a powered result, not an illustrative point estimate. Across
128 independent seeds the isolated accuracy is 0.326
(95% CI 0.320–0.332) and the communicating accuracy is
0.493 (95% CI 0.487–0.499), both clearing
the 0.167 chance baseline — isolated agents are *not* at
chance, since a partial FOV plus majority voting still carries some signal.
The paired Wilcoxon signed-rank test (communicating vs. isolated, matched by
seed) gives $p = 0.0000$, effect size $r = 1.000$
(large): communicating beats isolated on every one of the
128 seeds, which is also why the p-value is at the smallest a
128-seed paired sign test can report — it should be read as "every
seed agreed," not as a precise magnitude of evidence beyond that floor. Given
isolated performance is above chance, the precise claim is not that
communication is *logically* necessary for any signal at all, but that it is
necessary to approach the communicating-level accuracy under fully disjoint
observations: the gap between the two conditions is significant, large, and
reproducible, unlike the binary-complement contrast above.

We separately quantify EFE-guided navigation rather than asserting an
unquantified "widens the gap" effect. In a matched but smaller-scale
disjoint-FOV movement-policy comparison (2 agents,
4-position binary-state grid, belief sharing active in
both arms), EFE-guided accuracy is 0.975 versus
0.977 for random movement ($p = 0.1046$,
negligible effect): the two movement policies are not
significantly different, because both are already near ceiling once belief
sharing is active. We report this as the null result it is rather than
claiming an unmeasured EFE benefit. [@fig:disjoint-fov-world] summarizes the
necessity result.

![Disjoint-field-of-view communication and navigation comparison.
Source relation: source-inspired original project extension of the moving-world
mechanism; estimand: condition-level consensus accuracy in the two declared
disjoint-FOV protocols; uncertainty: across-seed standard-deviation error bars.
Disjoint-FOV extension of the moving sentinel world, as a two-panel figure
whose panels come from two separately configured experiments. Left panel
(communication necessity, 3 agents each observing a
2-position non-overlapping window of the
6-position world): the x-axis is the condition (isolated
vs.\ communicating); the y-axis is consensus accuracy — drawn as accuracy,
the fraction of trials whose pooled argmax matches the true state. The accuracy gap between communicating and isolated conditions quantifies
the necessity of belief sharing under fully disjoint fields of view, now backed
by the paired Wilcoxon test in the text ($p = 0.0000$) rather than
a single point estimate. Right panel (EFE vs random navigation, a smaller
2-agent, 4-position configuration):
the x-axis is the movement policy (EFE-guided vs.\ random); the y-axis is
final consensus accuracy — the panel is titled as a null result because that
is what it shows. Both policies sit near ceiling once belief sharing is active, so the
EFE-guided vs.\ random contrast is the null result reported in the text
($p = 0.1046$). In both panels, bars show accuracy averaged
across seeds; error bars
show the across-seed standard deviation.](../output/figures/disjoint_fov_world.png){#fig:disjoint-fov-world width=80%}
