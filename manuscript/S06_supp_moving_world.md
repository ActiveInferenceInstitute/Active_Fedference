## Supplement: moving-world methods and condition definitions {#sec:supp-moving}

This supplement supplies the construction that [@sec:results-moving] defers here:
exactly how the moving-sentinel world is built, how the three actions and the
expected-free-energy policy move an agent, and how the *isolated*,
*communicating*, and *EFE-guided* conditions differ. It answers the mechanical
question left open in the main section — what precisely is held fixed and what
varies across the three conditions whose accuracies are contrasted there — so
the reported binary-complement numbers can be read against their generative
model rather than taken on trust.

The moving-world generative model is built by `build_moving_world`. A linear
grid of {{MOVING_N_POSITIONS}} cells holds one binary hidden state — the half of
the grid (left = state 0, right = state 1) that contains the threat. The
{{MOVING_N_AGENTS}} sentinels start at evenly tiled positions
($i \cdot \lfloor n_{\text{positions}} / n_{\text{agents}} \rfloor$) and each
observe a half-open field-of-view window. With the default setup the two FOVs
are disjoint, one per half. Each agent's likelihood is a $2 \times 2$ matrix
over outcomes (detected / not_detected) given the binary state, with a confident
signed reading for the half the agent watches; the transition tensor encodes
three deterministic control paths — stay, left (reflecting at cell 0), and right
(reflecting at the last cell). The hidden-state prior is uniform.

Action selection has two regimes. The random conditions draw each agent's move
uniformly from the three controls. The EFE-guided condition uses
`efe_policy_select`: for every candidate move it lands the agent at the
deterministic next position, reconstructs the likelihood from that viewpoint,
and scores the move by the expected posterior entropy after one observation,
$H = \sum_o P(o)\,H(P(s \mid o))$ — taking the entropy-minimizing
(information-seeking) step.

We compare three conditions — *isolated* (random moves, no sharing),
*communicating* (random moves plus a per-step log-linear-pool consensus), and
*EFE-guided* (information-seeking moves plus the same per-step sharing) — over
{{MOVING_N_TRIALS}} trials of {{MOVING_N_STEPS}} steps each, scoring the pooled
consensus against the true state. All numerics are deterministic given the run
seed.
