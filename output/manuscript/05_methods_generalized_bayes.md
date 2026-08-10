## Generalized Bayes: the route back to standard Bayes {#sec:method-genbayes}

The inference engine FedGVI federates is the generalized (Gibbs) posterior
[@bissiri2016general; @jiang2008gibbs; @jewson2018divergence;
@knoblauch2022generalized], which trades the likelihood for a loss $L$ and the
KL regularizer for a general divergence $\mathcal D$:

$$
q_n^\ast(s) \;=\; \arg\min_{q_n} \; \mathbb{E}_{q_n}\!\Big[\textstyle\sum_i L(s; o_i)\Big]
\;+\; \tfrac{1}{\tau}\, \mathcal D\!\big(q_n \,\|\, \pi_0\big),
$$ {#eq:gen-bayes}

with prior $\pi_0$, learning rate $\tau$, and regularizing divergence
$\mathcal D$. The
learning rate is part of the inferential specification, not a cosmetic constant;
coarsened-posterior and safe-Bayes work show why calibration of that temperature
matters under misspecification [@miller2018coarsening; @grunwald2012safe], where
ordinary Bayes concentrates around a KL pseudo-truth rather than literal truth
when the model family is wrong [@kleijn2012misspecification]. We name the object
[@eq:gen-bayes] defines.

\begin{definition}[Generalized-(Gibbs)-Bayes posterior]\label{def:generalized-bayes}
For a loss $L$, prior $\pi_0$, learning rate $\tau>0$, and divergence
$\mathcal D$, the generalized-Bayes posterior is the minimizer $q_n^\ast$ of
(\ref{eq:gen-bayes}). For
$\mathcal D=\mathrm{KL}$ the minimizer is the tempered softmax
$$
q_n^\ast(s) \;\propto\; \pi_0(s)\,\exp\!\big(-\tau \textstyle\sum_i L(s; o_i)\big),
$$
\label{eq:tempered-softmax}
implemented in \texttt{generalized\_bayes.generalized\_posterior}.
\end{definition}

The tempered softmax of [@eq:tempered-softmax], stated in the definition above,
is not an approximation: it
is the exact closed-form minimizer of [@eq:gen-bayes] when the regularizer is the
KL divergence, because the categorical support is finite and the objective is
strictly convex in $q$. The recovery to standard Bayes follows by choosing the
loss. With $L=\mathrm{NLL}$, $\mathrm{NLL}(p, o) = -\log p(o)$, the exponential
in that tempered softmax becomes a product of likelihoods and the minimizer is
*exactly* standard Bayes; [@eq:standard-bayes] in [@sec:method-aggregation]
states that corner, and Corollary \ref{cor:closed-form-bayes} there pins it to the closed-form
prior-times-likelihood product. The largest observed discrepancy between
`generalized_posterior` in this regime and the analytic Bayes posterior is
5.55e-17, reported in [@sec:results-recovery] — exact to
machine precision (a maximum deviation of about one ULP), not merely close.

FedGVI computes each client update against a *cavity* rather than the full
posterior, so a contributing agent does not double-count its own previous
message. We name that operation.

\begin{definition}[Cavity / PVI factor update]\label{def:cavity}
The cavity removes agent $n$'s factor from the colony posterior in
natural-parameter (log) space,
$$
q_{-n}(s)
\;=\;
\frac{q(s)/t_n(s)}{\sum_{s'} q(s')/t_n(s')}
\;=\;
\operatorname{softmax}\!\big(\log q(s)-\log t_n(s)\big),
$$
\label{eq:cavity}
where the final expression makes the normalization explicit; the
partitioned-variational-inference (PVI) update
re-multiplies a refreshed factor onto the cavity of (\ref{eq:cavity}). Taking a
cavity and re-multiplying the original site factor restores the global posterior
\begin{equation}
q(s)=\frac{q_{-n}(s)t_n(s)}{\sum_{s'}q_{-n}(s')t_n(s')},
\label{eq:factor-replacement}
\end{equation}
with the original site factor, the recombination identity, the
property \texttt{generalized\_bayes.cavity} and
\texttt{generalized\_bayes.update\_factor} satisfy.
\end{definition}

The numbered recombination identity is [@eq:factor-replacement].

The cavity of [@eq:cavity] is the discrete analogue of the expectation-
propagation / partitioned-VI cavity used outside active inference
[@ashman2022partitioned; @bui2018partitioned], imported here so that the per-agent generalized-Bayes
update of [@eq:gen-bayes] is computed against the colony belief with the agent's
own contribution removed — exactly the sensory-attenuation discipline the
belief-sharing round of [@sec:method-belief-sharing] requires. What remains
unspecified in [@eq:gen-bayes] are its two ingredients — the divergence
$\mathcal D$ and
the loss $L$ — whose robust members and standard-Bayes limits
[@sec:method-divergences] develops next; the aggregation identity
([@sec:method-aggregation]) then federates the resulting per-agent posteriors.

The authoritative notation supplement makes the same normalization and
recombination contract explicit in [@eq:notation-cavity] and
[@eq:notation-factor-replacement]; those equations govern the symbols used by
the implementation and all later supplements.

## Conjugate likelihood learning for the shared model {#sec:method-learning}

Active-inference agents learn the parameters of their generative model, not just
plan with them [@smith2020active; @friston2024federated]. The likelihood matrix
$A$ carries a Dirichlet prior with concentration $a$ over each column, updated
conjugately by accumulating observation-state co-occurrence counts
([@eq:dirichlet-update]), giving the column-normalized expected likelihood. The update of [@eq:dirichlet-update]
is driven by the expected sufficient statistics under the data-generating model,
so as the concentrations accumulate $\mathbb{E}[A]$ converges to the true
likelihood. Convergence is measured by the per-column KL divergence summed over
hidden states, which decreases monotonically toward the standard-Bayes fixed
point; [@sec:results-language] reports the learning curve, where the KL falls
from 3.4231 to 0.0027 across
24 count batches. A forgetting hyperprior optionally decays
the running mass toward an asymptote so the agent does not become infinitely
confident; with the hyperprior disabled the classical unbounded accumulation of
[@eq:dirichlet-update] is recovered. The implementation is
`dirichlet_learning.learn_likelihood`.

## Bayesian model reduction for structure comparison {#sec:method-bmr}

Structure learning in the active-inference frame proceeds by Bayesian model
reduction (BMR): given a full model with Dirichlet posterior `post` under prior
`prior`, the change in negative variational free energy from swapping in a
*reduced* prior — for example one that prunes a redundant column toward zero —
is available in closed form without re-running inference
[@friston2011post; @smith2020active]. Because the likelihood is shared, the
reduced posterior is `post + reduced_prior - prior`, and the free-energy
difference is a difference of log multivariate Beta functions ([@eq:bmr-deltaf]),
where $\ln B(a) = \sum_k \ln\Gamma(a_k) - \ln\Gamma(\sum_k a_k)$ is the log
Dirichlet normalizer. A positive $\Delta F$ in [@eq:bmr-deltaf] means the reduced
model has more evidence — the pruned structure was redundant and should be
adopted; a negative $\Delta F$ means the reduction destroyed something the data
support. When the reduced prior equals the prior the score is identically zero,
the no-reduction fixed point. [@sec:results-emergence] reports
$\Delta F = 3.68$ for a redundant reduction (accepted)
against $\Delta F = -27.67$ for a supported one
(rejected). The implementation is `bayesian_model_reduction.reduce`.
