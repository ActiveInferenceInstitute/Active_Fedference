# Parameter recovery: acuity selection on the tested grid {#sec:results-parameter-recovery}

Parameter recovery probes whether the executed observation model contains enough
information to distinguish sensor acuity under the study design. We sweep acuity
values {{PARAM_RECOVERY_ACUITY_GRID}}: for each true acuity the model generates
{{PARAM_RECOVERY_N_OBSERVATIONS}} synthetic observations per trial across
{{PARAM_RECOVERY_N_TRIALS}} independent trials, fits acuity by marginal-likelihood
grid search, and compares the recovered value with ground truth.

Across the sweep the mean absolute recovery error is {{PARAM_RECOVERY_MEAN_ABS_ERROR}} and the coefficient of determination of mean-recovered versus true acuity is $R^2$ = {{PARAM_RECOVERY_R_SQUARED}}.
Within this finite grid and observation budget, recovered acuity tracks the
identity line with the reported error ([@fig:parameter-recovery]). This is
evidence of practical acuity recoverability for the executed design, not a proof
of global or structural identifiability and not an acuity-by-colony-size study.

![Two-panel parameter-recovery figure. Source relation: original project parameter-recovery diagnostic; estimand: recovered acuity and absolute error in probability units; uncertainty: empirical percentile intervals across independent trials. In the left panel, the x-axis is true acuity and the y-axis is recovered acuity, both in probability units; error bars show the {{PARAM_RECOVERY_INTERVAL_PERCENT}}% empirical percentile interval across {{PARAM_RECOVERY_N_TRIALS}} trials per condition, and the diagonal is the identity reference. This interval is a descriptive quantile of the independent-trial estimates, not a bootstrap confidence interval or Bayesian credible interval. In the right panel, the x-axis is tested true acuity and the y-axis is absolute acuity error in probability units; the horizontal line is the global mean absolute error. These finite-grid results quantify acuity recovery for {{PARAM_RECOVERY_N_OBSERVATIONS}} observations per trial; they do not establish global structural identifiability.](../output/figures/parameter_recovery.png){#fig:parameter-recovery width=90%}
