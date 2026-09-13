# Exact values for accessibility fallbacks

These tables are generated from the same typed reports consumed by the figures. They preserve numerical access without changing the caption or claim boundary.

## Belief Quality {#fig-values-belief-quality}

Source report: `belief_quality.json`. Log-score intervals resample configured seeds; empty reliability bins are omitted.

| series | row_kind | bin_center | mean_confidence | empirical_accuracy | mean_log_score_nats | ci_lo | ci_hi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | log_score_summary | — | — | — | 0 | 0 | 0 |
| oracle | reliability_bin | 0.95 | 1 | 1 | — | — | — |
| uniform | log_score_summary | — | — | — | -1.38629436112 | -1.38629436112 | -1.38629436112 |
| uniform | reliability_bin | 0.25 | 0.25 | 0.25 | — | — | — |
| confident_wrong | log_score_summary | — | — | — | -5.70378247466 | -5.70378247466 | -5.70378247466 |
| confident_wrong | reliability_bin | 0.95 | 0.99 | 0 | — | — | — |

## Composite two-configuration logistic-regression proxy {#fig-values-bnn-robustness}

Source report: `bnn_robustness.json`. Exploratory point-estimate proxy comparing joint NLL/L2=0.05 and RCCE/L2=0.10 configurations; because loss and shrinkage change together, the contrast cannot identify an RCCE-only effect. Intervals resample synthetic-data seeds.

| configuration | contamination_fraction | held_out_accuracy | ci_lo | ci_hi |
| --- | --- | --- | --- | --- |
| nll / L2=0.05 (standard proxy) | 0 | 0.863955078125 | 0.862265380859 | 0.865683837891 |
| nll / L2=0.05 (standard proxy) | 0.1 | 0.864892578125 | 0.863310546875 | 0.866455078125 |
| nll / L2=0.05 (standard proxy) | 0.2 | 0.862568359375 | 0.860458984375 | 0.8645703125 |
| nll / L2=0.05 (standard proxy) | 0.3 | 0.842646484375 | 0.837031005859 | 0.847724609375 |
| nll / L2=0.05 (standard proxy) | 0.35 | 0.530244140625 | 0.489843261719 | 0.569824462891 |
| nll / L2=0.05 (standard proxy) | 0.4 | 0.15939453125 | 0.153075439453 | 0.166494140625 |
| rcce / L2=0.10 (exploratory proxy) | 0 | 0.864189453125 | 0.862529052734 | 0.86583984375 |
| rcce / L2=0.10 (exploratory proxy) | 0.1 | 0.86494140625 | 0.86333984375 | 0.866474853516 |
| rcce / L2=0.10 (exploratory proxy) | 0.2 | 0.864609375 | 0.86259765625 | 0.866523681641 |
| rcce / L2=0.10 (exploratory proxy) | 0.3 | 0.859599609375 | 0.857333984375 | 0.86177734375 |
| rcce / L2=0.10 (exploratory proxy) | 0.35 | 0.558017578125 | 0.501249023437 | 0.613111328125 |
| rcce / L2=0.10 (exploratory proxy) | 0.4 | 0.145478515625 | 0.14234375 | 0.148740234375 |

## Complexity Scaling {#fig-values-complexity-scaling}

Source report: `complexity_scaling.json`. Min-max spans are timing-repeat ranges, not confidence intervals.

| method | axis | size | median_seconds | min_seconds | max_seconds | expected_exponent | observed_log_log_slope |
| --- | --- | --- | --- | --- | --- | --- | --- |
| log_linear_pool | agents | 4 | 0.000230500008911 | 0.000225124997087 | 0.000252583995461 | 1 | 0.746672196134 |
| log_linear_pool | agents | 8 | 0.000438332965132 | 0.000312083982863 | 0.000441041018348 | 1 | 0.746672196134 |
| log_linear_pool | agents | 16 | 0.00034141598735 | 0.000322791980579 | 0.000671832996886 | 1 | 0.746672196134 |
| log_linear_pool | agents | 32 | 0.000734665954951 | 0.000689374981448 | 0.000905332970433 | 1 | 0.746672196134 |
| log_linear_pool | agents | 64 | 0.00236799998675 | 0.00180329201976 | 0.00342387502315 | 1 | 0.746672196134 |
| log_linear_pool | states | 256 | 0.0171315409825 | 0.0167679580045 | 0.0201625829795 | 1 | 0.979704138395 |
| log_linear_pool | states | 512 | 0.0344316670089 | 0.0312537500286 | 0.0407036669785 | 1 | 0.979704138395 |
| log_linear_pool | states | 1024 | 0.062473708007 | 0.058720333036 | 0.0727368330117 | 1 | 0.979704138395 |
| log_linear_pool | states | 2048 | 0.13364229101 | 0.117980958021 | 0.152571832994 | 1 | 0.979704138395 |
| log_linear_pool | states | 4096 | 0.259361292003 | 0.234668624995 | 0.263946334017 | 1 | 0.979704138395 |
| robust_aggregate | agents | 4 | 0.00244775001192 | 0.00147583399666 | 0.00387070898432 | 1 | 0.869609276547 |
| robust_aggregate | agents | 8 | 0.00263845798327 | 0.00262595899403 | 0.00271837500622 | 1 | 0.869609276547 |
| robust_aggregate | agents | 16 | 0.00610974995652 | 0.00508658302715 | 0.00720562500646 | 1 | 0.869609276547 |
| robust_aggregate | agents | 32 | 0.0135586669785 | 0.0100141669973 | 0.014812999987 | 1 | 0.869609276547 |
| robust_aggregate | agents | 64 | 0.0219900410157 | 0.0212012919947 | 0.0316731670173 | 1 | 0.869609276547 |
| robust_aggregate | states | 256 | 0.237209292012 | 0.236437917047 | 0.244118958013 | 1 | 0.966995735111 |
| robust_aggregate | states | 512 | 0.455050708028 | 0.447153375018 | 0.458817375009 | 1 | 0.966995735111 |
| robust_aggregate | states | 1024 | 0.890730874962 | 0.875468000013 | 0.896320750006 | 1 | 0.966995735111 |
| robust_aggregate | states | 2048 | 1.73759445798 | 1.728917167 | 1.85330699995 | 1 | 0.966995735111 |
| robust_aggregate | states | 4096 | 3.46466295799 | 3.44916991697 | 3.67859966704 | 1 | 0.966995735111 |
| variational_aggregate | agents | 4 | 0.00338524999097 | 0.0033686250099 | 0.0034472909756 | 1 | 0.821885867076 |
| variational_aggregate | agents | 8 | 0.00509441603208 | 0.00505662499927 | 0.00517545797629 | 1 | 0.821885867076 |
| variational_aggregate | agents | 16 | 0.00874566700077 | 0.00859016698087 | 0.00884745904477 | 1 | 0.821885867076 |
| variational_aggregate | agents | 32 | 0.0161082499544 | 0.0160230410402 | 0.0164900840027 | 1 | 0.821885867076 |
| variational_aggregate | agents | 64 | 0.0328605830437 | 0.0322022909531 | 0.0333112500375 | 1 | 0.821885867076 |
| variational_aggregate | states | 256 | 0.35500874999 | 0.352529749973 | 0.411684791034 | 1 | 1.00503652551 |
| variational_aggregate | states | 512 | 0.663354374992 | 0.653461582959 | 0.678140665987 | 1 | 1.00503652551 |
| variational_aggregate | states | 1024 | 1.31547458295 | 1.30797554197 | 1.330850792 | 1 | 1.00503652551 |
| variational_aggregate | states | 2048 | 2.89508258295 | 2.575369292 | 2.92936362501 | 1 | 1.00503652551 |
| variational_aggregate | states | 4096 | 5.533655042 | 5.29192329204 | 5.68821654201 | 1 | 1.00503652551 |
| share_round_naive | agents | 4 | 0.000897624995559 | 0.000805874995422 | 0.0013543330133 | 2 | 1.62615685813 |
| share_round_naive | agents | 8 | 0.00230241601821 | 0.00216312502744 | 0.00308454199694 | 2 | 1.62615685813 |
| share_round_naive | agents | 16 | 0.0080759159755 | 0.00764725002227 | 0.014992374985 | 2 | 1.62615685813 |
| share_round_naive | agents | 32 | 0.0253026250284 | 0.0245224160026 | 0.02573987504 | 2 | 1.62615685813 |
| share_round_robust | agents | 4 | 0.0271247079945 | 0.0265557079692 | 0.0279245840502 | 2 | 1.9285573635 |
| share_round_robust | agents | 8 | 0.0962406659964 | 0.0947537919856 | 0.0998489999911 | 2 | 1.9285573635 |
| share_round_robust | agents | 16 | 0.371671999979 | 0.365836749959 | 0.411904000037 | 2 | 1.9285573635 |
| share_round_robust | agents | 32 | 1.489166916 | 1.47228491702 | 1.78867491701 | 2 | 1.9285573635 |
| infer_states | modalities | 1 | 9.38329612836e-05 | 9.20409802347e-05 | 9.54589922912e-05 | 1 | 0.710635093486 |
| infer_states | modalities | 2 | 0.000130916014314 | 0.000130417000037 | 0.000136625021696 | 1 | 0.710635093486 |
| infer_states | modalities | 4 | 0.000209207995795 | 0.000208417011891 | 0.000216290995013 | 1 | 0.710635093486 |
| infer_states | modalities | 8 | 0.000414541980717 | 0.000379082979634 | 0.000421833014116 | 1 | 0.710635093486 |

## Conditional World {#fig-values-conditional-world}

Source report: `conditional_world.json`. Contrast is robust-minus-reference true-state mass; the seeded world/scenario row is the independent unit and trials are nested within that row.

| scenario_id | attack | true_state | target_state | observability | adversary_weight | contrast_mean | ci_lo | ci_hi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s0_t1_o45_clean_full | clean | 0 | 1 | 0.45 | 1 | -0.00239421251435 | -0.00239437392663 | -0.00239405364899 |
| s0_t1_o45_clean_half | clean | 0 | 1 | 0.45 | 0.5 | -0.00239404145796 | -0.00239419509138 | -0.00239387644853 |
| s0_t1_o45_confident_wrong_full | confident_wrong | 0 | 1 | 0.45 | 1 | -0.000988545166589 | -0.000991044432826 | -0.000986009152422 |
| s0_t1_o45_confident_wrong_half | confident_wrong | 0 | 1 | 0.45 | 0.5 | -0.225953853564 | -0.226371484627 | -0.225521910911 |
| s0_t1_o45_label_noise_full | label_noise | 0 | 1 | 0.45 | 1 | -0.488870221281 | -0.488897724061 | -0.488838874888 |
| s0_t1_o45_label_noise_half | label_noise | 0 | 1 | 0.45 | 0.5 | -0.488871322135 | -0.48890061614 | -0.488840824225 |
| s0_t1_o45_permutation_full | permutation | 0 | 1 | 0.45 | 1 | 0.023860773766 | 0.0237998322068 | 0.0239213522168 |
| s0_t1_o45_permutation_half | permutation | 0 | 1 | 0.45 | 0.5 | 0.00549902086446 | 0.00548282299486 | 0.00551554040875 |
| s0_t1_o45_uniform_full | uniform | 0 | 1 | 0.45 | 1 | -0.000654455788683 | -0.000660386652698 | -0.00064853677668 |
| s0_t1_o45_uniform_half | uniform | 0 | 1 | 0.45 | 0.5 | -0.000649036525058 | -0.000653957231618 | -0.000644170812572 |
| s0_t1_o70_clean_full | clean | 0 | 1 | 0.7 | 1 | -8.55843167833e-08 | -8.77901477608e-08 | -8.33571636047e-08 |
| s0_t1_o70_clean_half | clean | 0 | 1 | 0.7 | 0.5 | -8.48889808945e-08 | -8.69645331747e-08 | -8.28771171314e-08 |
| s0_t1_o70_confident_wrong_full | confident_wrong | 0 | 1 | 0.7 | 1 | -0.160304212965 | -0.160666135988 | -0.159918671023 |
| s0_t1_o70_confident_wrong_half | confident_wrong | 0 | 1 | 0.7 | 0.5 | 0.0131801069005 | 0.0131330257074 | 0.0132274949843 |
| s0_t1_o70_label_noise_full | label_noise | 0 | 1 | 0.7 | 1 | -0.00131218843489 | -0.00131286641751 | -0.00131148486189 |
| s0_t1_o70_label_noise_half | label_noise | 0 | 1 | 0.7 | 0.5 | -0.00168291587902 | -0.00168405039242 | -0.00168182490422 |
| s0_t1_o70_permutation_full | permutation | 0 | 1 | 0.7 | 1 | 0.00277938816578 | 0.00277084946149 | 0.00278807396825 |
| s0_t1_o70_permutation_half | permutation | 0 | 1 | 0.7 | 0.5 | 0.000331617976668 | 0.000330603694416 | 0.000332680316595 |
| s0_t1_o70_uniform_full | uniform | 0 | 1 | 0.7 | 1 | -3.94597746383e-07 | -4.05423858628e-07 | -3.83457376431e-07 |
| s0_t1_o70_uniform_half | uniform | 0 | 1 | 0.7 | 0.5 | -3.87494384035e-07 | -4.00273118112e-07 | -3.74355908041e-07 |
| s1_t2_o45_clean_full | clean | 1 | 2 | 0.45 | 1 | -0.00239422341167 | -0.00239437383672 | -0.00239406033145 |
| s1_t2_o45_clean_half | clean | 1 | 2 | 0.45 | 0.5 | -0.00239423467171 | -0.00239435815405 | -0.00239410396257 |
| s1_t2_o45_confident_wrong_full | confident_wrong | 1 | 2 | 0.45 | 1 | -0.000990972372864 | -0.000993527169273 | -0.000988507165419 |
| s1_t2_o45_confident_wrong_half | confident_wrong | 1 | 2 | 0.45 | 0.5 | -0.225884871744 | -0.226366722201 | -0.225392607793 |
| s1_t2_o45_label_noise_full | label_noise | 1 | 2 | 0.45 | 1 | -0.488890249812 | -0.488920623705 | -0.4888602026 |
| s1_t2_o45_label_noise_half | label_noise | 1 | 2 | 0.45 | 0.5 | -0.48887363825 | -0.488901571882 | -0.488846865355 |
| s1_t2_o45_permutation_full | permutation | 1 | 2 | 0.45 | 1 | 0.0238824510851 | 0.0238324531263 | 0.0239308574962 |
| s1_t2_o45_permutation_half | permutation | 1 | 2 | 0.45 | 0.5 | 0.00547870492137 | 0.00546088706195 | 0.00549684893573 |
| s1_t2_o45_uniform_full | uniform | 1 | 2 | 0.45 | 1 | -0.000654055681628 | -0.000658928757195 | -0.00064885781519 |
| s1_t2_o45_uniform_half | uniform | 1 | 2 | 0.45 | 0.5 | -0.000651618184329 | -0.000656715693734 | -0.000646185681921 |
| s1_t2_o70_clean_full | clean | 1 | 2 | 0.7 | 1 | -8.42841255778e-08 | -8.69972870861e-08 | -8.16777197319e-08 |
| s1_t2_o70_clean_half | clean | 1 | 2 | 0.7 | 0.5 | -8.5195033036e-08 | -8.78470105181e-08 | -8.25061101915e-08 |
| s1_t2_o70_confident_wrong_full | confident_wrong | 1 | 2 | 0.7 | 1 | -0.160199124008 | -0.160556516943 | -0.159830480042 |
| s1_t2_o70_confident_wrong_half | confident_wrong | 1 | 2 | 0.7 | 0.5 | 0.0131728751597 | 0.0131293976345 | 0.0132192029463 |
| s1_t2_o70_label_noise_full | label_noise | 1 | 2 | 0.7 | 1 | -0.00131115085973 | -0.00131185416077 | -0.00131045782543 |
| s1_t2_o70_label_noise_half | label_noise | 1 | 2 | 0.7 | 0.5 | -0.00168248212736 | -0.0016837340055 | -0.00168123503952 |
| s1_t2_o70_permutation_full | permutation | 1 | 2 | 0.7 | 1 | 0.00279113193228 | 0.0027840064631 | 0.00279852112477 |
| s1_t2_o70_permutation_half | permutation | 1 | 2 | 0.7 | 0.5 | 0.000329874966596 | 0.000328842098867 | 0.00033088840783 |
| s1_t2_o70_uniform_full | uniform | 1 | 2 | 0.7 | 1 | -3.86710807352e-07 | -3.96901809059e-07 | -3.76457518579e-07 |
| s1_t2_o70_uniform_half | uniform | 1 | 2 | 0.7 | 0.5 | -3.94838140361e-07 | -4.06834099211e-07 | -3.83481350345e-07 |

## Cross Study Summary {#fig-values-cross-study-summary}

Source report: `cross_study_summary.json`. Native units remain separate; intervals come from the harmonized seed-level rerun. Study 4 is a within-run display-selected maximum across non-reference server presets at the declared worst rate, not a preselected method or inferential winner.

| study | label | metric | unit | mean | ci_lo | ci_hi |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Study 1 Belief sharing | Accuracy gain (comm − iso) | fraction | 0.436956505809 | 0.403402914702 | 0.46746235977 |
| 2 | Study 2 Language acquisition | KL reduction (initial − final) | nats | 3.42047973214 | 3.42045106214 | 3.4205071003 |
| 3 | Study 3 Emergence (BMR) | ΔF (redundant pruning) | nats | 3.71621102364 | 3.71029104144 | 3.72186452035 |
| 4 | Study 4 Robustness sweep | Trial-mean accuracy gain (best robust − naive, worst rate) | fraction | 0.117536707282 | 0.10901564649 | 0.126243002247 |
| 5 | Study 5 Moving world (EFE) | Accuracy gain (EFE − isolated) | fraction | -0.017578125 | -0.023046875 | -0.0125 |
| 6 | Study 6 2-level hierarchical POMDP | Location accuracy gap (hier − flat) | fraction | -0.0015625 | -0.008203125 | 0.0046875 |
| 7 | Study 7 3-level hierarchical POMDP | Location accuracy gap (3-level − flat) | fraction | -0.004296875 | -0.012109375 | 0.00390625 |
| 8 | Study 8 Sensitivity sweep | Mean accuracy gap (comm - iso, 5x5 grid) | fraction | 0.243302199836 | 0.241138212395 | 0.245469635246 |
| 9 | Study 9 Parameter recovery | R-squared (acuity identifiability) | R-sq | 0.886109418283 | 0.871630042936 | 0.900536738227 |

## Robustness Review Grid {#fig-values-robustness-review-grid}

Source report: `robustness_review_grid.json`. Conditional display-summary rows reproduce the heatmap's grouped mean and half min-max span; all predeclared preset-rate rows are retained without result-dependent selection.

| row_kind | mechanism | rate | preset | scenario_id | adversary_weight | mean_contrast | half_min_max_span | group_cell_count | ci_lo | ci_hi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conditional_cell | clean | — | — | s0_t1_o45_clean_full | 1 | -0.00239415268878 | — | — | — | — |
| conditional_cell | clean | — | — | s0_t1_o45_clean_half | 0.5 | -0.00239420198845 | — | — | — | — |
| conditional_cell | confident_wrong | — | — | s0_t1_o45_confident_wrong_full | 1 | -0.000987889264314 | — | — | — | — |
| conditional_cell | confident_wrong | — | — | s0_t1_o45_confident_wrong_half | 0.5 | -0.226454978863 | — | — | — | — |
| conditional_cell | label_noise | — | — | s0_t1_o45_label_noise_full | 1 | -0.488891648708 | — | — | — | — |
| conditional_cell | label_noise | — | — | s0_t1_o45_label_noise_half | 0.5 | -0.488883249708 | — | — | — | — |
| conditional_cell | permutation | — | — | s0_t1_o45_permutation_full | 1 | 0.0238716933689 | — | — | — | — |
| conditional_cell | permutation | — | — | s0_t1_o45_permutation_half | 0.5 | 0.0054889806201 | — | — | — | — |
| conditional_cell | uniform | — | — | s0_t1_o45_uniform_full | 1 | -0.000649524334185 | — | — | — | — |
| conditional_cell | uniform | — | — | s0_t1_o45_uniform_half | 0.5 | -0.000651777817296 | — | — | — | — |
| conditional_cell | clean | — | — | s0_t1_o70_clean_full | 1 | -8.32777834449e-08 | — | — | — | — |
| conditional_cell | clean | — | — | s0_t1_o70_clean_half | 0.5 | -8.48110937314e-08 | — | — | — | — |
| conditional_cell | confident_wrong | — | — | s0_t1_o70_confident_wrong_full | 1 | -0.16007511644 | — | — | — | — |
| conditional_cell | confident_wrong | — | — | s0_t1_o70_confident_wrong_half | 0.5 | 0.0131762615618 | — | — | — | — |
| conditional_cell | label_noise | — | — | s0_t1_o70_label_noise_full | 1 | -0.00131161644006 | — | — | — | — |
| conditional_cell | label_noise | — | — | s0_t1_o70_label_noise_half | 0.5 | -0.00168252947342 | — | — | — | — |
| conditional_cell | permutation | — | — | s0_t1_o70_permutation_full | 1 | 0.00278742189179 | — | — | — | — |
| conditional_cell | permutation | — | — | s0_t1_o70_permutation_half | 0.5 | 0.00033134026175 | — | — | — | — |
| conditional_cell | uniform | — | — | s0_t1_o70_uniform_full | 1 | -3.92773127249e-07 | — | — | — | — |
| conditional_cell | uniform | — | — | s0_t1_o70_uniform_half | 0.5 | -3.89696531293e-07 | — | — | — | — |
| conditional_cell | clean | — | — | s1_t2_o45_clean_full | 1 | -0.00239415732863 | — | — | — | — |
| conditional_cell | clean | — | — | s1_t2_o45_clean_half | 0.5 | -0.00239410235409 | — | — | — | — |
| conditional_cell | confident_wrong | — | — | s1_t2_o45_confident_wrong_full | 1 | -0.000988782013958 | — | — | — | — |
| conditional_cell | confident_wrong | — | — | s1_t2_o45_confident_wrong_half | 0.5 | -0.226164156848 | — | — | — | — |
| conditional_cell | label_noise | — | — | s1_t2_o45_label_noise_full | 1 | -0.488885054036 | — | — | — | — |
| conditional_cell | label_noise | — | — | s1_t2_o45_label_noise_half | 0.5 | -0.488888657197 | — | — | — | — |
| conditional_cell | permutation | — | — | s1_t2_o45_permutation_full | 1 | 0.023888921753 | — | — | — | — |
| conditional_cell | permutation | — | — | s1_t2_o45_permutation_half | 0.5 | 0.00548376679063 | — | — | — | — |
| conditional_cell | uniform | — | — | s1_t2_o45_uniform_full | 1 | -0.000651011429238 | — | — | — | — |
| conditional_cell | uniform | — | — | s1_t2_o45_uniform_half | 0.5 | -0.000652254292401 | — | — | — | — |
| conditional_cell | clean | — | — | s1_t2_o70_clean_full | 1 | -8.45641847162e-08 | — | — | — | — |
| conditional_cell | clean | — | — | s1_t2_o70_clean_half | 0.5 | -8.39255074296e-08 | — | — | — | — |
| conditional_cell | confident_wrong | — | — | s1_t2_o70_confident_wrong_full | 1 | -0.160255224327 | — | — | — | — |
| conditional_cell | confident_wrong | — | — | s1_t2_o70_confident_wrong_half | 0.5 | 0.0131702304697 | — | — | — | — |
| conditional_cell | label_noise | — | — | s1_t2_o70_label_noise_full | 1 | -0.00131160642477 | — | — | — | — |
| conditional_cell | label_noise | — | — | s1_t2_o70_label_noise_half | 0.5 | -0.00168318338655 | — | — | — | — |
| conditional_cell | permutation | — | — | s1_t2_o70_permutation_full | 1 | 0.00278400349634 | — | — | — | — |
| conditional_cell | permutation | — | — | s1_t2_o70_permutation_half | 0.5 | 0.000330742443902 | — | — | — | — |
| conditional_cell | uniform | — | — | s1_t2_o70_uniform_full | 1 | -3.92750744168e-07 | — | — | — | — |
| conditional_cell | uniform | — | — | s1_t2_o70_uniform_half | 0.5 | -3.88937453183e-07 | — | — | — | — |
| conditional_display_summary | clean | — | — | — | 0.5 | -0.00119711826979 | 0.00119705903147 | 4 | — | — |
| conditional_display_summary | clean | — | — | — | 1 | -0.00119711946484 | 0.00119703702542 | 4 | — | — |
| conditional_display_summary | confident_wrong | — | — | — | 0.5 | -0.10656816092 | 0.119815620212 | 4 | — | — |
| conditional_display_summary | confident_wrong | — | — | — | 1 | -0.0805767530113 | 0.0796336675314 | 4 | — | — |
| conditional_display_summary | label_noise | — | — | — | 0.5 | -0.245284404941 | 0.243603063862 | 4 | — | — |
| conditional_display_summary | label_noise | — | — | — | 1 | -0.245099981402 | 0.243790021142 | 4 | — | — |
| conditional_display_summary | permutation | — | — | — | 0.5 | 0.0029087075291 | 0.0025791190881 | 4 | — | — |
| conditional_display_summary | permutation | — | — | — | 1 | 0.0133330101275 | 0.0105524591283 | 4 | — | — |
| conditional_display_summary | uniform | — | — | — | 0.5 | -0.00032620268592 | 0.000325932677474 | 4 | — | — |
| conditional_display_summary | uniform | — | — | — | 1 | -0.000325330321824 | 0.000325309339247 | 4 | — | — |
| rate_profile | confident_wrong | 0 | RKL | — | — | -0.0030873319709 | — | — | -0.00309966317429 | -0.00307531918443 |
| rate_profile | confident_wrong | 0 | AR | — | — | -0.0020376072976 | — | — | -0.00204739451814 | -0.00202761842691 |
| rate_profile | confident_wrong | 0 | beta | — | — | -0.00444376332873 | — | — | -0.00445769853381 | -0.00443035650228 |
| rate_profile | confident_wrong | 0 | rcce | — | — | -0.00372382087257 | — | — | -0.0037368079166 | -0.00371087504316 |
| rate_profile | confident_wrong | 0.2 | RKL | — | — | -0.00583644267488 | — | — | -0.00586257671926 | -0.00581100494234 |
| rate_profile | confident_wrong | 0.2 | AR | — | — | -0.00398452075568 | — | — | -0.00400473223963 | -0.00396438905871 |
| rate_profile | confident_wrong | 0.2 | beta | — | — | -0.00813891742789 | — | — | -0.00816954951067 | -0.00810871546739 |
| rate_profile | confident_wrong | 0.2 | rcce | — | — | -0.00692749994345 | — | — | -0.00695596789094 | -0.00689874163731 |
| rate_profile | confident_wrong | 0.4 | RKL | — | — | -0.00853631844269 | — | — | -0.00855383792691 | -0.00851976539468 |
| rate_profile | confident_wrong | 0.4 | AR | — | — | -0.00571098216865 | — | — | -0.00572338744926 | -0.00569882419375 |
| rate_profile | confident_wrong | 0.4 | beta | — | — | -0.0119845742239 | — | — | -0.0120064692932 | -0.011962615889 |
| rate_profile | confident_wrong | 0.4 | rcce | — | — | -0.0101785788114 | — | — | -0.0101981918045 | -0.0101590542093 |
| rate_profile | confident_wrong | 0.5 | RKL | — | — | -0.00642357890219 | — | — | -0.00647333239382 | -0.00637159597851 |
| rate_profile | confident_wrong | 0.5 | AR | — | — | -0.00339806420931 | — | — | -0.00344987261939 | -0.00334625857616 |
| rate_profile | confident_wrong | 0.5 | beta | — | — | -0.010175710987 | — | — | -0.0102221298883 | -0.0101283104345 |
| rate_profile | confident_wrong | 0.5 | rcce | — | — | -0.00820629274816 | — | — | -0.00825519712501 | -0.00815690072831 |
| rate_profile | confident_wrong | 0.6 | RKL | — | — | 0.000892416199917 | — | — | 0.000715031288465 | 0.00107431264948 |
| rate_profile | confident_wrong | 0.6 | AR | — | — | 0.00411740295935 | — | — | 0.0039365322267 | 0.00430331238256 |
| rate_profile | confident_wrong | 0.6 | beta | — | — | -0.00315325844891 | — | — | -0.00333300935206 | -0.00298035620703 |
| rate_profile | confident_wrong | 0.6 | rcce | — | — | -0.00103038739735 | — | — | -0.00120273136507 | -0.000851924189252 |
| rate_profile | confident_wrong | 0.7 | RKL | — | — | 0.0203749470184 | — | — | 0.0198877579427 | 0.0208667923001 |
| rate_profile | confident_wrong | 0.7 | AR | — | — | 0.0238690322028 | — | — | 0.0233766739097 | 0.0243806041541 |
| rate_profile | confident_wrong | 0.7 | beta | — | — | 0.016060886158 | — | — | 0.0156014113636 | 0.0165532480997 |
| rate_profile | confident_wrong | 0.7 | rcce | — | — | 0.0183094914705 | — | — | 0.0178096350541 | 0.0187959112377 |
| rate_profile | confident_wrong | 0.8 | RKL | — | — | 0.0790040306345 | — | — | 0.0777020114807 | 0.0803404969214 |
| rate_profile | confident_wrong | 0.8 | AR | — | — | 0.0827478680744 | — | — | 0.0814853973942 | 0.0840306075889 |
| rate_profile | confident_wrong | 0.8 | beta | — | — | 0.0745416995119 | — | — | 0.0732815091318 | 0.0758391912802 |
| rate_profile | confident_wrong | 0.8 | rcce | — | — | 0.0768410390424 | — | — | 0.0755597537692 | 0.0781447134368 |
| rate_profile | confident_wrong | 0.9 | RKL | — | — | -0.032684640794 | — | — | -0.0448391681789 | -0.0206527338949 |
| rate_profile | confident_wrong | 0.9 | AR | — | — | -0.223666256977 | — | — | -0.236108640628 | -0.210844466879 |
| rate_profile | confident_wrong | 0.9 | beta | — | — | 0.112306498947 | — | — | 0.102151387075 | 0.122144741853 |
| rate_profile | confident_wrong | 0.9 | rcce | — | — | 0.04461582441 | — | — | 0.0340993147092 | 0.0554435624844 |
| rate_profile | byzantine | 0 | RKL | — | — | -0.0030873319709 | — | — | -0.00309950034027 | -0.00307507328782 |
| rate_profile | byzantine | 0 | AR | — | — | -0.0020376072976 | — | — | -0.00204774633218 | -0.00202799669981 |
| rate_profile | byzantine | 0 | beta | — | — | -0.00444376332873 | — | — | -0.0044570991048 | -0.00443034864266 |
| rate_profile | byzantine | 0 | rcce | — | — | -0.00372382087257 | — | — | -0.00373623740712 | -0.00371044912477 |
| rate_profile | byzantine | 0.2 | RKL | — | — | -0.00675789943638 | — | — | -0.00679068082733 | -0.00672406011809 |
| rate_profile | byzantine | 0.2 | AR | — | — | -0.00463797686656 | — | — | -0.0046628573117 | -0.00461250631838 |
| rate_profile | byzantine | 0.2 | beta | — | — | -0.00937203594607 | — | — | -0.00941311370988 | -0.00933158574733 |
| rate_profile | byzantine | 0.2 | rcce | — | — | -0.00799924628541 | — | — | -0.0080365121744 | -0.00796257266001 |
| rate_profile | byzantine | 0.4 | RKL | — | — | 0.0100374755654 | — | — | 0.00965611953356 | 0.0104145557349 |
| rate_profile | byzantine | 0.4 | AR | — | — | 0.0134027294258 | — | — | 0.0130237475595 | 0.0137806579353 |
| rate_profile | byzantine | 0.4 | beta | — | — | 0.00583812003164 | — | — | 0.00548105608213 | 0.00620952234193 |
| rate_profile | byzantine | 0.4 | rcce | — | — | 0.00803512276167 | — | — | 0.00766322698644 | 0.00840864389266 |
| rate_profile | byzantine | 0.5 | RKL | — | — | 0.0953472772782 | — | — | 0.0936818865643 | 0.0970124823442 |
| rate_profile | byzantine | 0.5 | AR | — | — | 0.0973568731689 | — | — | 0.0955733262213 | 0.0991264288348 |
| rate_profile | byzantine | 0.5 | beta | — | — | 0.0908767572562 | — | — | 0.0892479263587 | 0.0925136001297 |
| rate_profile | byzantine | 0.5 | rcce | — | — | 0.093176910969 | — | — | 0.0915158687093 | 0.0948449514563 |
| rate_profile | byzantine | 0.6 | RKL | — | — | -0.127012253411 | — | — | -0.139773367933 | -0.114535044837 |
| rate_profile | byzantine | 0.6 | AR | — | — | -0.307829144255 | — | — | -0.320404451702 | -0.2948104139 |
| rate_profile | byzantine | 0.6 | beta | — | — | 0.0248248882658 | — | — | 0.0138710022732 | 0.0358505171574 |
| rate_profile | byzantine | 0.6 | rcce | — | — | -0.0443811955481 | — | — | -0.0562676568645 | -0.0327103480139 |
| rate_profile | byzantine | 0.7 | RKL | — | — | -0.276962061756 | — | — | -0.280388859612 | -0.273565448272 |
| rate_profile | byzantine | 0.7 | AR | — | — | -0.277301796956 | — | — | -0.280623458558 | -0.274021487838 |
| rate_profile | byzantine | 0.7 | beta | — | — | -0.276365748585 | — | — | -0.279768973896 | -0.273011611751 |
| rate_profile | byzantine | 0.7 | rcce | — | — | -0.276661964991 | — | — | -0.280142559157 | -0.273198583311 |
| rate_profile | byzantine | 0.8 | RKL | — | — | -0.0760921815523 | — | — | -0.0773321020987 | -0.07483356972 |
| rate_profile | byzantine | 0.8 | AR | — | — | -0.0761049543182 | — | — | -0.0773292972965 | -0.0749008068667 |
| rate_profile | byzantine | 0.8 | beta | — | — | -0.0760790516561 | — | — | -0.0772540806738 | -0.074841151672 |
| rate_profile | byzantine | 0.8 | rcce | — | — | -0.0760856578647 | — | — | -0.0773299029063 | -0.0748756890833 |
| rate_profile | byzantine | 0.9 | RKL | — | — | -0.0166672705683 | — | — | -0.016959410138 | -0.016369390168 |
| rate_profile | byzantine | 0.9 | AR | — | — | -0.0166688374909 | — | — | -0.0169460984099 | -0.0163853706533 |
| rate_profile | byzantine | 0.9 | beta | — | — | -0.0166656498467 | — | — | -0.0169508006779 | -0.0163865185591 |
| rate_profile | byzantine | 0.9 | rcce | — | — | -0.0166664668163 | — | — | -0.0169581451547 | -0.0163798851882 |
| rate_profile | drift | 0 | RKL | — | — | -0.0030873319709 | — | — | -0.00309972677184 | -0.00307529420097 |
| rate_profile | drift | 0 | AR | — | — | -0.0020376072976 | — | — | -0.00204758789729 | -0.00202810620575 |
| rate_profile | drift | 0 | beta | — | — | -0.00444376332873 | — | — | -0.00445692572522 | -0.00443036700073 |
| rate_profile | drift | 0 | rcce | — | — | -0.00372382087257 | — | — | -0.00373657287793 | -0.00371089872358 |
| rate_profile | drift | 0.2 | RKL | — | — | -0.00583644267488 | — | — | -0.00586173550936 | -0.00581109552897 |
| rate_profile | drift | 0.2 | AR | — | — | -0.00398452075568 | — | — | -0.00400484683186 | -0.00396527194974 |
| rate_profile | drift | 0.2 | beta | — | — | -0.00813891742789 | — | — | -0.00817022170807 | -0.0081083193561 |
| rate_profile | drift | 0.2 | rcce | — | — | -0.00692749994345 | — | — | -0.00695569211751 | -0.00689925391326 |
| rate_profile | drift | 0.4 | RKL | — | — | -0.00853631844269 | — | — | -0.00855350603013 | -0.00851937200351 |
| rate_profile | drift | 0.4 | AR | — | — | -0.00571098216865 | — | — | -0.00572302220651 | -0.0056988917203 |
| rate_profile | drift | 0.4 | beta | — | — | -0.0119845742239 | — | — | -0.0120063982747 | -0.0119616470291 |
| rate_profile | drift | 0.4 | rcce | — | — | -0.0101785788114 | — | — | -0.0101982857446 | -0.0101593815068 |
| rate_profile | drift | 0.5 | RKL | — | — | -0.00642357890219 | — | — | -0.0064721224657 | -0.00637257254423 |
| rate_profile | drift | 0.5 | AR | — | — | -0.00339806420931 | — | — | -0.0034482072327 | -0.00334577712849 |
| rate_profile | drift | 0.5 | beta | — | — | -0.010175710987 | — | — | -0.0102241295907 | -0.0101275615709 |
| rate_profile | drift | 0.5 | rcce | — | — | -0.00820629274816 | — | — | -0.00825475316615 | -0.00815801337307 |
| rate_profile | drift | 0.6 | RKL | — | — | 0.000892416199917 | — | — | 0.000710099699468 | 0.00107579976022 |
| rate_profile | drift | 0.6 | AR | — | — | 0.00411740295935 | — | — | 0.00394256207628 | 0.00430818437737 |
| rate_profile | drift | 0.6 | beta | — | — | -0.00315325844891 | — | — | -0.00332521702079 | -0.00298075091317 |
| rate_profile | drift | 0.6 | rcce | — | — | -0.00103038739735 | — | — | -0.00120631676023 | -0.000844304128053 |
| rate_profile | drift | 0.7 | RKL | — | — | 0.0203749470184 | — | — | 0.0198759673561 | 0.0208745882498 |
| rate_profile | drift | 0.7 | AR | — | — | 0.0238690322028 | — | — | 0.0233621843111 | 0.0243704656315 |
| rate_profile | drift | 0.7 | beta | — | — | 0.016060886158 | — | — | 0.0155617741647 | 0.0165421829818 |
| rate_profile | drift | 0.7 | rcce | — | — | 0.0183094914705 | — | — | 0.0178190444644 | 0.018806661827 |
| rate_profile | drift | 0.8 | RKL | — | — | 0.0790040306345 | — | — | 0.0777190825067 | 0.0802995319046 |
| rate_profile | drift | 0.8 | AR | — | — | 0.0827478680744 | — | — | 0.0814037671383 | 0.084101532778 |
| rate_profile | drift | 0.8 | beta | — | — | 0.0745416995119 | — | — | 0.0732293327835 | 0.0758627332997 |
| rate_profile | drift | 0.8 | rcce | — | — | 0.0768410390424 | — | — | 0.0755136274126 | 0.0781331095169 |
| rate_profile | drift | 0.9 | RKL | — | — | -0.032684640794 | — | — | -0.0449036186388 | -0.020222202441 |
| rate_profile | drift | 0.9 | AR | — | — | -0.223666256977 | — | — | -0.236600329151 | -0.210996327321 |
| rate_profile | drift | 0.9 | beta | — | — | 0.112306498947 | — | — | 0.102544379969 | 0.1221821731 |
| rate_profile | drift | 0.9 | rcce | — | — | 0.04461582441 | — | — | 0.034051524372 | 0.0551575859822 |

## Sensitivity Heatmap {#fig-values-sensitivity-heatmap}

Source report: `sensitivity.json`. Display-band membership is not a confidence interval, significance test, or zero-effect claim.

| panel | acuity | n_agents | accuracy_gap | inside_display_band |
| --- | --- | --- | --- | --- |
| belief_sharing | 0.4 | 2 | 0 | yes |
| belief_sharing | 0.4 | 4 | 0.188880960082 | no |
| belief_sharing | 0.4 | 6 | 0.402147711908 | no |
| belief_sharing | 0.4 | 8 | 0.447257165652 | no |
| belief_sharing | 0.4 | 10 | 0.526521492458 | no |
| belief_sharing | 0.55 | 2 | 0 | yes |
| belief_sharing | 0.55 | 4 | 0.313183369029 | no |
| belief_sharing | 0.55 | 6 | 0.447275953466 | no |
| belief_sharing | 0.55 | 8 | 0.506588377435 | no |
| belief_sharing | 0.55 | 10 | 0.58612492916 | no |
| belief_sharing | 0.7 | 2 | 0 | yes |
| belief_sharing | 0.7 | 4 | 0.247299165884 | no |
| belief_sharing | 0.7 | 6 | 0.40026092271 | no |
| belief_sharing | 0.7 | 8 | 0.441561560648 | no |
| belief_sharing | 0.7 | 10 | 0.429437733587 | no |
| belief_sharing | 0.85 | 2 | 0 | yes |
| belief_sharing | 0.85 | 4 | 0.179203494815 | no |
| belief_sharing | 0.85 | 6 | 0.267753333833 | no |
| belief_sharing | 0.85 | 8 | 0.230216930531 | no |
| belief_sharing | 0.85 | 10 | 0.231898182269 | no |
| belief_sharing | 0.95 | 2 | 0 | yes |
| belief_sharing | 0.95 | 4 | 0.0502426769649 | no |
| belief_sharing | 0.95 | 6 | 0.0313477923101 | yes |
| belief_sharing | 0.95 | 8 | 0.0736923599291 | no |
| belief_sharing | 0.95 | 10 | 0.061410959365 | no |
| hierarchical | 0.4 | 2 | 0.1 | no |
| hierarchical | 0.4 | 4 | 0 | yes |
| hierarchical | 0.4 | 6 | -0.45 | no |
| hierarchical | 0.4 | 8 | -0.15 | no |
| hierarchical | 0.4 | 10 | -0.3 | no |
| hierarchical | 0.55 | 2 | -0.05 | yes |
| hierarchical | 0.55 | 4 | -0.2 | no |
| hierarchical | 0.55 | 6 | -0.15 | no |
| hierarchical | 0.55 | 8 | -0.15 | no |
| hierarchical | 0.55 | 10 | -0.35 | no |
| hierarchical | 0.7 | 2 | 0.15 | no |
| hierarchical | 0.7 | 4 | -0.1 | no |
| hierarchical | 0.7 | 6 | -0.15 | no |
| hierarchical | 0.7 | 8 | -0.1 | no |
| hierarchical | 0.7 | 10 | -0.05 | yes |
| hierarchical | 0.85 | 2 | 0.2 | no |
| hierarchical | 0.85 | 4 | 0 | yes |
| hierarchical | 0.85 | 6 | 0 | yes |
| hierarchical | 0.85 | 8 | 0 | yes |
| hierarchical | 0.85 | 10 | 0 | yes |
| hierarchical | 0.95 | 2 | 0.05 | no |
| hierarchical | 0.95 | 4 | 0 | yes |
| hierarchical | 0.95 | 6 | 0 | yes |
| hierarchical | 0.95 | 8 | 0 | yes |
| hierarchical | 0.95 | 10 | 0 | yes |
