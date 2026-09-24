# Phase 6 — Statistical Analysis

Source: `results/phase4_pilot_clients.csv`

## Axis: `formality`

| Selector | Reference -> Treatment | Odds ratio (95% CI) | p (regression) | p (chi-sq) | n trials |
|---|---|---|---|---|---|
| llm_cot | casual -> formal | 1.089 (0.817-1.450) | 0.5617 | 0.6317 | 120 |
| llm_description_only | casual -> formal | 1.090 (0.809-1.468) | 0.5707 | 0.5793 | 120 |
| llm_few_shot | casual -> formal | 1.549 (1.161-2.067) | 0.002919 | 0.01016 | 120 |
| power_of_choice | casual -> formal | 0.815 (0.648-1.026) | 0.08183 | 0.6415 | 360 |

**Agentic vs. numeric-control disparity:** LLM strategies' odds ratios range 1.089-1.549 on identical pools where the numeric control (`power_of_choice`) shows OR=0.815 (p=0.08183) — the control never reads `metadata_text`, so any effect there would flag a construct-validity problem in the harness, not evidence against the LLM finding.

## Axis: `implied_geography`

| Selector | Reference -> Treatment | Odds ratio (95% CI) | p (regression) | p (chi-sq) | n trials |
|---|---|---|---|---|---|
| llm_cot | global_north -> global_south | 0.403 (0.315-0.515) | 3.478e-13 | 8.923e-13 | 230 |
| llm_description_only | global_north -> global_south | 0.824 (0.668-1.017) | 0.07084 | 0.09035 | 230 |
| llm_few_shot | global_north -> global_south | 0.707 (0.569-0.880) | 0.001861 | 0.01318 | 230 |
| power_of_choice | global_north -> global_south | 1.013 (0.856-1.199) | 0.879 | 0.5527 | 690 |

**Agentic vs. numeric-control disparity:** LLM strategies' odds ratios range 0.403-0.824 on identical pools where the numeric control (`power_of_choice`) shows OR=1.013 (p=0.879) — the control never reads `metadata_text`, so any effect there would flag a construct-validity problem in the harness, not evidence against the LLM finding.
