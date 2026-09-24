# phase7_mitigation — Statistical Analysis

Source: `results/phase7_mitigation_clients.csv`

## Axis: `formality`

| Selector | Reference -> Treatment | Odds ratio (95% CI) | p (regression) | p (chi-sq) | n trials |
|---|---|---|---|---|---|
| llm_cot | casual -> formal | 1.042 (0.678-1.601) | 0.8524 | 0.4216 | 60 |
| llm_description_only | casual -> formal | 1.375 (0.866-2.183) | 0.1771 | 0.261 | 60 |
| llm_few_shot | casual -> formal | 1.165 (0.714-1.902) | 0.5402 | 0.9431 | 60 |
| power_of_choice | casual -> formal | 0.792 (0.571-1.098) | 0.1616 | 1 | 180 |

**Agentic vs. numeric-control disparity:** LLM strategies' odds ratios range 1.042-1.375 on identical pools where the numeric control (`power_of_choice`) shows OR=0.792 (p=0.1616) — the control never reads `metadata_text`, so any effect there would flag a construct-validity problem in the harness, not evidence against the LLM finding.

## Axis: `implied_geography`

| Selector | Reference -> Treatment | Odds ratio (95% CI) | p (regression) | p (chi-sq) | n trials |
|---|---|---|---|---|---|
| llm_cot | global_north -> global_south | 0.955 (0.668-1.367) | 0.8028 | 0.7212 | 60 |
| llm_description_only | global_north -> global_south | 0.999 (0.653-1.529) | 0.9977 | 0.7478 | 60 |
| llm_few_shot | global_north -> global_south | 1.132 (0.704-1.818) | 0.6097 | 0.4863 | 60 |
| power_of_choice | global_north -> global_south | 0.877 (0.637-1.206) | 0.4198 | 0.2621 | 180 |

**Agentic vs. numeric-control disparity:** LLM strategies' odds ratios range 0.955-1.132 on identical pools where the numeric control (`power_of_choice`) shows OR=0.877 (p=0.4198) — the control never reads `metadata_text`, so any effect there would flag a construct-validity problem in the harness, not evidence against the LLM finding.
