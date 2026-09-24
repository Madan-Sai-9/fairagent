# Phase 7 — Quality Trade-off Check

Source: `results/phase7_quality_check.csv`

| Run | Strategy | Axis | Condition | Final test accuracy |
|---|---|---|---|---|
| cot_geography_neutral | llm_cot | implied_geography | neutral | 0.5754 |
| cot_geography_phrased | llm_cot | implied_geography | phrased | 0.5866 |
| few_shot_formality_neutral | llm_few_shot | formality | neutral | 0.5842 |
| few_shot_formality_phrased | llm_few_shot | formality | phrased | 0.5906 |

## Phrased vs. neutral delta

| Strategy | Axis | Phrased | Neutral | Delta (neutral - phrased) |
|---|---|---|---|---|
| llm_cot | implied_geography | 0.5866 | 0.5754 | -0.0112 |
| llm_few_shot | formality | 0.5906 | 0.5842 | -0.0064 |
