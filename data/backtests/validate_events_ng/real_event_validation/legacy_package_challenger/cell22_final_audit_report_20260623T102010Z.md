# RozviDrought Legacy Challenger Final Audit

Generated: 20260623T102010Z

---

## Executive Summary

Objective: determine whether the legacy challenger pipeline should replace the validated fusion package.

Result: challenger does not outperform the validated fusion model.

Recommendation: retain the validated fusion package as the operational candidate.

## Architecture Investigation

- Atmospheric subsystem supports vectorized inference.
- Soil subsystem supports vectorized inference.
- Vegetation subsystem supports vectorized inference.
- Hydrology subsystem supports vectorized inference.
- Fusion model supports vectorized inference.
- SubsystemService tail(1) operation destroys hydrology vectorization.
- FusionService run() performs single-row fusion only.
- Original challenger implementation was heavily bottlenecked by per-pixel loops.

## Cell 18 Batch Parity Validation

- Batch parity passed.
- No numerical differences detected between batch and single execution.
- National inference approved.
- Hybrid eligibility: 512 / 512.

## Cell 19 Fusion Schema Validation

- Final fusion probabilities located successfully.
- Valid probability path:
  fusion_result.probs[0][0:3]
- Fusion output sums to 1.0.
- Fusion extraction contract confirmed.

## Cell 20 National Legacy Challenger

- National inference completed successfully.
- Pixels evaluated: 26,775.
- Monthly outputs generated.
- Event outputs generated.
- No inference failures.
- Runtime instability observed across months.
- Significant throughput variation despite identical workloads.

## Legacy Challenger Event Results

- 2015-2016: mean_class=0.156, class1+=12.73%, class2+=2.75%, class3=0.10%
- 2018-2019: mean_class=0.118, class1+=8.12%, class2+=3.68%, class3=0.01%
- 2023-2024: mean_class=0.084, class1+=8.25%, class2+=0.10%, class3=0.00%

Finding: challenger identifies event onset but loses persistence.
Finding: severe drought probabilities collapse after initial months.
Finding: national signal remains weak.

## Excel Severity Comparison

| Event | Excel Severity | Fusion Mean Severity |
|------|------|------|
| 2015/16 | 4 Extreme | 1.704 |
| 2018/19 | 2 Moderate | 1.079 |
| 2023/24 | 5 Catastrophic | 1.100 |

Fusion ranking:
2015/16 > 2023/24 > 2018/19

Excel ranking:
2023/24 > 2015/16 > 2018/19

Only one of three event rankings matches exactly.
Top two events are reversed.

## Interpretation

The fusion model detects all three major drought periods.
The challenger detects drought onset but struggles with persistence.
The fusion model better reproduces historical event behaviour.
Severity calibration remains compressed.
Physical drought intensity and humanitarian impact severity are not equivalent targets.
Excel severity scores incorporate impacts not directly represented in the model feature space.

## Final Decision

PASS: validated fusion package
PASS: batch parity
PASS: national-scale execution
PASS: event detection
WARN: severity ranking mismatch
WARN: compressed severity range
FAIL: legacy challenger does not outperform validated fusion package

Operational Recommendation:
Retain the validated fusion package as the primary drought model.
Archive the legacy challenger as an investigated alternative.