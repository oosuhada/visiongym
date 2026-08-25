# VisionGym Evaluation Summary

- Model: `Qwen/Qwen3-VL-2B-Instruct`
- Prompt mode: `direct`
- Samples: 1400
- Overall accuracy: 49.29%
- ID accuracy: 53.25%
- OOD accuracy: 47.70%
- OOD gap: 5.55%
- Invalid output rate: 0.00%

## Error distribution

- spatial_inversion: 273
- distance_reasoning_error: 136
- relation_chain_failure: 101
- object_confusion: 96
- counting_error: 78
- relation_error: 16
- other: 10

## OOD error distribution

- spatial_inversion: 214
- distance_reasoning_error: 89
- relation_chain_failure: 78
- object_confusion: 66
- counting_error: 57
- relation_error: 14
- other: 5
