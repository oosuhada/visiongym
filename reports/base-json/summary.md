# VisionGym Evaluation Summary

- Model: `Qwen/Qwen3-VL-2B-Instruct`
- Prompt mode: `json`
- Samples: 1400
- Overall accuracy: 48.29%
- ID accuracy: 50.50%
- OOD accuracy: 47.40%
- OOD gap: 3.10%
- Invalid output rate: 0.00%

## Error distribution

- spatial_inversion: 288
- distance_reasoning_error: 142
- relation_chain_failure: 114
- object_confusion: 82
- counting_error: 74
- other: 20
- relation_error: 4

## OOD error distribution

- spatial_inversion: 220
- distance_reasoning_error: 95
- relation_chain_failure: 84
- object_confusion: 59
- counting_error: 55
- other: 9
- relation_error: 4
