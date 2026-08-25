# VisionGym Evaluation Summary

- Model: `Qwen/Qwen3-VL-2B-Instruct`
- Prompt mode: `fewshot`
- Samples: 1400
- Overall accuracy: 54.50%
- ID accuracy: 60.50%
- OOD accuracy: 52.10%
- OOD gap: 8.40%
- Invalid output rate: 0.00%

## Error distribution

- spatial_inversion: 211
- distance_reasoning_error: 127
- relation_chain_failure: 94
- object_confusion: 91
- counting_error: 78
- relation_error: 30
- other: 6

## OOD error distribution

- spatial_inversion: 166
- distance_reasoning_error: 87
- relation_chain_failure: 76
- object_confusion: 64
- counting_error: 58
- relation_error: 24
- other: 4
