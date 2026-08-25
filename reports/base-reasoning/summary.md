# VisionGym Evaluation Summary

- Model: `Qwen/Qwen3-VL-2B-Instruct`
- Prompt mode: `reasoning`
- Samples: 1400
- Overall accuracy: 47.93%
- ID accuracy: 55.25%
- OOD accuracy: 45.00%
- OOD gap: 10.25%
- Invalid output rate: 0.00%

## Error distribution

- spatial_inversion: 263
- distance_reasoning_error: 144
- object_confusion: 133
- relation_chain_failure: 96
- counting_error: 82
- other: 6
- relation_error: 5

## OOD error distribution

- spatial_inversion: 204
- object_confusion: 100
- distance_reasoning_error: 96
- relation_chain_failure: 78
- counting_error: 63
- other: 5
- relation_error: 4
