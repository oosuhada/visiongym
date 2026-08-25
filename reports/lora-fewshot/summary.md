# VisionGym Evaluation Summary

- Model: `Qwen/Qwen3-VL-2B-Instruct+VisionGym-LoRA`
- Prompt mode: `fewshot`
- Samples: 1400
- Overall accuracy: 56.43%
- ID accuracy: 66.50%
- OOD accuracy: 52.40%
- OOD gap: 14.10%
- Invalid output rate: 0.00%

## Error distribution

- spatial_inversion: 207
- distance_reasoning_error: 139
- object_confusion: 102
- relation_chain_failure: 87
- counting_error: 67
- relation_error: 8

## OOD error distribution

- spatial_inversion: 165
- distance_reasoning_error: 96
- object_confusion: 81
- relation_chain_failure: 72
- counting_error: 54
- relation_error: 8
