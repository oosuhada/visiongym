# VisionGym Evaluation Summary

- Model: `Qwen/Qwen3-VL-2B-Instruct+VisionGym-LoRA`
- Prompt mode: `direct`
- Samples: 1400
- Overall accuracy: 58.64%
- ID accuracy: 65.50%
- OOD accuracy: 55.90%
- OOD gap: 9.60%
- Invalid output rate: 0.00%

## Error distribution

- spatial_inversion: 189
- distance_reasoning_error: 122
- object_confusion: 103
- relation_chain_failure: 91
- counting_error: 66
- relation_error: 8

## OOD error distribution

- spatial_inversion: 148
- distance_reasoning_error: 81
- object_confusion: 78
- relation_chain_failure: 72
- counting_error: 54
- relation_error: 8
