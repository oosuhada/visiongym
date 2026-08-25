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

- ood_failure: 476
- distance_reasoning_error: 43
- spatial_inversion: 42
- object_confusion: 21
- relation_chain_failure: 15
- counting_error: 13
