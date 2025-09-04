# Reproducing Human-Model Correlation Results

This section reproduces the correlation analysis between model evaluations and human annotations as reported in the paper.

## Models Evaluated

The analysis includes seven models from the paper:

**Commercial Models:**

- GPT-4o Audio (snapshot: gpt-4o-audio-preview-2025-06-03)
- GPT-4o-Mini Audio (snapshot: gpt-4o-mini-audio-preview-2024-12-17)
- Doubao's end-to-end real-time speech dialogue model

**Open-Source End-to-End Speech Language Models:**

- Step-Audio
- Kimi-Audio
- Baichuan-Audio
- Qwen-2.5 Omni

## Quick Start

```bash
# Download evaluation results for all seven models
huggingface-cli download --repo-type dataset --local-dir-use-symlinks False zhanjun/VoiceGenEval-eval-results --local-dir VoiceGenEval-eval-results

# Compute Spearman correlation coefficients between models and human evaluations
python human_align/compute_model_human_spearman_r.py
```

## What This Does

The script calculates Spearman correlation coefficients between human annotations and model scores across:

- Overall correlations (human average vs model, individual annotators vs model)
- Inter-annotator agreement (human vs human)
- Category-specific correlations for different ability types (instruction, reasoning, etc.)
- Results for both English and Chinese evaluations