# Evaluation Tool

A Python tool for evaluating voice synthesis quality using the Gemini API. This tool processes voice samples, scores them using AI evaluation, and provides comprehensive analysis with visualization.

## Features

- **Concurrent API Calls**: Multi-threaded evaluation for faster processing
- **Resume from Breakpoint**: Automatically skip already processed samples
- **Comprehensive Analysis**: Statistical analysis with weighted scoring
- **Visualization**: Generates charts for ability-based score analysis
- **Complete Output Logging**: Saves full Gemini responses for later review

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/alibaba/voice-gen-eval.git
cd voice-gen-eval
```

2. **Install dependencies**:

```bash
pip install google-generativeai matplotlib pandas tqdm
```

## Usage

### Example

We provide a complete example dataset to help you get started quickly. The repository includes sample model inference results located at:

opensource/VoiceGenEval/data/examples/model_res/en/
 ├── wav/                                    # Audio files directory
 │   ├── acoustic_properties-age-1-gpt-4o-audio-preview-2025-06-03.mp3
 │   ├── instruction-switch-381-gpt-4o-audio-preview-2025-06-03.mp3
 │   └── ...                                # More audio files
 └── metadata.jsonl                         # Metadata file

To run evaluation on the provided example data:

```bash
python gemini_eval.py \
    --root_dir ./data/examples/model_res/en/wav \
    --metadata_path ./data/examples/model_res/en/metadata.jsonl \
    --out_dir ./data/examples/eval_res/en \
    --gemini_api_key YOUR_API_KEY
```

This will use the default paths pointing to the example dataset and generate evaluation results in `./data/examples/eval_res/en/`.

### Custom Dataset

For your own dataset, specify the paths:

```bash
python gemini_eval.py \
    --root_dir /path/to/your/wav/files \
    --metadata_path /path/to/your/metadata.jsonl \
    --out_dir /path/to/output \
    --gemini_api_key YOUR_API_KEY \
    --concurrency 8
```

## Input Data Format

### Metadata JSONL Format

Each line in the metadata file should be a JSON object containing the following fields:

```json
{
    "id": 3801,
    "instruct_id": 381,
    "model_name": "gpt-4o-audio-preview-2025-06-03",
    "ability": "instruction/variation",
    "instruct_text": "Please start with a tone of tiredness and weariness, as if you've had a long, difficult day, and then shift to a tone of relief and satisfaction as you say the following sentence: \"It's been a long day--but now, it's over.\"",
    "response_audio_path": "instruction-switch-381-gpt-4o-audio-preview-2025-06-03.mp3"
}
```

#### Required Fields

- **`id`**: Unique identifier for the sample
- **`model_name`**: Name of the model that generated this response
- **`ability`**: Evaluation category in format `category/subcategory` (e.g., `instruction/variation`)
- **`instruct_text`**: The instruction text given to the model
- **`response_audio_path`**: Relative path to the generated audio file (from `root_dir`)
- **`instruct_id`**: Instruction identifier for grouping

## Parameters

| Parameter                 | Type | Default                                         | Description                                                                            |
| ------------------------- | ---- | ----------------------------------------------- | -------------------------------------------------------------------------------------- |
| `--root_dir`            | str  | `./data/examples/model_res/en/wav`            | Root directory containing WAV audio files                                              |
| `--metadata_path`       | str  | `./data/examples/model_res/en/metadata.jsonl` | Path to the metadata JSONL file containing sample information                          |
| `--prompts_dir`         | str  | `lalm_eval/eval_prompts/en`                   | Directory containing prompt template files (*.txt) for different evaluation categories |
| `--out_dir`             | str  | `./data/examples/eval_res/en`                 | Output directory for evaluation results and visualizations                             |
| `--gemini_api_key`      | str  | **Required**                              | Google Gemini API key for accessing the evaluation service                             |
| `--model_name`          | str  | `gemini-2.5-pro-preview-06-05`                | Gemini model name to use for evaluation                                                |
| `--max_retry_api`       | int  | `5`                                           | Maximum number of API retry attempts on failure                                        |
| `--sleep_between_retry` | int  | `5`                                           | Sleep time in seconds between API retry attempts                                       |
| `--max_tokens`          | int  | `4096`                                        | Maximum number of tokens for API responses                                             |
| `--max_per_ability`     | int  | `100000`                                      | Maximum number of samples to evaluate per ability category                             |
| `--concurrency`         | int  | `4`                                           | Number of concurrent threads for parallel processing                                   |
| `--overwrite`           | flag | `False`                                       | Overwrite existing results and restart evaluation from scratch                         |

## Output

### Generated Files

1. **metadata_with_score.jsonl**: Original metadata with Gemini scores
2. **Individual response files**: `<model>/<category>/<subcategory>/<id>.txt`
3. **Visualization charts**: `<model>/ability_scores.png`

## Resume Functionality

The tool automatically detects previously processed samples and skips them. To restart from scratch, use the `--overwrite` flag.

## Submit your results

To submit your results to VoiceGenEval, please send the results file (metadata_with_score.jsonl) to jzhan24@m.fudan.edu.cn.
