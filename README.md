# IST-VQA-Bench

Multilingual Indian Scene Text VQA evaluation benchmark and VLM-as-a-Judge tooling.

The Hugging Face dataset lives at `datasets/ist_vqa/IST-VQA-Bench` (393 test QA pairs, English / Hindi / Bengali / Tamil, embedded images + `image_sha256`).

## Setup

```bash
cd IST-VQA-Bench
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
hf auth login --token <your_huggingface_token>
hf download somusan/IST-VQA-Bench --repo-type=dataset --local-dir ./IST-VQA-Bench
```

## Predictions CSV

Produce a CSV with one row per IST-VQA-Bench sample:

| Column | Required | Description |
|---|---|---|
| `id` | yes | Sample id from the dataset |
| `generated_answer` | yes | Model prediction to judge |

Example:

```csv
id,generated_answer
24222ba4-c557-4781-9e10-937f6c023a1b_0,"গাড়ির পেছনের অংশে নীল রঙের ব্যানারে লেখা আছে ..."
```

You can dump ids from the dataset with:

```python
from datasets import load_dataset

ds = load_dataset("../datasets/ist_vqa/IST-VQA-Bench", split="test")
print(ds[0]["id"])
```

## Run the VLM judge

Ground-truth questions/answers and images are loaded from IST-VQA-Bench; only model answers come from the predictions CSV.

```bash
python scripts/vlm_judge.py \
  --dataset ../datasets/ist_vqa/IST-VQA-Bench \
  --predictions model_eval_results.csv \
  --output judge_eval_results.csv
```

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--dataset` | `../datasets/ist_vqa/IST-VQA-Bench` | Local path or Hub dataset id |
| `--predictions` | `model_eval_results.csv` | Model answers CSV |
| `--output` | `judge_eval_results.csv` | Judge scores CSV |
| `--judge-model` | `gemini-3.1-flash-lite` | Primary judge model |
| `--fallback-model` | `gemma-4-31b-it` | Fallback if primary fails |
| `--max-workers` | `8` | Concurrent judge threads |
| `--request-delay` | `3.0` | Seconds between API calls per worker |
| `--pipeline-name` | `model_answer` | Output column prefix for predictions |
| `--api-key` | `$GEMINI_API_KEY` | Gemini API key |

The script resumes from an existing `--output` file and retries rows marked `judge_model_used=ERROR`.

## Output

Each row includes `id`, `image_sha256`, `user_query`, `judge_response` (GT), `{pipeline}_response`, `language`, `data_source`, and scores:

- `visual_grounding` (0.35)
- `question_relevance` (0.25)
- `reference_alignment` (0.20)
- `language_correctness` (0.20)
- `overall_score` (weighted sum)
- plus per-dimension rationales, `judge_model_used`, `judge_latency`

## Dataset fields (reference)

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique sample id |
| `image` | `Image()` | Embedded JPG |
| `image_sha256` | string | SHA-256 of raw JPG bytes |
| `conversations` | list | `human` question + `gpt` answer |
| `language` | string | `english` / `hindi` / `bengali` / `tamil` |
| `data_source` | string | Origin corpus |
| `split` | string | `test` |
