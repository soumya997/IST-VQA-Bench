"""
VLM-as-a-Judge for IST-VQA-Bench (Gemini API, multi-threaded)
=============================================================
Loads ground-truth QA + embedded images from the IST-VQA-Bench Hugging Face
dataset, joins model predictions from a CSV, and scores each answer with a
multi-dimensional VQA judge prompt.

Predictions CSV (required columns):
  - id              sample id from IST-VQA-Bench
  - generated_answer  model prediction to judge

Optional legacy columns (ignored for GT/image; used only if --id matching fails
fallbacks are needed): user_query, image_path.

Output CSV: per sample — GT, prediction, language, data_source, judge scores.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import time
import concurrent.futures
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from google import genai
from google.genai import types
from google.genai.errors import APIError
from PIL import Image as PILImage

# ─── defaults ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    Path(__file__).resolve().parents[2] / "datasets" / "ist_vqa" / "IST-VQA-Bench"
)
DEFAULT_PRED_CSV = REPO_ROOT / "model_eval_results.csv"
DEFAULT_OUT_CSV = REPO_ROOT / "judge_eval_results.csv"

JUDGE_MODEL = "gemini-3.1-flash-lite"
FALLBACK_MODEL = "gemma-4-31b-it"
REQUEST_DELAY = 3.0
MAX_RETRIES = 3
RETRY_INITIAL_DELAY = 6.0
RETRY_MAX_DELAY = 50.0
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
MAX_WORKERS = 8
PIPELINE_NAME = "model_answer"

JUDGE_SCORE_KEYS = [
    "visual_grounding",
    "visual_grounding_rationale",
    "question_relevance",
    "question_relevance_rationale",
    "reference_alignment",
    "reference_alignment_rationale",
    "language_correctness",
    "language_correctness_rationale",
    "overall_score",
    "overall_rationale",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLM judge for IST-VQA-Bench")
    p.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path or Hub id for IST-VQA-Bench (default: sibling datasets/ist_vqa/IST-VQA-Bench)",
    )
    p.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PRED_CSV,
        help="CSV with columns: id, generated_answer",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help="Output judge CSV path",
    )
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--fallback-model", default=FALLBACK_MODEL)
    p.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    p.add_argument("--request-delay", type=float, default=REQUEST_DELAY)
    p.add_argument(
        "--pipeline-name",
        default=PIPELINE_NAME,
        help="Prefix for the prediction column in the output CSV",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="Gemini API key (or set GEMINI_API_KEY)",
    )
    return p.parse_args()


def strip_image_token(text: str) -> str:
    return text.replace("<image>", "").strip()


def qa_from_conversations(conversations) -> tuple[str, str]:
    """Extract (question, answer) from chat-style conversations."""
    question, answer = "", ""
    for turn in conversations:
        role = turn.get("from") or turn.get("role") or ""
        value = turn.get("value") or turn.get("content") or ""
        if role in ("human", "user"):
            question = strip_image_token(value)
        elif role in ("gpt", "assistant"):
            answer = value.strip()
    if not question or not answer:
        raise ValueError(f"Could not parse QA from conversations: {conversations!r}")
    return question, answer


def load_benchmark(dataset_path: Path | str) -> list[dict]:
    """Load IST-VQA-Bench test split into judge-ready rows (PIL images)."""
    path = Path(dataset_path)
    print(f"Loading IST-VQA-Bench from: {path}")

    parquet_files = sorted((path / "data").glob("test-*.parquet")) if path.is_dir() else []
    if parquet_files:
        # Local parquet layout — avoid Hugging Face datasets cache locks.
        from datasets import Dataset

        ds = Dataset.from_parquet([str(p) for p in parquet_files])
    else:
        ds = load_dataset(str(dataset_path), split="test")

    rows = []
    for ex in ds:
        question, answer = qa_from_conversations(ex["conversations"])
        rows.append(
            {
                "id": ex["id"],
                "image": ex["image"],  # PIL.Image via Image() feature
                "image_sha256": ex["image_sha256"],
                "user_query": question,
                "ground_truth": answer,
                "language": ex["language"],
                "data_source": ex["data_source"],
            }
        )
    print(f"  loaded {len(rows)} samples")
    return rows


def load_predictions(pred_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(pred_csv)
    if "generated_answer" not in df.columns:
        raise ValueError(
            f"{pred_csv} must contain a 'generated_answer' column. "
            f"Found: {list(df.columns)}"
        )
    if "id" not in df.columns:
        raise ValueError(
            f"{pred_csv} must contain an 'id' column matching IST-VQA-Bench sample ids. "
            f"Found: {list(df.columns)}"
        )
    # Keep first prediction per id
    df = df.drop_duplicates(subset=["id"], keep="first")
    return df.set_index("id", drop=False)


def merge_bench_and_preds(bench_rows: list[dict], pred_df: pd.DataFrame) -> list[dict]:
    merged = []
    missing = []
    for row in bench_rows:
        sid = row["id"]
        if sid not in pred_df.index:
            missing.append(sid)
            continue
        pred = pred_df.loc[sid]
        if isinstance(pred, pd.DataFrame):
            pred = pred.iloc[0]
        item = dict(row)
        item["generated_answer"] = str(pred["generated_answer"])
        merged.append(item)
    if missing:
        print(
            f"WARNING: {len(missing)}/{len(bench_rows)} bench samples have no prediction "
            f"(e.g. first ids: {missing[:5]})"
        )
    print(f"Judging {len(merged)} samples with predictions")
    return merged


def encode_image(image: PILImage.Image | bytes | str | Path) -> tuple[str, str]:
    """Return (base64_data, mime_type) for a PIL image, raw bytes, or file path."""
    if isinstance(image, PILImage.Image):
        buf = io.BytesIO()
        fmt = (image.format or "JPEG").upper()
        if fmt not in {"JPEG", "JPG", "PNG", "WEBP", "GIF"}:
            fmt = "JPEG"
        save_fmt = "JPEG" if fmt in {"JPEG", "JPG"} else fmt
        image.save(buf, format=save_fmt)
        mime = mimetypes.types_map.get(f".{save_fmt.lower()}", "image/jpeg")
        if save_fmt == "JPG":
            mime = "image/jpeg"
        return base64.b64encode(buf.getvalue()).decode(), mime

    if isinstance(image, (bytes, bytearray)):
        return base64.b64encode(bytes(image)).decode(), "image/jpeg"

    path = Path(image)
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


def make_client(api_key: str) -> genai.Client:
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        raise SystemExit(
            "Set GEMINI_API_KEY or pass --api-key before running the judge."
        )
    return genai.Client(
        api_key=api_key,
        http_options={
            "retry_options": types.HttpRetryOptions(
                attempts=8,
                initial_delay=5.0,
                max_delay=90.0,
            )
        },
    )


def generate_with_retry(
    client: genai.Client,
    model: str,
    fallback_model: str,
    contents: list,
    request_delay: float,
) -> tuple[str, str]:
    """Call Gemini with retries. Returns (text, model_name_actually_used)."""
    models_to_try = [model]
    if model != fallback_model:
        models_to_try.append(fallback_model)

    last_err = None
    for model_name in models_to_try:
        delay = RETRY_INITIAL_DELAY
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                t0 = time.perf_counter()
                response = client.models.generate_content(
                    model=model_name, contents=contents
                )
                api_latency = time.perf_counter() - t0
                if request_delay > 0:
                    time.sleep(request_delay)
                text = getattr(response, "text", None)
                if not text or not str(text).strip():
                    raise ValueError(f"Empty response.text from {model_name}")
                text = str(text).strip()
                used = getattr(response, "model", None) or model_name
                print(
                    f"  judge_model={used} (requested={model_name}) "
                    f"api_latency={api_latency:.2f}s"
                )
                return text, used
            except APIError as e:
                last_err = e
                if e.code not in RETRYABLE_STATUS_CODES:
                    raise
                if attempt == MAX_RETRIES:
                    print(f"  Model {model_name} unavailable after {MAX_RETRIES} attempts.")
                    break
                wait = min(delay, RETRY_MAX_DELAY)
                print(
                    f"  API {e.code} on {model_name}, "
                    f"retry {attempt}/{MAX_RETRIES} in {wait:.0f}s..."
                )
                time.sleep(wait)
                delay *= 2
            except ValueError as e:
                last_err = e
                if attempt == MAX_RETRIES:
                    print(f"  {e} after {MAX_RETRIES} attempts.")
                    break
                wait = min(delay, RETRY_MAX_DELAY)
                print(f"  {e}; retry {attempt}/{MAX_RETRIES} in {wait:.0f}s...")
                time.sleep(wait)
                delay *= 2
    raise last_err


def judge_compare(
    client: genai.Client,
    image: PILImage.Image,
    question: str,
    reference_answer: str,
    model_answer: str,
    judge_model: str,
    fallback_model: str,
    request_delay: float,
) -> dict:
    """Ask Gemini to score the model answer using the image + reference."""
    judge_prompt = f"""You are an expert visual question answering evaluator. You will be shown:
1. An image
2. A question about the image (in Bengali, Hindi, Tamil, or English)
3. A reference answer (ground truth)
4. A model's predicted answer

Your task is to evaluate the model's answer on multiple dimensions. IMPORTANT: Use the IMAGE as the primary source of truth, not just the reference answer. The reference answer may be incomplete or may describe only one aspect of the image.

## Evaluation Criteria

Score each dimension from 0.0 to 1.0:

### 1. Visual Grounding (weight: 0.35)
Is the information in the model's answer actually present and verifiable in the image?
- 1.0: Everything stated is visually verifiable in the image
- 0.5: Partially verifiable — some claims correct, some unverifiable or wrong
- 0.0: The answer contains information not present in the image

### 2. Question Relevance (weight: 0.25)
Does the answer address what the question is actually asking?
- 1.0: Directly and completely answers the question
- 0.5: Partially answers or answers a related but different aspect
- 0.0: Does not address the question at all

### 3. Reference Alignment (weight: 0.20)
How well does the answer align with the reference answer's content?
- 1.0: Covers the same information as the reference
- 0.5: Covers different but equally valid information from the same image region
- 0.0: Contradicts the reference or describes entirely different content

### 4. Language Correctness (weight: 0.20)
Is the answer in the correct language with proper grammar and script usage?
- 1.0: Correct language, natural grammar, proper script
- 0.5: Correct language but with errors, or mixed scripts
- 0.0: Wrong language entirely

## Inputs

**Question:** {question}

**Reference Answer:** {reference_answer}

**Model's Answer:** {model_answer}

## Special Instructions
- If the model's answer describes different but CORRECT text/content from the image that the reference answer did not mention, this is NOT a failure. Score Visual Grounding based on the image, not the reference.
- For scene text questions, verify whether the text quoted by the model actually appears in the image.
- Be sensitive to partial correctness — a model that correctly identifies some text but misses other text should get partial credit, not zero.

## Output Format
Respond ONLY with this JSON, no markdown fences, no other text:
{{
  "visual_grounding": <float 0.0-1.0>,
  "visual_grounding_rationale": "<one sentence>",
  "question_relevance": <float 0.0-1.0>,
  "question_relevance_rationale": "<one sentence>",
  "reference_alignment": <float 0.0-1.0>,
  "reference_alignment_rationale": "<one sentence>",
  "language_correctness": <float 0.0-1.0>,
  "language_correctness_rationale": "<one sentence>",
  "overall_score": <float 0.0-1.0>,
  "overall_rationale": "<one sentence>"
}}

The overall_score should be the weighted sum: (visual_grounding * 0.35) + (question_relevance * 0.25) + (reference_alignment * 0.20) + (language_correctness * 0.20)."""

    b64, mime = encode_image(image)
    t0 = time.perf_counter()
    text, used_model = generate_with_retry(
        client,
        judge_model,
        fallback_model,
        contents=[
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": mime, "data": b64}},
                    {"text": judge_prompt},
                ],
            }
        ],
        request_delay=request_delay,
    )
    latency = time.perf_counter() - t0
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    verdict = json.loads(text)
    verdict["judge_model_used"] = used_model
    verdict["judge_latency"] = round(latency, 3)
    return verdict


def process_row(args: tuple) -> dict:
    """Worker that judges a single merged bench+prediction row."""
    (
        i,
        row,
        total,
        pipeline_name,
        client,
        judge_model,
        fallback_model,
        request_delay,
    ) = args

    sid = row["id"]
    user_query = row["user_query"]
    print(f"[{i + 1}/{total}] Judging: {sid} | {user_query[:50]}...")

    try:
        verdict = judge_compare(
            client,
            row["image"],
            user_query,
            row["ground_truth"],
            row["generated_answer"],
            judge_model,
            fallback_model,
            request_delay,
        )
    except Exception as e:
        print(f"  ERROR on row {i + 1}: {e}")
        verdict = {
            k: (
                0.0
                if k.endswith("_score")
                or k
                in {
                    "visual_grounding",
                    "question_relevance",
                    "reference_alignment",
                    "language_correctness",
                    "overall_score",
                }
                else f"JUDGE_ERROR: {e}"
            )
            for k in JUDGE_SCORE_KEYS
        }
        verdict["judge_model_used"] = "ERROR"
        verdict["judge_latency"] = 0.0

    result_row = {
        "id": sid,
        "image_sha256": row.get("image_sha256", ""),
        "user_query": user_query,
        "judge_response": row["ground_truth"],
        f"{pipeline_name}_response": row["generated_answer"],
        "language": row.get("language", ""),
        "data_source": row.get("data_source", ""),
        "judge_model_used": verdict.get("judge_model_used", ""),
        "judge_latency": verdict.get("judge_latency", 0.0),
    }
    for key in JUDGE_SCORE_KEYS:
        result_row[key] = verdict.get(key)
    print(
        f"  -> model={result_row['judge_model_used']} "
        f"latency={result_row['judge_latency']}s "
        f"(requested={judge_model})"
    )
    return result_row


def resume_state(output_csv: Path) -> tuple[list[dict], int]:
    """Load existing output for resume; return (results, start_idx)."""
    if not output_csv.exists() or output_csv.stat().st_size <= 1:
        return [], 0
    try:
        existing_df = pd.read_csv(output_csv)
    except pd.errors.EmptyDataError:
        return [], 0
    if not len(existing_df):
        print(f"Output exists but empty — starting fresh: {output_csv}")
        return [], 0
    results = existing_df.to_dict("records")
    print(f"Resuming from {len(results)} completed rows in {output_csv}")
    return results, len(results)


def retry_error_rows(
    results: list[dict],
    merged_by_id: dict[str, dict],
    pipeline_name: str,
    client: genai.Client,
    judge_model: str,
    fallback_model: str,
    request_delay: float,
    output_csv: Path,
) -> None:
    err_idxs = [i for i, r in enumerate(results) if r.get("judge_model_used") == "ERROR"]
    if not err_idxs:
        return
    print(f"Retrying {len(err_idxs)} ERROR rows: {err_idxs}")
    for i in err_idxs:
        sid = results[i].get("id")
        if sid not in merged_by_id:
            print(f"  skip ERROR row {i}: id={sid} not in current merge set")
            continue
        results[i] = process_row(
            (
                i,
                merged_by_id[sid],
                len(results),
                pipeline_name,
                client,
                judge_model,
                fallback_model,
                request_delay,
            )
        )
        pd.DataFrame(results).to_csv(output_csv, index=False)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("\n=--------------------")
    print("DATASET:     ", args.dataset)
    print("PREDICTIONS: ", args.predictions)
    print("OUTPUT:      ", args.output)
    print("JUDGE_MODEL: ", args.judge_model)
    print("=--------------------\n")

    client = make_client(args.api_key)
    bench_rows = load_benchmark(args.dataset)
    pred_df = load_predictions(args.predictions)
    merged = merge_bench_and_preds(bench_rows, pred_df)
    merged_by_id = {r["id"]: r for r in merged}

    results, start_idx = resume_state(args.output)
    retry_error_rows(
        results,
        merged_by_id,
        args.pipeline_name,
        client,
        args.judge_model,
        args.fallback_model,
        args.request_delay,
        args.output,
    )

    remaining = merged[start_idx:]
    print(f"Total rows: {len(merged)}, remaining: {len(remaining)}")

    if not remaining:
        print("All rows already processed!")
    else:
        work_items = [
            (
                start_idx + i,
                row,
                len(merged),
                args.pipeline_name,
                client,
                args.judge_model,
                args.fallback_model,
                args.request_delay,
            )
            for i, row in enumerate(remaining)
        ]
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.max_workers
            ) as executor:
                for result_row in executor.map(process_row, work_items):
                    results.append(result_row)
                    pd.DataFrame(results).to_csv(args.output, index=False)
        except KeyboardInterrupt:
            print(f"\nInterrupted — saving {len(results)} rows to {args.output}")
            pd.DataFrame(results).to_csv(args.output, index=False)
            return

    print(f"\nDone! Results saved to {args.output}")

    results_df = pd.DataFrame(results)
    print(f"\n{'=' * 50}")
    n = len(results_df)
    scores = pd.to_numeric(results_df["overall_score"], errors="coerce")
    n_bad = int(scores.isna().sum())
    avg = scores.mean()
    print(
        f"{args.pipeline_name:20s} -> Avg overall_score: {avg:.2f} | n={n}"
        + (f" (excluded {n_bad} non-numeric)" if n_bad else "")
    )
    if "judge_latency" in results_df.columns:
        print(
            "Avg judge_latency: "
            f"{pd.to_numeric(results_df['judge_latency'], errors='coerce').mean():.2f}s"
        )
    if "judge_model_used" in results_df.columns:
        print("judge_model_used counts:")
        print(results_df["judge_model_used"].value_counts().to_string())

    if "language" in results_df.columns:
        print("\nAvg overall_score by language:")
        tmp = results_df.copy()
        tmp["overall_score"] = pd.to_numeric(tmp["overall_score"], errors="coerce")
        print(tmp.groupby("language")["overall_score"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
