#!/usr/bin/env python3
"""
Filter prompts in train.csv and val.csv using an LLM via vLLM endpoint.

Removes prompts that describe entire image changes while keeping editing-related
changes (color, contrast, perspective fixes, crops, etc.).

Outputs:
  - {split}_filtered.csv: Kept rows
  - {split}_removed.csv: Removed rows with removal reasons
"""

import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI


DEFAULT_SYSTEM_PROMPT = (
    "You are a prompt classifier for image editing datasets. Your task is to classify "
    "whether a prompt describes genuine editing operations or entire image replacements.\n\n"
    "KEEP prompts that describe:\n"
    "- Color/contrast/brightness/saturation adjustments\n"
    "- Perspective or lens distortion corrections\n"
    "- Cropping or composition tweaks\n"
    "- Lighting changes\n"
    "- Minor object removal/addition\n"
    "- Stylistic adjustments (e.g., filters, tones)\n"
    "- Sharpening, noise reduction, or similar enhancements\n\n"
    "REMOVE prompts that describe:\n"
    "- Complete scene changes\n"
    "- Entire image replacement\n"
    "- Wholesale content swaps\n"
    "- Transforming one subject into a completely different subject\n\n"
    "Respond ONLY with valid JSON in this exact format:\n"
    '{{"decision": "KEEP", "reason": "brief explanation"}} or '
    '{{"decision": "REMOVE", "reason": "brief explanation"}}'
)

DEFAULT_USER_PROMPT_TEMPLATE = (
    "Classify this image editing prompt:\n\n\"{prompt}\"\n\n"
    "Respond with JSON containing 'decision' (KEEP or REMOVE) and 'reason'."
)


def load_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Load CSV file into a list of dictionaries."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def partials_dir_for(csv_path: Path, suffix: str = "filter") -> Path:
    """Generate directory path for partial results."""
    return csv_path.parent / f".{csv_path.stem}_{suffix}_partials"


def load_partial_results(partial_dir: Path) -> Dict[int, Dict[str, str]]:
    """Load previously saved classification results."""
    results: Dict[int, Dict[str, str]] = {}
    if not partial_dir.exists():
        return results
    
    for entry in partial_dir.glob("*.json"):
        try:
            idx = int(entry.stem)
        except ValueError:
            continue
        try:
            with entry.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                if "decision" in data and "reason" in data:
                    results[idx] = data
        except (OSError, json.JSONDecodeError):
            continue
    return results


def persist_partial_result(
    partial_dir: Path, idx: int, decision: Optional[str], reason: Optional[str], error: Optional[str]
) -> None:
    """Save classification result for a single row."""
    partial_dir.mkdir(parents=True, exist_ok=True)
    
    payload: Dict[str, Any] = {}
    if decision is not None:
        payload["decision"] = decision
    if reason is not None:
        payload["reason"] = reason
    if error is not None:
        payload["error"] = error
    
    partial_path = partial_dir / f"{idx:08d}.json"
    if not payload:
        if partial_path.exists():
            try:
                partial_path.unlink()
            except OSError:
                pass
        return
    
    tmp_path = partial_path.with_name(partial_path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    tmp_path.replace(partial_path)


async def classify_prompt(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_prompt_template: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
    disable_thinking: bool,
    retries: int,
    retry_delay: float,
) -> Tuple[str, str]:
    """
    Classify a prompt using the LLM.
    
    Returns:
        Tuple of (decision, reason) where decision is "KEEP" or "REMOVE"
    """
    user_prompt = user_prompt_template.format(prompt=prompt)
    
    attempt = 0
    while True:
        attempt += 1
        try:
            async with semaphore:
                # Build extra_body for additional parameters
                extra_body = {
                    "top_k": top_k,
                }
                
                # For Qwen3 base models, disable thinking mode via chat_template_kwargs
                if disable_thinking and "Qwen3" in model and "Instruct-2507" not in model:
                    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    extra_body=extra_body,
                )
                
                content = response.choices[0].message.content.strip()
                
                # Try to parse JSON response
                try:
                    result = json.loads(content)
                    decision = result.get("decision", "").upper()
                    reason = result.get("reason", "")
                    
                    if decision not in ("KEEP", "REMOVE"):
                        raise ValueError(f"Invalid decision: {decision}")
                    
                    return decision, reason
                except (json.JSONDecodeError, ValueError) as parse_exc:
                    # If JSON parsing fails, try to extract decision from text
                    content_upper = content.upper()
                    if "KEEP" in content_upper and "REMOVE" not in content_upper:
                        return "KEEP", content
                    elif "REMOVE" in content_upper and "KEEP" not in content_upper:
                        return "REMOVE", content
                    else:
                        raise ValueError(f"Could not parse response: {content}") from parse_exc
        
        except Exception as exc:
            if attempt > retries:
                raise RuntimeError(
                    f"Failed to classify prompt after {retries} retries: {exc}"
                ) from exc
            await asyncio.sleep(retry_delay * attempt)


def write_filtered_csv(rows: List[Dict[str, Any]], decisions: List[str], output_path: Path) -> int:
    """Write filtered CSV with only KEEP rows."""
    if not rows:
        return 0
    
    kept_rows = [row for row, decision in zip(rows, decisions) if decision == "KEEP"]
    
    if not kept_rows:
        # Write empty CSV with headers
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
        return 0
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(kept_rows[0].keys()))
        writer.writeheader()
        writer.writerows(kept_rows)
    
    return len(kept_rows)


def write_removed_csv(
    rows: List[Dict[str, Any]], decisions: List[str], reasons: List[str], output_path: Path
) -> int:
    """Write removed CSV with REMOVE rows and removal reasons."""
    if not rows:
        return 0
    
    removed_data = [
        (row, reason)
        for row, decision, reason in zip(rows, decisions, reasons)
        if decision == "REMOVE"
    ]
    
    if not removed_data:
        # Write empty CSV with headers
        fieldnames = list(rows[0].keys()) + ["removal_reason"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
        return 0
    
    # Add removal_reason column
    fieldnames = list(removed_data[0][0].keys()) + ["removal_reason"]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, reason in removed_data:
            row_with_reason = dict(row)
            row_with_reason["removal_reason"] = reason
            writer.writerow(row_with_reason)
    
    return len(removed_data)


async def process_split(
    split_name: str,
    csv_path: Optional[Path],
    filtered_path: Optional[Path],
    removed_path: Optional[Path],
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
) -> None:
    """Process a single split (train or val)."""
    if not csv_path or not filtered_path or not removed_path:
        return
    
    rows = load_csv(csv_path)
    if not rows:
        print(f"[{split_name}] No rows found in {csv_path}, skipping.", file=sys.stderr)
        return
    
    total_rows = len(rows)
    print(f"[{split_name}] loaded {total_rows} rows from {csv_path}", file=sys.stderr)
    
    # Load partial results
    partial_dir = partials_dir_for(csv_path, "filter")
    partial_results = load_partial_results(partial_dir)
    
    decisions: List[Optional[str]] = [None] * total_rows
    reasons: List[Optional[str]] = [None] * total_rows
    errors: List[Optional[str]] = [None] * total_rows
    
    restored_count = 0
    for idx in range(total_rows):
        partial = partial_results.get(idx)
        if not partial:
            continue
        decision = partial.get("decision")
        reason = partial.get("reason")
        if decision and reason:
            decisions[idx] = decision
            reasons[idx] = reason
            restored_count += 1
    
    if restored_count:
        print(
            f"[{split_name}] restored {restored_count} decisions from {partial_dir}",
            file=sys.stderr,
        )
    
    async def run_row(idx: int, row: Dict[str, Any]) -> Tuple[int, Optional[str], Optional[str], Optional[str]]:
        """Classify a single row."""
        prompt = row.get("prompt", "")
        if not prompt:
            return idx, "REMOVE", "Empty prompt", None
        
        try:
            decision, reason = await classify_prompt(
                client=client,
                prompt=prompt,
                model=args.model,
                semaphore=semaphore,
                system_prompt=args.system_prompt,
                user_prompt_template=args.user_prompt_template,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                presence_penalty=args.presence_penalty,
                disable_thinking=args.disable_thinking,
                retries=args.retries,
                retry_delay=args.retry_delay,
            )
            return idx, decision, reason, None
        except Exception as exc:
            return idx, None, None, str(exc)
    
    # Process pending rows
    pending_indices = [idx for idx, decision in enumerate(decisions) if decision is None]
    pending_total = len(pending_indices)
    
    if pending_total:
        tasks = [asyncio.create_task(run_row(idx, rows[idx])) for idx in pending_indices]
        completed_total = restored_count
        progress_interval = max(1, args.progress_interval)
        
        for future in asyncio.as_completed(tasks):
            idx, decision, reason, error = await future
            if decision is not None:
                decisions[idx] = decision
                reasons[idx] = reason
                errors[idx] = None
            else:
                errors[idx] = error
            
            persist_partial_result(partial_dir, idx, decision, reason, errors[idx])
            
            completed_total += 1
            if completed_total % progress_interval == 0 or completed_total == total_rows:
                print(
                    f"[{split_name}] processed {completed_total}/{total_rows} prompts",
                    file=sys.stderr,
                )
            
            if error is not None:
                print(f"[{split_name}] row {idx} failed: {error}", file=sys.stderr)
    else:
        print(
            f"[{split_name}] all decisions restored from disk, skipping classification.",
            file=sys.stderr,
        )
    
    # Convert Optional[str] to str for output (default to REMOVE for errors)
    decisions_final = [d if d is not None else "REMOVE" for d in decisions]
    reasons_final = [r if r is not None else "Classification failed" for r in reasons]
    
    # Write outputs
    kept_count = write_filtered_csv(rows, decisions_final, filtered_path)
    removed_count = write_removed_csv(rows, decisions_final, reasons_final, removed_path)
    
    print(f"[{split_name}] kept {kept_count} prompts, wrote to {filtered_path}")
    print(f"[{split_name}] removed {removed_count} prompts, wrote to {removed_path}")
    
    failed_rows = [idx for idx, err in enumerate(errors) if err]
    if failed_rows:
        print(
            f"[{split_name}] {len(failed_rows)} prompts failed; marked as REMOVE with error reason",
            file=sys.stderr,
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter prompts using LLM classification via vLLM endpoint."
    )
    
    # Input CSVs
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("workspace/metadata-v12-recaptioned-long/train.csv"),
        help="Path to training CSV.",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path("workspace/metadata-v12-recaptioned-long/val.csv"),
        help="Path to validation CSV.",
    )
    
    # Output CSVs
    parser.add_argument(
        "--train-filtered-csv",
        type=Path,
        help="Output path for filtered training CSV (default: adds _filtered suffix).",
    )
    parser.add_argument(
        "--val-filtered-csv",
        type=Path,
        help="Output path for filtered validation CSV (default: adds _filtered suffix).",
    )
    parser.add_argument(
        "--train-removed-csv",
        type=Path,
        help="Output path for removed training CSV (default: adds _removed suffix).",
    )
    parser.add_argument(
        "--val-removed-csv",
        type=Path,
        help="Output path for removed validation CSV (default: adds _removed suffix).",
    )
    
    # vLLM endpoint configuration
    parser.add_argument(
        "--base-url",
        default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible base URL exposed by vLLM.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("VLLM_API_KEY", "EMPTY"),
        help="API key if the endpoint requires authentication.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B-Instruct-2507",
        help="Model name served by vLLM.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        default=True,
        help="Disable thinking mode for Qwen3 models (faster for classification).",
    )
    
    # LLM prompts
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for classification.",
    )
    parser.add_argument(
        "--user-prompt-template",
        default=DEFAULT_USER_PROMPT_TEMPLATE,
        help="User prompt template with {prompt} placeholder.",
    )
    
    # LLM parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens per LLM response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (recommended: 0.7 for Qwen3-Instruct-2507).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p sampling parameter.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=1.5,
        help="Presence penalty to reduce repetitions (0-2 recommended).",
    )
    
    # Async settings
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=32,
        help="Maximum concurrent requests to the endpoint.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry attempts per classification.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Base delay between retries (seconds).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Progress logging interval.",
    )
    
    args = parser.parse_args()
    
    # Set default output paths if not specified
    if not args.train_filtered_csv:
        args.train_filtered_csv = args.train_csv.parent / f"{args.train_csv.stem}_filtered{args.train_csv.suffix}"
    if not args.val_filtered_csv:
        args.val_filtered_csv = args.val_csv.parent / f"{args.val_csv.stem}_filtered{args.val_csv.suffix}"
    if not args.train_removed_csv:
        args.train_removed_csv = args.train_csv.parent / f"{args.train_csv.stem}_removed{args.train_csv.suffix}"
    if not args.val_removed_csv:
        args.val_removed_csv = args.val_csv.parent / f"{args.val_csv.stem}_removed{args.val_csv.suffix}"
    
    return args


async def main_async(args: argparse.Namespace) -> None:
    """Main async processing function."""
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.max_concurrency)
    
    await process_split(
        split_name="train",
        csv_path=args.train_csv,
        filtered_path=args.train_filtered_csv,
        removed_path=args.train_removed_csv,
        client=client,
        semaphore=semaphore,
        args=args,
    )
    
    await process_split(
        split_name="val",
        csv_path=args.val_csv,
        filtered_path=args.val_filtered_csv,
        removed_path=args.val_removed_csv,
        client=client,
        semaphore=semaphore,
        args=args,
    )


def main() -> None:
    """Main entry point."""
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

