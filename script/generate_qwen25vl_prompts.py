#!/usr/bin/env python3
"""
Regenerate edit prompts for Qwen Image Edit Plus datasets using Qwen2.5-VL via a vLLM OpenAI-compatible endpoint.

Default inputs target the repo's metadata JSONL files:
  - workspace/metadata-v12-recaptioned-long/train/metadata.jsonl  -> train.csv
  - workspace/metadata-v12-recaptioned-long/test/metadata.jsonl   -> val.csv
"""

import argparse
import asyncio
import base64
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import AsyncOpenAI
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert photo editor. Given an original control photograph and the final edited result, "
    "describe the exact visual adjustments that transform the original into the result. Focus on actionable "
    "editing instructions suitable for conditioning an image-edit diffusion model."
)

DEFAULT_USER_PROMPT = (
    "You are provided two images: the first is the control/original input, the second is the final edited output. "
    "Summarize the visual changes needed to convert the input into the output. Mention lighting or color shifts, "
    "object additions/removals, stylistic adjustments, prespective distortion fixes, lens distortion fixes, and composition tweaks. Keep the response deatiled (multiple "
    "sentences) and avoid referring to 'first/second image'."
)


@dataclass
class MetadataRow:
    output_file_name: str
    hdr_file_name: str
    input_file_names: List[str]
    raw: Dict[str, Any]


def load_metadata(jsonl_path: Path) -> List[MetadataRow]:
    rows: List[MetadataRow] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            record = line.strip()
            if not record:
                continue
            try:
                data = json.loads(record)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {jsonl_path}:{line_no}: {exc}") from exc

            output_path = data["output_file_name"]
            input_files = list(data.get("input_file_names", []))
            control_path = data.get("hdr_file_name")
            if not control_path:
                if input_files:
                    control_path = input_files[0]
                    input_files = input_files[1:]
                else:
                    raise ValueError(
                        f"Record at {jsonl_path}:{line_no} is missing both 'hdr_file_name' and 'input_file_names'."
                    )

            rows.append(
                MetadataRow(
                    output_file_name=output_path,
                    hdr_file_name=control_path,
                    input_file_names=input_files,
                    raw=data,
                )
            )
    return rows


def encode_image(path: Path, max_edge: int, jpeg_quality: int) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as img:
        img = img.convert("RGB")
        width, height = img.size
        largest_edge = max(width, height)
        if largest_edge > max_edge:
            scale = max_edge / float(largest_edge)
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=jpeg_quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


async def generate_prompt(
    client: AsyncOpenAI,
    model: str,
    control_image: Path,
    target_image: Path,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    max_edge: int,
    jpeg_quality: int,
    retries: int,
    retry_delay: float,
) -> str:
    attempt = 0
    while True:
        attempt += 1
        try:
            async with semaphore:
                control_b64 = await asyncio.to_thread(encode_image, control_image, max_edge, jpeg_quality)
                target_b64 = await asyncio.to_thread(encode_image, target_image, max_edge, jpeg_quality)

                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "text", "text": "Original image:"},
                            {"type": "image_url", "image_url": {"url": control_b64}},
                            {"type": "text", "text": "Edited result:"},
                            {"type": "image_url", "image_url": {"url": target_b64}},
                        ],
                    },
                ]

                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            if attempt > retries:
                raise RuntimeError(f"Failed to generate prompt after {retries} retries: {exc}") from exc
            await asyncio.sleep(retry_delay * attempt)


def write_csv(rows: Sequence[MetadataRow], prompts: Sequence[str], csv_path: Path, extra_columns: Sequence[str]) -> None:
    import csv

    if len(rows) != len(prompts):
        raise ValueError("Row count and prompt count do not match.")

    max_extra_controls = max((len(row.input_file_names) for row in rows), default=0)
    fieldnames: List[str] = ["path_target", "path_control", "prompt"]
    for idx in range(max_extra_controls):
        fieldnames.append(f"path_control_{idx + 1}")
    fieldnames.extend(extra_columns)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, prompt in zip(rows, prompts):
            record: Dict[str, Any] = {
                "path_target": row.output_file_name,
                "path_control": row.hdr_file_name,
                "prompt": prompt,
            }
            for idx, control_path in enumerate(row.input_file_names):
                record[f"path_control_{idx + 1}"] = control_path
            for column in extra_columns:
                record[column] = row.raw.get(column)
            writer.writerow(record)


def write_enriched_jsonl(rows: Sequence[MetadataRow], prompts: Sequence[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row, prompt in zip(rows, prompts):
            enriched = dict(row.raw)
            enriched["prompt"] = prompt
            handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")


def partials_dir_for(csv_path: Path) -> Path:
    return csv_path.parent / f".{csv_path.stem}_partials"


def load_partial_results(partial_dir: Path) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}
    if not partial_dir.exists():
        return results
    for entry in partial_dir.glob("*.json"):
        try:
            idx = int(entry.stem)
        except ValueError:
            continue
        try:
            with entry.open("r", encoding="utf-8") as handle:
                results[idx] = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
    return results


def persist_partial_result(partial_dir: Path, idx: int, prompt: Optional[str], error: Optional[str]) -> None:
    partial_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {}
    if prompt is not None:
        payload["prompt"] = prompt
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


async def process_split(
    name: str,
    jsonl_path: Optional[Path],
    csv_path: Optional[Path],
    enriched_jsonl: Optional[Path],
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
) -> None:
    if not jsonl_path or not csv_path:
        return
    rows = load_metadata(jsonl_path)
    if not rows:
        print(f"[{name}] No rows found in {jsonl_path}, skipping.", file=sys.stderr)
        return

    partial_dir = partials_dir_for(csv_path)
    partial_results = load_partial_results(partial_dir)

    total_rows = len(rows)
    prompts: List[Optional[str]] = [None] * total_rows
    errors: List[Optional[str]] = [None] * total_rows

    restored_prompts = 0
    restored_errors = 0
    for idx in range(total_rows):
        partial = partial_results.get(idx)
        if not partial:
            continue
        prompt = partial.get("prompt")
        error = partial.get("error")
        if prompt is not None:
            prompts[idx] = prompt
            restored_prompts += 1
        if error is not None and prompt is None:
            errors[idx] = error
            restored_errors += 1

    if restored_prompts or restored_errors:
        details: List[str] = []
        if restored_prompts:
            details.append(f"{restored_prompts} prompts")
        if restored_errors:
            details.append(f"{restored_errors} errors")
        print(f"[{name}] restored {' and '.join(details)} from {partial_dir}", file=sys.stderr)

    async def run_row(idx: int, row: MetadataRow) -> Tuple[int, Optional[str], Optional[str]]:
        try:
            prompt = await generate_prompt(
                client=client,
                model=args.model,
                control_image=Path(row.hdr_file_name),
                target_image=Path(row.output_file_name),
                semaphore=semaphore,
                system_prompt=args.system_prompt,
                user_prompt=args.user_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                max_edge=args.max_edge,
                jpeg_quality=args.jpeg_quality,
                retries=args.retries,
                retry_delay=args.retry_delay,
            )
            return idx, prompt, None
        except Exception as exc:
            return idx, None, str(exc)

    pending_indices = [idx for idx, prompt in enumerate(prompts) if prompt is None]
    pending_total = len(pending_indices)
    if pending_total:
        tasks = [asyncio.create_task(run_row(idx, rows[idx])) for idx in pending_indices]
        completed_total = restored_prompts
        progress_interval = max(1, args.progress_interval)

        for future in asyncio.as_completed(tasks):
            idx, prompt, error = await future
            if prompt is not None:
                prompts[idx] = prompt
                errors[idx] = None
            else:
                errors[idx] = error
            persist_partial_result(partial_dir, idx, prompt, errors[idx])

            completed_total += 1
            if completed_total % progress_interval == 0 or completed_total == total_rows:
                print(f"[{name}] processed {completed_total}/{total_rows} prompts", file=sys.stderr)
            if error is not None:
                print(f"[{name}] row {idx} failed: {error}", file=sys.stderr)
    else:
        print(f"[{name}] all prompts restored from disk, skipping generation.", file=sys.stderr)

    prompts_str = [p if p is not None else "" for p in prompts]
    write_csv(rows, prompts_str, csv_path, args.extra_columns)
    print(f"[{name}] wrote CSV to {csv_path}")

    if enriched_jsonl:
        write_enriched_jsonl(rows, prompts_str, enriched_jsonl)
        print(f"[{name}] wrote enriched JSONL to {enriched_jsonl}")

    failed_rows = [idx for idx, err in enumerate(errors) if err]
    if failed_rows:
        print(
            f"[{name}] {len(failed_rows)} prompts failed; per-row details saved under {partial_dir}",
            file=sys.stderr,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate edit prompts using Qwen2.5-VL via vLLM.")
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=Path("workspace/metadata-v12-recaptioned-long/train/metadata.jsonl"),
        help="Path to training metadata JSONL.",
    )
    parser.add_argument(
        "--val-jsonl",
        type=Path,
        default=Path("workspace/metadata-v12-recaptioned-long/test/metadata.jsonl"),
        help="Path to validation metadata JSONL (defaults to 'test').",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("workspace/metadata-v12-recaptioned-long/train.csv"),
        help="Output CSV path for training data.",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path("workspace/metadata-v12-recaptioned-long/val.csv"),
        help="Output CSV path for validation data.",
    )
    parser.add_argument(
        "--train-jsonl-out",
        type=Path,
        help="Optional enriched JSONL output for training data.",
    )
    parser.add_argument(
        "--val-jsonl-out",
        type=Path,
        help="Optional enriched JSONL output for validation data.",
    )
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
        default="Qwen/Qwen2.5-VL-72B-Instruct",
        help="Model name served by vLLM.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to steer the model.",
    )
    parser.add_argument(
        "--user-prompt",
        default=DEFAULT_USER_PROMPT,
        help="Instruction text sent alongside the images.",
    )
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens to sample per response.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--max-concurrency", type=int, default=32, help="Concurrent requests to the endpoint.")
    parser.add_argument("--max-edge", type=int, default=1024, help="Resize longest image edge before upload.")
    parser.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality for encoded images.")
    parser.add_argument("--retries", type=int, default=3, help="Retry attempts per prompt.")
    parser.add_argument("--retry-delay", type=float, default=2.0, help="Base delay between retries (seconds).")
    parser.add_argument("--progress-interval", type=int, default=10, help="Progress logging interval.")
    parser.add_argument(
        "--extra-columns",
        nargs="*",
        default=[],
        help="Optional metadata keys to copy into the CSV output.",
    )
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.max_concurrency)

    await process_split(
        name="train",
        jsonl_path=args.train_jsonl,
        csv_path=args.train_csv,
        enriched_jsonl=args.train_jsonl_out,
        client=client,
        semaphore=semaphore,
        args=args,
    )
    await process_split(
        name="val",
        jsonl_path=args.val_jsonl,
        csv_path=args.val_csv,
        enriched_jsonl=args.val_jsonl_out,
        client=client,
        semaphore=semaphore,
        args=args,
    )


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)


if __name__ == "__main__":
    main()
