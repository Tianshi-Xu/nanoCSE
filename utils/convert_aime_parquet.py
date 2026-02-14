#!/usr/bin/env python3
"""Convert AIME parquet datasets to repo-compatible JSON instances.

This script reads a parquet file and writes per-instance JSON files that are
compatible with perfagent.tasks.aime.AIMEInstance, plus a JSONL manifest and
sample instance(s).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def _first_non_empty(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        # Check raw key
        if key in row:
            value = row.get(key)
            if value is not None and str(value).strip() != "":
                return value
        # Check lowercase key if raw key not found
        for k in row:
            if k.lower() == key.lower():
                value = row.get(k)
                if value is not None and str(value).strip() != "":
                    return value
    return None


def _normalize_id(raw_id: Any, fallback_index: int) -> str:
    if raw_id is None:
        return f"aime_{fallback_index:04d}"
    raw = str(raw_id).strip()
    raw = re.sub(r"\s+", "_", raw)
    # Sanitize filename characters
    raw = re.sub(r"[^\w\-\.]", "_", raw)
    return raw if raw else f"aime_{fallback_index:04d}"


def _build_instance(row: dict[str, Any], idx: int) -> dict[str, Any]:
    # Try to find ID from various common fields
    instance_id = _normalize_id(
        _first_non_empty(
            row,
            ["id", "ID", "instance_id", "problem_id", "question_id", "uid", "qid"],
        ),
        idx,
    )

    problem = _first_non_empty(
        row,
        [
            "problem",
            "Problem",
            "question",
            "Question",
            "prompt",
            "problem_statement",
            "problem_text",
            "question_content",
            "statement",
        ],
    )
    answer = _first_non_empty(
        row,
        [
            "answer",
            "Answer",
            "final_answer",
            "ground_truth",
            "solution",
            "target",
        ],
    )

    instance = {
        "id": instance_id,
        "problem": str(problem or ""),
        "answer": None if answer is None else str(answer),
        "metadata": {},
    }

    # Exclude fields we already identified as core fields
    # Also exclude 'url' from metadata as requested
    core_values = {id(problem), id(answer), id(row.get("id"))}
    metadata = {}
    for k, v in row.items():
        if id(v) not in core_values and k.lower() != "url": 
             metadata[k] = v
             
    instance["metadata"] = metadata
    return instance


def _write_instance(instance: dict[str, Any], output_dir: Path, used_ids: dict[str, int]) -> Path:
    instance_dir = output_dir / "instances"
    instance_dir.mkdir(parents=True, exist_ok=True)

    base_id = instance.get("id", "unknown")
    if base_id in used_ids:
        used_ids[base_id] += 1
        file_id = f"{base_id}_{used_ids[base_id]}"
    else:
        used_ids[base_id] = 0
        file_id = base_id

    file_path = instance_dir / f"{file_id}.json"
    instance["id"] = file_id
    file_path.write_text(json.dumps(instance, ensure_ascii=False, indent=2), encoding="utf-8")
    return file_path


def convert(parquet_path: Path, output_dir: Path, sample_size: int = 3) -> None:
    df = pd.read_parquet(parquet_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    sample_path = output_dir / "sample_instances.json"

    used_ids: dict[str, int] = {}
    samples: list[dict[str, Any]] = []

    with manifest_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(df.to_dict(orient="records")):
            instance = _build_instance(row, idx)
            file_path = _write_instance(instance, output_dir, used_ids)
            manifest_entry = {
                "id": instance.get("id"),
                "file": str(file_path.relative_to(output_dir)),
                "problem": instance.get("problem", ""),
                "answer": instance.get("answer"),
            }
            f.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")

            if len(samples) < sample_size:
                samples.append(instance)

    sample_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert AIME parquet data to JSON instances")
    parser.add_argument("--input", required=True, type=Path, help="Path to the input parquet file")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for converted instances")
    parser.add_argument("--sample-size", type=int, default=3, help="How many sample instances to keep")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    convert(args.input, args.output, args.sample_size)


if __name__ == "__main__":
    main()
