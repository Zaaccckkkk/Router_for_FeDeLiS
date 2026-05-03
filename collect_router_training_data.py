#!/usr/bin/env python3
"""Collect correct router examples and clean query-strategy training data.

The default inputs are the two JSONL files in this repository:

  - RoG-webqsp_train_router_labels.jsonl
  - RoG-cwq_new.jsonl

Outputs:

  - correct_router_lines.jsonl: original JSONL rows copied unchanged if they
    are resolved or if an unresolved row has a prediction_llm acc of 1.0.
  - clean_router_training_data.jsonl: one {"query": ..., "strategy": ...}
    JSON object per training example.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUTS = (
    Path("RoG-webqsp_train_router_labels.jsonl"),
    Path("RoG-cwq_new.jsonl"),
)
DEFAULT_CORRECT_LINES = Path("correct_router_lines.jsonl")
DEFAULT_CLEAN_DATA = Path("clean_router_training_data.jsonl")
DEFAULT_EXCLUDED_LINES = {
    "RoG-cwq_new.jsonl": {50, 61, 93},
}


def prediction_llm_acc(attempt: dict[str, Any]) -> float | None:
    """Return attempts[*].metrics.prediction_llm.acc when it exists."""
    acc = (
        attempt.get("metrics", {})
        .get("prediction_llm", {})
        .get("acc")
    )
    return acc if isinstance(acc, (int, float)) else None


def successful_attempts(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return attempts whose prediction_llm accuracy is exactly 1.0."""
    return [
        attempt
        for attempt in record.get("attempts", [])
        if prediction_llm_acc(attempt) == 1.0
    ]


def strategy_from_mapping(mapping: dict[str, Any]) -> tuple[int, int]:
    """Extract a (max_length, top_k) strategy tuple."""
    return int(mapping["max_length"]), int(mapping["top_k"])


def attempt_sort_key(attempt: dict[str, Any]) -> tuple[int, int, int]:
    """Prefer the cheapest successful strategy, then shorter/smaller configs."""
    config = attempt.get("config", {})
    max_length = int(config.get("max_length", 10**9))
    top_k = int(config.get("top_k", 10**9))
    cost = int(config.get("cost", max_length * top_k))
    return cost, max_length, top_k


def choose_unresolved_attempt(
    attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Choose one successful unresolved attempt for one-label training data."""
    if not attempts:
        return []
    return [min(attempts, key=attempt_sort_key)]


def clean_examples(
    record: dict[str, Any],
    source_file: Path,
    unresolved_policy: str,
) -> list[dict[str, Any]]:
    """Build clean query-strategy examples from a selected input record."""
    status = record.get("status")
    question = record.get("question")
    if not isinstance(question, str) or not question:
        return []

    if status == "resolved":
        label = record.get("label")
        if not isinstance(label, dict):
            return []
        max_length, top_k = strategy_from_mapping(label)
        return [
            {
                "query": question,
                "strategy": f"({max_length},{top_k})",
            }
        ]

    if status == "unresolved":
        attempts = successful_attempts(record)
        if unresolved_policy == "cheapest":
            attempts = choose_unresolved_attempt(attempts)

        examples = []
        for attempt in attempts:
            config = attempt.get("config", {})
            max_length, top_k = strategy_from_mapping(config)
            examples.append(
                {
                    "query": question,
                    "strategy": f"({max_length},{top_k})",
                }
            )
        return examples

    return []


def is_correct_line(record: dict[str, Any]) -> bool:
    """Decide whether a full JSONL row should be copied to correct lines."""
    status = record.get("status")
    return status == "resolved" or (
        status == "unresolved" and bool(successful_attempts(record))
    )


def collect(
    input_paths: Iterable[Path],
    correct_lines_path: Path,
    clean_data_path: Path,
    unresolved_policy: str,
    excluded_lines: dict[str, set[int]],
) -> dict[str, int]:
    stats = {
        "input_lines": 0,
        "excluded_lines": 0,
        "correct_lines": 0,
        "clean_examples": 0,
        "resolved_correct": 0,
        "unresolved_correct": 0,
    }

    with correct_lines_path.open("w", encoding="utf-8") as correct_out, (
        clean_data_path.open("w", encoding="utf-8")
    ) as clean_out:
        for input_path in input_paths:
            with input_path.open("r", encoding="utf-8") as input_file:
                for line_number, raw_line in enumerate(input_file, start=1):
                    if not raw_line.strip():
                        continue
                    stats["input_lines"] += 1
                    if line_number in excluded_lines.get(input_path.name, set()):
                        stats["excluded_lines"] += 1
                        continue

                    try:
                        record = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in {input_path}:{line_number}: {exc}"
                        ) from exc

                    if not is_correct_line(record):
                        continue

                    correct_out.write(raw_line)
                    if not raw_line.endswith("\n"):
                        correct_out.write("\n")
                    stats["correct_lines"] += 1

                    if record.get("status") == "resolved":
                        stats["resolved_correct"] += 1
                    elif record.get("status") == "unresolved":
                        stats["unresolved_correct"] += 1

                    for example in clean_examples(
                        record, input_path, unresolved_policy
                    ):
                        clean_out.write(
                            json.dumps(example, ensure_ascii=False) + "\n"
                        )
                        stats["clean_examples"] += 1

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a raw correct-lines JSONL file and a clean query-strategy "
            "JSONL file from router-label JSONL data."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=list(DEFAULT_INPUTS),
        help="Input JSONL files. Defaults to the WebQSP and CWQ files.",
    )
    parser.add_argument(
        "--correct-lines-output",
        type=Path,
        default=DEFAULT_CORRECT_LINES,
        help=f"Raw selected-lines output. Default: {DEFAULT_CORRECT_LINES}",
    )
    parser.add_argument(
        "--clean-data-output",
        type=Path,
        default=DEFAULT_CLEAN_DATA,
        help=f"Clean query-strategy output. Default: {DEFAULT_CLEAN_DATA}",
    )
    parser.add_argument(
        "--unresolved-policy",
        choices=("cheapest", "all"),
        default="cheapest",
        help=(
            "For unresolved rows with multiple successful attempts, write only "
            "the cheapest successful strategy or write all successful strategies."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = collect(
        input_paths=args.inputs,
        correct_lines_path=args.correct_lines_output,
        clean_data_path=args.clean_data_output,
        unresolved_policy=args.unresolved_policy,
        excluded_lines=DEFAULT_EXCLUDED_LINES,
    )

    print(f"Wrote correct lines to {args.correct_lines_output}")
    print(f"Wrote clean data to {args.clean_data_output}")
    for name, value in stats.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
