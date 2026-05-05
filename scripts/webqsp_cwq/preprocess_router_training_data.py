#!/usr/bin/env python3
"""Preprocess router source JSONLs into action-level training tables.

This script is intentionally separate from the online data-generation pipeline.
It reads the existing source files with full attempt traces and writes richer
offline training tables for:

1. Query-level gating (`router_query_table.jsonl`)
2. Query-action scoring (`router_action_table.jsonl`)
3. Pairwise action ranking (`router_pairwise_table.jsonl`)

The source files remain unchanged.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROUTER_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INPUTS = (
    ROUTER_ROOT / "raw_data" / "webqsp_cwq" / "RoG-webqsp_train_router_labels.jsonl",
    ROUTER_ROOT / "raw_data" / "webqsp_cwq" / "RoG-cwq_new.jsonl",
)
DEFAULT_OUTPUT_DIR = ROUTER_ROOT / "preprocessed" / "webqsp_cwq"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build offline router training tables from the current source JSONL "
            "files without modifying the existing collection pipeline."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=list(DEFAULT_INPUTS),
        help="Source JSONL files with attempt traces.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the preprocessed JSONL files will be written.",
    )
    parser.add_argument(
        "--baseline-depth",
        type=int,
        default=1,
        help="Depth used for the cheap baseline gate target.",
    )
    parser.add_argument(
        "--baseline-width",
        type=int,
        default=1,
        help="Width used for the cheap baseline gate target.",
    )
    parser.add_argument(
        "--best-epsilon",
        type=float,
        default=0.0,
        help=(
            "Tolerance for treating the baseline as effectively as good as the "
            "best observed action."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_nested_metric(attempt: dict[str, Any], branch: str, metric: str) -> float | None:
    value = attempt.get("metrics", {}).get(branch, {}).get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def action_key(attempt: dict[str, Any]) -> tuple[int, int]:
    config = attempt.get("config", {})
    return int(config["max_length"]), int(config["top_k"])


def action_cost(attempt: dict[str, Any]) -> int:
    config = attempt.get("config", {})
    return int(config.get("cost", int(config["max_length"]) * int(config["top_k"])))


def runtime_sec(attempt: dict[str, Any]) -> float:
    runtime = attempt.get("runtime_sec", 0.0)
    return float(runtime) if isinstance(runtime, (int, float)) else 0.0


def best_attempt_by_llm_f1(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    def sort_key(attempt: dict[str, Any]) -> tuple[float, int, float, int, int]:
        depth, width = action_key(attempt)
        f1 = get_nested_metric(attempt, "prediction_llm", "f1") or 0.0
        return (-f1, action_cost(attempt), runtime_sec(attempt), depth, width)

    return min(attempts, key=sort_key)


def action_record(
    row: dict[str, Any],
    attempt: dict[str, Any],
    baseline_attempt: dict[str, Any],
    best_attempt: dict[str, Any],
    best_epsilon: float,
) -> dict[str, Any]:
    source_label = row.get("label") or {}
    depth, width = action_key(attempt)
    baseline_depth, baseline_width = action_key(baseline_attempt)
    best_depth, best_width = action_key(best_attempt)

    llm_f1 = get_nested_metric(attempt, "prediction_llm", "f1") or 0.0
    baseline_llm_f1 = get_nested_metric(baseline_attempt, "prediction_llm", "f1") or 0.0
    best_llm_f1 = get_nested_metric(best_attempt, "prediction_llm", "f1") or 0.0

    return {
        "query_id": row["id"],
        "benchmark": row["benchmark"],
        "split": row.get("split"),
        "status": row.get("status"),
        "query": row["question"],
        "q_entities": row.get("q_entities", []),
        "q_entity_count": len(row.get("q_entities", [])),
        "ground_truth": row.get("ground_truth", []),
        "hop": row.get("hop"),
        "source_label_depth": source_label.get("max_length"),
        "source_label_width": source_label.get("top_k"),
        "action_depth": depth,
        "action_width": width,
        "action_cost": action_cost(attempt),
        "runtime_sec": runtime_sec(attempt),
        "is_correct_flag": bool(attempt.get("is_correct")),
        "llm_acc": get_nested_metric(attempt, "prediction_llm", "acc"),
        "llm_hit": get_nested_metric(attempt, "prediction_llm", "hit"),
        "llm_f1": llm_f1,
        "llm_precision": get_nested_metric(attempt, "prediction_llm", "precision"),
        "llm_recall": get_nested_metric(attempt, "prediction_llm", "recall"),
        "direct_acc": get_nested_metric(attempt, "prediction_direct_answer", "acc"),
        "direct_hit": get_nested_metric(attempt, "prediction_direct_answer", "hit"),
        "direct_f1": get_nested_metric(attempt, "prediction_direct_answer", "f1"),
        "direct_precision": get_nested_metric(attempt, "prediction_direct_answer", "precision"),
        "direct_recall": get_nested_metric(attempt, "prediction_direct_answer", "recall"),
        "prediction_llm": attempt.get("prediction_llm", ""),
        "prediction_direct_answer": attempt.get("prediction_direct_answer", ""),
        "reasoning_path": attempt.get("reasoning_path", []),
        "baseline_depth": baseline_depth,
        "baseline_width": baseline_width,
        "baseline_llm_f1": baseline_llm_f1,
        "best_depth": best_depth,
        "best_width": best_width,
        "best_llm_f1": best_llm_f1,
        "f1_gap_to_best": best_llm_f1 - llm_f1,
        "f1_gain_over_baseline": llm_f1 - baseline_llm_f1,
        "is_best_by_llm_f1_cost": (depth, width) == (best_depth, best_width),
        "baseline_is_within_best_epsilon": (best_llm_f1 - baseline_llm_f1) <= best_epsilon,
        "action_is_within_best_epsilon": (best_llm_f1 - llm_f1) <= best_epsilon,
    }


def query_record(
    row: dict[str, Any],
    attempts: list[dict[str, Any]],
    baseline_attempt: dict[str, Any],
    best_attempt: dict[str, Any],
    best_epsilon: float,
) -> dict[str, Any]:
    source_label = row.get("label") or {}
    baseline_depth, baseline_width = action_key(baseline_attempt)
    best_depth, best_width = action_key(best_attempt)
    baseline_llm_f1 = get_nested_metric(baseline_attempt, "prediction_llm", "f1") or 0.0
    best_llm_f1 = get_nested_metric(best_attempt, "prediction_llm", "f1") or 0.0

    return {
        "query_id": row["id"],
        "benchmark": row["benchmark"],
        "split": row.get("split"),
        "status": row.get("status"),
        "query": row["question"],
        "q_entities": row.get("q_entities", []),
        "q_entity_count": len(row.get("q_entities", [])),
        "ground_truth": row.get("ground_truth", []),
        "hop": row.get("hop"),
        "num_attempts": len(attempts),
        "source_label_depth": source_label.get("max_length"),
        "source_label_width": source_label.get("top_k"),
        "baseline_depth": baseline_depth,
        "baseline_width": baseline_width,
        "baseline_action_cost": action_cost(baseline_attempt),
        "baseline_runtime_sec": runtime_sec(baseline_attempt),
        "baseline_llm_acc": get_nested_metric(baseline_attempt, "prediction_llm", "acc"),
        "baseline_llm_hit": get_nested_metric(baseline_attempt, "prediction_llm", "hit"),
        "baseline_llm_f1": baseline_llm_f1,
        "baseline_llm_precision": get_nested_metric(baseline_attempt, "prediction_llm", "precision"),
        "baseline_llm_recall": get_nested_metric(baseline_attempt, "prediction_llm", "recall"),
        "best_depth": best_depth,
        "best_width": best_width,
        "best_action_cost": action_cost(best_attempt),
        "best_runtime_sec": runtime_sec(best_attempt),
        "best_llm_acc": get_nested_metric(best_attempt, "prediction_llm", "acc"),
        "best_llm_hit": get_nested_metric(best_attempt, "prediction_llm", "hit"),
        "best_llm_f1": best_llm_f1,
        "best_llm_precision": get_nested_metric(best_attempt, "prediction_llm", "precision"),
        "best_llm_recall": get_nested_metric(best_attempt, "prediction_llm", "recall"),
        "best_minus_baseline_f1": best_llm_f1 - baseline_llm_f1,
        "baseline_is_best_by_f1_cost": (baseline_depth, baseline_width) == (best_depth, best_width),
        "baseline_is_within_best_epsilon": (best_llm_f1 - baseline_llm_f1) <= best_epsilon,
        "gate_target_need_better_than_baseline": (best_llm_f1 - baseline_llm_f1) > best_epsilon,
        "gate_target_baseline_f1_eq_1": baseline_llm_f1 == 1.0,
        "gate_target_best_f1_eq_1": best_llm_f1 == 1.0,
    }


def pairwise_preference(
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[str, float]:
    left_f1 = get_nested_metric(left, "prediction_llm", "f1") or 0.0
    right_f1 = get_nested_metric(right, "prediction_llm", "f1") or 0.0
    if left_f1 > right_f1:
        return "left", left_f1 - right_f1
    if right_f1 > left_f1:
        return "right", right_f1 - left_f1

    left_cost = action_cost(left)
    right_cost = action_cost(right)
    if left_cost < right_cost:
        return "left", 0.0
    if right_cost < left_cost:
        return "right", 0.0

    left_runtime = runtime_sec(left)
    right_runtime = runtime_sec(right)
    if left_runtime < right_runtime:
        return "left", 0.0
    if right_runtime < left_runtime:
        return "right", 0.0

    return "tie", 0.0


def pairwise_record(row: dict[str, Any], left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
    preferred_side, f1_margin = pairwise_preference(left, right)
    if preferred_side == "tie":
        return None

    left_depth, left_width = action_key(left)
    right_depth, right_width = action_key(right)

    return {
        "query_id": row["id"],
        "benchmark": row["benchmark"],
        "split": row.get("split"),
        "status": row.get("status"),
        "query": row["question"],
        "q_entities": row.get("q_entities", []),
        "q_entity_count": len(row.get("q_entities", [])),
        "ground_truth": row.get("ground_truth", []),
        "hop": row.get("hop"),
        "left_depth": left_depth,
        "left_width": left_width,
        "left_cost": action_cost(left),
        "left_runtime_sec": runtime_sec(left),
        "left_llm_f1": get_nested_metric(left, "prediction_llm", "f1"),
        "left_llm_acc": get_nested_metric(left, "prediction_llm", "acc"),
        "right_depth": right_depth,
        "right_width": right_width,
        "right_cost": action_cost(right),
        "right_runtime_sec": runtime_sec(right),
        "right_llm_f1": get_nested_metric(right, "prediction_llm", "f1"),
        "right_llm_acc": get_nested_metric(right, "prediction_llm", "acc"),
        "preferred_side": preferred_side,
        "preferred_depth": left_depth if preferred_side == "left" else right_depth,
        "preferred_width": left_width if preferred_side == "left" else right_width,
        "f1_margin": f1_margin,
        "cost_margin_left_minus_right": action_cost(left) - action_cost(right),
    }


def preprocess(
    input_paths: list[Path],
    baseline_depth: int,
    baseline_width: int,
    best_epsilon: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    query_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []

    for path in input_paths:
        for row in read_jsonl(path):
            attempts = row.get("attempts", [])
            if not attempts:
                continue

            baseline_attempt = None
            unique_attempts: dict[tuple[int, int], dict[str, Any]] = {}
            for attempt in attempts:
                key = action_key(attempt)
                unique_attempts[key] = attempt
                if key == (baseline_depth, baseline_width):
                    baseline_attempt = attempt

            if baseline_attempt is None:
                continue

            deduped_attempts = list(unique_attempts.values())
            best_attempt = best_attempt_by_llm_f1(deduped_attempts)

            query_rows.append(
                query_record(
                    row=row,
                    attempts=deduped_attempts,
                    baseline_attempt=baseline_attempt,
                    best_attempt=best_attempt,
                    best_epsilon=best_epsilon,
                )
            )

            for attempt in deduped_attempts:
                action_rows.append(
                    action_record(
                        row=row,
                        attempt=attempt,
                        baseline_attempt=baseline_attempt,
                        best_attempt=best_attempt,
                        best_epsilon=best_epsilon,
                    )
                )

            for left, right in combinations(deduped_attempts, 2):
                record = pairwise_record(row, left, right)
                if record is not None:
                    pairwise_rows.append(record)

    return query_rows, action_rows, pairwise_rows


def main() -> None:
    args = parse_args()
    query_rows, action_rows, pairwise_rows = preprocess(
        input_paths=args.inputs,
        baseline_depth=args.baseline_depth,
        baseline_width=args.baseline_width,
        best_epsilon=args.best_epsilon,
    )

    output_dir: Path = args.output_dir
    query_path = output_dir / "router_query_table.jsonl"
    action_path = output_dir / "router_action_table.jsonl"
    pairwise_path = output_dir / "router_pairwise_table.jsonl"

    write_jsonl(query_path, query_rows)
    write_jsonl(action_path, action_rows)
    write_jsonl(pairwise_path, pairwise_rows)

    print(f"Wrote query-level table to {query_path}")
    print(f"Wrote action-level table to {action_path}")
    print(f"Wrote pairwise table to {pairwise_path}")
    print(f"query_rows: {len(query_rows)}")
    print(f"action_rows: {len(action_rows)}")
    print(f"pairwise_rows: {len(pairwise_rows)}")


if __name__ == "__main__":
    main()
