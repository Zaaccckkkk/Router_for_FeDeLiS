#!/usr/bin/env python3
"""Build CR-LT support-alignment training tables for router training.

This script is intentionally separate from the online FiDeLiS data generation
pipeline and from the WebQSP/CWQ preprocessing script. It converts the current
CR-LT router label traces plus the annotated CR-LT dataset into action-level
offline tables for support-faithful training.

Outputs:
1. Query-level rows (`router_query_table.jsonl`)
2. Query-action rows (`router_action_table.jsonl`)
3. Pairwise preference rows (`router_pairwise_table.jsonl`)
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROUTER_ROOT = SCRIPT_DIR.parents[1]
RAW_DATA_DIR = ROUTER_ROOT / "raw_data" / "crlt"

DEFAULT_LABELS_PATH = RAW_DATA_DIR / "CL-LT-KGQA_train_router_labels.jsonl"
DEFAULT_RAW_QA_PATH = RAW_DATA_DIR / "CR-LT-QA.json"
DEFAULT_RAW_CLAIM_PATH = RAW_DATA_DIR / "CR-LT-ClaimVerification.json"
DEFAULT_OUTPUT_DIR = ROUTER_ROOT / "preprocessed" / "crlt"

FACTS_USED_KEY = "facts used in this step"
TRIPLE_GROUP_PATTERN = re.compile(r"\(([^()]*)\)")
LEADING_INDEX_PATTERN = re.compile(r"^\s*\d+\s*[-:]\s*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build offline CR-LT router training tables using support-evidence "
            "alignment instead of answer-only supervision."
        )
    )
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--raw-qa-path", type=Path, default=DEFAULT_RAW_QA_PATH)
    parser.add_argument("--raw-claim-path", type=Path, default=DEFAULT_RAW_CLAIM_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--baseline-depth",
        type=int,
        default=1,
        help="Depth for the cheap baseline action.",
    )
    parser.add_argument(
        "--baseline-width",
        type=int,
        default=1,
        help="Width for the cheap baseline action.",
    )
    parser.add_argument(
        "--support-beta",
        type=float,
        default=2.0,
        help="Beta used for support F-beta scoring.",
    )
    parser.add_argument(
        "--label-cost-weight",
        type=float,
        default=0.05,
        help="Cost penalty weight used to choose the default support label.",
    )
    parser.add_argument(
        "--best-epsilon",
        type=float,
        default=0.0,
        help="Tolerance for treating the baseline as close enough to the best support score.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def normalize_text(value: Any) -> str:
    text = str(value)
    text = text.strip().strip('"').strip("'")
    text = LEADING_INDEX_PATTERN.sub("", text)
    text = " ".join(text.split())
    return text.casefold()


def parse_triple_group(group_text: str) -> tuple[str, str, str] | None:
    parts = [normalize_text(part) for part in group_text.split(",", 2)]
    if len(parts) != 3 or any(part == "" for part in parts):
        return None
    return tuple(parts)  # type: ignore[return-value]


def parse_triples_from_text(text: str) -> list[tuple[str, str, str]]:
    stripped = LEADING_INDEX_PATTERN.sub("", text.strip())
    triples: list[tuple[str, str, str]] = []

    groups = TRIPLE_GROUP_PATTERN.findall(stripped)
    if groups:
        for group in groups:
            triple = parse_triple_group(group)
            if triple is not None:
                triples.append(triple)
        return triples

    # Fallback for strings that are already bare comma-separated triples.
    bare = stripped.strip("()")
    triple = parse_triple_group(bare)
    return [triple] if triple is not None else []


def parse_gold_fact_field(value: Any) -> list[tuple[str, str, str]]:
    values = value if isinstance(value, list) else [value]
    triples: list[tuple[str, str, str]] = []
    for item in values:
        if isinstance(item, str):
            triples.extend(parse_triples_from_text(item))
    return triples


def extract_gold_support(raw_row: dict[str, Any]) -> list[tuple[str, str, str]]:
    support: list[tuple[str, str, str]] = []
    for step in raw_row.get("Reasoning Steps", []):
        facts_key = None
        for key in step.keys():
            if FACTS_USED_KEY in key.lower():
                facts_key = key
                break
        if facts_key is not None:
            support.extend(parse_gold_fact_field(step[facts_key]))

    if not support:
        kg_triples = raw_row.get("KG Triples", "")
        if isinstance(kg_triples, str):
            support.extend(parse_triples_from_text(kg_triples))

    deduped: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for triple in support:
        if triple not in seen:
            seen.add(triple)
            deduped.append(triple)
    return deduped


def parse_reasoning_path(path: str) -> list[tuple[str, str, str]]:
    tokens = [normalize_text(token) for token in path.split("->")]
    tokens = [token for token in tokens if token]
    triples: list[tuple[str, str, str]] = []
    for index in range(0, len(tokens) - 2, 2):
        triples.append((tokens[index], tokens[index + 1], tokens[index + 2]))
    return triples


def parse_retrieved_support(reasoning_paths: list[str]) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    for path in reasoning_paths:
        triples.extend(parse_reasoning_path(path))
    deduped: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for triple in triples:
        if triple not in seen:
            seen.add(triple)
            deduped.append(triple)
    return deduped


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


def support_metrics(
    gold_support: list[tuple[str, str, str]],
    retrieved_support: list[tuple[str, str, str]],
    beta: float,
) -> dict[str, float | int]:
    gold_set = set(gold_support)
    retrieved_set = set(retrieved_support)
    matched = len(gold_set & retrieved_set)
    gold_count = len(gold_set)
    retrieved_count = len(retrieved_set)

    precision = matched / retrieved_count if retrieved_count else 0.0
    recall = matched / gold_count if gold_count else 0.0
    if precision == 0.0 and recall == 0.0:
        f_beta = 0.0
    else:
        beta_sq = beta * beta
        f_beta = (1.0 + beta_sq) * precision * recall / ((beta_sq * precision) + recall)

    return {
        "gold_support_count": gold_count,
        "retrieved_support_count": retrieved_count,
        "matched_support_count": matched,
        "support_precision": precision,
        "support_recall": recall,
        "support_fbeta": f_beta,
    }


def support_reward(f_beta: float, cost: int, max_cost: int, cost_weight: float) -> float:
    normalized_cost = cost / max(1, max_cost)
    return f_beta - (cost_weight * normalized_cost)


def best_attempt_by_support(
    attempts: list[dict[str, Any]],
    support_by_attempt: dict[int, dict[str, float | int]],
    max_cost: int,
    beta: float,
    cost_weight: float,
) -> dict[str, Any]:
    del beta  # already reflected in support_by_attempt

    def sort_key(item: tuple[int, dict[str, Any]]) -> tuple[float, float, float, float, float, int, float, int, int]:
        index, attempt = item
        metrics = support_by_attempt[index]
        cost = action_cost(attempt)
        reward = support_reward(float(metrics["support_fbeta"]), cost, max_cost, cost_weight)
        return (
            reward,
            float(metrics["support_fbeta"]),
            float(metrics["support_recall"]),
            float(metrics["support_precision"]),
            float(get_nested_metric(attempt, "prediction_llm", "f1") or 0.0),
            -cost,
            -runtime_sec(attempt),
            -action_key(attempt)[0],
            -action_key(attempt)[1],
        )

    return max(enumerate(attempts), key=sort_key)[1]


def find_baseline_attempt(
    attempts: list[dict[str, Any]],
    baseline_depth: int,
    baseline_width: int,
) -> dict[str, Any]:
    for attempt in attempts:
        if action_key(attempt) == (baseline_depth, baseline_width):
            return attempt
    return min(
        attempts,
        key=lambda attempt: (action_cost(attempt), action_key(attempt)[0], action_key(attempt)[1]),
    )


def action_record(
    row: dict[str, Any],
    raw_row: dict[str, Any],
    attempt: dict[str, Any],
    baseline_attempt: dict[str, Any],
    best_attempt: dict[str, Any],
    support_stats: dict[str, float | int],
    baseline_support_stats: dict[str, float | int],
    best_support_stats: dict[str, float | int],
    max_cost: int,
    support_beta: float,
    label_cost_weight: float,
    best_epsilon: float,
) -> dict[str, Any]:
    del support_beta  # persisted indirectly through support_fbeta fields
    answer_label = row.get("label") or {}
    depth, width = action_key(attempt)
    baseline_depth, baseline_width = action_key(baseline_attempt)
    best_depth, best_width = action_key(best_attempt)

    llm_f1 = get_nested_metric(attempt, "prediction_llm", "f1") or 0.0
    baseline_llm_f1 = get_nested_metric(baseline_attempt, "prediction_llm", "f1") or 0.0
    best_llm_f1 = get_nested_metric(best_attempt, "prediction_llm", "f1") or 0.0

    support_fbeta = float(support_stats["support_fbeta"])
    baseline_support_fbeta = float(baseline_support_stats["support_fbeta"])
    best_support_fbeta = float(best_support_stats["support_fbeta"])

    return {
        "query_id": row["id"],
        "benchmark": row["benchmark"],
        "task_type": raw_row["task_type"],
        "split": row.get("split"),
        "status": row.get("status"),
        "query": row["question"],
        "q_entities": row.get("q_entities", []),
        "q_entity_count": len(row.get("q_entities", [])),
        "ground_truth": row.get("ground_truth", []),
        "hop": row.get("hop"),
        "inference_rule": raw_row.get("Inference Rule", ""),
        "reasoning_strategy": raw_row.get("Reasoning Strategy", []),
        "gold_support_triples": raw_row["gold_support_triples"],
        "answer_source_label_depth": answer_label.get("max_length"),
        "answer_source_label_width": answer_label.get("top_k"),
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
        "prediction_llm": attempt.get("prediction_llm", ""),
        "reasoning_path": attempt.get("reasoning_path", []),
        "retrieved_support_triples": support_stats["retrieved_support_triples"],
        "gold_support_count": int(support_stats["gold_support_count"]),
        "retrieved_support_count": int(support_stats["retrieved_support_count"]),
        "matched_support_count": int(support_stats["matched_support_count"]),
        "support_precision": float(support_stats["support_precision"]),
        "support_recall": float(support_stats["support_recall"]),
        "support_fbeta": support_fbeta,
        "support_reward": support_reward(
            support_fbeta,
            action_cost(attempt),
            max_cost=max_cost,
            cost_weight=label_cost_weight,
        ),
        "baseline_depth": baseline_depth,
        "baseline_width": baseline_width,
        "baseline_llm_f1": baseline_llm_f1,
        "baseline_support_fbeta": baseline_support_fbeta,
        "best_depth": best_depth,
        "best_width": best_width,
        "best_llm_f1": best_llm_f1,
        "best_support_fbeta": best_support_fbeta,
        "support_fbeta_gap_to_best": best_support_fbeta - support_fbeta,
        "support_fbeta_gain_over_baseline": support_fbeta - baseline_support_fbeta,
        "llm_f1_gap_to_best": best_llm_f1 - llm_f1,
        "llm_f1_gain_over_baseline": llm_f1 - baseline_llm_f1,
        "is_best_by_support_cost": (depth, width) == (best_depth, best_width),
        "baseline_is_within_best_epsilon": (best_support_fbeta - baseline_support_fbeta) <= best_epsilon,
        "action_is_within_best_epsilon": (best_support_fbeta - support_fbeta) <= best_epsilon,
    }


def query_record(
    row: dict[str, Any],
    raw_row: dict[str, Any],
    attempts: list[dict[str, Any]],
    baseline_attempt: dict[str, Any],
    best_attempt: dict[str, Any],
    baseline_support_stats: dict[str, float | int],
    best_support_stats: dict[str, float | int],
    best_epsilon: float,
) -> dict[str, Any]:
    answer_label = row.get("label") or {}
    baseline_depth, baseline_width = action_key(baseline_attempt)
    best_depth, best_width = action_key(best_attempt)
    baseline_support_fbeta = float(baseline_support_stats["support_fbeta"])
    best_support_fbeta = float(best_support_stats["support_fbeta"])
    has_positive_support = int(best_support_stats["matched_support_count"]) > 0

    return {
        "query_id": row["id"],
        "benchmark": row["benchmark"],
        "task_type": raw_row["task_type"],
        "split": row.get("split"),
        "status": row.get("status"),
        "query": row["question"],
        "q_entities": row.get("q_entities", []),
        "q_entity_count": len(row.get("q_entities", [])),
        "ground_truth": row.get("ground_truth", []),
        "hop": row.get("hop"),
        "num_attempts": len(attempts),
        "inference_rule": raw_row.get("Inference Rule", ""),
        "reasoning_strategy": raw_row.get("Reasoning Strategy", []),
        "gold_support_triples": raw_row["gold_support_triples"],
        "gold_support_count": len(raw_row["gold_support_triples"]),
        "answer_source_label_depth": answer_label.get("max_length"),
        "answer_source_label_width": answer_label.get("top_k"),
        "support_label_depth": best_depth if has_positive_support else None,
        "support_label_width": best_width if has_positive_support else None,
        "baseline_depth": baseline_depth,
        "baseline_width": baseline_width,
        "baseline_action_cost": action_cost(baseline_attempt),
        "baseline_runtime_sec": runtime_sec(baseline_attempt),
        "baseline_llm_f1": get_nested_metric(baseline_attempt, "prediction_llm", "f1") or 0.0,
        "baseline_support_precision": float(baseline_support_stats["support_precision"]),
        "baseline_support_recall": float(baseline_support_stats["support_recall"]),
        "baseline_support_fbeta": baseline_support_fbeta,
        "best_depth": best_depth,
        "best_width": best_width,
        "best_action_cost": action_cost(best_attempt),
        "best_runtime_sec": runtime_sec(best_attempt),
        "best_llm_f1": get_nested_metric(best_attempt, "prediction_llm", "f1") or 0.0,
        "best_support_precision": float(best_support_stats["support_precision"]),
        "best_support_recall": float(best_support_stats["support_recall"]),
        "best_support_fbeta": best_support_fbeta,
        "best_support_minus_baseline": best_support_fbeta - baseline_support_fbeta,
        "baseline_is_best_by_support_cost": (baseline_depth, baseline_width) == (best_depth, best_width),
        "baseline_is_within_best_epsilon": (best_support_fbeta - baseline_support_fbeta) <= best_epsilon,
        "gate_target_need_better_than_baseline": (best_support_fbeta - baseline_support_fbeta) > best_epsilon,
        "has_positive_support_label": has_positive_support,
    }


def pairwise_preference(
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[str, float] | None:
    left_support = float(left["support_fbeta"])
    right_support = float(right["support_fbeta"])
    if left_support > right_support:
        return "left", left_support - right_support
    if right_support > left_support:
        return "right", right_support - left_support

    left_recall = float(left["support_recall"])
    right_recall = float(right["support_recall"])
    if left_recall > right_recall:
        return "left", left_recall - right_recall
    if right_recall > left_recall:
        return "right", right_recall - left_recall

    left_precision = float(left["support_precision"])
    right_precision = float(right["support_precision"])
    if left_precision > right_precision:
        return "left", left_precision - right_precision
    if right_precision > left_precision:
        return "right", right_precision - left_precision

    if left_support == right_support == left_recall == right_recall == left_precision == right_precision == 0.0:
        return None

    left_cost = int(left["action_cost"])
    right_cost = int(right["action_cost"])
    if left_cost < right_cost:
        return "left", 0.0
    if right_cost < left_cost:
        return "right", 0.0

    left_runtime = float(left["runtime_sec"])
    right_runtime = float(right["runtime_sec"])
    if left_runtime < right_runtime:
        return "left", 0.0
    if right_runtime < left_runtime:
        return "right", 0.0

    return None


def pairwise_record(
    query_row: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
    preferred_side: str,
    margin: float,
) -> dict[str, Any]:
    preferred_depth, preferred_width = (
        (left["action_depth"], left["action_width"])
        if preferred_side == "left"
        else (right["action_depth"], right["action_width"])
    )

    return {
        "query_id": query_row["query_id"],
        "benchmark": query_row["benchmark"],
        "task_type": query_row["task_type"],
        "query": query_row["query"],
        "hop": query_row["hop"],
        "q_entities": query_row["q_entities"],
        "q_entity_count": query_row["q_entity_count"],
        "ground_truth": query_row["ground_truth"],
        "gold_support_triples": query_row["gold_support_triples"],
        "left_depth": left["action_depth"],
        "left_width": left["action_width"],
        "left_cost": left["action_cost"],
        "left_runtime_sec": left["runtime_sec"],
        "left_support_precision": left["support_precision"],
        "left_support_recall": left["support_recall"],
        "left_support_fbeta": left["support_fbeta"],
        "left_llm_f1": left["llm_f1"],
        "right_depth": right["action_depth"],
        "right_width": right["action_width"],
        "right_cost": right["action_cost"],
        "right_runtime_sec": right["runtime_sec"],
        "right_support_precision": right["support_precision"],
        "right_support_recall": right["support_recall"],
        "right_support_fbeta": right["support_fbeta"],
        "right_llm_f1": right["llm_f1"],
        "preferred_side": preferred_side,
        "preferred_depth": preferred_depth,
        "preferred_width": preferred_width,
        "support_fbeta_margin": margin,
        "support_recall_margin": abs(float(left["support_recall"]) - float(right["support_recall"])),
        "support_precision_margin": abs(float(left["support_precision"]) - float(right["support_precision"])),
        "cost_margin_left_minus_right": int(left["action_cost"]) - int(right["action_cost"]),
    }


def load_raw_examples(raw_qa_path: Path, raw_claim_path: Path) -> dict[str, dict[str, Any]]:
    raw_rows: dict[str, dict[str, Any]] = {}
    for task_type, path in (
        ("qa", raw_qa_path),
        ("claim_verification", raw_claim_path),
    ):
        for row in read_json(path):
            enriched = dict(row)
            enriched["task_type"] = task_type
            enriched["gold_support_triples"] = extract_gold_support(row)
            raw_rows[row["id"]] = enriched
    return raw_rows


def main() -> None:
    args = parse_args()

    label_rows = read_jsonl(args.labels_path)
    raw_rows = load_raw_examples(args.raw_qa_path, args.raw_claim_path)

    max_cost = 1
    for row in label_rows:
        for attempt in row.get("attempts", []):
            if attempt.get("config"):
                max_cost = max(max_cost, action_cost(attempt))

    query_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []

    for row in label_rows:
        raw_row = raw_rows.get(row["id"])
        if raw_row is None:
            raise KeyError(f"Missing raw CR-LT annotation for router label id `{row['id']}`")

        attempts = [
            attempt
            for attempt in (row.get("attempts") or [])
            if attempt.get("config") is not None
        ]
        if not attempts:
            continue

        baseline_attempt = find_baseline_attempt(
            attempts,
            baseline_depth=args.baseline_depth,
            baseline_width=args.baseline_width,
        )

        support_by_attempt: dict[int, dict[str, float | int | list[tuple[str, str, str]]]] = {}
        gold_support = raw_row["gold_support_triples"]
        for index, attempt in enumerate(attempts):
            retrieved_support = parse_retrieved_support(attempt.get("reasoning_path", []) or [])
            stats = support_metrics(gold_support, retrieved_support, beta=args.support_beta)
            stats["retrieved_support_triples"] = retrieved_support
            support_by_attempt[index] = stats

        best_attempt = best_attempt_by_support(
            attempts=attempts,
            support_by_attempt=support_by_attempt,  # type: ignore[arg-type]
            max_cost=max_cost,
            beta=args.support_beta,
            cost_weight=args.label_cost_weight,
        )

        baseline_index = attempts.index(baseline_attempt)
        best_index = attempts.index(best_attempt)
        baseline_support_stats = support_by_attempt[baseline_index]
        best_support_stats = support_by_attempt[best_index]

        query_row = query_record(
            row=row,
            raw_row=raw_row,
            attempts=attempts,
            baseline_attempt=baseline_attempt,
            best_attempt=best_attempt,
            baseline_support_stats=baseline_support_stats,
            best_support_stats=best_support_stats,
            best_epsilon=args.best_epsilon,
        )
        query_rows.append(query_row)

        query_action_rows: list[dict[str, Any]] = []
        for index, attempt in enumerate(attempts):
            action_row = action_record(
                row=row,
                raw_row=raw_row,
                attempt=attempt,
                baseline_attempt=baseline_attempt,
                best_attempt=best_attempt,
                support_stats=support_by_attempt[index],
                baseline_support_stats=baseline_support_stats,
                best_support_stats=best_support_stats,
                max_cost=max_cost,
                support_beta=args.support_beta,
                label_cost_weight=args.label_cost_weight,
                best_epsilon=args.best_epsilon,
            )
            action_rows.append(action_row)
            query_action_rows.append(action_row)

        for left, right in combinations(query_action_rows, 2):
            preference = pairwise_preference(left, right)
            if preference is None:
                continue
            preferred_side, margin = preference
            pairwise_rows.append(
                pairwise_record(
                    query_row=query_row,
                    left=left,
                    right=right,
                    preferred_side=preferred_side,
                    margin=margin,
                )
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "router_query_table.jsonl", query_rows)
    write_jsonl(args.output_dir / "router_action_table.jsonl", action_rows)
    write_jsonl(args.output_dir / "router_pairwise_table.jsonl", pairwise_rows)

    summary = {
        "query_rows": len(query_rows),
        "action_rows": len(action_rows),
        "pairwise_rows": len(pairwise_rows),
        "labels_path": str(args.labels_path),
        "raw_qa_path": str(args.raw_qa_path),
        "raw_claim_path": str(args.raw_claim_path),
        "output_dir": str(args.output_dir),
        "support_beta": args.support_beta,
        "label_cost_weight": args.label_cost_weight,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
