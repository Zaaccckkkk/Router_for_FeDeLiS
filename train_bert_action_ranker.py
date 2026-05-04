#!/usr/bin/env python3
"""Train a BERT action scorer with pointwise regression and pairwise ranking.

This trainer is separate from the original notebook-based multiclass router
experiments. It consumes the offline preprocessed tables and keeps the original
evaluation metrics for apples-to-apples comparison on the comparable subset.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PREPROCESSED_DIR = SCRIPT_DIR / "preprocessed"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "bert_action_ranker"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a BERT-based action scorer for the router using pointwise "
            "regression and pairwise ranking losses."
        )
    )
    parser.add_argument("--query-path", type=Path, default=DEFAULT_PREPROCESSED_DIR / "router_query_table.jsonl")
    parser.add_argument("--action-path", type=Path, default=DEFAULT_PREPROCESSED_DIR / "router_action_table.jsonl")
    parser.add_argument("--pairwise-path", type=Path, default=DEFAULT_PREPROCESSED_DIR / "router_pairwise_table.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--freeze-mode", choices=("frozen", "last1", "full"), default="last1")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--pointwise-batch-size", type=int, default=32)
    parser.add_argument("--pairwise-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pointwise-loss-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-loss-weight", type=float, default=0.5)
    parser.add_argument("--reward-cost-weight", type=float, default=0.05)
    parser.add_argument("--pairwise-margin-weight", type=float, default=1.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_for_run() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_action_text(row: dict[str, Any]) -> str:
    return (
        f"dataset {row['benchmark']} "
        f"depth {row['action_depth']} "
        f"width {row['action_width']} "
        f"cost {row['action_cost']}"
    )


def build_numeric_features(depth: int, width: int, cost: int) -> list[float]:
    return [
        depth / 4.0,
        width / 5.0,
        cost / 20.0,
    ]


class PointwiseActionDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int, max_cost: int, reward_cost_weight: float):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_cost = max(1, max_cost)
        self.reward_cost_weight = reward_cost_weight

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        encoded = self.tokenizer(
            row["query"],
            build_action_text(row),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        reward = float(row["llm_f1"]) - self.reward_cost_weight * (float(row["action_cost"]) / self.max_cost)
        features = build_numeric_features(row["action_depth"], row["action_width"], row["action_cost"])
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "numeric_features": torch.tensor(features, dtype=torch.float32),
            "reward": torch.tensor(reward, dtype=torch.float32),
        }


class PairwiseActionDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        left_text = (
            f"dataset {row['benchmark']} depth {row['left_depth']} "
            f"width {row['left_width']} cost {row['left_cost']}"
        )
        right_text = (
            f"dataset {row['benchmark']} depth {row['right_depth']} "
            f"width {row['right_width']} cost {row['right_cost']}"
        )
        left_encoded = self.tokenizer(
            row["query"],
            left_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        right_encoded = self.tokenizer(
            row["query"],
            right_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        sign = 1.0 if row["preferred_side"] == "left" else -1.0
        weight = 1.0 + float(row.get("f1_margin", 0.0))
        return {
            "left_input_ids": left_encoded["input_ids"].squeeze(0),
            "left_attention_mask": left_encoded["attention_mask"].squeeze(0),
            "left_numeric_features": torch.tensor(
                build_numeric_features(row["left_depth"], row["left_width"], row["left_cost"]),
                dtype=torch.float32,
            ),
            "right_input_ids": right_encoded["input_ids"].squeeze(0),
            "right_attention_mask": right_encoded["attention_mask"].squeeze(0),
            "right_numeric_features": torch.tensor(
                build_numeric_features(row["right_depth"], row["right_width"], row["right_cost"]),
                dtype=torch.float32,
            ),
            "sign": torch.tensor(sign, dtype=torch.float32),
            "weight": torch.tensor(weight, dtype=torch.float32),
        }


class QueryActionScorer(nn.Module):
    def __init__(self, model_name: str, dropout: float, freeze_mode: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self._configure_trainable_layers(freeze_mode)
        hidden_size = self.encoder.config.hidden_size
        self.numeric_proj = nn.Linear(3, 32)
        self.dropout = nn.Dropout(dropout)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size + 32, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def _configure_trainable_layers(self, freeze_mode: str) -> None:
        if freeze_mode == "full":
            return
        for param in self.encoder.parameters():
            param.requires_grad = False
        if freeze_mode == "last1" and hasattr(self.encoder, "transformer"):
            for param in self.encoder.transformer.layer[-1].parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask, numeric_features):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]
        numeric_embedding = F.gelu(self.numeric_proj(numeric_features))
        fused = torch.cat([cls_embedding, numeric_embedding], dim=-1)
        score = self.scorer(self.dropout(fused)).squeeze(-1)
        return score


def make_query_groups(
    query_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, tuple[int, int] | None]]:
    query_meta = {row["query_id"]: copy.deepcopy(row) for row in query_rows}
    actions_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        actions_by_query[row["query_id"]].append(row)

    comparable_targets: dict[str, tuple[int, int] | None] = {}
    for query_id, meta in query_meta.items():
        target = None
        if meta.get("source_label_depth") is not None and meta.get("source_label_width") is not None:
            target = (int(meta["source_label_depth"]), int(meta["source_label_width"]))
        else:
            successful = [
                row for row in actions_by_query.get(query_id, [])
                if row.get("llm_acc") == 1.0
            ]
            if successful:
                successful.sort(key=lambda row: (int(row["action_cost"]), int(row["action_depth"]), int(row["action_width"])))
                target = (int(successful[0]["action_depth"]), int(successful[0]["action_width"]))
        comparable_targets[query_id] = target
        meta["actions"] = actions_by_query.get(query_id, [])

    return query_meta, sorted(query_meta), comparable_targets


def make_grouped_stratified_like_folds(
    query_ids: list[str],
    query_meta: dict[str, dict[str, Any]],
    comparable_targets: dict[str, tuple[int, int] | None],
    n_splits: int,
    seed: int,
) -> list[tuple[list[str], list[str]]]:
    rng = np.random.default_rng(seed)
    label_to_query_ids: dict[str, list[str]] = defaultdict(list)
    for query_id in query_ids:
        target = comparable_targets.get(query_id)
        if target is not None:
            label = f"target:{target[0]},{target[1]}"
        else:
            meta = query_meta[query_id]
            label = f"best:{meta['best_depth']},{meta['best_width']}"
        label_to_query_ids[label].append(query_id)

    folds: list[list[str]] = [[] for _ in range(n_splits)]
    for offset, label in enumerate(sorted(label_to_query_ids)):
        ids = np.array(label_to_query_ids[label], dtype=object)
        rng.shuffle(ids)
        for position, query_id in enumerate(ids.tolist()):
            folds[(offset + position) % n_splits].append(query_id)

    all_ids = set(query_ids)
    splits: list[tuple[list[str], list[str]]] = []
    for val_query_ids in folds:
        val_ids = sorted(val_query_ids)
        train_ids = sorted(all_ids - set(val_ids))
        splits.append((train_ids, val_ids))
    return splits


def train_pointwise_epoch(
    model: QueryActionScorer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weight: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        numeric_features = batch["numeric_features"].to(device)
        reward = batch["reward"].to(device)

        optimizer.zero_grad(set_to_none=True)
        score = model(input_ids=input_ids, attention_mask=attention_mask, numeric_features=numeric_features)
        loss = F.smooth_l1_loss(score, reward)
        weighted_loss = loss_weight * loss
        weighted_loss.backward()
        optimizer.step()

        batch_size = reward.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
    return total_loss / max(1, total_examples)


def train_pairwise_epoch(
    model: QueryActionScorer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weight: float,
    margin_weight: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        left_input_ids = batch["left_input_ids"].to(device)
        left_attention_mask = batch["left_attention_mask"].to(device)
        left_numeric_features = batch["left_numeric_features"].to(device)
        right_input_ids = batch["right_input_ids"].to(device)
        right_attention_mask = batch["right_attention_mask"].to(device)
        right_numeric_features = batch["right_numeric_features"].to(device)
        sign = batch["sign"].to(device)
        weight = batch["weight"].to(device)

        optimizer.zero_grad(set_to_none=True)
        left_score = model(
            input_ids=left_input_ids,
            attention_mask=left_attention_mask,
            numeric_features=left_numeric_features,
        )
        right_score = model(
            input_ids=right_input_ids,
            attention_mask=right_attention_mask,
            numeric_features=right_numeric_features,
        )
        pair_loss = F.softplus(-(left_score - right_score) * sign) * (1.0 + margin_weight * (weight - 1.0))
        loss = pair_loss.mean()
        weighted_loss = loss_weight * loss
        weighted_loss.backward()
        optimizer.step()

        batch_size = sign.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
    return total_loss / max(1, total_examples)


@torch.no_grad()
def evaluate_query_actions(
    model: QueryActionScorer,
    val_query_ids: list[str],
    query_meta: dict[str, dict[str, Any]],
    comparable_targets: dict[str, tuple[int, int] | None],
    tokenizer,
    max_length: int,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    predictions: list[dict[str, Any]] = []

    comparable_gold_depth: list[int] = []
    comparable_gold_width: list[int] = []
    comparable_pred_depth: list[int] = []
    comparable_pred_width: list[int] = []

    predicted_f1: list[float] = []
    best_f1: list[float] = []
    baseline_f1: list[float] = []

    for query_id in val_query_ids:
        meta = query_meta[query_id]
        actions = meta.get("actions", [])
        if not actions:
            continue

        scored_actions: list[tuple[float, dict[str, Any]]] = []
        for action in actions:
            encoded = tokenizer(
                action["query"],
                build_action_text(action),
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            numeric_features = torch.tensor(
                [build_numeric_features(action["action_depth"], action["action_width"], action["action_cost"])],
                dtype=torch.float32,
                device=device,
            )
            score = model(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
                numeric_features=numeric_features,
            ).item()
            scored_actions.append((score, action))

        scored_actions.sort(
            key=lambda item: (
                -item[0],
                int(item[1]["action_cost"]),
                int(item[1]["action_depth"]),
                int(item[1]["action_width"]),
            )
        )
        best_scored, chosen = scored_actions[0]
        del best_scored

        target = comparable_targets.get(query_id)
        comparable = target is not None
        if comparable:
            comparable_gold_depth.append(int(target[0]))
            comparable_gold_width.append(int(target[1]))
            comparable_pred_depth.append(int(chosen["action_depth"]))
            comparable_pred_width.append(int(chosen["action_width"]))

        predicted_f1.append(float(chosen["llm_f1"]))
        best_f1.append(float(meta["best_llm_f1"]))
        baseline_f1.append(float(meta["baseline_llm_f1"]))

        predictions.append(
            {
                "query_id": query_id,
                "benchmark": meta["benchmark"],
                "status": meta["status"],
                "query": meta["query"],
                "hop": meta["hop"],
                "comparable_target_exists": comparable,
                "gold_depth": target[0] if comparable else "",
                "gold_width": target[1] if comparable else "",
                "pred_depth": int(chosen["action_depth"]),
                "pred_width": int(chosen["action_width"]),
                "pred_cost": int(chosen["action_cost"]),
                "pred_score": float(scored_actions[0][0]),
                "pred_llm_f1": float(chosen["llm_f1"]),
                "best_depth": int(meta["best_depth"]),
                "best_width": int(meta["best_width"]),
                "best_llm_f1": float(meta["best_llm_f1"]),
                "baseline_llm_f1": float(meta["baseline_llm_f1"]),
                "best_minus_pred_f1": float(meta["best_llm_f1"]) - float(chosen["llm_f1"]),
                "pred_minus_baseline_f1": float(chosen["llm_f1"]) - float(meta["baseline_llm_f1"]),
            }
        )

    metrics = compute_original_metrics(
        gold_depth=comparable_gold_depth,
        gold_width=comparable_gold_width,
        pred_depth=comparable_pred_depth,
        pred_width=comparable_pred_width,
    )
    metrics["comparable_query_count"] = float(len(comparable_gold_depth))
    metrics["mean_pred_llm_f1"] = float(np.mean(predicted_f1)) if predicted_f1 else 0.0
    metrics["mean_best_llm_f1"] = float(np.mean(best_f1)) if best_f1 else 0.0
    metrics["mean_baseline_llm_f1"] = float(np.mean(baseline_f1)) if baseline_f1 else 0.0
    metrics["mean_best_minus_pred_f1"] = float(np.mean([b - p for b, p in zip(best_f1, predicted_f1)])) if predicted_f1 else 0.0
    metrics["mean_pred_minus_baseline_f1"] = float(np.mean([p - b for p, b in zip(predicted_f1, baseline_f1)])) if predicted_f1 else 0.0
    return metrics, predictions


def compute_original_metrics(
    gold_depth: list[int],
    gold_width: list[int],
    pred_depth: list[int],
    pred_width: list[int],
) -> dict[str, float]:
    if not gold_depth:
        return {
            "exact_accuracy": 0.0,
            "depth_accuracy": 0.0,
            "width_accuracy": 0.0,
            "depth_macro_f1": 0.0,
            "width_macro_f1": 0.0,
            "strategy_macro_f1": 0.0,
            "depth_abs_error": 0.0,
            "width_abs_error": 0.0,
            "cost_sensitive_error": 0.0,
        }

    exact = [
        gd == pd and gw == pw
        for gd, gw, pd, pw in zip(gold_depth, gold_width, pred_depth, pred_width)
    ]
    gold_strategy = [f"({d},{w})" for d, w in zip(gold_depth, gold_width)]
    pred_strategy = [f"({d},{w})" for d, w in zip(pred_depth, pred_width)]
    return {
        "exact_accuracy": float(np.mean(exact)),
        "depth_accuracy": accuracy_score(gold_depth, pred_depth),
        "width_accuracy": accuracy_score(gold_width, pred_width),
        "depth_macro_f1": f1_score(gold_depth, pred_depth, average="macro", zero_division=0),
        "width_macro_f1": f1_score(gold_width, pred_width, average="macro", zero_division=0),
        "strategy_macro_f1": f1_score(gold_strategy, pred_strategy, average="macro", zero_division=0),
        "depth_abs_error": float(np.mean([abs(a - b) for a, b in zip(gold_depth, pred_depth)])),
        "width_abs_error": float(np.mean([abs(a - b) for a, b in zip(gold_width, pred_width)])),
        "cost_sensitive_error": float(np.mean([
            abs(gd - pd) + 0.5 * abs(gw - pw)
            for gd, gw, pd, pw in zip(gold_depth, gold_width, pred_depth, pred_width)
        ])),
    }


def summarize_folds(rows: list[dict[str, Any]], metric_names: list[str]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for metric in metric_names:
        values = [float(row[metric]) for row in rows]
        summary.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = device_for_run()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    query_rows = read_jsonl(args.query_path)
    action_rows = read_jsonl(args.action_path)
    pairwise_rows = read_jsonl(args.pairwise_path)

    query_meta, query_ids, comparable_targets = make_query_groups(query_rows, action_rows)
    max_cost = max(int(row["action_cost"]) for row in action_rows)

    splits = make_grouped_stratified_like_folds(
        query_ids=query_ids,
        query_meta=query_meta,
        comparable_targets=comparable_targets,
        n_splits=args.n_splits,
        seed=args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    fold_results: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []

    for fold_id, (train_query_ids, val_query_ids) in enumerate(splits, start=1):
        print(f"\n===== Fold {fold_id}/{args.n_splits} =====")
        set_seed(args.seed + fold_id)

        train_query_set = set(train_query_ids)
        val_query_set = set(val_query_ids)
        train_action_rows = [row for row in action_rows if row["query_id"] in train_query_set]
        train_pairwise_rows = [row for row in pairwise_rows if row["query_id"] in train_query_set]
        val_action_rows = [row for row in action_rows if row["query_id"] in val_query_set]
        val_pairwise_rows = [row for row in pairwise_rows if row["query_id"] in val_query_set]

        print(
            f"Train queries: {len(train_query_ids)} | Val queries: {len(val_query_ids)} | "
            f"Train actions: {len(train_action_rows)} | Train pairs: {len(train_pairwise_rows)}"
        )

        train_pointwise_dataset = PointwiseActionDataset(
            rows=train_action_rows,
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_cost=max_cost,
            reward_cost_weight=args.reward_cost_weight,
        )
        train_pairwise_dataset = PairwiseActionDataset(
            rows=train_pairwise_rows,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
        val_pointwise_dataset = PointwiseActionDataset(
            rows=val_action_rows,
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_cost=max_cost,
            reward_cost_weight=args.reward_cost_weight,
        )
        val_pairwise_dataset = PairwiseActionDataset(
            rows=val_pairwise_rows,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )

        train_point_loader = DataLoader(train_pointwise_dataset, batch_size=args.pointwise_batch_size, shuffle=True)
        train_pair_loader = DataLoader(train_pairwise_dataset, batch_size=args.pairwise_batch_size, shuffle=True)
        val_point_loader = DataLoader(val_pointwise_dataset, batch_size=args.pointwise_batch_size, shuffle=False)
        val_pair_loader = DataLoader(val_pairwise_dataset, batch_size=args.pairwise_batch_size, shuffle=False)

        model = QueryActionScorer(
            model_name=args.model_name,
            dropout=args.dropout,
            freeze_mode=args.freeze_mode,
        ).to(device)

        encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
        head_params = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
        param_groups = []
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": args.encoder_learning_rate})
        if head_params:
            param_groups.append({"params": head_params, "lr": args.head_learning_rate})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

        best_state = None
        best_epoch = 0
        best_score = -math.inf
        best_loss = math.inf
        patience_used = 0

        for epoch in range(1, args.epochs + 1):
            point_train_loss = train_pointwise_epoch(
                model=model,
                loader=train_point_loader,
                optimizer=optimizer,
                device=device,
                loss_weight=args.pointwise_loss_weight,
            )
            pair_train_loss = train_pairwise_epoch(
                model=model,
                loader=train_pair_loader,
                optimizer=optimizer,
                device=device,
                loss_weight=args.pairwise_loss_weight,
                margin_weight=args.pairwise_margin_weight,
            )

            val_metrics, fold_predictions = evaluate_query_actions(
                model=model,
                val_query_ids=val_query_ids,
                query_meta=query_meta,
                comparable_targets=comparable_targets,
                tokenizer=tokenizer,
                max_length=args.max_length,
                device=device,
            )
            val_metrics["loss"] = point_train_loss * args.pointwise_loss_weight + pair_train_loss * args.pairwise_loss_weight

            score = float(val_metrics["exact_accuracy"])
            loss = float(val_metrics["loss"])
            improved = score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and loss < best_loss)
            if improved:
                best_score = score
                best_loss = loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                best_predictions = fold_predictions
                best_metrics = copy.deepcopy(val_metrics)
                patience_used = 0
            else:
                patience_used += 1

            print(
                f"Epoch {epoch:02d} | point_loss={point_train_loss:.4f} | "
                f"pair_loss={pair_train_loss:.4f} | exact_acc={val_metrics['exact_accuracy']:.4f} | "
                f"strategy_macro_f1={val_metrics['strategy_macro_f1']:.4f}"
            )
            if patience_used >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        assert best_state is not None
        model.load_state_dict(best_state)
        print(f"Best epoch: {best_epoch} | best exact_accuracy={best_score:.4f}")

        fold_result = {"fold": fold_id, "best_epoch": best_epoch, **best_metrics}
        fold_results.append(fold_result)
        for record in best_predictions:
            enriched = dict(record)
            enriched["fold"] = fold_id
            all_predictions.append(enriched)

    metric_names = [
        "exact_accuracy",
        "depth_accuracy",
        "width_accuracy",
        "depth_macro_f1",
        "width_macro_f1",
        "strategy_macro_f1",
        "depth_abs_error",
        "width_abs_error",
        "cost_sensitive_error",
        "loss",
        "mean_pred_llm_f1",
        "mean_best_llm_f1",
        "mean_baseline_llm_f1",
        "mean_best_minus_pred_f1",
        "mean_pred_minus_baseline_f1",
    ]
    summary_rows = summarize_folds(fold_results, metric_names)

    write_csv(args.output_dir / "bert_action_ranker_cv_results.csv", fold_results)
    write_csv(args.output_dir / "bert_action_ranker_cv_summary.csv", summary_rows)
    write_csv(args.output_dir / "bert_action_ranker_cv_predictions.csv", all_predictions)

    print("\nSaved fold metrics to bert_action_ranker_cv_results.csv")
    print("Saved summary metrics to bert_action_ranker_cv_summary.csv")
    print("Saved validation predictions to bert_action_ranker_cv_predictions.csv")
    print("\nSummary:")
    for row in summary_rows:
        print(f"{row['metric']}: mean={row['mean']:.6f} std={row['std']:.6f}")


if __name__ == "__main__":
    main()
