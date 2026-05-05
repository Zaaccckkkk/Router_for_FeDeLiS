#!/usr/bin/env python3
from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED = ROOT / "preprocessed"
BERT = ROOT / "BERT"
OUTDIR = ROOT / "analysis" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_tables_from_notebook(nb_path: Path) -> list[pd.DataFrame]:
    nb = json.loads(nb_path.read_text())
    tables: list[pd.DataFrame] = []
    for cell in nb["cells"]:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            html = "".join(data.get("text/html", [])) if "text/html" in data else None
            if html and "<table" in html:
                try:
                    dfs = pd.read_html(StringIO(html))
                except Exception:
                    continue
                tables.extend(dfs)
    return tables


def pick_result_table(tables: list[pd.DataFrame], required_cols: list[str]) -> pd.DataFrame | None:
    for df in tables:
        cols = set(df.columns)
        if all(col in cols for col in required_cols):
            return df.copy()
    return None


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def depth_width_count_heatmap(df: pd.DataFrame, depth_col: str, width_col: str) -> pd.DataFrame:
    return (
        df.groupby([depth_col, width_col]).size().unstack(fill_value=0).sort_index().sort_index(axis=1)
    )


def depth_width_mean_heatmap(df: pd.DataFrame, depth_col: str, width_col: str, score_col: str) -> pd.DataFrame:
    return (
        df.groupby([depth_col, width_col])[score_col].mean().unstack().sort_index().sort_index(axis=1)
    )


def plot_group(
    *,
    group_name: str,
    action_df: pd.DataFrame,
    query_df: pd.DataFrame,
    query_depth_col: str,
    query_width_col: str,
    score_col: str,
    score_label: str,
    results_df: pd.DataFrame | None,
) -> None:
    count_heat = depth_width_count_heatmap(action_df, "action_depth", "action_width")
    score_heat = depth_width_mean_heatmap(action_df, "action_depth", "action_width", score_col)
    query_dist = depth_width_count_heatmap(
        query_df.dropna(subset=[query_depth_col, query_width_col]),
        query_depth_col,
        query_width_col,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{group_name}: training distribution and available result distribution", fontsize=14)

    im0 = axes[0, 0].imshow(count_heat.values, aspect="auto", cmap="Blues")
    axes[0, 0].set_title("Action-row count by depth/width")
    axes[0, 0].set_xticks(range(len(count_heat.columns)), labels=[str(c) for c in count_heat.columns])
    axes[0, 0].set_yticks(range(len(count_heat.index)), labels=[str(i) for i in count_heat.index])
    axes[0, 0].set_xlabel("Width")
    axes[0, 0].set_ylabel("Depth")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(score_heat.values, aspect="auto", cmap="YlGn")
    axes[0, 1].set_title(f"Mean {score_label} by depth/width")
    axes[0, 1].set_xticks(range(len(score_heat.columns)), labels=[str(c) for c in score_heat.columns])
    axes[0, 1].set_yticks(range(len(score_heat.index)), labels=[str(i) for i in score_heat.index])
    axes[0, 1].set_xlabel("Width")
    axes[0, 1].set_ylabel("Depth")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(query_dist.values, aspect="auto", cmap="Purples")
    axes[1, 0].set_title("Best-action query distribution by depth/width")
    axes[1, 0].set_xticks(range(len(query_dist.columns)), labels=[str(c) for c in query_dist.columns])
    axes[1, 0].set_yticks(range(len(query_dist.index)), labels=[str(i) for i in query_dist.index])
    axes[1, 0].set_xlabel("Width")
    axes[1, 0].set_ylabel("Depth")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    if results_df is not None and not results_df.empty:
        show = results_df.copy()
        show = to_numeric(show, ["exact_accuracy", "depth_accuracy", "width_accuracy"])
        x = range(len(show))
        ax.plot(x, show["exact_accuracy"], marker="o", label="exact_accuracy")
        ax.plot(x, show["depth_accuracy"], marker="o", label="depth_accuracy")
        ax.plot(x, show["width_accuracy"], marker="o", label="width_accuracy")
        ax.set_xticks(list(x), labels=[str(r) for r in show["run_name"]], rotation=35, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_title("Available sweep result metrics")
        ax.set_ylabel("Accuracy")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No local result table available", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout()
    path = OUTDIR / f"{group_name.lower().replace('+', '_').replace(' ', '_')}_distribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    web_query = pd.DataFrame(read_jsonl(PREPROCESSED / "webqsp_cwq" / "router_query_table.jsonl"))
    web_action = pd.DataFrame(read_jsonl(PREPROCESSED / "webqsp_cwq" / "router_action_table.jsonl"))
    crlt_query = pd.DataFrame(read_jsonl(PREPROCESSED / "crlt" / "router_query_table.jsonl"))
    crlt_action = pd.DataFrame(read_jsonl(PREPROCESSED / "crlt" / "router_action_table.jsonl"))

    web_nb_tables = extract_tables_from_notebook(BERT / "webqsp_cwq" / "06_action_ranker_pointwise_pairwise.ipynb")
    crlt_nb_tables = extract_tables_from_notebook(BERT / "crlt" / "01_crlt_support_action_ranker.ipynb")

    web_results = pick_result_table(
        web_nb_tables,
        ["run_name", "exact_accuracy", "depth_accuracy", "width_accuracy"],
    )
    crlt_results = pick_result_table(
        crlt_nb_tables,
        ["run_name", "exact_accuracy", "depth_accuracy", "width_accuracy", "mean_pred_support_fbeta"],
    )

    if web_results is not None:
        web_results = web_results.drop(columns=[c for c in web_results.columns if str(c).startswith("Unnamed:")], errors="ignore")
    if crlt_results is not None:
        crlt_results = crlt_results.drop(columns=[c for c in crlt_results.columns if str(c).startswith("Unnamed:")], errors="ignore")

    plot_group(
        group_name="WebQSP+CWQ",
        action_df=web_action,
        query_df=web_query,
        query_depth_col="best_depth",
        query_width_col="best_width",
        score_col="llm_f1",
        score_label="LLM F1",
        results_df=web_results,
    )

    plot_group(
        group_name="CR-LT",
        action_df=crlt_action,
        query_df=crlt_query,
        query_depth_col="best_depth",
        query_width_col="best_width",
        score_col="support_fbeta",
        score_label="support F_beta",
        results_df=crlt_results,
    )

    summary_rows = []
    for name, qdf, adf, score_col in [
        ("WebQSP+CWQ", web_query, web_action, "llm_f1"),
        ("CR-LT", crlt_query, crlt_action, "support_fbeta"),
    ]:
        summary_rows.append(
            {
                "group": name,
                "queries": len(qdf),
                "actions": len(adf),
                "best_depth_mode": qdf["best_depth"].mode().iloc[0],
                "best_width_mode": qdf["best_width"].mode().iloc[0],
                "action_score_mean": adf[score_col].mean(),
            }
        )
    pd.DataFrame(summary_rows).to_csv(OUTDIR / "distribution_summary.csv", index=False)


if __name__ == "__main__":
    main()
