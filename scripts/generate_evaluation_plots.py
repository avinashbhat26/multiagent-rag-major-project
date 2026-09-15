from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_summary(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows[row["metric"]] = {
                "baseline": float(row["baseline"]),
                "proposed": float(row["proposed"]),
            }
    return rows


def load_document_metrics(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    data: dict[str, dict[str, dict[str, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            doc = row["document_id"]
            metric = row["metric"]
            data.setdefault(doc, {})[metric] = {
                "baseline": float(row["baseline"]),
                "proposed": float(row["proposed"]),
            }
    return data


def add_labels(ax: plt.Axes, bars) -> None:
    for bar in bars:
        y = bar.get_height()
        ax.annotate(
            f"{y:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, y),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def bar_pair_plot(
    title: str,
    ylabel: str,
    baseline: float,
    proposed: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bars = ax.bar(
        ["Baseline RAG", "Proposed Multi-Agent RAG"],
        [baseline, proposed],
        color=["#1f77b4", "#ff7f0e"],
        width=0.56,
    )
    add_labels(ax, bars)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, format="png")
    plt.close(fig)


def grouped_document_plot(
    title: str,
    ylabel: str,
    metric_name: str,
    document_data: dict[str, dict[str, dict[str, float]]],
    out_path: Path,
) -> None:
    docs = sorted(document_data.keys())
    baseline_vals = [document_data[d][metric_name]["baseline"] for d in docs]
    proposed_vals = [document_data[d][metric_name]["proposed"] for d in docs]

    x = list(range(len(docs)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    bars1 = ax.bar(
        [i - width / 2 for i in x], baseline_vals, width, label="Baseline", color="#1f77b4"
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x], proposed_vals, width, label="Proposed", color="#ff7f0e"
    )
    add_labels(ax, bars1)
    add_labels(ax, bars2)
    ax.set_xticks(x)
    ax.set_xticklabels(docs, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, format="png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-quality evaluation plots.")
    parser.add_argument(
        "--summary",
        default="eval/results/summary_metrics.csv",
        help="Path to summary metrics CSV.",
    )
    parser.add_argument(
        "--document-metrics",
        default="eval/results/document_wise_metrics.csv",
        help="Path to document-wise metrics CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="eval/results/plots",
        help="Directory to save PNG plots.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    document_path = Path(args.document_metrics)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary metrics file not found: {summary_path}")
    if not document_path.exists():
        raise FileNotFoundError(f"Document metrics file not found: {document_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    summary = load_summary(summary_path)
    doc_metrics = load_document_metrics(document_path)

    bar_pair_plot(
        "Context Reduction Comparison",
        "Context Reduction (%)",
        summary["avg_context_reduction_percentage"]["baseline"],
        summary["avg_context_reduction_percentage"]["proposed"],
        out_dir / "context_reduction_comparison.png",
    )
    bar_pair_plot(
        "Answer Relevance Comparison",
        "Answer Relevance Score",
        summary["avg_answer_relevance_score"]["baseline"],
        summary["avg_answer_relevance_score"]["proposed"],
        out_dir / "answer_relevance_comparison.png",
    )
    bar_pair_plot(
        "Faithfulness Comparison",
        "Faithfulness Score",
        summary["avg_faithfulness_score"]["baseline"],
        summary["avg_faithfulness_score"]["proposed"],
        out_dir / "faithfulness_comparison.png",
    )
    bar_pair_plot(
        "Latency Comparison",
        "Latency (ms)",
        summary["avg_latency_ms"]["baseline"],
        summary["avg_latency_ms"]["proposed"],
        out_dir / "latency_comparison.png",
    )
    bar_pair_plot(
        "Evidence Coverage Comparison",
        "Evidence Coverage Score",
        summary["avg_evidence_coverage_score"]["baseline"],
        summary["avg_evidence_coverage_score"]["proposed"],
        out_dir / "evidence_coverage_comparison.png",
    )
    bar_pair_plot(
        "Selected Chunks Comparison",
        "Average Selected Chunks",
        summary["avg_selected_chunks"]["baseline"],
        summary["avg_selected_chunks"]["proposed"],
        out_dir / "selected_chunks_comparison.png",
    )

    grouped_document_plot(
        "Document-wise Context Reduction",
        "Context Reduction (%)",
        "avg_context_reduction_percentage",
        doc_metrics,
        out_dir / "document_wise_context_reduction.png",
    )
    grouped_document_plot(
        "Document-wise Answer Relevance",
        "Answer Relevance Score",
        "avg_answer_relevance_score",
        doc_metrics,
        out_dir / "document_wise_relevance.png",
    )

    print(f"Plots generated in: {out_dir}")


if __name__ == "__main__":
    main()
