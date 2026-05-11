from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import statistics
import time

from app.services.rag_service import MultiAgentRAGService


@dataclass(slots=True)
class EvaluationSample:
    question: str
    expected_keywords: list[str]


@dataclass(slots=True)
class EvaluationRow:
    question: str
    mode: str
    latency_ms: float
    retrieved_context_count: int
    selected_context_count: int
    context_reduction_ratio: float
    retrieval_quality: float
    faithfulness: float
    confidence: float
    verified: bool
    regenerated: bool


class EvaluationRunner:
    """Runs baseline vs multi-agent evaluation and emits structured artifacts."""

    def __init__(self, service: MultiAgentRAGService) -> None:
        self.service = service

    def load_samples(self, dataset_path: str) -> list[EvaluationSample]:
        payload = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
        samples: list[EvaluationSample] = []
        for item in payload.get("samples", []):
            samples.append(
                EvaluationSample(
                    question=str(item["question"]),
                    expected_keywords=[str(v).lower() for v in item.get("expected_keywords", [])],
                )
            )
        return samples

    def evaluate(self, samples: list[EvaluationSample]) -> list[EvaluationRow]:
        rows: list[EvaluationRow] = []
        for sample in samples:
            for mode in ("baseline", "multi_agent"):
                start = time.perf_counter()
                response = self.service.ask(sample.question, mode=mode, top_k=5)
                latency_ms = (time.perf_counter() - start) * 1000.0
                retrieval_quality = self._retrieval_quality(
                    response.contexts, sample.expected_keywords
                )
                reduction_ratio = 0.0
                if response.retrieved_context_count > 0:
                    reduction_ratio = 1.0 - (
                        response.selected_context_count / response.retrieved_context_count
                    )
                rows.append(
                    EvaluationRow(
                        question=sample.question,
                        mode=mode,
                        latency_ms=latency_ms,
                        retrieved_context_count=response.retrieved_context_count,
                        selected_context_count=response.selected_context_count,
                        context_reduction_ratio=max(0.0, reduction_ratio),
                        retrieval_quality=retrieval_quality,
                        faithfulness=response.semantic_similarity,
                        confidence=response.confidence,
                        verified=response.verified,
                        regenerated=response.regenerated,
                    )
                )
        return rows

    def write_outputs(self, rows: list[EvaluationRow], output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        details_path = out / "evaluation_details.csv"
        with details_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "question",
                    "mode",
                    "latency_ms",
                    "retrieved_context_count",
                    "selected_context_count",
                    "context_reduction_ratio",
                    "retrieval_quality",
                    "faithfulness",
                    "confidence",
                    "verified",
                    "regenerated",
                ]
            )
            for row in rows:
                writer.writerow(
                    [
                        row.question,
                        row.mode,
                        f"{row.latency_ms:.2f}",
                        row.retrieved_context_count,
                        row.selected_context_count,
                        f"{row.context_reduction_ratio:.4f}",
                        f"{row.retrieval_quality:.4f}",
                        f"{row.faithfulness:.4f}",
                        f"{row.confidence:.4f}",
                        row.verified,
                        row.regenerated,
                    ]
                )

        summary_rows = self._summaries(rows)
        summary_path = out / "evaluation_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "mode",
                    "avg_latency_ms",
                    "avg_retrieval_quality",
                    "avg_context_reduction",
                    "avg_faithfulness",
                    "avg_confidence",
                    "verification_rate",
                    "regeneration_rate",
                ]
            )
            for row in summary_rows:
                writer.writerow(row)

        table_path = out / "evaluation_table.md"
        with table_path.open("w", encoding="utf-8") as f:
            f.write(
                "| mode | avg_latency_ms | avg_retrieval_quality | avg_context_reduction | "
                "avg_faithfulness | avg_confidence | verification_rate | regeneration_rate |\n"
            )
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in summary_rows:
                f.write(
                    f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | "
                    f"{row[5]} | {row[6]} | {row[7]} |\n"
                )

    def _summaries(self, rows: list[EvaluationRow]) -> list[list[str]]:
        grouped: dict[str, list[EvaluationRow]] = {"baseline": [], "multi_agent": []}
        for row in rows:
            grouped[row.mode].append(row)

        summary_rows: list[list[str]] = []
        for mode, values in grouped.items():
            if not values:
                continue
            summary_rows.append(
                [
                    mode,
                    f"{statistics.mean(v.latency_ms for v in values):.2f}",
                    f"{statistics.mean(v.retrieval_quality for v in values):.4f}",
                    f"{statistics.mean(v.context_reduction_ratio for v in values):.4f}",
                    f"{statistics.mean(v.faithfulness for v in values):.4f}",
                    f"{statistics.mean(v.confidence for v in values):.4f}",
                    f"{statistics.mean(1.0 if v.verified else 0.0 for v in values):.4f}",
                    f"{statistics.mean(1.0 if v.regenerated else 0.0 for v in values):.4f}",
                ]
            )
        return summary_rows

    @staticmethod
    def _retrieval_quality(contexts: list, expected_keywords: list[str]) -> float:
        if not expected_keywords or not contexts:
            return 0.0
        combined = " ".join(c.text.lower() for c in contexts)
        hits = sum(1 for keyword in expected_keywords if keyword in combined)
        return hits / len(expected_keywords)
