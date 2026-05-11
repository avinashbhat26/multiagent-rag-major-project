from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json
import statistics
import time

from app.schemas.rag import AskResponse
from app.services.rag_service import MultiAgentRAGService


@dataclass(slots=True)
class EvaluationQuestion:
    question_id: str
    question: str
    expected_answer: str
    expected_source_keywords: list[str]
    difficulty: str
    query_type: str


@dataclass(slots=True)
class EvaluationRecord:
    question_id: str
    question: str
    difficulty: str
    query_type: str
    mode: str
    answer: str
    retrieved_chunks_count: int
    selected_chunks_count: int
    context_reduction_percentage: float
    average_similarity_score: float
    max_similarity_score: float
    min_similarity_score: float
    evidence_coverage_score: float
    reranking_gain: float
    duplicate_or_redundant_chunks_removed: int
    answer_length: int
    answer_relevance_score: float
    faithfulness_score: float
    unsupported_claim_count: int
    hallucination_flag: bool
    verified: bool
    confidence_score: float
    supported_claims_count: int
    unsupported_claims_count: int
    regeneration_triggered: bool
    latency_ms: float
    token_context_length_estimate: int


class PaperEvaluationRunner:
    """Paper-oriented evaluator for baseline vs proposed multi-agent RAG."""

    def __init__(self, service: MultiAgentRAGService) -> None:
        self.service = service

    def load_questions(self, path: str) -> list[EvaluationQuestion]:
        questions: list[EvaluationQuestion] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            questions.append(
                EvaluationQuestion(
                    question_id=str(payload["question_id"]),
                    question=str(payload["question"]),
                    expected_answer=str(payload.get("expected_answer", "")),
                    expected_source_keywords=[
                        str(keyword).lower()
                        for keyword in payload.get("expected_source_keywords", [])
                    ],
                    difficulty=str(payload.get("difficulty", "medium")),
                    query_type=str(payload.get("query_type", "factual")),
                )
            )
        return questions

    def run(self, questions: list[EvaluationQuestion], mode: str) -> list[EvaluationRecord]:
        modes: list[str]
        if mode == "both":
            modes = ["baseline", "proposed"]
        else:
            modes = [mode]

        results: list[EvaluationRecord] = []
        for question in questions:
            for selected_mode in modes:
                api_mode = "baseline" if selected_mode == "baseline" else "multi_agent"
                start = time.perf_counter()
                response = self.service.ask(question.question, mode=api_mode, top_k=5)
                latency_ms = (time.perf_counter() - start) * 1000.0
                results.append(
                    self._build_record(
                        question=question,
                        mode=selected_mode,
                        response=response,
                        latency_ms=latency_ms,
                    )
                )
        return results

    def write_outputs(self, records: list[EvaluationRecord], output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        detailed_path = out / "detailed_results.csv"
        with detailed_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=list(asdict(records[0]).keys()) if records else []
            )
            if records:
                writer.writeheader()
                for record in records:
                    writer.writerow(asdict(record))

        summary = self._build_summary(records)
        summary_path = out / "summary_metrics.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["metric", "baseline", "proposed", "baseline_vs_proposed_comparison"])
            for metric_name, values in summary.items():
                writer.writerow(
                    [metric_name, values["baseline"], values["proposed"], values["comparison"]]
                )

        report_path = out / "evaluation_report.md"
        with report_path.open("w", encoding="utf-8") as file:
            file.write("# Evaluation Report\n\n")
            file.write("## Summary Comparison Table\n\n")
            file.write("| Metric | Baseline | Proposed | Comparison |\n")
            file.write("|---|---:|---:|---|\n")
            for metric_name, values in summary.items():
                file.write(
                    f"| {metric_name} | {values['baseline']} | {values['proposed']} | {values['comparison']} |\n"
                )
            file.write("\n## Notes\n")
            file.write("- Proposed corresponds to multi-agent mode.\n")
            file.write("- Faithfulness is measured via semantic similarity from verification.\n")
            file.write("- Relevance is keyword overlap + semantic similarity blend.\n")

    def _build_record(
        self,
        question: EvaluationQuestion,
        mode: str,
        response: AskResponse,
        latency_ms: float,
    ) -> EvaluationRecord:
        scores = [chunk.score for chunk in response.contexts]
        avg_score = statistics.mean(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0

        reduction_pct = self.context_reduction_percentage(
            response.retrieved_context_count, response.selected_context_count
        )
        relevance = self.answer_relevance_score(
            answer=response.answer,
            expected_keywords=question.expected_source_keywords,
            semantic_similarity=response.semantic_similarity,
        )
        unsupported_count = len(response.unsupported_claims)

        return EvaluationRecord(
            question_id=question.question_id,
            question=question.question,
            difficulty=question.difficulty,
            query_type=question.query_type,
            mode=mode,
            answer=response.answer,
            retrieved_chunks_count=response.retrieved_context_count,
            selected_chunks_count=response.selected_context_count,
            context_reduction_percentage=reduction_pct,
            average_similarity_score=round(avg_score, 4),
            max_similarity_score=round(max_score, 4),
            min_similarity_score=round(min_score, 4),
            evidence_coverage_score=round(response.evidence_coverage_score, 4),
            reranking_gain=round(response.reranking_gain, 4),
            duplicate_or_redundant_chunks_removed=response.removed_redundant_chunks,
            answer_length=len(response.answer),
            answer_relevance_score=relevance,
            faithfulness_score=round(response.semantic_similarity, 4),
            unsupported_claim_count=unsupported_count,
            hallucination_flag=unsupported_count > 0,
            verified=response.verified,
            confidence_score=round(response.confidence, 4),
            supported_claims_count=len(response.supported_claims),
            unsupported_claims_count=unsupported_count,
            regeneration_triggered=response.regenerated,
            latency_ms=round(latency_ms, 2),
            token_context_length_estimate=self.token_context_length_estimate(response),
        )

    @staticmethod
    def token_context_length_estimate(response: AskResponse) -> int:
        text = " ".join(chunk.text for chunk in response.contexts)
        # Approximation: 1 token ~= 0.75 words
        words = len(text.split())
        return int(words / 0.75)

    @staticmethod
    def context_reduction_percentage(retrieved_count: int, selected_count: int) -> float:
        if retrieved_count <= 0:
            return 0.0
        reduced = (retrieved_count - selected_count) / retrieved_count
        return round(max(0.0, reduced) * 100.0, 2)

    @staticmethod
    def answer_relevance_score(
        answer: str, expected_keywords: list[str], semantic_similarity: float
    ) -> float:
        if not answer.strip():
            return 0.0
        keyword_overlap = 0.0
        if expected_keywords:
            answer_lower = answer.lower()
            matches = sum(1 for keyword in expected_keywords if keyword in answer_lower)
            keyword_overlap = matches / len(expected_keywords)
        score = 0.6 * keyword_overlap + 0.4 * semantic_similarity
        return round(min(1.0, max(0.0, score)), 4)

    def _build_summary(self, records: list[EvaluationRecord]) -> dict[str, dict[str, str]]:
        baseline = [row for row in records if row.mode == "baseline"]
        proposed = [row for row in records if row.mode == "proposed"]

        def avg(values: list[float]) -> float:
            return statistics.mean(values) if values else 0.0

        def fmt(value: float) -> str:
            return f"{value:.4f}"

        metrics = {
            "Avg Context Reduction %": (
                avg([row.context_reduction_percentage for row in baseline]),
                avg([row.context_reduction_percentage for row in proposed]),
            ),
            "Avg Confidence Score": (
                avg([row.confidence_score for row in baseline]),
                avg([row.confidence_score for row in proposed]),
            ),
            "Verification Pass Rate": (
                avg([1.0 if row.verified else 0.0 for row in baseline]),
                avg([1.0 if row.verified else 0.0 for row in proposed]),
            ),
            "Avg Faithfulness Score": (
                avg([row.faithfulness_score for row in baseline]),
                avg([row.faithfulness_score for row in proposed]),
            ),
            "Avg Answer Relevance Score": (
                avg([row.answer_relevance_score for row in baseline]),
                avg([row.answer_relevance_score for row in proposed]),
            ),
            "Avg Latency": (
                avg([row.latency_ms for row in baseline]),
                avg([row.latency_ms for row in proposed]),
            ),
            "Avg Retrieved Chunks": (
                avg([float(row.retrieved_chunks_count) for row in baseline]),
                avg([float(row.retrieved_chunks_count) for row in proposed]),
            ),
            "Avg Selected Chunks": (
                avg([float(row.selected_chunks_count) for row in baseline]),
                avg([float(row.selected_chunks_count) for row in proposed]),
            ),
            "Unsupported Claims Count": (
                avg([float(row.unsupported_claims_count) for row in baseline]),
                avg([float(row.unsupported_claims_count) for row in proposed]),
            ),
            "Avg Evidence Coverage Score": (
                avg([row.evidence_coverage_score for row in baseline]),
                avg([row.evidence_coverage_score for row in proposed]),
            ),
            "Avg Reranking Gain": (
                avg([row.reranking_gain for row in baseline]),
                avg([row.reranking_gain for row in proposed]),
            ),
        }

        summary: dict[str, dict[str, str]] = {}
        for metric_name, (baseline_value, proposed_value) in metrics.items():
            delta = proposed_value - baseline_value
            summary[metric_name] = {
                "baseline": fmt(baseline_value),
                "proposed": fmt(proposed_value),
                "comparison": f"{delta:+.4f}",
            }
        return summary
