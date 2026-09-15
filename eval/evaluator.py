from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
import csv
import json
import statistics
import time

from fastapi import UploadFile

from app.schemas.rag import AskResponse
from app.services.rag_service import MultiAgentRAGService


@dataclass(slots=True)
class EvaluationDocument:
    document_id: str
    document_name: str
    domain: str
    file_path: str
    description: str


@dataclass(slots=True)
class EvaluationQuestion:
    question_id: str
    document_id: str
    question: str
    expected_answer_keywords: list[str]
    query_type: str
    difficulty: str


@dataclass(slots=True)
class EvaluationRecord:
    document_id: str
    document_name: str
    domain: str
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
    query_analysis_ms: float
    planning_ms: float
    query_embedding_ms: float
    retrieval_ms: float
    reranking_ms: float
    context_selection_ms: float
    generation_ms: float
    verification_ms: float
    evidence_recovery_ms: float
    total_ms: float
    reranking_applied: bool
    reranking_reason: str
    query_complexity: str
    retrieval_candidate_k: int


class MultiDocumentEvaluationRunner:
    def __init__(self, service: MultiAgentRAGService) -> None:
        self.service = service

    def load_documents(self, path: str) -> tuple[list[EvaluationDocument], list[str]]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        docs: list[EvaluationDocument] = []
        warnings: list[str] = []
        seen_paths: set[str] = set()

        for item in payload.get("documents", []):
            file_path = str(item["file_path"])
            normalized = str(Path(file_path).resolve()).lower()
            if normalized in seen_paths:
                warnings.append(
                    f"Duplicate document path detected and skipped: {file_path} (document_id={item.get('document_id')})"
                )
                continue
            seen_paths.add(normalized)
            docs.append(
                EvaluationDocument(
                    document_id=str(item["document_id"]),
                    document_name=str(item["document_name"]),
                    domain=str(item["domain"]),
                    file_path=file_path,
                    description=str(item.get("description", "")),
                )
            )
        return docs, warnings

    def load_questions_dir(self, questions_dir: str) -> dict[str, list[EvaluationQuestion]]:
        root = Path(questions_dir)
        questions_by_doc: dict[str, list[EvaluationQuestion]] = {}

        for file in root.glob("*_questions.jsonl"):
            for line in file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                question = EvaluationQuestion(
                    question_id=str(row["question_id"]),
                    document_id=str(row["document_id"]),
                    question=str(row["question"]),
                    expected_answer_keywords=[
                        str(v).lower() for v in row.get("expected_answer_keywords", [])
                    ],
                    query_type=str(row.get("query_type", "factual")),
                    difficulty=str(row.get("difficulty", "medium")),
                )
                questions_by_doc.setdefault(question.document_id, []).append(question)
        return questions_by_doc

    def run(
        self,
        documents: list[EvaluationDocument],
        questions_by_doc: dict[str, list[EvaluationQuestion]],
        mode: str,
    ) -> list[EvaluationRecord]:
        modes = ["baseline", "proposed"] if mode == "both" else [mode]
        records: list[EvaluationRecord] = []

        for document in documents:
            if not Path(document.file_path).exists():
                raise FileNotFoundError(f"Document not found: {document.file_path}")

            uploads = [
                UploadFile(
                    filename=Path(document.file_path).name,
                    file=BytesIO(Path(document.file_path).read_bytes()),
                )
            ]
            self.service.index_documents(uploads, reset=True)
            doc_questions = questions_by_doc.get(document.document_id, [])

            for question in doc_questions:
                for selected_mode in modes:
                    api_mode = "baseline" if selected_mode == "baseline" else "multi_agent"
                    start = time.perf_counter()
                    response = self.service.ask(question.question, mode=api_mode, top_k=5)
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    records.append(
                        self._build_record(
                            response=response,
                            question=question,
                            document=document,
                            mode=selected_mode,
                            latency_ms=latency_ms,
                        )
                    )
        return records

    def write_outputs(self, records: list[EvaluationRecord], output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self._write_detailed(records, out / "detailed_results.csv")

        summary_rows = self._compute_summary(records)
        self._write_summary(
            summary_rows,
            out / "summary_metrics.csv",
            ["metric", "baseline", "proposed", "baseline_vs_proposed_comparison"],
        )

        document_rows = self._grouped_metrics(records, group_field="document_id")
        self._write_grouped(document_rows, out / "document_wise_metrics.csv")

        domain_rows = self._grouped_metrics(records, group_field="domain")
        self._write_grouped(domain_rows, out / "domain_wise_metrics.csv")

        self._write_markdown_report(summary_rows, out / "evaluation_report.md")
        self._write_latex_table(summary_rows, out / "latex_results_table.tex")

    def _build_record(
        self,
        response: AskResponse,
        question: EvaluationQuestion,
        document: EvaluationDocument,
        mode: str,
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
            response.answer,
            question.expected_answer_keywords,
            response.semantic_similarity,
        )
        unsupported_count = len(response.unsupported_claims)
        timings = response.timings
        return EvaluationRecord(
            document_id=document.document_id,
            document_name=document.document_name,
            domain=document.domain,
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
            query_analysis_ms=timings.query_analysis_ms,
            planning_ms=timings.planning_ms,
            query_embedding_ms=timings.query_embedding_ms,
            retrieval_ms=timings.retrieval_ms,
            reranking_ms=timings.reranking_ms,
            context_selection_ms=timings.context_selection_ms,
            generation_ms=timings.generation_ms,
            verification_ms=timings.verification_ms,
            evidence_recovery_ms=timings.evidence_recovery_ms,
            total_ms=timings.total_ms,
            reranking_applied=response.reranking_applied,
            reranking_reason=response.reranking_reason,
            query_complexity=response.query_complexity,
            retrieval_candidate_k=response.retrieval_candidate_k,
        )

    @staticmethod
    def token_context_length_estimate(response: AskResponse) -> int:
        words = len(" ".join(chunk.text for chunk in response.contexts).split())
        return int(words / 0.75)

    @staticmethod
    def context_reduction_percentage(retrieved_count: int, selected_count: int) -> float:
        if retrieved_count <= 0:
            return 0.0
        return round(max(0.0, (retrieved_count - selected_count) / retrieved_count) * 100.0, 2)

    @staticmethod
    def answer_relevance_score(
        answer: str, keywords: list[str], semantic_similarity: float
    ) -> float:
        if not answer.strip():
            return 0.0
        overlap = 0.0
        if keywords:
            answer_lower = answer.lower()
            hits = sum(1 for keyword in keywords if keyword in answer_lower)
            overlap = hits / len(keywords)
        return round(min(1.0, max(0.0, 0.6 * overlap + 0.4 * semantic_similarity)), 4)

    def _compute_summary(self, records: list[EvaluationRecord]) -> list[dict[str, str]]:
        baseline = [r for r in records if r.mode == "baseline"]
        proposed = [r for r in records if r.mode == "proposed"]
        metrics = self._metric_pairs(baseline, proposed)
        rows: list[dict[str, str]] = []
        for name, (b, p) in metrics.items():
            rows.append(
                {
                    "metric": name,
                    "baseline": f"{b:.4f}",
                    "proposed": f"{p:.4f}",
                    "baseline_vs_proposed_comparison": f"{(p - b):+.4f}",
                }
            )
        return rows

    def _grouped_metrics(
        self, records: list[EvaluationRecord], group_field: str
    ) -> list[dict[str, str]]:
        groups = sorted({getattr(r, group_field) for r in records})
        rows: list[dict[str, str]] = []
        for group in groups:
            baseline = [
                r for r in records if getattr(r, group_field) == group and r.mode == "baseline"
            ]
            proposed = [
                r for r in records if getattr(r, group_field) == group and r.mode == "proposed"
            ]
            metrics = self._metric_pairs(baseline, proposed)
            for metric_name, (b, p) in metrics.items():
                rows.append(
                    {
                        group_field: str(group),
                        "metric": metric_name,
                        "baseline": f"{b:.4f}",
                        "proposed": f"{p:.4f}",
                        "baseline_vs_proposed_comparison": f"{(p - b):+.4f}",
                    }
                )
        return rows

    @staticmethod
    def _metric_pairs(
        baseline: list[EvaluationRecord], proposed: list[EvaluationRecord]
    ) -> dict[str, tuple[float, float]]:
        def avg(values: list[float]) -> float:
            return statistics.mean(values) if values else 0.0

        return {
            "avg_context_reduction_percentage": (
                avg([r.context_reduction_percentage for r in baseline]),
                avg([r.context_reduction_percentage for r in proposed]),
            ),
            "avg_confidence_score": (
                avg([r.confidence_score for r in baseline]),
                avg([r.confidence_score for r in proposed]),
            ),
            "verification_pass_rate": (
                avg([1.0 if r.verified else 0.0 for r in baseline]),
                avg([1.0 if r.verified else 0.0 for r in proposed]),
            ),
            "avg_faithfulness_score": (
                avg([r.faithfulness_score for r in baseline]),
                avg([r.faithfulness_score for r in proposed]),
            ),
            "avg_answer_relevance_score": (
                avg([r.answer_relevance_score for r in baseline]),
                avg([r.answer_relevance_score for r in proposed]),
            ),
            "avg_latency_ms": (
                avg([r.latency_ms for r in baseline]),
                avg([r.latency_ms for r in proposed]),
            ),
            "avg_retrieved_chunks": (
                avg([float(r.retrieved_chunks_count) for r in baseline]),
                avg([float(r.retrieved_chunks_count) for r in proposed]),
            ),
            "avg_selected_chunks": (
                avg([float(r.selected_chunks_count) for r in baseline]),
                avg([float(r.selected_chunks_count) for r in proposed]),
            ),
            "unsupported_claims_count": (
                avg([float(r.unsupported_claims_count) for r in baseline]),
                avg([float(r.unsupported_claims_count) for r in proposed]),
            ),
            "avg_evidence_coverage_score": (
                avg([r.evidence_coverage_score for r in baseline]),
                avg([r.evidence_coverage_score for r in proposed]),
            ),
            "avg_reranking_gain": (
                avg([r.reranking_gain for r in baseline]),
                avg([r.reranking_gain for r in proposed]),
            ),
            "reranked_query_rate": (
                avg([1.0 if r.reranking_applied else 0.0 for r in baseline]),
                avg([1.0 if r.reranking_applied else 0.0 for r in proposed]),
            ),
            "mean_latency_when_reranking_applied": (
                avg([r.total_ms for r in baseline if r.reranking_applied]),
                avg([r.total_ms for r in proposed if r.reranking_applied]),
            ),
            "mean_latency_when_reranking_skipped": (
                avg([r.total_ms for r in baseline if not r.reranking_applied]),
                avg([r.total_ms for r in proposed if not r.reranking_applied]),
            ),
        }

    @staticmethod
    def _write_detailed(records: list[EvaluationRecord], output_path: Path) -> None:
        with output_path.open("w", newline="", encoding="utf-8") as file:
            fieldnames = list(asdict(records[0]).keys()) if records else []
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if records:
                writer.writeheader()
                for record in records:
                    writer.writerow(asdict(record))

    @staticmethod
    def _write_summary(rows: list[dict[str, str]], output_path: Path, headers: list[str]) -> None:
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def _write_grouped(rows: list[dict[str, str]], output_path: Path) -> None:
        if not rows:
            return
        headers = list(rows[0].keys())
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def _write_markdown_report(summary_rows: list[dict[str, str]], output_path: Path) -> None:
        with output_path.open("w", encoding="utf-8") as file:
            file.write("# Evaluation Report\n\n")
            file.write("## Summary Metrics (Baseline vs Proposed)\n\n")
            file.write("| Metric | Baseline | Proposed | Comparison |\n")
            file.write("|---|---:|---:|---:|\n")
            for row in summary_rows:
                file.write(
                    f"| {row['metric']} | {row['baseline']} | {row['proposed']} | {row['baseline_vs_proposed_comparison']} |\n"
                )

    @staticmethod
    def _write_latex_table(summary_rows: list[dict[str, str]], output_path: Path) -> None:
        with output_path.open("w", encoding="utf-8") as file:
            file.write("\\begin{table}[t]\n")
            file.write("\\caption{Baseline vs Proposed Multi-Agent RAG Results}\n")
            file.write("\\label{tab:rag_results}\n")
            file.write("\\centering\n")
            file.write("\\resizebox{\\columnwidth}{!}{%\n")
            file.write("\\begin{tabular}{lccc}\n")
            file.write("\\hline\n")
            file.write("Metric & Baseline & Proposed & Delta \\\\\n")
            file.write("\\hline\n")
            for row in summary_rows:
                metric = row["metric"].replace("_", "\\_")
                file.write(
                    f"{metric} & {row['baseline']} & {row['proposed']} & {row['baseline_vs_proposed_comparison']} \\\\\n"
                )
            file.write("\\hline\n")
            file.write("\\end{tabular}%\n")
            file.write("}\n")
            file.write("\\end{table}\n")
