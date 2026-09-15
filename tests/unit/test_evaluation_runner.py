import json
from pathlib import Path
from types import SimpleNamespace

from eval.evaluator import (
    EvaluationDocument,
    EvaluationQuestion,
    MultiDocumentEvaluationRunner,
)


def fake_timings() -> SimpleNamespace:
    return SimpleNamespace(
        query_analysis_ms=1.0,
        planning_ms=1.0,
        query_embedding_ms=1.0,
        retrieval_ms=1.0,
        reranking_ms=1.0,
        context_selection_ms=1.0,
        generation_ms=1.0,
        verification_ms=1.0,
        evidence_recovery_ms=0.0,
        total_ms=7.0,
    )


class FakeService:
    def index_documents(self, files, reset=False):  # noqa: ANN001,ANN202
        _ = (files, reset)
        return SimpleNamespace(indexed_files=1, indexed_chunks=3, total_chunks=3, sources=["x.pdf"])

    def ask(self, question: str, mode: str, top_k: int = 5):  # noqa: ANN201
        _ = (question, top_k)
        if mode == "baseline":
            return SimpleNamespace(
                answer="Attendance must be 75 percent",
                retrieved_context_count=5,
                selected_context_count=5,
                evidence_coverage_score=0.0,
                reranking_gain=0.0,
                contexts=[SimpleNamespace(text="attendance 75% minimum rule", score=0.9)],
                removed_redundant_chunks=0,
                semantic_similarity=0.60,
                unsupported_claims=["extra claim"],
                supported_claims=["Attendance must be 75 percent"],
                verified=False,
                confidence=0.65,
                regenerated=False,
                timings=fake_timings(),
                reranking_applied=False,
                reranking_reason="baseline_mode",
                query_complexity="simple",
                retrieval_candidate_k=10,
            )
        return SimpleNamespace(
            answer="Minimum attendance is 75 percent and low attendance leads to W grade",
            retrieved_context_count=5,
            selected_context_count=3,
            evidence_coverage_score=0.78,
            reranking_gain=0.12,
            contexts=[SimpleNamespace(text="attendance 75% W grade", score=0.94)],
            removed_redundant_chunks=2,
            semantic_similarity=0.82,
            unsupported_claims=[],
            supported_claims=["Minimum attendance is 75 percent"],
            verified=True,
            confidence=0.84,
            regenerated=True,
            timings=fake_timings(),
            reranking_applied=True,
            reranking_reason="ambiguous_retrieval_scores",
            query_complexity="simple",
            retrieval_candidate_k=10,
        )


def test_load_documents_detects_duplicates(tmp_path: Path) -> None:
    doc_path = tmp_path / "docs.json"
    same_pdf = str((tmp_path / "a.pdf").resolve())
    (tmp_path / "a.pdf").write_bytes(b"fake")
    doc_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_id": "d1",
                        "document_name": "Doc1",
                        "domain": "x",
                        "file_path": same_pdf,
                        "description": "",
                    },
                    {
                        "document_id": "d2",
                        "document_name": "Doc2",
                        "domain": "x",
                        "file_path": same_pdf,
                        "description": "",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = MultiDocumentEvaluationRunner(FakeService())
    docs, warnings = runner.load_documents(str(doc_path))
    assert len(docs) == 1
    assert warnings


def test_metric_calculations() -> None:
    runner = MultiDocumentEvaluationRunner(FakeService())
    reduction = runner.context_reduction_percentage(5, 3)
    relevance = runner.answer_relevance_score(
        answer="attendance is 75%", keywords=["attendance", "75%"], semantic_similarity=0.8
    )
    assert reduction == 40.0
    assert 0.0 <= relevance <= 1.0


def test_evaluation_outputs_are_created(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake pdf")

    documents = [
        EvaluationDocument(
            document_id="academic_rulebook",
            document_name="Academic",
            domain="academics",
            file_path=str(pdf_path),
            description="",
        )
    ]
    questions_by_doc = {
        "academic_rulebook": [
            EvaluationQuestion(
                question_id="Q1",
                document_id="academic_rulebook",
                question="What is attendance rule?",
                expected_answer_keywords=["attendance", "75%"],
                difficulty="easy",
                query_type="factual",
            )
        ]
    }

    runner = MultiDocumentEvaluationRunner(FakeService())
    records = runner.run(documents, questions_by_doc, mode="both")
    out = tmp_path / "results"
    runner.write_outputs(records, str(out))

    assert (out / "detailed_results.csv").exists()
    assert (out / "summary_metrics.csv").exists()
    assert (out / "document_wise_metrics.csv").exists()
    assert (out / "domain_wise_metrics.csv").exists()
    assert (out / "evaluation_report.md").exists()
    assert (out / "latex_results_table.tex").exists()
