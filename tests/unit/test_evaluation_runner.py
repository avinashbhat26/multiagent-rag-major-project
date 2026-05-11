from pathlib import Path
from types import SimpleNamespace

from eval.evaluator import EvaluationQuestion, PaperEvaluationRunner


class FakeService:
    def ask(self, question: str, mode: str, top_k: int = 5) -> SimpleNamespace:
        _ = (question, top_k)
        if mode == "baseline":
            return SimpleNamespace(
                answer="Attendance must be 75 percent",
                retrieved_context_count=5,
                selected_context_count=5,
                contexts=[SimpleNamespace(text="attendance 75% minimum rule", score=0.9)],
                removed_redundant_chunks=0,
                semantic_similarity=0.60,
                unsupported_claims=["extra claim"],
                supported_claims=["Attendance must be 75 percent"],
                verified=False,
                confidence=0.65,
                regenerated=False,
            )
        return SimpleNamespace(
            answer="Minimum attendance is 75 percent and low attendance leads to W grade",
            retrieved_context_count=5,
            selected_context_count=3,
            contexts=[SimpleNamespace(text="attendance 75% W grade", score=0.94)],
            removed_redundant_chunks=2,
            semantic_similarity=0.82,
            unsupported_claims=[],
            supported_claims=["Minimum attendance is 75 percent"],
            verified=True,
            confidence=0.84,
            regenerated=True,
        )


def test_load_questions_from_jsonl(tmp_path: Path) -> None:
    dataset_path = tmp_path / "questions.jsonl"
    dataset_path.write_text(
        '{"question_id":"Q1","question":"What is attendance?","expected_answer":"75%",'
        '"expected_source_keywords":["attendance","75%"],"difficulty":"easy","query_type":"factual"}\n',
        encoding="utf-8",
    )
    runner = PaperEvaluationRunner(FakeService())
    questions = runner.load_questions(str(dataset_path))
    assert len(questions) == 1
    assert questions[0].question_id == "Q1"


def test_metric_calculations() -> None:
    runner = PaperEvaluationRunner(FakeService())
    reduction = runner.context_reduction_percentage(5, 3)
    relevance = runner.answer_relevance_score(
        answer="attendance is 75%", expected_keywords=["attendance", "75%"], semantic_similarity=0.8
    )
    assert reduction == 40.0
    assert 0.0 <= relevance <= 1.0


def test_evaluation_outputs_are_created(tmp_path: Path) -> None:
    runner = PaperEvaluationRunner(FakeService())
    questions = [
        EvaluationQuestion(
            question_id="Q1",
            question="What is attendance rule?",
            expected_answer="75%",
            expected_source_keywords=["attendance", "75%"],
            difficulty="easy",
            query_type="factual",
        )
    ]
    records = runner.run(questions, mode="both")
    out = tmp_path / "results"
    runner.write_outputs(records, str(out))

    assert (out / "detailed_results.csv").exists()
    assert (out / "summary_metrics.csv").exists()
    assert (out / "evaluation_report.md").exists()
