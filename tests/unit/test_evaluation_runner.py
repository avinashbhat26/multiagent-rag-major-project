from pathlib import Path
from types import SimpleNamespace

from eval.evaluator import EvaluationRunner, EvaluationSample


class FakeService:
    def ask(self, question: str, mode: str, top_k: int = 5) -> SimpleNamespace:
        _ = (question, top_k)
        if mode == "baseline":
            return SimpleNamespace(
                retrieved_context_count=5,
                selected_context_count=5,
                semantic_similarity=0.6,
                confidence=0.65,
                verified=False,
                regenerated=False,
                contexts=[SimpleNamespace(text="attendance 75% course grade")],
            )
        return SimpleNamespace(
            retrieved_context_count=5,
            selected_context_count=3,
            semantic_similarity=0.8,
            confidence=0.82,
            verified=True,
            regenerated=True,
            contexts=[SimpleNamespace(text="attendance 75% course grade W grade")],
        )


def test_evaluation_runner_writes_outputs(tmp_path: Path) -> None:
    runner = EvaluationRunner(FakeService())
    rows = runner.evaluate(
        [EvaluationSample(question="attendance rules", expected_keywords=["attendance", "75%"])]
    )
    out_dir = tmp_path / "eval_out"
    runner.write_outputs(rows, str(out_dir))

    assert (out_dir / "evaluation_details.csv").exists()
    assert (out_dir / "evaluation_summary.csv").exists()
    assert (out_dir / "evaluation_table.md").exists()
