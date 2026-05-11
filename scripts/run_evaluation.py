from __future__ import annotations

import argparse
import os

from app.agents.generator import GeneratorAgent
from app.llm.providers import ExtractiveProvider
from app.services.rag_service import MultiAgentRAGService
from eval.evaluator import PaperEvaluationRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible paper evaluation for RAG.")
    parser.add_argument(
        "--mode",
        default="both",
        choices=["both", "baseline", "proposed"],
        help="Evaluation mode: baseline, proposed, or both.",
    )
    parser.add_argument(
        "--questions",
        default="eval/questions.jsonl",
        help="Path to JSONL questions file.",
    )
    parser.add_argument(
        "--output-dir",
        default="eval/results",
        help="Directory where result artifacts will be written.",
    )
    args = parser.parse_args()

    service = MultiAgentRAGService()
    if not os.getenv("OPENAI_API_KEY"):
        service.generator = GeneratorAgent(provider=ExtractiveProvider())

    runner = PaperEvaluationRunner(service)
    questions = runner.load_questions(args.questions)
    records = runner.run(questions, mode=args.mode)
    runner.write_outputs(records, args.output_dir)
    print(f"Evaluation completed for mode={args.mode}. Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
