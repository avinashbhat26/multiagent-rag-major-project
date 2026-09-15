from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from app.agents.generator import GeneratorAgent
    from app.llm.providers import ExtractiveProvider
    from app.services.rag_service import MultiAgentRAGService
    from eval.evaluator import MultiDocumentEvaluationRunner

    parser = argparse.ArgumentParser(description="Run multi-document paper evaluation for RAG.")
    parser.add_argument(
        "--mode",
        default="both",
        choices=["both", "baseline", "proposed"],
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--documents",
        default="eval/evaluation_documents.json",
        help="Path to document configuration JSON.",
    )
    parser.add_argument(
        "--questions-dir",
        default="eval/questions",
        help="Path to directory containing document question JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        default="eval/results",
        help="Directory for generated evaluation outputs.",
    )
    args = parser.parse_args()

    service = MultiAgentRAGService()
    if not os.getenv("OPENAI_API_KEY"):
        service.generator = GeneratorAgent(provider=ExtractiveProvider())

    runner = MultiDocumentEvaluationRunner(service)
    documents, warnings = runner.load_documents(args.documents)
    for warning in warnings:
        print(f"WARNING: {warning}")

    questions_by_doc = runner.load_questions_dir(args.questions_dir)
    records = runner.run(documents, questions_by_doc, mode=args.mode)
    runner.write_outputs(records, args.output_dir)
    print(f"Evaluation completed for mode={args.mode}. Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
