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
    from eval.evaluator import PaperEvaluationRunner

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
    parser.add_argument(
        "--pdf",
        action="append",
        default=[],
        help="PDF path to index before evaluation. Can be provided multiple times.",
    )
    parser.add_argument(
        "--reset-index",
        action="store_true",
        help="Reset index before indexing provided PDFs.",
    )
    args = parser.parse_args()

    service = MultiAgentRAGService()
    if not os.getenv("OPENAI_API_KEY"):
        service.generator = GeneratorAgent(provider=ExtractiveProvider())

    if args.pdf:
        from io import BytesIO

        from fastapi import UploadFile

        uploads: list[UploadFile] = []
        for pdf_path in args.pdf:
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_file}")
            uploads.append(
                UploadFile(
                    filename=pdf_file.name,
                    file=BytesIO(pdf_file.read_bytes()),
                )
            )
        index_result = service.index_documents(uploads, reset=args.reset_index)
        print(
            "Indexed before evaluation:",
            f"files={index_result.indexed_files}, chunks={index_result.indexed_chunks}, total={index_result.total_chunks}",
        )

    runner = PaperEvaluationRunner(service)
    questions = runner.load_questions(args.questions)
    records = runner.run(questions, mode=args.mode)
    runner.write_outputs(records, args.output_dir)
    print(f"Evaluation completed for mode={args.mode}. Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
