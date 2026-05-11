from __future__ import annotations

import argparse

from eval.evaluator import EvaluationRunner
from app.services.rag_service import MultiAgentRAGService


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline vs multi-agent evaluation.")
    parser.add_argument(
        "--dataset",
        default="eval/sample_eval_dataset.json",
        help="Path to evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default="eval/results",
        help="Directory where CSV and table outputs will be written.",
    )
    args = parser.parse_args()

    service = MultiAgentRAGService()
    runner = EvaluationRunner(service)
    samples = runner.load_samples(args.dataset)
    rows = runner.evaluate(samples)
    runner.write_outputs(rows, args.output_dir)
    print(f"Evaluation completed. Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
