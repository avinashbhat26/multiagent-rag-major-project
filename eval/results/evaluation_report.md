# Evaluation Report

## Summary Comparison Table

| Metric | Baseline | Proposed | Comparison |
|---|---:|---:|---|
| Avg Context Reduction % | 0.0000 | 60.0000 | +60.0000 |
| Avg Confidence Score | 0.9486 | 0.8813 | -0.0673 |
| Verification Pass Rate | 0.8000 | 0.7500 | -0.0500 |
| Avg Faithfulness Score | 0.8765 | 0.8772 | +0.0007 |
| Avg Answer Relevance Score | 0.7506 | 0.7509 | +0.0003 |
| Avg Latency | 73.6635 | 78.8070 | +5.1435 |
| Avg Retrieved Chunks | 5.0000 | 5.0000 | +0.0000 |
| Avg Selected Chunks | 5.0000 | 2.0000 | -3.0000 |
| Unsupported Claims Count | 0.2000 | 0.2500 | +0.0500 |

## Notes
- Proposed corresponds to multi-agent mode.
- Faithfulness is measured via semantic similarity from verification.
- Relevance is keyword overlap + semantic similarity blend.
