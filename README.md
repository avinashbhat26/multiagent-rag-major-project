# Multi-Agent RAG Major Project

**Project Title:** Design and Implementation of a Multi-Agent Retrieval-Augmented Generation System with Self-Verification and Adaptive Context Selection for Reliable Question Answering

## Overview
This repository contains a production-style research prototype for a multi-agent RAG system. It extends a baseline document question-answering pipeline into a modular architecture with:
- Query analysis
- Planning / task decomposition
- Retrieval using embeddings + FAISS
- Adaptive context selection
- Answer generation
- Self-verification
- Optional regeneration loop

## Architecture
Baseline path:
- PDF ingestion -> chunking -> embedding -> FAISS retrieval -> answer generation

Proposed multi-agent path:
- Query Analyzer Agent
- Planner Agent
- Retriever Agent
- Adaptive Context Selection Agent (dynamic top-k + redundancy removal)
- Reranking Layer (optional cross-encoder with coverage-aware scoring)
- Generator Agent (provider abstraction: OpenAI / Ollama / extractive fallback)
- Verification Agent (supported/unsupported claims + confidence + regeneration trigger)

## Tech Stack
- Python 3.11
- FastAPI
- SentenceTransformers
- FAISS
- Pydantic
- pytest
- Ruff
- Black
- Docker
- GitHub Actions

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create environment file:
```bash
cp .env.example .env
```
Windows PowerShell:
```powershell
Copy-Item .env.example .env
```

## Web Frontend (Academic Demo UI)
The demo UI is served by FastAPI at `/ui`.

Run the backend:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Then open `http://127.0.0.1:8000/ui` and:
- Upload one or more PDF files
- Click **Build / Update Knowledge Base**
- Ask questions and inspect:
  - generated answer
  - verification status
  - confidence score
  - retrieved chunks (source, page, score, text)
  - reranking, evidence coverage, timings, and agent trace

## Browser Frontend (No Paid Hosting Required)
The project includes a static browser UI served directly by FastAPI.

Run:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open:
```text
http://127.0.0.1:8000/ui/
```

The browser UI supports:
- PDF upload and indexing
- baseline vs proposed multi-agent mode
- answer generation
- retrieval preview
- verification metrics
- retrieved evidence cards
- pipeline timings
- agent trace display

## Environment
Copy `.env.example` to `.env` and fill values as needed.

Recommended values:
- `LLM_PROVIDER=extractive` for offline local demo
- `LLM_PROVIDER=openai` with `OPENAI_API_KEY` for cloud LLM
- `LLM_PROVIDER=ollama` with `OLLAMA_BASE_URL` for local model serving

## API Demo Flow
1. `POST /rag/index` with one or more PDFs
2. `POST /rag/retrieve` with a question to inspect retrieved chunks
3. `POST /rag/ask` with:
   - `mode=baseline` for fixed top-k baseline
   - `mode=multi_agent` for adaptive context selection + verification loop

`/rag/ask` response includes:
- `answer`
- `verified`
- `confidence`
- `supported_claims`
- `unsupported_claims`
- `contexts`
- `dynamic_top_k`
- `removed_redundant_chunks`
- `regenerated`

## Testing and Quality
Run formatting/lint/tests locally:
```bash
python -m ruff format .
python -m ruff check .
python -m pytest -q
```

## Multi-Document Evaluation for Paper Results
This repository supports reproducible multi-document evaluation across local PDFs with per-document isolation.

Document config:
- `eval/evaluation_documents.json`

Question files:
- `eval/questions/academic_rulebook_questions.jsonl`
- `eval/questions/ipr_policy_questions.jsonl`

Run evaluation:
```bash
python scripts/run_evaluation.py --mode both --documents eval/evaluation_documents.json --questions-dir eval/questions --output-dir eval/results
```

Optional:
```bash
python scripts/run_evaluation.py --mode baseline --documents eval/evaluation_documents.json --questions-dir eval/questions --output-dir eval/results
python scripts/run_evaluation.py --mode proposed --documents eval/evaluation_documents.json --questions-dir eval/questions --output-dir eval/results
```

Generated outputs:
- `eval/results/detailed_results.csv`
- `eval/results/summary_metrics.csv`
- `eval/results/document_wise_metrics.csv`
- `eval/results/domain_wise_metrics.csv`
- `eval/results/evaluation_report.md`
- `eval/results/latex_results_table.tex`

Generate plots:
```bash
python scripts/generate_evaluation_plots.py --summary eval/results/summary_metrics.csv --document-metrics eval/results/document_wise_metrics.csv --output-dir eval/results/plots
```

## CI/CD
- CI: `ruff check` + `ruff format --check` + `pytest` + Docker build via GitHub Actions
- CD: ready for deployment to Render / Hugging Face Spaces / other services

## Docker
Build and run locally:
```bash
docker build -t multiagent-rag:latest .
docker run -p 8000:8000 --env-file .env multiagent-rag:latest
```

## Suggested Deployment
- Backend API: Render free web service or another low-cost service
- Demo UI: Hugging Face Spaces (free CPU basic)
