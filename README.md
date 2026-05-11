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

## Streamlit Frontend (Academic Demo UI)
Run backend and frontend in separate terminals:

Terminal 1 (FastAPI backend):
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Terminal 2 (Streamlit frontend):
```bash
streamlit run frontend/streamlit_app.py
```

Then open `http://localhost:8501` and:
- Upload one or more PDF files
- Click **Build / Update Knowledge Base**
- Ask questions and inspect:
  - generated answer
  - verification status
  - confidence score
  - retrieved chunks (source, page, score, text)

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

## Evaluation Framework
The project includes a modular evaluation runner for baseline vs multi-agent comparison.

Inputs:
- `eval/questions.jsonl` with:
  - `question_id`
  - `question`
  - `expected_answer`
  - `expected_source_keywords`
  - `difficulty`
  - `query_type`
- indexed knowledge base already loaded in the backend service

Run:
```bash
python scripts/run_evaluation.py --mode both
python scripts/run_evaluation.py --mode baseline
python scripts/run_evaluation.py --mode proposed
python scripts/run_evaluation.py --mode both --questions eval/questions.jsonl --output-dir eval/results
```

Generated structured outputs:
- `eval/results/detailed_results.csv` (per-question detailed metrics)
- `eval/results/summary_metrics.csv` (baseline vs proposed summary)
- `eval/results/evaluation_report.md` (human-readable report and comparison table)

Metrics included:
- retrieval: chunk counts, context reduction %, similarity stats, redundancy removed
- generation: answer length, relevance score, faithfulness score, unsupported claims, hallucination flag
- verification: verified status, confidence, supported/unsupported claim counts, regeneration trigger
- system: latency, context token estimate, baseline-vs-proposed comparison deltas

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
