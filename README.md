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
uvicorn app.main:app --reload
```

## Environment
Copy `.env.example` to `.env` and fill values as needed.

## CI/CD
- CI: lint + unit tests + integration tests + Docker build via GitHub Actions
- CD: ready for deployment to Render / Hugging Face Spaces / other services

## Suggested Deployment
- Backend API: Render free web service or another low-cost service
- Demo UI: Hugging Face Spaces (free CPU basic)
