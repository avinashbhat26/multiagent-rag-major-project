from typing import Literal

from pydantic import BaseModel, Field


class ContextChunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    page: int
    score: float


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: Literal["baseline", "multi_agent"] = "baseline"
    top_k: int | None = Field(default=None, ge=1, le=20)


class AskResponse(BaseModel):
    question: str
    mode: Literal["baseline", "multi_agent"]
    answer: str
    llm_provider: str
    verified: bool
    confidence: float
    retrieved_context_count: int
    selected_context_count: int
    contexts: list[ContextChunk] = Field(default_factory=list)


class IndexResponse(BaseModel):
    indexed_files: int
    indexed_chunks: int
    total_chunks: int
    sources: list[str]


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class RetrieveResponse(BaseModel):
    question: str
    retrieved_context_count: int
    contexts: list[ContextChunk] = Field(default_factory=list)
