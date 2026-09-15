from typing import Literal

from pydantic import BaseModel, Field


class ContextChunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    page: int
    score: float


class PipelineTimings(BaseModel):
    query_analysis_ms: float = 0.0
    planning_ms: float = 0.0
    query_embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    context_selection_ms: float = 0.0
    generation_ms: float = 0.0
    verification_ms: float = 0.0
    evidence_recovery_ms: float = 0.0
    total_ms: float = 0.0


class AgentTraceStep(BaseModel):
    agent: str
    status: str
    duration_ms: float
    detail: str = ""


class ClaimVerificationSchema(BaseModel):
    claim: str
    status: str
    support_score: float
    lexical_support: float
    semantic_support: float
    supporting_chunk_id: str
    source: str
    page: int


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
    semantic_similarity: float = 0.0
    supported_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    claim_verifications: list[ClaimVerificationSchema] = Field(default_factory=list)
    retrieved_context_count: int
    selected_context_count: int
    retrieved_before_reranking_count: int = 0
    retrieved_after_reranking_count: int = 0
    evidence_coverage_score: float = 0.0
    reranking_gain: float = 0.0
    reranking_applied: bool = False
    reranking_reason: str = ""
    dynamic_top_k: int = 0
    removed_redundant_chunks: int = 0
    redundancy_ratio: float = 0.0
    evidence_sufficiency_score: float = 0.0
    selection_stop_reason: str = ""
    query_complexity: str = ""
    retrieval_candidate_k: int = 0
    evidence_recovery_triggered: bool = False
    recovery_query: str = ""
    additional_evidence_count: int = 0
    regenerated: bool = False
    timings: PipelineTimings = Field(default_factory=PipelineTimings)
    agent_trace: list[AgentTraceStep] = Field(default_factory=list)
    contexts: list[ContextChunk] = Field(default_factory=list)


class IndexResponse(BaseModel):
    indexed_files: int
    indexed_chunks: int
    total_chunks: int
    sources: list[str]


class IndexedDocument(BaseModel):
    document_id: str
    source: str
    chunk_count: int
    indexed_at: str


class RagStatusResponse(BaseModel):
    total_chunks: int
    indexed_documents: list[IndexedDocument] = Field(default_factory=list)
    llm_provider: str
    embedding_backend: str
    reranking_enabled: bool


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class RetrieveResponse(BaseModel):
    question: str
    retrieved_context_count: int
    contexts: list[ContextChunk] = Field(default_factory=list)
