from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.schemas.rag import (
    AskRequest,
    AskResponse,
    CreateKnowledgeBaseRequest,
    IndexResponse,
    KnowledgeBaseListResponse,
    KnowledgeBaseSummary,
    RagStatusResponse,
    RenameKnowledgeBaseRequest,
    RetrieveRequest,
    RetrieveResponse,
)
from app.services.rag_service import MultiAgentRAGService

router = APIRouter()
service = MultiAgentRAGService()


@router.post("/index", response_model=IndexResponse)
def index_documents(
    files: list[UploadFile] = File(...),
    reset: bool = Query(default=False),
    knowledge_base_id: str | None = Query(default=None),
) -> IndexResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    try:
        return service.index_documents(
            files=files,
            reset=reset,
            knowledge_base_id=knowledge_base_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    try:
        return service.ask(
            payload.question,
            mode=payload.mode,
            top_k=payload.top_k,
            knowledge_base_id=payload.knowledge_base_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_context(payload: RetrieveRequest) -> RetrieveResponse:
    try:
        return service.retrieve(
            payload.question,
            top_k=payload.top_k,
            knowledge_base_id=payload.knowledge_base_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/status", response_model=RagStatusResponse)
def get_status(knowledge_base_id: str | None = Query(default=None)) -> RagStatusResponse:
    try:
        return service.status(knowledge_base_id=knowledge_base_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/documents")
def list_documents(knowledge_base_id: str | None = Query(default=None)) -> dict[str, object]:
    try:
        return {"documents": service.status(knowledge_base_id=knowledge_base_id).indexed_documents}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/reset", response_model=RagStatusResponse)
def reset_knowledge_base(knowledge_base_id: str | None = Query(default=None)) -> RagStatusResponse:
    try:
        return service.reset(knowledge_base_id=knowledge_base_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/knowledge-bases", response_model=KnowledgeBaseListResponse)
def list_knowledge_bases() -> KnowledgeBaseListResponse:
    return service.list_knowledge_bases()


@router.post("/knowledge-bases", response_model=KnowledgeBaseSummary)
def create_knowledge_base(payload: CreateKnowledgeBaseRequest) -> KnowledgeBaseSummary:
    return service.create_knowledge_base(payload)


@router.post("/knowledge-bases/{knowledge_base_id}/select", response_model=RagStatusResponse)
def select_knowledge_base(knowledge_base_id: str) -> RagStatusResponse:
    try:
        return service.select_knowledge_base(knowledge_base_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/knowledge-bases/{knowledge_base_id}", response_model=KnowledgeBaseSummary)
def rename_knowledge_base(
    knowledge_base_id: str, payload: RenameKnowledgeBaseRequest
) -> KnowledgeBaseSummary:
    try:
        return service.rename_knowledge_base(knowledge_base_id, payload.name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/knowledge-bases/{knowledge_base_id}", response_model=KnowledgeBaseListResponse)
def delete_knowledge_base(knowledge_base_id: str) -> KnowledgeBaseListResponse:
    try:
        return service.delete_knowledge_base(knowledge_base_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
