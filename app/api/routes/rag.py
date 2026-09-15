from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.schemas.rag import (
    AskRequest,
    AskResponse,
    IndexResponse,
    RagStatusResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from app.services.rag_service import MultiAgentRAGService

router = APIRouter()
service = MultiAgentRAGService()


@router.post("/index", response_model=IndexResponse)
def index_documents(
    files: list[UploadFile] = File(...), reset: bool = Query(default=False)
) -> IndexResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    return service.index_documents(files=files, reset=reset)


@router.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    return service.ask(payload.question, mode=payload.mode, top_k=payload.top_k)


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_context(payload: RetrieveRequest) -> RetrieveResponse:
    return service.retrieve(payload.question, top_k=payload.top_k)


@router.get("/status", response_model=RagStatusResponse)
def get_status() -> RagStatusResponse:
    return service.status()


@router.get("/documents")
def list_documents() -> dict[str, object]:
    return {"documents": service.status().indexed_documents}


@router.post("/reset", response_model=RagStatusResponse)
def reset_knowledge_base() -> RagStatusResponse:
    return service.reset()
