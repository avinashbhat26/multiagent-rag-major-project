from fastapi import APIRouter

from app.schemas.rag import AskRequest, AskResponse
from app.services.rag_service import MultiAgentRAGService

router = APIRouter()
service = MultiAgentRAGService()


@router.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    return service.ask(payload.question)
