from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    question: str
    answer: str
    verified: bool
    confidence: float
    selected_context_count: int
