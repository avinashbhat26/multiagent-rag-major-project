from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.rag import router as rag_router
from app.core.config import settings

app = FastAPI(title=settings.app_name)
app.include_router(health_router, prefix="/health", tags=["health"])
app.include_router(rag_router, prefix="/rag", tags=["rag"])
