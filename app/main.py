from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes.evaluation import router as evaluation_router
from app.api.routes.health import router as health_router
from app.api.routes.rag import router as rag_router
from app.core.config import settings

app = FastAPI(title=settings.app_name)
allowed_origins = [
    origin.strip() for origin in settings.cors_allowed_origins.split(",") if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(health_router, prefix="/health", tags=["health"])
app.include_router(rag_router, prefix="/rag", tags=["rag"])
app.include_router(evaluation_router, prefix="/evaluation", tags=["evaluation"])

web_ui_dir = Path(__file__).resolve().parents[1] / "frontend" / "web"
if web_ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=web_ui_dir, html=True), name="web-ui")
