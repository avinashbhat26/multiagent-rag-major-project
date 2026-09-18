from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.routes.rag import service
from app.main import app
from app.services.rag_service import (
    DEFAULT_KNOWLEDGE_BASE_ID,
    DEFAULT_KNOWLEDGE_BASE_NAME,
    KnowledgeBaseState,
)


def test_index_retrieve_and_ask_endpoints(tmp_path: Path) -> None:
    client = TestClient(app)
    service._storage_dir = tmp_path / "kb"  # noqa: SLF001
    service._knowledge_bases = {  # noqa: SLF001
        DEFAULT_KNOWLEDGE_BASE_ID: KnowledgeBaseState(
            knowledge_base_id=DEFAULT_KNOWLEDGE_BASE_ID,
            name=DEFAULT_KNOWLEDGE_BASE_NAME,
        )
    }
    service.active_knowledge_base_id = DEFAULT_KNOWLEDGE_BASE_ID
    service.embedder.backend = "hash"
    service.parser.extract_pages = lambda _bytes: [(1, "alpha beta gamma " * 100)]  # type: ignore[method-assign]

    files = {"files": ("demo.pdf", BytesIO(b"%PDF-1.4 fake"), "application/pdf")}
    index_response = client.post("/rag/index?reset=true", files=files)
    assert index_response.status_code == 200
    assert index_response.json()["indexed_files"] == 1

    retrieve_response = client.post(
        "/rag/retrieve", json={"question": "What is alpha?", "top_k": 3}
    )
    assert retrieve_response.status_code == 200
    assert retrieve_response.json()["retrieved_context_count"] >= 1

    ask_response = client.post(
        "/rag/ask",
        json={"question": "What is alpha?", "mode": "baseline", "top_k": 3},
    )
    assert ask_response.status_code == 200
    body = ask_response.json()
    assert body["mode"] == "baseline"
    assert body["selected_context_count"] >= 1
    assert "supported_claims" in body
    assert "timings" in body
    assert "agent_trace" in body
    assert body["timings"]["total_ms"] >= 0.0

    status_response = client.get("/rag/status")
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["total_chunks"] >= 1
    assert status["indexed_documents"]
