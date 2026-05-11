from io import BytesIO

from fastapi.testclient import TestClient

from app.api.routes.rag import service
from app.main import app


def test_index_retrieve_and_ask_endpoints() -> None:
    client = TestClient(app)
    service.store.reset()
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
