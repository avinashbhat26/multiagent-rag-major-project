from io import BytesIO

from fastapi import UploadFile

from app.services.rag_service import MultiAgentRAGService


def test_index_and_ask_baseline_flow() -> None:
    service = MultiAgentRAGService()
    service.embedder.backend = "hash"

    fake_pdf = UploadFile(filename="demo.pdf", file=BytesIO(b"%PDF-1.4 fake"))
    service.parser.extract_pages = lambda _bytes: [(1, "alpha beta gamma " * 100)]  # type: ignore[method-assign]

    index_response = service.index_documents([fake_pdf], reset=True)
    ask_response = service.ask("What is alpha?", mode="baseline", top_k=3)
    retrieve_response = service.retrieve("What is alpha?", top_k=3)

    assert index_response.indexed_files == 1
    assert index_response.indexed_chunks >= 1
    assert ask_response.mode == "baseline"
    assert ask_response.llm_provider
    assert ask_response.retrieved_context_count >= 1
    assert ask_response.selected_context_count >= 1
    assert retrieve_response.retrieved_context_count >= 1
