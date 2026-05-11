from app.agents.context_selector import AdaptiveContextSelectionAgent
from app.models.document import RetrievedChunk


def test_context_selector_returns_non_empty_selection() -> None:
    agent = AdaptiveContextSelectionAgent()
    chunks = [
        RetrievedChunk(chunk_id="x:1", text="A", source="x", page=1, score=0.95),
        RetrievedChunk(chunk_id="x:2", text="B", source="x", page=1, score=0.90),
        RetrievedChunk(chunk_id="x:3", text="C", source="x", page=1, score=0.80),
    ]
    selection = agent.select(chunks)
    assert len(selection.selected_chunks) >= 1
    assert selection.dynamic_top_k >= 1
