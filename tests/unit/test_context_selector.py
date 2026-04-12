from app.agents.context_selector import AdaptiveContextSelectionAgent
from app.models.document import RetrievedChunk


def test_context_selector_returns_non_empty_selection() -> None:
    agent = AdaptiveContextSelectionAgent()
    chunks = [
        RetrievedChunk(text="A", source="x", score=0.95),
        RetrievedChunk(text="B", source="x", score=0.90),
        RetrievedChunk(text="C", source="x", score=0.80),
    ]
    selected = agent.select(chunks)
    assert len(selected) >= 1
