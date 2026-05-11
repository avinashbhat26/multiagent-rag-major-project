from app.agents.reranker import RerankerAgent
from app.models.document import RetrievedChunk


def test_reranker_returns_coverage_and_gain() -> None:
    agent = RerankerAgent()
    chunks = [
        RetrievedChunk(
            chunk_id="a",
            text="attendance minimum 75 percent",
            source="doc.pdf",
            page=1,
            score=0.7,
        ),
        RetrievedChunk(
            chunk_id="b",
            text="hostel rules and discipline",
            source="doc.pdf",
            page=2,
            score=0.8,
        ),
    ]
    result = agent.rerank("attendance rule", chunks, top_k=2)

    assert len(result.reranked_chunks) == 2
    assert 0.0 <= result.evidence_coverage_score <= 1.0
    assert isinstance(result.reranking_gain, float)
