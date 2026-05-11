from app.agents.verifier import VerificationAgent
from app.models.document import RetrievedChunk
from app.retrieval.embedding_service import EmbeddingService


def test_verifier_produces_supported_and_unsupported_claims() -> None:
    agent = VerificationAgent(EmbeddingService(backend="hash"))
    context = [
        RetrievedChunk(
            chunk_id="doc:p1:c1",
            text="Students must maintain minimum 75 percent attendance in each course.",
            source="doc.pdf",
            page=1,
            score=0.9,
        )
    ]
    answer = "Students must maintain 75 percent attendance. They can skip all exams."

    result = agent.verify(answer, context, threshold=0.75)

    assert result.supported_claims
    assert result.unsupported_claims
    assert 0.0 <= result.confidence <= 0.99
    assert 0.0 <= result.semantic_similarity <= 1.0
