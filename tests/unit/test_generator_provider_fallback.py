from app.agents.generator import GeneratorAgent
from app.models.document import RetrievedChunk


class BrokenProvider:
    name = "broken"

    def generate(self, question: str, context: list[RetrievedChunk]) -> str:
        raise RuntimeError("simulated provider failure")


def test_generator_falls_back_when_provider_fails() -> None:
    agent = GeneratorAgent(provider=BrokenProvider())
    answer = agent.generate(
        "What is alpha?",
        [RetrievedChunk(chunk_id="x:1", text="alpha context", source="x.pdf", page=1, score=0.9)],
    )

    assert answer.startswith("Answer:")
    assert "Source: x.pdf, page 1" in answer
    assert agent.last_used_provider == "extractive"


def test_extractive_provider_prefers_definition_sentence() -> None:
    from app.llm.providers import ExtractiveProvider

    provider = ExtractiveProvider()
    answer = provider.generate(
        "word history is derived from which word?",
        [
            RetrievedChunk(
                chunk_id="history:p1:c1",
                text=(
                    "History helps us understand the past. "
                    "The word history is derived from the Greek word historia. "
                    "It links the present with the past."
                ),
                source="history.pdf",
                page=1,
                score=0.91,
            )
        ],
    )

    assert "Greek word historia" in answer
    assert "Source: history.pdf, page 1" in answer
