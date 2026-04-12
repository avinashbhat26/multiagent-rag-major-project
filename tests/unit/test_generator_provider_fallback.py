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

    assert "Based on retrieved evidence" in answer
    assert agent.last_used_provider == "extractive"
