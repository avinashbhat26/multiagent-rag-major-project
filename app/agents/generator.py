from app.core.logging import get_logger
from app.llm.providers import ExtractiveProvider, LLMProvider, build_provider
from app.models.document import RetrievedChunk

logger = get_logger(__name__)


class GeneratorAgent:
    """Answer generator backed by a pluggable LLM provider abstraction."""

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider or build_provider()
        self.fallback_provider = ExtractiveProvider()
        self.last_used_provider = self.provider.name

    def generate(self, question: str, context: list[RetrievedChunk]) -> str:
        try:
            answer = self.provider.generate(question, context)
            if answer:
                self.last_used_provider = self.provider.name
                return answer
        except Exception as exc:  # pragma: no cover - provider errors vary by environment
            logger.warning(
                "Provider '%s' failed (%s). Falling back to extractive provider.",
                self.provider.name,
                exc,
            )
        self.last_used_provider = self.fallback_provider.name
        return self.fallback_provider.generate(question, context)
