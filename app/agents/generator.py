from app.models.document import RetrievedChunk


class GeneratorAgent:
    """Placeholder generator; swap with OpenAI or Ollama-backed implementation."""

    def generate(self, question: str, context: list[RetrievedChunk]) -> str:
        if not context:
            return "No relevant context was found to answer the question."
        return f"Draft answer based on {len(context)} selected context chunks for: {question}"
