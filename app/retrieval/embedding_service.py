class EmbeddingService:
    """Embedding service stub. Replace with SentenceTransformers-backed implementation."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 5 for _ in texts]
