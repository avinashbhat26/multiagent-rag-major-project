from app.models.document import RetrievedChunk


class FaissStore:
    """Vector store stub. Replace with real FAISS indexing/retrieval implementation."""

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                text=f"Context chunk {i+1} for {query}", source="demo.pdf", score=1.0 - i * 0.05
            )
            for i in range(top_k)
        ]
