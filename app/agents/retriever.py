from app.retrieval.embedding_service import EmbeddingService
from app.retrieval.faiss_store import FaissStore
from app.models.document import RetrievedChunk


class RetrieverAgent:
    """Retrieves candidate chunks from FAISS for a given query string."""

    def __init__(self, embedder: EmbeddingService, store: FaissStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self.embedder.backend == "hash":
            lexical_results = self.store.lexical_search(query, top_k=top_k)
            if lexical_results:
                return lexical_results

        query_vector = self.embedder.embed_query(query)
        return self.retrieve_by_embedding(query_vector, top_k=top_k)

    def retrieve_by_embedding(self, query_vector, top_k: int) -> list[RetrievedChunk]:  # noqa: ANN001
        return self.store.search(query_vector, top_k=top_k)
