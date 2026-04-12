from threading import Lock

import faiss
import numpy as np

from app.models.document import DocumentChunk, RetrievedChunk


class FaissStore:
    """In-memory FAISS index with chunk metadata tracking."""

    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[DocumentChunk] = []
        self._dimension: int | None = None
        self._lock = Lock()

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    def reset(self) -> None:
        with self._lock:
            self._index = None
            self._chunks = []
            self._dimension = None

    def add(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> int:
        if len(chunks) == 0:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length.")

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("embeddings must be a 2D array.")

        with self._lock:
            if self._index is None:
                self._dimension = vectors.shape[1]
                self._index = faiss.IndexFlatIP(self._dimension)
            elif vectors.shape[1] != self._dimension:
                raise ValueError("Embedding dimension mismatch with FAISS index.")

            self._index.add(np.ascontiguousarray(vectors))
            self._chunks.extend(chunks)

        return len(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[RetrievedChunk]:
        if self._index is None or not self._chunks:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if self._dimension is not None and query.shape[1] != self._dimension:
            raise ValueError("Query embedding dimension mismatch with FAISS index.")

        k = min(top_k, len(self._chunks))
        scores, indices = self._index.search(query, k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    page=chunk.page,
                    metadata=chunk.metadata,
                    score=float(score),
                )
            )
        return results
