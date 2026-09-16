from dataclasses import asdict
import json
from pathlib import Path
import re
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

    @property
    def chunks(self) -> list[DocumentChunk]:
        return list(self._chunks)

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

    def lexical_search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Search chunks with keyword coverage for offline/hash embedding mode."""
        query_terms = self._terms(query)
        if not query_terms or not self._chunks:
            return []

        scored: list[RetrievedChunk] = []
        for chunk in self._chunks:
            chunk_terms = self._terms(chunk.text)
            if not chunk_terms:
                continue
            overlap = query_terms & chunk_terms
            if not overlap:
                continue
            coverage = len(overlap) / len(query_terms)
            density = len(overlap) / max(1, len(chunk_terms))
            score = (0.85 * coverage) + (0.15 * density)
            scored.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    page=chunk.page,
                    metadata=chunk.metadata,
                    score=round(score, 4),
                )
            )

        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def save(self, index_path: Path, chunks_path: Path) -> None:
        """Persist FAISS index and chunk metadata to disk."""
        with self._lock:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            chunks_path.parent.mkdir(parents=True, exist_ok=True)
            if self._index is not None:
                faiss.write_index(self._index, str(index_path))
            elif index_path.exists():
                index_path.unlink()

            chunks_path.write_text(
                json.dumps([asdict(chunk) for chunk in self._chunks], indent=2),
                encoding="utf-8",
            )

    def load(self, index_path: Path, chunks_path: Path) -> None:
        """Load FAISS index and chunk metadata from disk if present."""
        if not chunks_path.exists():
            return

        chunk_rows = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [DocumentChunk(**row) for row in chunk_rows]

        with self._lock:
            self._chunks = chunks
            if index_path.exists() and chunks:
                self._index = faiss.read_index(str(index_path))
                self._dimension = self._index.d
            else:
                self._index = None
                self._dimension = None

    @staticmethod
    def _terms(text: str) -> set[str]:
        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "by",
            "for",
            "from",
            "how",
            "in",
            "is",
            "of",
            "on",
            "or",
            "the",
            "there",
            "to",
            "what",
            "when",
            "where",
            "which",
            "who",
            "with",
        }
        tokens = re.findall(r"[a-zA-Z0-9%]+", text.lower())
        return {token for token in tokens if token not in stop_words}
