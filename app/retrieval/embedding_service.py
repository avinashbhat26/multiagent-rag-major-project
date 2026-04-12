import hashlib

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - handled via runtime fallback
    SentenceTransformer = None  # type: ignore[assignment]

logger = get_logger(__name__)


class EmbeddingService:
    """
    Embedding service with two modes:
    - sentence_transformer: semantic embeddings via SentenceTransformers
    - hash: deterministic local fallback for CI/offline reliability
    """

    def __init__(self, model_name: str | None = None, backend: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self.backend = (backend or settings.embedding_backend).lower()
        self.dimension = settings.embedding_dimension
        self._model: SentenceTransformer | None = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        if self.backend == "sentence_transformer":
            model = self._get_model()
            if model is not None:
                vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                return np.asarray(vectors, dtype=np.float32)

        return np.vstack([self._hash_embedding(text) for text in texts])

    def _get_model(self) -> SentenceTransformer | None:
        if self._model is not None:
            return self._model
        if SentenceTransformer is None:
            logger.warning("SentenceTransformers not importable; using hash embeddings.")
            return None
        try:
            self._model = SentenceTransformer(self.model_name)
            return self._model
        except Exception as exc:  # pragma: no cover - network/model errors are environment specific
            logger.warning("Could not load embedding model '%s': %s", self.model_name, exc)
            return None

    def _hash_embedding(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self.dimension).astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector if norm == 0 else vector / norm
