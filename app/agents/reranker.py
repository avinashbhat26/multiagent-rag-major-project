from __future__ import annotations

from dataclasses import dataclass
import re

from app.core.config import settings
from app.core.logging import get_logger
from app.models.document import RetrievedChunk

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore[assignment]

logger = get_logger(__name__)


@dataclass(slots=True)
class RerankResult:
    reranked_chunks: list[RetrievedChunk]
    evidence_coverage_score: float
    reranking_gain: float
    used_cross_encoder: bool


class RerankerAgent:
    """Optional reranking stage with cross-encoder or lightweight lexical fallback."""

    def __init__(self) -> None:
        self.backend = settings.reranker_backend.lower()
        self.model_name = settings.reranker_model
        self._model: CrossEncoder | None = None

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> RerankResult:
        if not chunks:
            return RerankResult([], 0.0, 0.0, False)

        original_scores = [chunk.score for chunk in chunks[:top_k]]
        used_cross_encoder = False
        reranked = chunks

        if self.backend in {"auto", "cross_encoder"}:
            model = self._get_model()
            if model is not None:
                pairs = [(query, chunk.text) for chunk in chunks]
                ce_scores = model.predict(pairs)
                rescored = []
                for chunk, ce_score in zip(chunks, ce_scores, strict=False):
                    blended = 0.55 * float(chunk.score) + 0.45 * float(ce_score)
                    rescored.append(
                        RetrievedChunk(
                            chunk_id=chunk.chunk_id,
                            text=chunk.text,
                            source=chunk.source,
                            page=chunk.page,
                            metadata=chunk.metadata,
                            score=blended,
                        )
                    )
                reranked = sorted(rescored, key=lambda c: c.score, reverse=True)
                used_cross_encoder = True
            elif self.backend == "cross_encoder":
                logger.warning(
                    "Cross-encoder backend requested but unavailable; using fallback scoring."
                )

        if not used_cross_encoder:
            reranked = self._fallback_rerank(query, chunks)

        top = reranked[:top_k]
        new_scores = [chunk.score for chunk in top]
        gain = (sum(new_scores) - sum(original_scores)) / max(1, len(original_scores))
        coverage = self._coverage_score(query, top)

        return RerankResult(
            reranked_chunks=top,
            evidence_coverage_score=coverage,
            reranking_gain=round(gain, 4),
            used_cross_encoder=used_cross_encoder,
        )

    def _get_model(self) -> CrossEncoder | None:
        if self._model is not None:
            return self._model
        if CrossEncoder is None:
            return None
        try:
            self._model = CrossEncoder(self.model_name)
            return self._model
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not load reranker model '%s': %s", self.model_name, exc)
            return None

    def _fallback_rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        query_terms = self._terms(query)
        rescored: list[RetrievedChunk] = []
        for chunk in chunks:
            text_terms = self._terms(chunk.text)
            overlap = len(query_terms & text_terms) / max(1, len(query_terms))
            blended = 0.7 * chunk.score + 0.3 * overlap
            rescored.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    page=chunk.page,
                    metadata=chunk.metadata,
                    score=blended,
                )
            )
        return sorted(rescored, key=lambda c: c.score, reverse=True)

    def _coverage_score(self, query: str, selected: list[RetrievedChunk]) -> float:
        query_terms = self._terms(query)
        if not query_terms:
            return 0.0
        evidence_terms: set[str] = set()
        for chunk in selected:
            evidence_terms |= self._terms(chunk.text)
        covered = len(query_terms & evidence_terms) / len(query_terms)
        return round(min(1.0, max(0.0, covered)), 4)

    @staticmethod
    def _terms(text: str) -> set[str]:
        stop = {"the", "is", "are", "a", "an", "of", "to", "for", "and", "in", "on", "with"}
        tokens = re.findall(r"[a-zA-Z0-9%]+", text.lower())
        return {token for token in tokens if token not in stop}
