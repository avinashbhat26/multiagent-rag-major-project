from dataclasses import dataclass
import re

import numpy as np

from app.core.config import settings
from app.models.document import RetrievedChunk
from app.retrieval.embedding_service import EmbeddingService


@dataclass(slots=True)
class ClaimVerification:
    claim: str
    status: str
    support_score: float
    lexical_support: float
    semantic_support: float
    supporting_chunk_id: str
    source: str
    page: int


@dataclass(slots=True)
class VerificationResult:
    verified: bool
    confidence: float
    semantic_similarity: float
    supported_claims: list[str]
    unsupported_claims: list[str]
    claim_verifications: list[ClaimVerification]


class VerificationAgent:
    """Evidence-grounded verifier using claim support heuristics."""

    def __init__(self, embedder: EmbeddingService) -> None:
        self.embedder = embedder

    def verify(
        self, answer: str, context: list[RetrievedChunk], threshold: float
    ) -> VerificationResult:
        if not context:
            return VerificationResult(
                verified=False,
                confidence=0.0,
                semantic_similarity=0.0,
                supported_claims=[],
                unsupported_claims=[answer.strip()] if answer.strip() else [],
                claim_verifications=[],
            )

        claims = self._extract_claims(answer)
        if not claims:
            claims = [answer.strip()] if answer.strip() else []

        supported: list[str] = []
        unsupported: list[str] = []
        weakly_supported: list[str] = []
        claim_verifications: list[ClaimVerification] = []

        for claim in claims:
            verification = self._verify_claim(claim, context)
            if not verification:
                continue
            claim_verifications.append(verification)
            if verification.status == "supported":
                supported.append(claim)
            elif verification.status == "weakly_supported":
                weakly_supported.append(claim)
            else:
                unsupported.append(claim)

        total_claims = max(1, len(supported) + len(weakly_supported) + len(unsupported))
        support_ratio = len(supported) / total_claims
        weak_support_ratio = len(weakly_supported) / total_claims
        semantic_similarity = self._semantic_similarity(
            answer=answer,
            context_text=" ".join(chunk.text for chunk in context),
        )
        avg_claim_support = (
            sum(item.support_score for item in claim_verifications) / len(claim_verifications)
            if claim_verifications
            else 0.0
        )
        unsupported_penalty = min(0.35, len(unsupported) * 0.12)
        confidence = min(
            1.0,
            max(
                0.0,
                (0.45 * support_ratio)
                + (0.15 * weak_support_ratio)
                + (0.25 * avg_claim_support)
                + (0.15 * semantic_similarity)
                - unsupported_penalty,
            ),
        )
        verified = confidence >= threshold and len(unsupported) == 0

        return VerificationResult(
            verified=verified,
            confidence=confidence,
            semantic_similarity=semantic_similarity,
            supported_claims=supported,
            unsupported_claims=unsupported,
            claim_verifications=claim_verifications,
        )

    def _verify_claim(self, claim: str, context: list[RetrievedChunk]) -> ClaimVerification | None:
        claim_terms = self._terms(claim)
        if not claim_terms:
            return None

        best: ClaimVerification | None = None
        claim_vector = self.embedder.embed_query(claim)
        for chunk in context:
            lexical_support = self._lexical_support(claim_terms, chunk.text)
            semantic_support = self._vector_similarity(
                claim_vector, self.embedder.embed([chunk.text])[0]
            )
            support_score = (0.6 * lexical_support) + (0.4 * semantic_support)
            if support_score >= settings.claim_support_threshold:
                status = "supported"
            elif support_score >= settings.claim_weak_support_threshold:
                status = "weakly_supported"
            else:
                status = "unsupported"
            current = ClaimVerification(
                claim=claim,
                status=status,
                support_score=round(support_score, 4),
                lexical_support=round(lexical_support, 4),
                semantic_support=round(semantic_support, 4),
                supporting_chunk_id=chunk.chunk_id,
                source=chunk.source,
                page=chunk.page,
            )
            if best is None or current.support_score > best.support_score:
                best = current
        return best

    def _lexical_support(self, claim_terms: list[str], chunk_text: str) -> float:
        chunk_terms = set(self._terms(chunk_text))
        if not claim_terms or not chunk_terms:
            return 0.0
        return len(set(claim_terms) & chunk_terms) / len(set(claim_terms))

    def _semantic_similarity(self, answer: str, context_text: str) -> float:
        if not answer.strip() or not context_text.strip():
            return 0.0
        vectors = self.embedder.embed([answer, context_text])
        answer_vec = vectors[0]
        context_vec = vectors[1]
        numerator = float(np.dot(answer_vec, context_vec))
        denominator = float(np.linalg.norm(answer_vec) * np.linalg.norm(context_vec))
        if denominator == 0:
            return 0.0
        cosine = numerator / denominator
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    def _vector_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        numerator = float(np.dot(left, right))
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator == 0:
            return 0.0
        cosine = numerator / denominator
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    @staticmethod
    def _extract_claims(answer: str) -> list[str]:
        parts = re.split(r"[.\n;]+", answer)
        return [
            part.strip() for part in parts if VerificationAgent._is_meaningful_claim(part.strip())
        ]

    @staticmethod
    def _is_meaningful_claim(claim: str) -> bool:
        if len(claim) < 12:
            return False
        if re.fullmatch(r"[\d\W]+", claim):
            return False
        tokens = re.findall(r"[a-zA-Z]+", claim)
        return len(tokens) >= 3

    @staticmethod
    def _terms(text: str) -> list[str]:
        stop = {
            "the",
            "is",
            "are",
            "a",
            "an",
            "of",
            "to",
            "for",
            "and",
            "in",
            "on",
            "with",
            "that",
            "this",
        }
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return [token for token in tokens if token not in stop]
