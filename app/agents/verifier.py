from dataclasses import dataclass
import re

from app.models.document import RetrievedChunk


@dataclass(slots=True)
class VerificationResult:
    verified: bool
    confidence: float
    supported_claims: list[str]
    unsupported_claims: list[str]


class VerificationAgent:
    """Evidence-grounded verifier using claim support heuristics."""

    def verify(
        self, answer: str, context: list[RetrievedChunk], threshold: float
    ) -> VerificationResult:
        if not context:
            return VerificationResult(
                verified=False,
                confidence=0.0,
                supported_claims=[],
                unsupported_claims=[answer.strip()] if answer.strip() else [],
            )

        claims = self._extract_claims(answer)
        if not claims:
            claims = [answer.strip()] if answer.strip() else []

        context_text = " ".join(chunk.text.lower() for chunk in context)
        supported: list[str] = []
        unsupported: list[str] = []

        for claim in claims:
            claim_terms = self._terms(claim)
            if not claim_terms:
                continue
            overlap_count = sum(1 for token in claim_terms if token in context_text)
            overlap_ratio = overlap_count / len(claim_terms)
            if overlap_ratio >= 0.5:
                supported.append(claim)
            else:
                unsupported.append(claim)

        total_claims = max(1, len(supported) + len(unsupported))
        support_ratio = len(supported) / total_claims
        evidence_bonus = min(0.15, len(context) * 0.02)
        confidence = min(0.99, max(0.0, support_ratio + evidence_bonus))
        verified = confidence >= threshold and len(unsupported) == 0

        return VerificationResult(
            verified=verified,
            confidence=confidence,
            supported_claims=supported,
            unsupported_claims=unsupported,
        )

    @staticmethod
    def _extract_claims(answer: str) -> list[str]:
        parts = re.split(r"[.\n;]+", answer)
        return [part.strip() for part in parts if part.strip()]

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
