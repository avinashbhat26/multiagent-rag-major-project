from app.models.document import RetrievedChunk


class VerificationAgent:
    """Simple evidence-grounded verifier placeholder."""

    def verify(self, answer: str, context: list[RetrievedChunk]) -> tuple[bool, float]:
        if not context:
            return False, 0.0
        confidence = min(0.95, 0.55 + 0.1 * len(context))
        return confidence >= 0.75, confidence
