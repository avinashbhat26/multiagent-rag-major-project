from dataclasses import dataclass
import re


@dataclass(slots=True)
class QueryAnalysis:
    normalized_question: str
    is_multi_part: bool
    detected_intent: str
    keywords: list[str]


class QueryAnalyzerAgent:
    """Normalizes query text and extracts lightweight intent structure."""

    _stop_words = {
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
        "what",
        "how",
        "when",
        "where",
        "why",
        "which",
    }

    def analyze(self, question: str) -> QueryAnalysis:
        normalized = " ".join(question.strip().split())
        lower = normalized.lower()
        is_multi_part = (" and " in lower) or ("; " in lower) or ("?" in normalized[:-1])

        if any(token in lower for token in ["compare", "difference", "versus", "vs"]):
            intent = "comparison"
        elif any(token in lower for token in ["summarize", "summary", "overview"]):
            intent = "summarization"
        else:
            intent = "factoid"

        tokens = re.findall(r"[a-zA-Z0-9]+", lower)
        keywords = [token for token in tokens if token not in self._stop_words]
        keywords = keywords[:8]

        return QueryAnalysis(
            normalized_question=normalized,
            is_multi_part=is_multi_part,
            detected_intent=intent,
            keywords=keywords,
        )
