from dataclasses import dataclass
import re

from app.models.document import RetrievedChunk


@dataclass(slots=True)
class ContextSelectionResult:
    selected_chunks: list[RetrievedChunk]
    dynamic_top_k: int
    removed_as_redundant: int
    score_threshold: float


class AdaptiveContextSelectionAgent:
    """Selects useful retrieved chunks using dynamic thresholding and redundancy control."""

    def __init__(self, min_top_k: int = 2, max_top_k: int = 6) -> None:
        self.min_top_k = min_top_k
        self.max_top_k = max_top_k

    def select(self, chunks: list[RetrievedChunk]) -> ContextSelectionResult:
        if not chunks:
            return ContextSelectionResult(
                selected_chunks=[],
                dynamic_top_k=0,
                removed_as_redundant=0,
                score_threshold=0.0,
            )

        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        top_score = max(sorted_chunks[0].score, 1e-6)
        score_threshold = top_score * 0.72

        selected: list[RetrievedChunk] = []
        removed_as_redundant = 0
        dynamic_top_k = min(self.max_top_k, max(self.min_top_k, len(sorted_chunks) // 2))

        for chunk in sorted_chunks:
            if chunk.score < score_threshold and len(selected) >= self.min_top_k:
                continue
            if self._is_redundant(chunk, selected):
                removed_as_redundant += 1
                continue
            selected.append(chunk)
            if len(selected) >= dynamic_top_k:
                break

        if len(selected) < self.min_top_k:
            for chunk in sorted_chunks:
                if chunk in selected:
                    continue
                if self._is_redundant(chunk, selected):
                    removed_as_redundant += 1
                    continue
                selected.append(chunk)
                if len(selected) >= self.min_top_k:
                    break

        return ContextSelectionResult(
            selected_chunks=selected,
            dynamic_top_k=dynamic_top_k,
            removed_as_redundant=removed_as_redundant,
            score_threshold=score_threshold,
        )

    def _is_redundant(self, candidate: RetrievedChunk, selected: list[RetrievedChunk]) -> bool:
        candidate_terms = self._token_set(candidate.text)
        if not candidate_terms:
            return False
        for chosen in selected:
            chosen_terms = self._token_set(chosen.text)
            if not chosen_terms:
                continue
            overlap = len(candidate_terms & chosen_terms) / len(candidate_terms | chosen_terms)
            if overlap >= 0.82:
                return True
        return False

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
