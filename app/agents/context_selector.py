from dataclasses import dataclass
import math
import re

from app.models.document import RetrievedChunk


@dataclass(slots=True)
class ContextSelectionResult:
    selected_chunks: list[RetrievedChunk]
    dynamic_top_k: int
    removed_as_redundant: int
    score_threshold: float
    evidence_sufficiency_score: float
    selection_stop_reason: str
    redundancy_ratio: float


class AdaptiveContextSelectionAgent:
    """Selects useful retrieved chunks using dynamic thresholding and redundancy control."""

    def __init__(self, min_top_k: int = 3, max_top_k: int = 5) -> None:
        self.min_top_k = min_top_k
        self.max_top_k = max_top_k

    def select(
        self, chunks: list[RetrievedChunk], query_terms: list[str] | None = None
    ) -> ContextSelectionResult:
        if not chunks:
            return ContextSelectionResult(
                selected_chunks=[],
                dynamic_top_k=0,
                removed_as_redundant=0,
                score_threshold=0.0,
                evidence_sufficiency_score=0.0,
                selection_stop_reason="no_candidates",
                redundancy_ratio=0.0,
            )

        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        top_score = max(sorted_chunks[0].score, 1e-6)
        score_threshold = top_score * 0.60

        selected: list[RetrievedChunk] = []
        removed_as_redundant = 0
        dynamic_top_k = min(
            self.max_top_k,
            max(self.min_top_k, math.ceil(len(sorted_chunks) / 3)),
        )
        covered_terms: set[str] = set()
        target_terms = {term.lower() for term in query_terms or []}
        stop_reason = "max_context_reached"

        for chunk in sorted_chunks:
            chunk_terms = self._token_set(chunk.text)
            adds_new_query_evidence = bool(
                target_terms and (chunk_terms & target_terms - covered_terms)
            )
            if (
                chunk.score < score_threshold
                and len(selected) >= self.min_top_k
                and not adds_new_query_evidence
            ):
                stop_reason = "remaining_candidates_below_threshold"
                continue
            if self._is_redundant(chunk, selected) and not adds_new_query_evidence:
                removed_as_redundant += 1
                continue
            selected.append(chunk)
            covered_terms |= chunk_terms
            if (
                target_terms
                and self._coverage(target_terms, covered_terms) >= 0.85
                and len(selected) >= self.min_top_k
            ):
                stop_reason = "coverage_sufficient"
                break
            if len(selected) >= dynamic_top_k:
                stop_reason = "max_context_reached"
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
                    stop_reason = "minimum_context_satisfied"
                    break

        if not selected and removed_as_redundant:
            stop_reason = "all_remaining_redundant"

        sufficiency = self._evidence_sufficiency(target_terms, selected)
        redundancy_ratio = removed_as_redundant / len(sorted_chunks)

        return ContextSelectionResult(
            selected_chunks=selected,
            dynamic_top_k=dynamic_top_k,
            removed_as_redundant=removed_as_redundant,
            score_threshold=score_threshold,
            evidence_sufficiency_score=sufficiency,
            selection_stop_reason=stop_reason,
            redundancy_ratio=round(redundancy_ratio, 4),
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
            if overlap >= 0.90:
                return True
        return False

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))

    def _evidence_sufficiency(
        self, target_terms: set[str], selected: list[RetrievedChunk]
    ) -> float:
        if not target_terms:
            return 0.0
        evidence_terms: set[str] = set()
        for chunk in selected:
            evidence_terms |= self._token_set(chunk.text)
        return round(self._coverage(target_terms, evidence_terms), 4)

    @staticmethod
    def _coverage(target_terms: set[str], evidence_terms: set[str]) -> float:
        if not target_terms:
            return 0.0
        return len(target_terms & evidence_terms) / len(target_terms)
