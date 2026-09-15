from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

import httpx

from app.core.config import settings
from app.models.document import RetrievedChunk

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]


class LLMProvider(Protocol):
    """Interface for pluggable answer generation providers."""

    name: str

    def generate(self, question: str, context: list[RetrievedChunk]) -> str:
        """Generate an answer grounded in retrieved context."""


@dataclass(slots=True)
class ExtractiveProvider:
    """Local deterministic fallback provider for testing and offline execution."""

    name: str = "extractive"

    def generate(self, question: str, context: list[RetrievedChunk]) -> str:
        if not context:
            return "No relevant context was found to answer the question."

        best_sentence, source, page = self._best_evidence_sentence(question, context)
        if best_sentence:
            return (
                f"Based on retrieved evidence, the answer to '{question}' is: "
                f"{best_sentence} (Source: {source}, page {page})"
            )

        top_chunks = context[: min(2, len(context))]
        evidence = " ".join(chunk.text for chunk in top_chunks)
        return f"Based on retrieved evidence, the answer to '{question}' is: {evidence[:500]}"

    def _best_evidence_sentence(
        self, question: str, context: list[RetrievedChunk]
    ) -> tuple[str, str, int]:
        question_terms = self._terms(question)
        question_lower = question.lower()
        cue_terms = self._cue_terms(question_lower)

        best: tuple[float, str, str, int] = (0.0, "", "", 0)
        for chunk in context:
            for sentence in self._sentences(chunk.text):
                sentence_terms = self._terms(sentence)
                if not sentence_terms:
                    continue

                overlap = len(question_terms & sentence_terms) / max(1, len(question_terms))
                cue_bonus = 0.15 * len(cue_terms & sentence_terms)
                source_bonus = min(0.15, max(0.0, chunk.score) * 0.15)
                definition_bonus = self._definition_bonus(question_lower, sentence.lower())
                score = overlap + cue_bonus + source_bonus + definition_bonus

                if score > best[0]:
                    best = (score, sentence, chunk.source, chunk.page)

        if best[0] < 0.25:
            return "", "", 0
        return best[1], best[2], best[3]

    @staticmethod
    def _sentences(text: str) -> list[str]:
        normalized = " ".join(text.split())
        parts = re.split(r"(?<=[.!?])\s+", normalized)
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _terms(text: str) -> set[str]:
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
            "which",
            "what",
            "from",
            "word",
        }
        return {token for token in re.findall(r"[a-zA-Z0-9%]+", text.lower()) if token not in stop}

    @staticmethod
    def _cue_terms(question_lower: str) -> set[str]:
        cues = set()
        if any(term in question_lower for term in ["derived", "origin", "etymology"]):
            cues |= {"derived", "origin", "greek", "latin", "word", "means", "meaning"}
        if any(term in question_lower for term in ["minimum", "requirement", "rule"]):
            cues |= {"must", "minimum", "required", "rule", "shall"}
        if any(term in question_lower for term in ["list", "types", "forms"]):
            cues |= {"include", "includes", "following", "types"}
        return cues

    @staticmethod
    def _definition_bonus(question_lower: str, sentence_lower: str) -> float:
        if "derived" in question_lower and any(
            phrase in sentence_lower for phrase in ["derived from", "comes from", "origin"]
        ):
            return 0.5
        if "means" in question_lower and any(
            phrase in sentence_lower for phrase in ["means", "meaning", "defined as"]
        ):
            return 0.35
        return 0.0


@dataclass(slots=True)
class OpenAIProvider:
    """OpenAI chat-completions provider."""

    model: str
    api_key: str
    base_url: str = ""
    name: str = "openai"
    _client: OpenAI | None = None

    def _client_or_raise(self) -> OpenAI:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed.")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        if self._client is None:
            kwargs: dict[str, str] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def generate(self, question: str, context: list[RetrievedChunk]) -> str:
        if not context:
            return "No relevant context was found to answer the question."
        client = self._client_or_raise()
        context_block = "\n\n".join(
            f"[source={chunk.source} page={chunk.page} score={chunk.score:.3f}] {chunk.text}"
            for chunk in context
        )
        system_prompt = (
            "You are a retrieval-augmented QA assistant. Use only the provided context. "
            "If evidence is insufficient, explicitly say so."
        )
        user_prompt = f"Question:\n{question}\n\nContext:\n{context_block}\n\nAnswer:"
        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text.strip()


@dataclass(slots=True)
class OllamaProvider:
    """Ollama provider via local HTTP API."""

    model: str
    base_url: str
    timeout_seconds: float = 60.0
    name: str = "ollama"

    def generate(self, question: str, context: list[RetrievedChunk]) -> str:
        if not context:
            return "No relevant context was found to answer the question."
        context_block = "\n\n".join(
            f"[source={chunk.source} page={chunk.page} score={chunk.score:.3f}] {chunk.text}"
            for chunk in context
        )
        prompt = (
            "Use only the following context to answer the question. "
            "If the answer is not in context, say evidence is insufficient.\n\n"
            f"Question:\n{question}\n\nContext:\n{context_block}\n\nAnswer:"
        )
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url.rstrip('/')}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
        payload = response.json()
        return payload.get("message", {}).get("content", "").strip()


def build_provider(provider_name: str | None = None) -> LLMProvider:
    """Build an LLM provider by name with config-based defaults."""
    target = (provider_name or settings.llm_provider).lower()

    if target == "openai":
        return OpenAIProvider(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    if target == "ollama":
        return OllamaProvider(model=settings.llm_model, base_url=settings.ollama_base_url)
    return ExtractiveProvider()
