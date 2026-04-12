from __future__ import annotations

from dataclasses import dataclass
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
        top_chunks = context[: min(2, len(context))]
        evidence = " ".join(chunk.text for chunk in top_chunks)
        return f"Based on retrieved evidence, the answer to '{question}' is: {evidence[:500]}"


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
