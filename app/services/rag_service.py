from collections.abc import Sequence
from typing import Literal

from fastapi import UploadFile

from app.agents.context_selector import AdaptiveContextSelectionAgent
from app.agents.generator import GeneratorAgent
from app.agents.planner import PlannerAgent
from app.agents.query_analyzer import QueryAnalyzerAgent
from app.agents.verifier import VerificationAgent
from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.pdf_parser import PDFParser
from app.processing.text_chunker import TextChunker
from app.retrieval.embedding_service import EmbeddingService
from app.retrieval.faiss_store import FaissStore
from app.models.document import RetrievedChunk
from app.schemas.rag import AskResponse, ContextChunk, IndexResponse, RetrieveResponse

logger = get_logger(__name__)


class MultiAgentRAGService:
    def __init__(self) -> None:
        self.query_analyzer = QueryAnalyzerAgent()
        self.planner = PlannerAgent()
        self.embedder = EmbeddingService()
        self.retriever = FaissStore()
        self.parser = PDFParser()
        self.chunker = TextChunker(
            chunk_size_words=settings.chunk_size_words,
            overlap_words=settings.chunk_overlap_words,
            min_words=settings.min_chunk_words,
        )
        self.selector = AdaptiveContextSelectionAgent()
        self.generator = GeneratorAgent()
        self.verifier = VerificationAgent()

    def _retrieve_chunks(
        self, question: str, top_k: int | None = None
    ) -> tuple[str, list[RetrievedChunk]]:
        normalized = self.query_analyzer.analyze(question)
        if self.retriever.total_chunks == 0:
            return normalized, []

        query_vector = self.embedder.embed([normalized])[0]
        retrieval_k = top_k or settings.top_k
        candidates = self.retriever.search(query_vector, top_k=retrieval_k)
        return normalized, candidates

    def index_documents(self, files: Sequence[UploadFile], reset: bool = False) -> IndexResponse:
        if reset:
            self.retriever.reset()

        all_chunks = []
        indexed_files = 0
        sources: list[str] = []

        for uploaded_file in files:
            filename = uploaded_file.filename or "uploaded.pdf"
            if not filename.lower().endswith(".pdf"):
                logger.info("Skipping non-PDF file: %s", filename)
                continue

            file_bytes = uploaded_file.file.read()
            pages = self.parser.extract_pages(file_bytes)
            chunks = self.chunker.chunk_pages(source=filename, pages=pages)

            if not chunks:
                logger.info("No chunks generated for file: %s", filename)
                continue

            indexed_files += 1
            sources.append(filename)
            all_chunks.extend(chunks)

        if not all_chunks:
            return IndexResponse(
                indexed_files=0,
                indexed_chunks=0,
                total_chunks=self.retriever.total_chunks,
                sources=[],
            )

        embeddings = self.embedder.embed([chunk.text for chunk in all_chunks])
        indexed_chunk_count = self.retriever.add(all_chunks, embeddings)

        return IndexResponse(
            indexed_files=indexed_files,
            indexed_chunks=indexed_chunk_count,
            total_chunks=self.retriever.total_chunks,
            sources=sources,
        )

    def retrieve(self, question: str, top_k: int | None = None) -> RetrieveResponse:
        normalized, candidates = self._retrieve_chunks(question, top_k=top_k)
        if not candidates:
            return RetrieveResponse(question=normalized, retrieved_context_count=0, contexts=[])
        contexts = [
            ContextChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source=chunk.source,
                page=chunk.page,
                score=chunk.score,
            )
            for chunk in candidates
        ]
        return RetrieveResponse(
            question=normalized,
            retrieved_context_count=len(candidates),
            contexts=contexts,
        )

    def ask(
        self,
        question: str,
        mode: Literal["baseline", "multi_agent"] = "multi_agent",
        top_k: int | None = None,
    ) -> AskResponse:
        normalized, candidates = self._retrieve_chunks(question, top_k=top_k)
        if not candidates:
            return AskResponse(
                question=normalized,
                mode=mode,
                answer="Knowledge base is empty. Upload PDFs before asking questions.",
                llm_provider=self.generator.last_used_provider,
                verified=False,
                confidence=0.0,
                retrieved_context_count=0,
                selected_context_count=0,
                contexts=[],
            )

        if mode == "baseline":
            selected = candidates
        else:
            self.planner.plan(normalized)
            selected = self.selector.select(candidates)

        answer = self.generator.generate(normalized, selected)
        verified, confidence = self.verifier.verify(answer, selected)

        if mode == "multi_agent" and not verified and selected:
            retry_context = selected[: max(1, min(2, len(selected)))]
            retry_answer = self.generator.generate(normalized, retry_context)
            retry_verified, retry_confidence = self.verifier.verify(retry_answer, retry_context)
            if retry_confidence > confidence:
                selected = retry_context
                answer = retry_answer
                verified = retry_verified
                confidence = retry_confidence

        response_contexts = [
            ContextChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source=chunk.source,
                page=chunk.page,
                score=chunk.score,
            )
            for chunk in selected
        ]

        return AskResponse(
            question=normalized,
            mode=mode,
            answer=answer,
            llm_provider=self.generator.last_used_provider,
            verified=verified,
            confidence=confidence,
            retrieved_context_count=len(candidates),
            selected_context_count=len(selected),
            contexts=response_contexts,
        )
