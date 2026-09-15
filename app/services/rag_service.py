from collections.abc import Sequence
from datetime import datetime, timezone
from time import perf_counter
from typing import Literal
import uuid

from fastapi import UploadFile

from app.agents.context_selector import AdaptiveContextSelectionAgent
from app.agents.generator import GeneratorAgent
from app.agents.planner import PlannerAgent, QueryPlan
from app.agents.query_analyzer import QueryAnalysis, QueryAnalyzerAgent
from app.agents.reranker import RerankerAgent
from app.agents.retriever import RetrieverAgent
from app.agents.verifier import VerificationResult, VerificationAgent
from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.pdf_parser import PDFParser
from app.models.document import RetrievedChunk
from app.processing.text_chunker import TextChunker
from app.retrieval.embedding_service import EmbeddingService
from app.retrieval.faiss_store import FaissStore
from app.schemas.rag import (
    AgentTraceStep,
    AskResponse,
    ClaimVerificationSchema,
    ContextChunk,
    IndexResponse,
    IndexedDocument,
    PipelineTimings,
    RagStatusResponse,
    RetrieveResponse,
)

logger = get_logger(__name__)


class MultiAgentRAGService:
    def __init__(self) -> None:
        self.query_analyzer = QueryAnalyzerAgent()
        self.planner = PlannerAgent()
        self.embedder = EmbeddingService()
        self.store = FaissStore()
        self.retriever = RetrieverAgent(self.embedder, self.store)
        self.reranker = RerankerAgent()
        self.parser = PDFParser()
        self.chunker = TextChunker(
            chunk_size_words=settings.chunk_size_words,
            overlap_words=settings.chunk_overlap_words,
            min_words=settings.min_chunk_words,
        )
        self.selector = AdaptiveContextSelectionAgent()
        self.generator = GeneratorAgent()
        self.verifier = VerificationAgent(self.embedder)
        self._documents: list[IndexedDocument] = []

    def status(self) -> RagStatusResponse:
        return RagStatusResponse(
            total_chunks=self.store.total_chunks,
            indexed_documents=self._documents,
            llm_provider=self.generator.last_used_provider,
            embedding_backend=self.embedder.backend,
            reranking_enabled=settings.enable_reranking,
        )

    def reset(self) -> RagStatusResponse:
        self.store.reset()
        self._documents = []
        return self.status()

    def index_documents(self, files: Sequence[UploadFile], reset: bool = False) -> IndexResponse:
        if reset:
            self.reset()

        all_chunks = []
        indexed_files = 0
        sources: list[str] = []
        chunk_counts: dict[str, int] = {}

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
            chunk_counts[filename] = len(chunks)
            all_chunks.extend(chunks)

        if not all_chunks:
            return IndexResponse(
                indexed_files=0,
                indexed_chunks=0,
                total_chunks=self.store.total_chunks,
                sources=[],
            )

        embeddings = self.embedder.embed([chunk.text for chunk in all_chunks])
        indexed_chunk_count = self.store.add(all_chunks, embeddings)
        indexed_at = datetime.now(timezone.utc).isoformat()
        for source in sources:
            self._documents.append(
                IndexedDocument(
                    document_id=str(uuid.uuid4()),
                    source=source,
                    chunk_count=chunk_counts[source],
                    indexed_at=indexed_at,
                )
            )

        return IndexResponse(
            indexed_files=indexed_files,
            indexed_chunks=indexed_chunk_count,
            total_chunks=self.store.total_chunks,
            sources=sources,
        )

    def retrieve(self, question: str, top_k: int | None = None) -> RetrieveResponse:
        analysis = self.query_analyzer.analyze(question)
        retrieval_k = self._candidate_k(analysis, top_k or settings.top_k)
        if self.store.total_chunks == 0:
            return RetrieveResponse(
                question=analysis.normalized_question,
                retrieved_context_count=0,
                contexts=[],
            )
        query_vector = self.embedder.embed_query(analysis.normalized_question)
        candidates = self.retriever.retrieve_by_embedding(query_vector, top_k=retrieval_k)
        return RetrieveResponse(
            question=analysis.normalized_question,
            retrieved_context_count=len(candidates),
            contexts=[self._context_schema(chunk) for chunk in candidates],
        )

    def ask(
        self,
        question: str,
        mode: Literal["baseline", "multi_agent"] = "multi_agent",
        top_k: int | None = None,
    ) -> AskResponse:
        total_start = perf_counter()
        timings = PipelineTimings()
        trace: list[AgentTraceStep] = []

        start = perf_counter()
        analysis = self.query_analyzer.analyze(question)
        timings.query_analysis_ms = self._elapsed(start)
        trace.append(
            AgentTraceStep(
                agent="Query Analyzer",
                status="completed",
                duration_ms=timings.query_analysis_ms,
                detail=f"intent={analysis.detected_intent}; complexity={analysis.complexity}",
            )
        )

        plan: QueryPlan | None = None
        if mode == "multi_agent":
            start = perf_counter()
            plan = self.planner.plan(analysis)
            timings.planning_ms = self._elapsed(start)
            trace.append(
                AgentTraceStep(
                    agent="Planner",
                    status="completed",
                    duration_ms=timings.planning_ms,
                    detail=f"steps={len(plan.steps)}; sub_queries={len(plan.sub_queries)}",
                )
            )

        top_k_value = top_k or settings.top_k
        retrieval_candidate_k = self._candidate_k(analysis, top_k_value)
        if self.store.total_chunks == 0:
            timings.total_ms = self._elapsed(total_start)
            trace.append(
                AgentTraceStep(
                    agent="Retriever", status="skipped", duration_ms=0.0, detail="empty_index"
                )
            )
            return self._empty_response(
                analysis=analysis,
                mode=mode,
                timings=timings,
                trace=trace,
                retrieval_candidate_k=retrieval_candidate_k,
            )

        start = perf_counter()
        query_vector = self.embedder.embed_query(analysis.normalized_question)
        timings.query_embedding_ms = self._elapsed(start)

        start = perf_counter()
        candidates = self.retriever.retrieve_by_embedding(query_vector, top_k=retrieval_candidate_k)
        timings.retrieval_ms = self._elapsed(start)
        trace.append(
            AgentTraceStep(
                agent="Retriever",
                status="completed",
                duration_ms=timings.retrieval_ms,
                detail=f"candidates={len(candidates)}; requested_k={retrieval_candidate_k}",
            )
        )

        selected = candidates[:top_k_value]
        evidence_coverage_score = 0.0
        reranking_gain = 0.0
        reranking_applied = False
        reranking_reason = "baseline_mode"
        dynamic_top_k = len(selected)
        removed_redundant_chunks = 0
        redundancy_ratio = 0.0
        evidence_sufficiency_score = 0.0
        selection_stop_reason = "fixed_top_k_baseline"
        retrieved_after_reranking_count = len(selected)

        if mode == "multi_agent":
            reranked_candidates = candidates
            start = perf_counter()
            rerank = self.reranker.rerank(
                analysis.normalized_question, candidates, top_k=top_k_value
            )
            timings.reranking_ms = self._elapsed(start)
            reranked_candidates = rerank.reranked_chunks
            evidence_coverage_score = rerank.evidence_coverage_score
            reranking_gain = rerank.reranking_gain
            reranking_applied = rerank.reranking_applied
            reranking_reason = rerank.reranking_reason
            retrieved_after_reranking_count = len(reranked_candidates)
            trace.append(
                AgentTraceStep(
                    agent="Reranker",
                    status="completed" if reranking_applied else "skipped",
                    duration_ms=timings.reranking_ms,
                    detail=reranking_reason,
                )
            )

            start = perf_counter()
            selection = self.selector.select(reranked_candidates, query_terms=analysis.keywords)
            timings.context_selection_ms = self._elapsed(start)
            selected = selection.selected_chunks
            dynamic_top_k = selection.dynamic_top_k
            removed_redundant_chunks = selection.removed_as_redundant
            redundancy_ratio = selection.redundancy_ratio
            evidence_sufficiency_score = selection.evidence_sufficiency_score
            selection_stop_reason = selection.selection_stop_reason
            trace.append(
                AgentTraceStep(
                    agent="Adaptive Context Selector",
                    status="completed",
                    duration_ms=timings.context_selection_ms,
                    detail=f"{len(reranked_candidates)} candidates -> {len(selected)} selected; {selection_stop_reason}",
                )
            )

        start = perf_counter()
        answer = self.generator.generate(analysis.normalized_question, selected)
        timings.generation_ms = self._elapsed(start)
        trace.append(
            AgentTraceStep(
                agent="Generator",
                status="completed",
                duration_ms=timings.generation_ms,
                detail=f"provider={self.generator.last_used_provider}",
            )
        )

        start = perf_counter()
        verification = self.verifier.verify(answer, selected, threshold=settings.verify_threshold)
        timings.verification_ms = self._elapsed(start)
        trace.append(
            AgentTraceStep(
                agent="Verifier",
                status="completed",
                duration_ms=timings.verification_ms,
                detail=f"confidence={verification.confidence:.3f}; verified={verification.verified}",
            )
        )

        evidence_recovery_triggered = False
        recovery_query = ""
        additional_evidence_count = 0
        regenerated = False
        if (
            mode == "multi_agent"
            and verification.confidence < settings.verify_threshold
            and selected
        ):
            start = perf_counter()
            (
                answer,
                selected,
                verification,
                evidence_recovery_triggered,
                recovery_query,
                additional_evidence_count,
                regenerated,
            ) = self._recover_and_regenerate(analysis, selected, verification, top_k_value, answer)
            timings.evidence_recovery_ms = self._elapsed(start)
            trace.append(
                AgentTraceStep(
                    agent="Evidence Recovery",
                    status="completed" if evidence_recovery_triggered else "skipped",
                    duration_ms=timings.evidence_recovery_ms,
                    detail=f"additional_evidence={additional_evidence_count}",
                )
            )

        timings.total_ms = self._elapsed(total_start)
        return AskResponse(
            question=analysis.normalized_question,
            mode=mode,
            answer=answer,
            llm_provider=self.generator.last_used_provider,
            verified=verification.verified,
            confidence=verification.confidence,
            semantic_similarity=verification.semantic_similarity,
            supported_claims=verification.supported_claims,
            unsupported_claims=verification.unsupported_claims,
            claim_verifications=[
                self._claim_schema(item) for item in verification.claim_verifications
            ],
            retrieved_context_count=len(candidates),
            selected_context_count=len(selected),
            retrieved_before_reranking_count=len(candidates),
            retrieved_after_reranking_count=retrieved_after_reranking_count,
            evidence_coverage_score=evidence_coverage_score,
            reranking_gain=reranking_gain,
            reranking_applied=reranking_applied,
            reranking_reason=reranking_reason,
            dynamic_top_k=dynamic_top_k,
            removed_redundant_chunks=removed_redundant_chunks,
            redundancy_ratio=redundancy_ratio,
            evidence_sufficiency_score=evidence_sufficiency_score,
            selection_stop_reason=selection_stop_reason,
            query_complexity=analysis.complexity,
            retrieval_candidate_k=retrieval_candidate_k,
            evidence_recovery_triggered=evidence_recovery_triggered,
            recovery_query=recovery_query,
            additional_evidence_count=additional_evidence_count,
            regenerated=regenerated,
            timings=timings,
            agent_trace=trace,
            contexts=[self._context_schema(chunk) for chunk in selected],
        )

    def _recover_and_regenerate(
        self,
        analysis: QueryAnalysis,
        selected: list[RetrievedChunk],
        verification: VerificationResult,
        top_k_value: int,
        answer: str,
    ) -> tuple[str, list[RetrievedChunk], VerificationResult, bool, str, int, bool]:
        if not verification.unsupported_claims:
            return answer, selected, verification, False, "", 0, False

        recovery_query = verification.unsupported_claims[0]
        query_vector = self.embedder.embed_query(recovery_query)
        recovery_candidates = self.retriever.retrieve_by_embedding(
            query_vector, top_k=max(2, top_k_value)
        )
        rerank = self.reranker.rerank(
            recovery_query, recovery_candidates, top_k=min(3, top_k_value)
        )
        selection = self.selector.select(rerank.reranked_chunks, query_terms=analysis.keywords)
        known_ids = {chunk.chunk_id for chunk in selected}
        new_evidence = [
            chunk for chunk in selection.selected_chunks if chunk.chunk_id not in known_ids
        ]
        if not new_evidence:
            return answer, selected, verification, True, recovery_query, 0, False

        recovered_context = selected + new_evidence[:2]
        retry_answer = self.generator.generate(analysis.normalized_question, recovered_context)
        retry_verification = self.verifier.verify(
            retry_answer,
            recovered_context,
            threshold=settings.verify_threshold,
        )
        if retry_verification.confidence > verification.confidence:
            return (
                retry_answer,
                recovered_context,
                retry_verification,
                True,
                recovery_query,
                len(new_evidence[:2]),
                True,
            )
        return answer, selected, verification, True, recovery_query, len(new_evidence[:2]), False

    def _empty_response(
        self,
        analysis: QueryAnalysis,
        mode: Literal["baseline", "multi_agent"],
        timings: PipelineTimings,
        trace: list[AgentTraceStep],
        retrieval_candidate_k: int,
    ) -> AskResponse:
        return AskResponse(
            question=analysis.normalized_question,
            mode=mode,
            answer="Knowledge base is empty. Upload PDFs before asking questions.",
            llm_provider=self.generator.last_used_provider,
            verified=False,
            confidence=0.0,
            semantic_similarity=0.0,
            supported_claims=[],
            unsupported_claims=[],
            claim_verifications=[],
            retrieved_context_count=0,
            selected_context_count=0,
            retrieved_before_reranking_count=0,
            retrieved_after_reranking_count=0,
            evidence_coverage_score=0.0,
            reranking_gain=0.0,
            reranking_applied=False,
            reranking_reason="empty_index",
            dynamic_top_k=0,
            removed_redundant_chunks=0,
            redundancy_ratio=0.0,
            evidence_sufficiency_score=0.0,
            selection_stop_reason="empty_index",
            query_complexity=analysis.complexity,
            retrieval_candidate_k=retrieval_candidate_k,
            evidence_recovery_triggered=False,
            recovery_query="",
            additional_evidence_count=0,
            regenerated=False,
            timings=timings,
            agent_trace=trace,
            contexts=[],
        )

    def _candidate_k(self, analysis: QueryAnalysis, top_k: int) -> int:
        if analysis.complexity == "simple":
            multiplier = settings.simple_query_pool_multiplier
        elif analysis.complexity == "complex":
            multiplier = settings.complex_query_pool_multiplier
        else:
            multiplier = settings.retrieval_pool_multiplier
        return max(top_k, top_k * max(1, multiplier))

    @staticmethod
    def _elapsed(start: float) -> float:
        return round((perf_counter() - start) * 1000.0, 4)

    @staticmethod
    def _context_schema(chunk: RetrievedChunk) -> ContextChunk:
        return ContextChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            source=chunk.source,
            page=chunk.page,
            score=chunk.score,
        )

    @staticmethod
    def _claim_schema(item) -> ClaimVerificationSchema:  # noqa: ANN001
        return ClaimVerificationSchema(
            claim=item.claim,
            status=item.status,
            support_score=item.support_score,
            lexical_support=item.lexical_support,
            semantic_support=item.semantic_support,
            supporting_chunk_id=item.supporting_chunk_id,
            source=item.source,
            page=item.page,
        )
