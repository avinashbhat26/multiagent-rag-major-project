from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from time import perf_counter
from typing import Literal
import uuid

from fastapi import UploadFile

from app.agents.context_selector import AdaptiveContextSelectionAgent
from app.agents.generator import GeneratorAgent
from app.agents.planner import QueryPlan, PlannerAgent
from app.agents.query_analyzer import QueryAnalysis, QueryAnalyzerAgent
from app.agents.reranker import RerankerAgent
from app.agents.retriever import RetrieverAgent
from app.agents.verifier import VerificationAgent, VerificationResult
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
    CreateKnowledgeBaseRequest,
    IndexResponse,
    IndexedDocument,
    KnowledgeBaseListResponse,
    KnowledgeBaseSummary,
    PipelineTimings,
    RagStatusResponse,
    RetrieveResponse,
)

logger = get_logger(__name__)

DEFAULT_KNOWLEDGE_BASE_ID = "default"
DEFAULT_KNOWLEDGE_BASE_NAME = "Default Knowledge Base"


@dataclass(slots=True)
class KnowledgeBaseState:
    knowledge_base_id: str
    name: str
    store: FaissStore = field(default_factory=FaissStore)
    documents: list[IndexedDocument] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MultiAgentRAGService:
    def __init__(self, storage_dir: str | Path | None = None) -> None:
        self.query_analyzer = QueryAnalyzerAgent()
        self.planner = PlannerAgent()
        self.embedder = EmbeddingService()
        self._storage_dir = Path(storage_dir or settings.knowledge_base_storage_dir)
        self._knowledge_bases: dict[str, KnowledgeBaseState] = {
            DEFAULT_KNOWLEDGE_BASE_ID: KnowledgeBaseState(
                knowledge_base_id=DEFAULT_KNOWLEDGE_BASE_ID,
                name=DEFAULT_KNOWLEDGE_BASE_NAME,
            )
        }
        self.active_knowledge_base_id = DEFAULT_KNOWLEDGE_BASE_ID
        self._load_knowledge_bases()
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

    @property
    def store(self) -> FaissStore:
        return self._active_kb().store

    def list_knowledge_bases(self) -> KnowledgeBaseListResponse:
        return KnowledgeBaseListResponse(
            active_knowledge_base_id=self.active_knowledge_base_id,
            knowledge_bases=[self._kb_summary(kb) for kb in self._knowledge_bases.values()],
        )

    def create_knowledge_base(self, payload: CreateKnowledgeBaseRequest) -> KnowledgeBaseSummary:
        kb_id = self._unique_kb_id(payload.name)
        kb = KnowledgeBaseState(knowledge_base_id=kb_id, name=payload.name.strip())
        self._knowledge_bases[kb_id] = kb
        self.active_knowledge_base_id = kb_id
        self._persist_knowledge_bases()
        return self._kb_summary(kb)

    def select_knowledge_base(self, knowledge_base_id: str) -> RagStatusResponse:
        self._get_kb(knowledge_base_id)
        self.active_knowledge_base_id = knowledge_base_id
        self._persist_knowledge_bases()
        return self.status(knowledge_base_id=knowledge_base_id)

    def rename_knowledge_base(self, knowledge_base_id: str, name: str) -> KnowledgeBaseSummary:
        kb = self._get_kb(knowledge_base_id)
        kb.name = name.strip()
        kb.updated_at = datetime.now(timezone.utc).isoformat()
        self._persist_knowledge_bases()
        return self._kb_summary(kb)

    def delete_knowledge_base(self, knowledge_base_id: str) -> KnowledgeBaseListResponse:
        if knowledge_base_id == DEFAULT_KNOWLEDGE_BASE_ID:
            self.reset(knowledge_base_id)
            return self.list_knowledge_bases()
        self._get_kb(knowledge_base_id)
        del self._knowledge_bases[knowledge_base_id]
        self._delete_knowledge_base_storage(knowledge_base_id)
        if self.active_knowledge_base_id == knowledge_base_id:
            self.active_knowledge_base_id = DEFAULT_KNOWLEDGE_BASE_ID
        self._persist_knowledge_bases()
        return self.list_knowledge_bases()

    def status(self, knowledge_base_id: str | None = None) -> RagStatusResponse:
        kb = self._resolve_kb(knowledge_base_id)
        return RagStatusResponse(
            active_knowledge_base_id=kb.knowledge_base_id,
            active_knowledge_base_name=kb.name,
            total_chunks=kb.store.total_chunks,
            indexed_documents=kb.documents,
            knowledge_bases=[self._kb_summary(item) for item in self._knowledge_bases.values()],
            llm_provider=self.generator.last_used_provider,
            embedding_backend=self.embedder.backend,
            reranking_enabled=settings.enable_reranking,
        )

    def reset(self, knowledge_base_id: str | None = None) -> RagStatusResponse:
        kb = self._resolve_kb(knowledge_base_id)
        kb.store.reset()
        kb.documents = []
        kb.updated_at = datetime.now(timezone.utc).isoformat()
        self._persist_knowledge_bases()
        return self.status(kb.knowledge_base_id)

    def index_documents(
        self,
        files: Sequence[UploadFile],
        reset: bool = False,
        knowledge_base_id: str | None = None,
    ) -> IndexResponse:
        kb = self._resolve_kb(knowledge_base_id)
        if reset:
            self.reset(kb.knowledge_base_id)

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
                knowledge_base_id=kb.knowledge_base_id,
                knowledge_base_name=kb.name,
                indexed_files=0,
                indexed_chunks=0,
                total_chunks=kb.store.total_chunks,
                sources=[],
            )

        embeddings = self.embedder.embed([chunk.text for chunk in all_chunks])
        indexed_chunk_count = kb.store.add(all_chunks, embeddings)
        indexed_at = datetime.now(timezone.utc).isoformat()
        for source in sources:
            kb.documents.append(
                IndexedDocument(
                    document_id=str(uuid.uuid4()),
                    source=source,
                    chunk_count=chunk_counts[source],
                    indexed_at=indexed_at,
                )
            )
        kb.updated_at = indexed_at
        self._persist_knowledge_bases()

        return IndexResponse(
            knowledge_base_id=kb.knowledge_base_id,
            knowledge_base_name=kb.name,
            indexed_files=indexed_files,
            indexed_chunks=indexed_chunk_count,
            total_chunks=kb.store.total_chunks,
            sources=sources,
        )

    def retrieve(
        self, question: str, top_k: int | None = None, knowledge_base_id: str | None = None
    ) -> RetrieveResponse:
        kb = self._resolve_kb(knowledge_base_id)
        analysis = self.query_analyzer.analyze(question)
        retrieval_k = self._candidate_k(analysis, top_k or settings.top_k)
        if kb.store.total_chunks == 0:
            return RetrieveResponse(
                knowledge_base_id=kb.knowledge_base_id,
                knowledge_base_name=kb.name,
                question=analysis.normalized_question,
                retrieved_context_count=0,
                contexts=[],
            )

        self.retriever.store = kb.store
        candidates = self.retriever.retrieve(analysis.normalized_question, top_k=retrieval_k)
        return RetrieveResponse(
            knowledge_base_id=kb.knowledge_base_id,
            knowledge_base_name=kb.name,
            question=analysis.normalized_question,
            retrieved_context_count=len(candidates),
            contexts=[self._context_schema(chunk) for chunk in candidates],
        )

    def ask(
        self,
        question: str,
        mode: Literal["baseline", "multi_agent"] = "multi_agent",
        top_k: int | None = None,
        knowledge_base_id: str | None = None,
    ) -> AskResponse:
        kb = self._resolve_kb(knowledge_base_id)
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
        retrieval_candidate_k = (
            top_k_value if mode == "baseline" else self._candidate_k(analysis, top_k_value)
        )
        if kb.store.total_chunks == 0:
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
                knowledge_base=kb,
            )

        start = perf_counter()
        self.retriever.store = kb.store
        candidates = self.retriever.retrieve(
            analysis.normalized_question, top_k=retrieval_candidate_k
        )
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
            ) = self._recover_and_regenerate(
                analysis, selected, verification, top_k_value, answer, kb
            )
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
            knowledge_base_id=kb.knowledge_base_id,
            knowledge_base_name=kb.name,
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
        knowledge_base: KnowledgeBaseState,
    ) -> tuple[str, list[RetrievedChunk], VerificationResult, bool, str, int, bool]:
        recovery_query = (
            verification.unsupported_claims[0]
            if verification.unsupported_claims
            else analysis.normalized_question
        )
        self.retriever.store = knowledge_base.store
        recovery_candidates = self.retriever.retrieve(
            recovery_query, top_k=max(top_k_value + 2, settings.top_k + 2)
        )
        rerank = self.reranker.rerank(
            recovery_query, recovery_candidates, top_k=min(top_k_value + 2, settings.top_k + 2)
        )
        selection = self.selector.select(rerank.reranked_chunks, query_terms=analysis.keywords)
        known_ids = {chunk.chunk_id for chunk in selected}
        new_evidence = [
            chunk for chunk in selection.selected_chunks if chunk.chunk_id not in known_ids
        ]
        if not new_evidence:
            return answer, selected, verification, True, recovery_query, 0, False

        recovered_context = selected + new_evidence[: max(1, min(3, top_k_value // 2))]
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
        knowledge_base: KnowledgeBaseState,
    ) -> AskResponse:
        return AskResponse(
            knowledge_base_id=knowledge_base.knowledge_base_id,
            knowledge_base_name=knowledge_base.name,
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

    def _resolve_kb(self, knowledge_base_id: str | None) -> KnowledgeBaseState:
        if knowledge_base_id:
            return self._get_kb(knowledge_base_id)
        return self._active_kb()

    def _active_kb(self) -> KnowledgeBaseState:
        return self._get_kb(self.active_knowledge_base_id)

    def _get_kb(self, knowledge_base_id: str) -> KnowledgeBaseState:
        if knowledge_base_id not in self._knowledge_bases:
            raise ValueError(f"Knowledge base not found: {knowledge_base_id}")
        return self._knowledge_bases[knowledge_base_id]

    def _unique_kb_id(self, name: str) -> str:
        base = self._slugify(name)
        kb_id = base
        suffix = 2
        while kb_id in self._knowledge_bases:
            kb_id = f"{base}-{suffix}"
            suffix += 1
        return kb_id

    @staticmethod
    def _kb_summary(kb: KnowledgeBaseState) -> KnowledgeBaseSummary:
        return KnowledgeBaseSummary(
            knowledge_base_id=kb.knowledge_base_id,
            name=kb.name,
            total_chunks=kb.store.total_chunks,
            document_count=len(kb.documents),
            created_at=kb.created_at,
            updated_at=kb.updated_at,
        )

    @staticmethod
    def _slugify(value: str) -> str:
        slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
        slug = "-".join(part for part in slug.split("-") if part)
        return slug or f"knowledge-base-{uuid.uuid4().hex[:8]}"

    def _candidate_k(self, analysis: QueryAnalysis, top_k: int) -> int:
        if analysis.complexity == "simple":
            multiplier = settings.simple_query_pool_multiplier
        elif analysis.complexity == "complex":
            multiplier = settings.complex_query_pool_multiplier
        else:
            multiplier = settings.retrieval_pool_multiplier
        return max(top_k, top_k * max(1, multiplier))

    def _metadata_path(self) -> Path:
        return self._storage_dir / "metadata.json"

    def _kb_storage_dir(self, knowledge_base_id: str) -> Path:
        return self._storage_dir / knowledge_base_id

    def _persist_knowledge_bases(self) -> None:
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "active_knowledge_base_id": self.active_knowledge_base_id,
            "knowledge_bases": [
                {
                    "knowledge_base_id": kb.knowledge_base_id,
                    "name": kb.name,
                    "created_at": kb.created_at,
                    "updated_at": kb.updated_at,
                }
                for kb in self._knowledge_bases.values()
            ],
        }
        self._metadata_path().write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        for kb in self._knowledge_bases.values():
            kb_dir = self._kb_storage_dir(kb.knowledge_base_id)
            kb.store.save(kb_dir / "index.faiss", kb_dir / "chunks.json")
            documents = [document.model_dump() for document in kb.documents]
            (kb_dir / "documents.json").write_text(
                json.dumps(documents, indent=2), encoding="utf-8"
            )

    def _load_knowledge_bases(self) -> None:
        metadata_path = self._metadata_path()
        if not metadata_path.exists():
            return

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            loaded: dict[str, KnowledgeBaseState] = {}
            for row in metadata.get("knowledge_bases", []):
                kb_id = row["knowledge_base_id"]
                kb = KnowledgeBaseState(
                    knowledge_base_id=kb_id,
                    name=row.get("name") or kb_id,
                    created_at=row.get("created_at") or datetime.now(timezone.utc).isoformat(),
                    updated_at=row.get("updated_at") or datetime.now(timezone.utc).isoformat(),
                )
                kb_dir = self._kb_storage_dir(kb_id)
                kb.store.load(kb_dir / "index.faiss", kb_dir / "chunks.json")
                documents_path = kb_dir / "documents.json"
                if documents_path.exists():
                    document_rows = json.loads(documents_path.read_text(encoding="utf-8"))
                    kb.documents = [
                        IndexedDocument.model_validate(document) for document in document_rows
                    ]
                loaded[kb_id] = kb

            if DEFAULT_KNOWLEDGE_BASE_ID not in loaded:
                loaded[DEFAULT_KNOWLEDGE_BASE_ID] = self._knowledge_bases[DEFAULT_KNOWLEDGE_BASE_ID]
            self._knowledge_bases = loaded
            active_id = metadata.get("active_knowledge_base_id", DEFAULT_KNOWLEDGE_BASE_ID)
            self.active_knowledge_base_id = (
                active_id if active_id in self._knowledge_bases else DEFAULT_KNOWLEDGE_BASE_ID
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Failed to load persisted knowledge bases: %s", exc)

    def _delete_knowledge_base_storage(self, knowledge_base_id: str) -> None:
        kb_dir = self._kb_storage_dir(knowledge_base_id)
        if kb_dir.exists():
            shutil.rmtree(kb_dir)

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
