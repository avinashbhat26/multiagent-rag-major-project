from app.agents.context_selector import AdaptiveContextSelectionAgent
from app.agents.generator import GeneratorAgent
from app.agents.planner import PlannerAgent
from app.agents.query_analyzer import QueryAnalyzerAgent
from app.agents.verifier import VerificationAgent
from app.retrieval.faiss_store import FaissStore
from app.schemas.rag import AskResponse


class MultiAgentRAGService:
    def __init__(self) -> None:
        self.query_analyzer = QueryAnalyzerAgent()
        self.planner = PlannerAgent()
        self.retriever = FaissStore()
        self.selector = AdaptiveContextSelectionAgent()
        self.generator = GeneratorAgent()
        self.verifier = VerificationAgent()

    def ask(self, question: str) -> AskResponse:
        normalized = self.query_analyzer.analyze(question)
        _plan = self.planner.plan(normalized)
        candidates = self.retriever.search(normalized, top_k=5)
        selected = self.selector.select(candidates)
        answer = self.generator.generate(normalized, selected)
        verified, confidence = self.verifier.verify(answer, selected)
        return AskResponse(
            question=normalized,
            answer=answer,
            verified=verified,
            confidence=confidence,
            selected_context_count=len(selected),
        )
