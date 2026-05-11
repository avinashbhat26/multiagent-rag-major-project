from dataclasses import dataclass

from app.agents.query_analyzer import QueryAnalysis


@dataclass(slots=True)
class PlanStep:
    step_id: int
    description: str


@dataclass(slots=True)
class QueryPlan:
    steps: list[PlanStep]
    sub_queries: list[str]


class PlannerAgent:
    """Builds a simple plan and optional sub-query decomposition."""

    def plan(self, analysis: QueryAnalysis) -> QueryPlan:
        steps = [
            PlanStep(1, "Retrieve candidate evidence chunks."),
            PlanStep(2, "Select non-redundant high-relevance context."),
            PlanStep(3, "Generate an evidence-grounded answer."),
            PlanStep(4, "Verify claims against selected evidence."),
        ]

        sub_queries = [analysis.normalized_question]
        if analysis.is_multi_part and " and " in analysis.normalized_question.lower():
            split_queries = [
                part.strip() for part in analysis.normalized_question.split(" and ") if part.strip()
            ]
            if split_queries:
                sub_queries = split_queries

        return QueryPlan(steps=steps, sub_queries=sub_queries)
