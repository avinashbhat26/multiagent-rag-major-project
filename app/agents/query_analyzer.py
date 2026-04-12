class QueryAnalyzerAgent:
    """Normalizes and lightly analyzes incoming user queries."""

    def analyze(self, question: str) -> str:
        return question.strip()
