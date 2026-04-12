from app.models.document import RetrievedChunk


class AdaptiveContextSelectionAgent:
    """Selects useful retrieved chunks using dynamic thresholding and redundancy control."""

    def select(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        selected: list[RetrievedChunk] = []
        seen = set()
        for chunk in sorted_chunks:
            key = chunk.text[:120]
            if key in seen:
                continue
            seen.add(key)
            selected.append(chunk)
            if len(selected) >= 3 and chunk.score < sorted_chunks[0].score * 0.85:
                break
        return selected
