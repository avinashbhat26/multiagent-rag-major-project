from app.models.document import DocumentChunk


class TextChunker:
    """Splits page text into overlapping word windows for retrieval."""

    def __init__(
        self, chunk_size_words: int = 220, overlap_words: int = 40, min_words: int = 40
    ) -> None:
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words
        self.min_words = min_words

    def chunk_pages(self, source: str, pages: list[tuple[int, str]]) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []

        for page_number, page_text in pages:
            words = page_text.split()
            if not words:
                continue

            step = max(1, self.chunk_size_words - self.overlap_words)
            chunk_counter = 0
            for start in range(0, len(words), step):
                chunk_words = words[start : start + self.chunk_size_words]
                if len(chunk_words) < self.min_words and chunk_counter > 0:
                    continue

                chunk_counter += 1
                chunk_text = " ".join(chunk_words).strip()
                if not chunk_text:
                    continue

                chunk_id = f"{source}:p{page_number}:c{chunk_counter}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source=source,
                        page=page_number,
                    )
                )

        return chunks
