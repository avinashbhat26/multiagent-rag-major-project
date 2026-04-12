import numpy as np

from app.models.document import DocumentChunk
from app.retrieval.faiss_store import FaissStore


def test_faiss_store_add_and_search() -> None:
    store = FaissStore()
    chunks = [
        DocumentChunk(chunk_id="doc:p1:c1", text="alpha context", source="doc.pdf", page=1),
        DocumentChunk(chunk_id="doc:p1:c2", text="beta context", source="doc.pdf", page=1),
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store.add(chunks, embeddings)

    results = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "doc:p1:c1"
    assert results[0].source == "doc.pdf"
