from __future__ import annotations

import json
from typing import Any

import httpx
import streamlit as st

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


def _backend_post(
    backend_url: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    endpoint = f"{backend_url.rstrip('/')}{path}"
    with httpx.Client(timeout=timeout) as client:
        response = client.post(endpoint, json=json_body, files=files)
        response.raise_for_status()
        return response.json()


def _render_contexts(contexts: list[dict[str, Any]]) -> None:
    st.subheader("Retrieved Chunks")
    if not contexts:
        st.info("No chunks retrieved.")
        return

    for idx, chunk in enumerate(contexts, start=1):
        score = float(chunk.get("score", 0.0))
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "-")
        chunk_id = chunk.get("chunk_id", "n/a")
        with st.expander(f"Chunk {idx}: {source} (page {page}) | score={score:.3f}"):
            st.caption(f"ID: {chunk_id}")
            st.write(chunk.get("text", ""))


def _render_verification(response: dict[str, Any]) -> None:
    st.subheader("Verification")
    verified = bool(response.get("verified", False))
    confidence = float(response.get("confidence", 0.0))
    status = "Verified" if verified else "Not Verified"
    if verified:
        st.success(status)
    else:
        st.warning(status)
    st.metric("Confidence Score", f"{confidence:.3f}")


def main() -> None:
    st.set_page_config(
        page_title="Multi-Agent RAG Demo",
        page_icon="📚",
        layout="wide",
    )
    st.title("Multi-Agent RAG Academic Demo")
    st.caption("Upload PDFs, build knowledge base, ask questions, and inspect evidence.")

    with st.sidebar:
        st.header("Configuration")
        backend_url = st.text_input("FastAPI Backend URL", value=DEFAULT_BACKEND_URL)
        mode = st.selectbox("QA Mode", options=["baseline", "multi_agent"], index=0)
        top_k = st.slider("Top-K Retrieval", min_value=1, max_value=20, value=5)
        reset_before_index = st.checkbox("Reset index before indexing", value=True)

    st.subheader("1) Upload Documents")
    uploaded_files = st.file_uploader(
        "Select one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="These files will be sent to /rag/index and chunked into the vector knowledge base.",
    )

    if st.button("Build / Update Knowledge Base", type="primary", use_container_width=True):
        if not uploaded_files:
            st.error("Please upload at least one PDF before indexing.")
        else:
            files_payload: list[tuple[str, tuple[str, bytes, str]]] = []
            for file in uploaded_files:
                files_payload.append(("files", (file.name, file.getvalue(), "application/pdf")))
            try:
                with st.spinner("Indexing documents..."):
                    data = _backend_post(
                        backend_url,
                        f"/rag/index?reset={'true' if reset_before_index else 'false'}",
                        files=files_payload,
                    )
                st.success("Indexing completed successfully.")
                col1, col2, col3 = st.columns(3)
                col1.metric("Indexed Files", int(data.get("indexed_files", 0)))
                col2.metric("Indexed Chunks", int(data.get("indexed_chunks", 0)))
                col3.metric("Total Chunks in Store", int(data.get("total_chunks", 0)))
                sources = data.get("sources", [])
                if sources:
                    st.caption("Sources: " + ", ".join(sources))
                st.session_state["last_index_response"] = data
            except httpx.HTTPStatusError as exc:
                st.error(f"Indexing failed: {exc.response.status_code} - {exc.response.text}")
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    st.divider()
    st.subheader("2) Ask Question")
    question = st.text_area(
        "Enter your question",
        value="What are the attendance rules?",
        height=100,
    )

    ask_col, retr_col = st.columns(2)
    ask_clicked = ask_col.button("Generate Answer", use_container_width=True)
    retr_clicked = retr_col.button("Preview Retrieval Only", use_container_width=True)

    if retr_clicked:
        if not question.strip():
            st.error("Please enter a non-empty question.")
        else:
            try:
                with st.spinner("Retrieving relevant chunks..."):
                    retrieval = _backend_post(
                        backend_url,
                        "/rag/retrieve",
                        json_body={"question": question.strip(), "top_k": int(top_k)},
                    )
                st.info(f"Retrieved {retrieval.get('retrieved_context_count', 0)} chunks.")
                _render_contexts(retrieval.get("contexts", []))
                st.session_state["last_retrieve_response"] = retrieval
            except httpx.HTTPStatusError as exc:
                st.error(f"Retrieval failed: {exc.response.status_code} - {exc.response.text}")
            except Exception as exc:
                st.error(f"Retrieval failed: {exc}")

    if ask_clicked:
        if not question.strip():
            st.error("Please enter a non-empty question.")
        else:
            payload = {"question": question.strip(), "mode": mode, "top_k": int(top_k)}
            try:
                with st.spinner("Generating answer..."):
                    answer_response = _backend_post(backend_url, "/rag/ask", json_body=payload)
                st.subheader("Generated Answer")
                st.write(answer_response.get("answer", ""))
                meta_col1, meta_col2, meta_col3 = st.columns(3)
                meta_col1.metric("Mode", str(answer_response.get("mode", "n/a")))
                meta_col2.metric("LLM Provider", str(answer_response.get("llm_provider", "n/a")))
                meta_col3.metric(
                    "Retrieved Chunks",
                    int(answer_response.get("retrieved_context_count", 0)),
                )
                _render_verification(answer_response)
                _render_contexts(answer_response.get("contexts", []))
                with st.expander("Show Raw JSON Response"):
                    st.code(json.dumps(answer_response, indent=2), language="json")
                st.session_state["last_ask_response"] = answer_response
            except httpx.HTTPStatusError as exc:
                st.error(f"QA request failed: {exc.response.status_code} - {exc.response.text}")
            except Exception as exc:
                st.error(f"QA request failed: {exc}")


if __name__ == "__main__":
    main()
