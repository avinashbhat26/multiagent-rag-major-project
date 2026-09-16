from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "multi-agent-rag"
    app_env: str = "development"
    openai_api_key: str = ""
    openai_base_url: str = ""
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_backend: str = "sentence_transformer"
    embedding_dimension: int = 384
    llm_provider: str = "extractive"
    llm_model: str = "gpt-4o-mini"
    top_k: int = 5
    retrieval_pool_multiplier: int = 3
    simple_query_pool_multiplier: int = 2
    complex_query_pool_multiplier: int = 4
    chunk_size_words: int = 220
    chunk_overlap_words: int = 40
    min_chunk_words: int = 40
    verify_threshold: float = 0.75
    enable_reranking: bool = True
    reranker_backend: str = "auto"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = 16
    rerank_min_candidates: int = 6
    rerank_score_gap_threshold: float = 0.08
    embedding_cache_enabled: bool = True
    embedding_cache_size: int = 128
    knowledge_base_storage_dir: str = ".rag_data/knowledge_bases"
    claim_support_threshold: float = 0.62
    claim_weak_support_threshold: float = 0.42
    cors_allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
