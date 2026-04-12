from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "multi-agent-rag"
    app_env: str = "development"
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    top_k: int = 5
    verify_threshold: float = 0.75

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
