from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration settings.
    Environment variables override defaults.
    """

    # OpenRouter
    openrouter_api_key: str = ""
    primary_model: str = "openai/gpt-4o-mini"
    fallback_model: str = "openai/gpt-4o-mini"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "production-api"

    # Application
    app_env: str = "development"
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = 300
    max_retries: int = 3

    # Groundwire persistence and dense retrieval
    database_url: str = ""
    groundwire_sqlite_path: str = "groundwire.db"
    groundwire_workspace_id: str = "workspace-default"
    pinecone_api_key: str = ""
    pinecone_index_host: str = ""
    pinecone_index_name: str = "groundwire-chunks"
    pinecone_namespace: str = "workspace-default"
    pinecone_text_field: str = "chunk_text"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance, loaded once and reused everywhere."""
    return Settings()
