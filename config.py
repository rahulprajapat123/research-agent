"""
Configuration management for RAG Research Intelligence System
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application
    app_name: str = "RAG Research Intelligence System"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: str = "INFO"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Upstash Redis (Optional)
    upstash_redis_url: Optional[str] = None
    upstash_redis_token: Optional[str] = None
    
    # LLM Providers
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_key: Optional[str] = None
    azure_openai_deployment: Optional[str] = None
    
    # LLM Configuration
    llm_provider: Literal["openai", "anthropic", "azure"] = "openai"
    llm_model: str = "gpt-4-turbo-preview"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4000
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    
    # Object Storage
    storage_type: Literal["s3", "azure", "minio"] = "azure"
    azure_storage_connection_string: Optional[str] = None
    azure_storage_container_name: str = "rag-research-documents"
    # AWS S3 (optional, if needed)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    s3_bucket_name: str = "rag-research-documents"
    
    # Ingestion
    ingestion_schedule_cron: str = "0 0 * * 0"
    max_document_size_mb: int = 50
    allowed_file_types: str = "pdf,html,txt"
    parse_timeout_seconds: int = 300
    
    # Claim Extraction
    claim_extraction_model: str = "gpt-4-turbo-preview"
    claim_extraction_temperature: float = 0.0
    claim_extraction_max_retries: int = 3
    min_claim_confidence: float = 0.6
    human_validation_required: bool = True
    human_validation_sample_rate: float = 0.1
    
    # Source Credibility
    tier_1_boost: int = 10
    tier_2_boost: int = 5
    tier_3_boost: int = 0
    min_citation_count: int = 10
    min_author_h_index: int = 20
    
    # Retrieval
    default_top_k: int = 20
    rerank_top_k: int = 10
    vector_search_weight: float = 0.7
    keyword_search_weight: float = 0.3
    recency_decay_days: int = 180
    
    # n8n Integration
    n8n_webhook_url: str = "https://n8n.example.com/webhook/ingest"
    n8n_validation_webhook_url: str = "https://n8n.example.com/webhook/validation-webhook"
    n8n_api_key: Optional[str] = None
    slack_webhook_url: Optional[str] = None
    
    # Rate Limiting
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    
    # Security
    secret_key: str = "change-this-in-production"
    allowed_origins: str = "*"  # Configure in production with specific domains
    
    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse comma-separated origins into list"""
        return [origin.strip() for origin in self.allowed_origins.split(",")]
    
    @property
    def allowed_file_types_list(self) -> list[str]:
        return [ft.strip() for ft in self.allowed_file_types.split(",")]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Source tier definitions
SOURCE_TIERS = {
    "tier_1": {
        "name": "High Authority",
        "boost": 10,
        "sources": [
            "arxiv.org/cs.IR",
            "arxiv.org/cs.CL",
            "ai.googleblog.com",
            "deepmind.google",
            "anthropic.com/research",
            "ai.meta.com/research"
        ]
    },
    "tier_2": {
        "name": "Industry Validated",
        "boost": 5,
        "sources": [
            "blog.langchain.dev",
            "docs.llamaindex.ai",
            "pinecone.io/learn",
            "weaviate.io/blog",
            "qdrant.tech/blog"
        ]
    },
    "tier_3": {
        "name": "Monitor, Lower Weight",
        "boost": 0,
        "sources": [
            "medium.com",
            "towardsdatascience.com"
        ]
    }
}

# Excluded sources
EXCLUDED_SOURCES = [
    "linkedin.com",
    "twitter.com",
    "x.com"
]
