import os
import yaml
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str = os.getenv("OPEN_API_KEY")
    embedding_model: str = config.get("EMBEDDING_MODEL")
    embedding_dimensions: int = config.get("EMBEDDING_DIMENSIONS")
    vlm_model: str = config.get("VLM_MODEL")
    generation_model: str = config.get("GENERATION_MODEL")
    intent_model: str = config.get("INTENT_MODEL")

    # Qdrant
    qdrant_host: str = os.getenv("QDRANT_HOST")
    qdrant_port: int = os.getenv("QDRANT_PORT")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY")
    collection_text: str = config.get("COLLECTION_TEXT")
    collection_images: str = config.get("COLLECTION_IMAGES")
    collection_tables: str = config.get("COLLECTION_TABLES")

    # Ingestion
    chunk_max_tokens: int = config.get("CHUNK_MAX_TOKENS")
    chunk_overlap_tokens: int = config.get("CHUNK_OVERLAP_TOKENS")
    bm25_index_path: str = config.get("BM25_INDEX_PATH")
    ingestion_data_dir: str = config.get("INGESTION_DATA_DIR")
    images_dir: str = config.get("IMAGES_DIR")

    # Retrieval
    dense_top_k: int = config.get("DENSE_TOP_K")
    sparse_top_k: int = config.get("SPARSE_TOP_K")
    rerank_top_k: int = config.get("RERANK_TOP_K")
    rrf_k: int = config.get("RRF_K")
    confidence_threshold: float = config.get("CONFIDENCE_THRESHOLD")

    # Logging
    log_level: str = config.get("LOG_LEVEL")
    app_name: str = config.get("APP_NAME")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
