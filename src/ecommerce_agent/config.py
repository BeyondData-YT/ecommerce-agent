from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
  model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
  
  # --- GROQ Configuration ---
  GROQ_API_KEY: str
  GROQ_LLM_MODEL: str = "llama-3.3-70b-versatile"
  GROQ_LLM_MODEL_CONTEXT_SUMMARY: str = "llama-3.1-8b-instant"
  
  # --- Postgres Configuration ---
  POSTGRES_USER: str
  POSTGRES_PASSWORD: str
  POSTGRES_DB: str
  POSTGRES_HOST: str
  POSTGRES_PORT: int = 5435
  
  # --- Vector Database Configuration ---
  VECTOR_DB_NAME: str = "ecommerce_db"
  VECTOR_DB_TABLE: str = "documents"
  
  # --- Embedding Configuration ---
  EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"
  EMBEDDING_DIMENSION: int = 1024
  
  # --- Data Configuration ---
  DATA_DIR: Path = Path(__file__).parent.parent.parent / "data"
  DATA_FAQS_DIR: Path = DATA_DIR / "faqs"

  # --- Splitter Configuration ---
  SMALL_CHUNK_SIZE: int = 150
  SMALL_CHUNK_OVERLAP: int = 20
  WINDOW_SIZE: int = 1000
  WINDOW_OVERLAP: int = 0
  
  # --- Telegram Configuration ---
  TELEGRAM_BOT_TOKEN: str
  WEBHOOK_URL: str
  
  # --- Langfuse Configuration ---
  LANGFUSE_SECRET_KEY: str
  LANGFUSE_PUBLIC_KEY: str
  LANGFUSE_HOST: str

settings = Settings()