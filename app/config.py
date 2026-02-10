from __future__ import annotations

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Защита от утечек: секреты читаем только из .env / переменных окружения.
    # .env не коммитим (он уже в .gitignore).
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Telegram
    bot_token: str = Field(..., alias="BOT_TOKEN")
    required_channel: str | None = Field(default=None, alias="REQUIRED_CHANNEL")  # например: @my_channel
    consult_url: str | None = Field(default=None, alias="CONSULT_URL")            # ссылка на запись

    # LLM (Ollama)
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1:8b-instruct", alias="OLLAMA_MODEL")

    # Storage
    sqlite_path: str = Field(default="data/bot.sqlite3", alias="SQLITE_PATH")

    # Limits
    daily_sessions_limit: int = Field(default=3, alias="DAILY_SESSIONS_LIMIT")
    session_max_followups: int = Field(default=6, alias="SESSION_MAX_FOLLOWUPS")
    max_user_message_chars: int = Field(default=2000, alias="MAX_USER_MESSAGE_CHARS")
    max_answer_chars: int = Field(default=4000, alias="MAX_ANSWER_CHARS")

    # RAG
    rag_top_k: int = Field(default=6, alias="RAG_TOP_K")
    rag_chunk_size: int = Field(default=800, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=120, alias="RAG_CHUNK_OVERLAP")
    embedding_model: str = Field(default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", alias="EMBEDDING_MODEL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
