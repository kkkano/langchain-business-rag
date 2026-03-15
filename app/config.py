from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"环境变量 {name} 必须是整数，当前值为: {value}") from exc


@dataclass(frozen=True)
class Settings:
    app_name: str = "Business RAG QA System"
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "paraphrase-multilingual-MiniLM-L12-v2",
    )
    vector_db_path: Path = BASE_DIR / "storage" / "chroma"
    upload_dir: Path = BASE_DIR / "data" / "uploads"
    sample_docs_dir: Path = BASE_DIR / "data" / "sample_docs"
    chunk_size: int = _env_int("RAG_CHUNK_SIZE", 320)
    chunk_overlap: int = _env_int("RAG_CHUNK_OVERLAP", 60)
    top_k: int = _env_int("RAG_TOP_K", 4)


def get_settings() -> Settings:
    settings = Settings()
    settings.vector_db_path.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.sample_docs_dir.mkdir(parents=True, exist_ok=True)
    return settings
