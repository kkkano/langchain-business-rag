from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _first_nonempty_env(*names: str, default: str = "") -> str:
    for name in names:
        value = _env(name)
        if value:
            return value
    return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"环境变量 {name} 必须是整数，当前值为: {value}") from exc


def _detect_llm_provider() -> str:
    explicit_provider = _env("LLM_PROVIDER").lower()
    if explicit_provider in {"deepseek", "openai"}:
        return explicit_provider

    if any(_env(name) for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")):
        return "openai"
    if any(_env(name) for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL")):
        return "deepseek"
    return "deepseek"


LLM_PROVIDER = _detect_llm_provider()


def _default_llm_api_key() -> str:
    if LLM_PROVIDER == "deepseek":
        return _first_nonempty_env("DEEPSEEK_API_KEY", "OPENAI_API_KEY")
    return _first_nonempty_env("OPENAI_API_KEY", "DEEPSEEK_API_KEY")


def _default_llm_base_url() -> str:
    if LLM_PROVIDER == "deepseek":
        return _first_nonempty_env(
            "DEEPSEEK_BASE_URL",
            "OPENAI_BASE_URL",
            default="https://api.deepseek.com",
        )
    return _first_nonempty_env("OPENAI_BASE_URL", "DEEPSEEK_BASE_URL")


def _default_llm_model() -> str:
    if LLM_PROVIDER == "deepseek":
        return _first_nonempty_env("DEEPSEEK_MODEL", "OPENAI_MODEL", default="deepseek-chat")
    return _first_nonempty_env("OPENAI_MODEL", "DEEPSEEK_MODEL", default="gpt-4o-mini")


@dataclass(frozen=True)
class Settings:
    app_name: str = "Business RAG QA System"
    llm_provider: str = LLM_PROVIDER
    llm_api_key: str = _default_llm_api_key()
    llm_base_url: str = _default_llm_base_url()
    llm_model: str = _default_llm_model()
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

    @property
    def provider_label(self) -> str:
        if self.llm_provider == "deepseek":
            return "DeepSeek"
        return "OpenAI Compatible"

    @property
    def primary_api_key_env(self) -> str:
        if self.llm_provider == "deepseek":
            return "DEEPSEEK_API_KEY"
        return "OPENAI_API_KEY"


def get_settings() -> Settings:
    settings = Settings()
    settings.vector_db_path.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.sample_docs_dir.mkdir(parents=True, exist_ok=True)
    return settings
