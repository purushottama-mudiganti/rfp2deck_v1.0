from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    model_reasoning: str = os.getenv("OPENAI_MODEL_REASONING", "gpt-5.2")
    model_fast: str = os.getenv("OPENAI_MODEL_FAST", "gpt-5-mini")
    embeddings_model: str = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large")
    data_dir: Path = Path(os.getenv("APP_DATA_DIR", ".data"))
    openai_timeout_s: float = float(os.getenv("OPENAI_TIMEOUT_S", "120"))
    sp_tenant_id: str = os.getenv("SP_TENANT_ID", "")
    sp_client_id: str = os.getenv("SP_CLIENT_ID", "")
    sp_scopes: str = os.getenv("SP_SCOPES", "Files.Read.All,Sites.Read.All")

    @property
    def rag_index_dir(self) -> Path:
        """Location of the persistent RAG index (e.g. built from SharePoint)."""
        configured = os.getenv("RAG_INDEX_DIR")
        if configured:
            return Path(configured)
        return self.data_dir / "indexes" / "default_rag"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "uploads").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "indexes").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "outputs").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "reports").mkdir(parents=True, exist_ok=True)


settings = Settings()
