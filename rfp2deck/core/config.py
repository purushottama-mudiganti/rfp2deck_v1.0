from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _coerce_env_path(value: str) -> Path:
    r"""Convert environment path strings into paths valid for this OS.

    A Windows absolute path such as ``C:\Users\...\file.potx`` is absolute on
    Windows but looks relative on Linux/WSL. When running outside Windows, map
    it to the standard WSL mount path: ``/mnt/c/Users/.../file.potx``.
    """
    cleaned = value.strip().strip('"').strip("'")
    if _WINDOWS_ABSOLUTE_PATH_RE.match(cleaned):
        if os.name == "nt":
            return Path(cleaned)
        win_path = PureWindowsPath(cleaned)
        drive = win_path.drive.rstrip(":").lower()
        return Path("/mnt") / drive / Path(*win_path.parts[1:])
    return Path(cleaned)


def _env_path(name: str, default: str) -> Path:
    raw = os.getenv(name)
    if raw:
        return _coerce_env_path(raw)
    return Path(default)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    model_reasoning: str = os.getenv("OPENAI_MODEL_REASONING", "gpt-5.2")
    model_fast: str = os.getenv("OPENAI_MODEL_FAST", "gpt-5-mini")
    image_model: str = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2")
    image_timeout_s: float = float(os.getenv("OPENAI_IMAGE_TIMEOUT_S", "240"))
    image_retry_attempts: int = int(os.getenv("OPENAI_IMAGE_RETRY_ATTEMPTS", "2"))
    embeddings_model: str = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large")
    data_dir: Path = _env_path("APP_DATA_DIR", ".data")
    openai_timeout_s: float = float(os.getenv("OPENAI_TIMEOUT_S", "120"))
    openai_retry_attempts: int = int(os.getenv("OPENAI_RETRY_ATTEMPTS", "3"))
    openai_retry_base_wait_s: float = float(os.getenv("OPENAI_RETRY_BASE_WAIT_S", "5"))
    openai_retry_max_wait_s: float = float(os.getenv("OPENAI_RETRY_MAX_WAIT_S", "90"))
    openai_structured_streaming: bool = _env_bool("OPENAI_STRUCTURED_STREAMING", False)
    reasoning_effort_high: str = os.getenv("OPENAI_REASONING_EFFORT_HIGH", "high")
    reasoning_effort_medium: str = os.getenv("OPENAI_REASONING_EFFORT_MEDIUM", "medium")
    reasoning_effort_low: str = os.getenv("OPENAI_REASONING_EFFORT_LOW", "low")
    reasoning_effort_deck_plan: str = os.getenv("OPENAI_REASONING_EFFORT_DECK_PLAN", "medium")
    deck_plan_timeout_s: float = float(os.getenv("OPENAI_DECK_PLAN_TIMEOUT_S", "420"))
    deck_plan_rag_max_chars: int = int(os.getenv("OPENAI_DECK_PLAN_RAG_MAX_CHARS", "18000"))
    deck_plan_layout_limit: int = int(os.getenv("OPENAI_DECK_PLAN_LAYOUT_LIMIT", "64"))
    deck_plan_prompt_max_chars: int = int(os.getenv("OPENAI_DECK_PLAN_PROMPT_MAX_CHARS", "30000"))
    deck_plan_chunked: bool = _env_bool("OPENAI_DECK_PLAN_CHUNKED", True)
    deck_plan_specialists: bool = _env_bool("OPENAI_DECK_PLAN_SPECIALISTS", False)
    deck_plan_batch_size: int = int(os.getenv("OPENAI_DECK_PLAN_BATCH_SIZE", "4"))
    understanding_direct_max_chars: int = int(
        os.getenv("OPENAI_UNDERSTANDING_DIRECT_MAX_CHARS", "180000")
    )
    understanding_evidence_chunk_chars: int = int(
        os.getenv("OPENAI_UNDERSTANDING_EVIDENCE_CHUNK_CHARS", "55000")
    )
    understanding_evidence_max_chars: int = int(
        os.getenv("OPENAI_UNDERSTANDING_EVIDENCE_MAX_CHARS", "180000")
    )
    understanding_evidence_workers: int = int(
        os.getenv("OPENAI_UNDERSTANDING_EVIDENCE_WORKERS", "2")
    )
    understanding_evidence_timeout_s: float = float(
        os.getenv("OPENAI_UNDERSTANDING_EVIDENCE_TIMEOUT_S", "300")
    )
    understanding_evidence_cache: bool = _env_bool(
        "OPENAI_UNDERSTANDING_EVIDENCE_CACHE", True
    )
    pipeline_cache: bool = _env_bool("RFP2DECK_PIPELINE_CACHE", True)
    pipeline_cache_version: str = os.getenv("RFP2DECK_PIPELINE_CACHE_VERSION", "v1")
    sp_tenant_id: str = os.getenv("SP_TENANT_ID", "")
    sp_client_id: str = os.getenv("SP_CLIENT_ID", "")
    sp_scopes: str = os.getenv("SP_SCOPES", "Files.Read.All,Sites.Read.All")

    @property
    def proposal_template_path(self) -> Path:
        """Configured corporate proposal template.

        Supports either a PPTX file or the official HCLTech POTX. POTX files are
        converted to a cached PPTX-compatible package before rendering.
        """
        configured = os.getenv("HCLTECH_TEMPLATE_PATH") or os.getenv("PROPOSAL_TEMPLATE_PATH")
        if configured:
            return _coerce_env_path(configured)
        # Bundled default (committed to the repo, resolved relative to PROJECT_ROOT)
        # so deployments without a configured path — e.g. Streamlit Cloud — still
        # render with the official HCLTech corporate template.
        return Path("templates") / "hcltech_expanded_v5.potx"

    @property
    def template_cache_dir(self) -> Path:
        configured = os.getenv("TEMPLATE_CACHE_DIR")
        if configured:
            return _coerce_env_path(configured)
        return self.data_dir / "templates"

    @property
    def rag_index_dir(self) -> Path:
        """Location of the persistent RAG index (e.g. built from SharePoint)."""
        configured = os.getenv("RAG_INDEX_DIR")
        if configured:
            return _coerce_env_path(configured)
        return self.data_dir / "indexes" / "default_rag"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "uploads").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "indexes").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "outputs").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "reports").mkdir(parents=True, exist_ok=True)
        self.template_cache_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "pipeline_cache").mkdir(parents=True, exist_ok=True)


settings = Settings()
