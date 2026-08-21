from __future__ import annotations

"""UI for generating proposal decks from RFP inputs."""

import logging
import os
import re
import sys
from pathlib import Path

import streamlit as st

try:
    from rfp2deck.agent.graph import build_graph
    from rfp2deck.agent.state import AgentState
    from rfp2deck.core.config import settings
    from rfp2deck.core.logging import setup_logging
    from rfp2deck.core.pipeline_cache import (
        build_diagram_cache_key,
        build_diagram_prompt_cache_key,
        build_plan_cache_key,
        hash_files,
        read_bytes_cache,
        read_json_cache,
        sha256_bytes,
        sha256_text,
        stable_hash,
        write_bytes_cache,
        write_json_cache,
    )
    from rfp2deck.core.schemas import (
        ClarificationRecord,
        DeckPlan,
        SourceDocument,
        TraceabilityReport,
    )
    from rfp2deck.diagrams.generator import generate_diagram_png
    from rfp2deck.ingestion.deck_analyzer import analyze_pptx_template
    from rfp2deck.ingestion.source_package import parse_source_document, render_source_package
    from rfp2deck.ingestion.template_resolver import resolve_pptx_template
    from rfp2deck.rag.indexer import build_faiss_index, chunk_text, load_index
    from rfp2deck.rag.retriever import retrieve
    from rfp2deck.rendering.pptx_renderer import render_deck_from_template, rendered_slide_count
except ModuleNotFoundError:
    # Ensure local package imports work when running via `streamlit run app/rfp2deck_app.py`.
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from rfp2deck.agent.graph import build_graph
    from rfp2deck.agent.state import AgentState
    from rfp2deck.core.config import settings
    from rfp2deck.core.logging import setup_logging
    from rfp2deck.core.pipeline_cache import (
        build_diagram_cache_key,
        build_diagram_prompt_cache_key,
        build_plan_cache_key,
        hash_files,
        read_bytes_cache,
        read_json_cache,
        sha256_bytes,
        sha256_text,
        stable_hash,
        write_bytes_cache,
        write_json_cache,
    )
    from rfp2deck.core.schemas import (
        ClarificationRecord,
        DeckPlan,
        SourceDocument,
        TraceabilityReport,
    )
    from rfp2deck.diagrams.generator import generate_diagram_png
    from rfp2deck.ingestion.deck_analyzer import analyze_pptx_template
    from rfp2deck.ingestion.source_package import parse_source_document, render_source_package
    from rfp2deck.ingestion.template_resolver import resolve_pptx_template
    from rfp2deck.rag.indexer import build_faiss_index, chunk_text, load_index
    from rfp2deck.rag.retriever import retrieve
    from rfp2deck.rendering.pptx_renderer import render_deck_from_template, rendered_slide_count

# Set APP_PASSWORD so that the public URL is not accessed by all 
# The password in the UI should match with the APP_PASSWORD set in the environment variables.
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

# Initialize auth state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# If not authenticated → show password input
if not st.session_state.authenticated:
    pwd = st.text_input("Enter password", type="password")

    if pwd:
        if pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()  #Important: refresh UI to remove input
        else:
            st.error("Incorrect password")
            st.stop()

    st.stop()  # stop app until correct password entered

# Map Streamlit secrets to environment variables (ignore if no secrets.toml present).
# COMMENTING THIS AS STREAMLIT.APP DOMAIN IS NOT ALLOWED IN HCLTECH.
# try:
#     for key, value in st.secrets.items():
#         os.environ[key] = str(value)
# except StreamlitSecretNotFoundError:
#     pass


# Project root path for local assets.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

setup_logging()
log = logging.getLogger("rfp2deck.app")
st.set_page_config(page_title="RFP → Proposal Deck Agent", layout="wide")


st.title("RFP → Proposal Deck Generator (Standard Template)")

# ----------------------------
# Session state defaults
# ----------------------------
st.session_state.setdefault("wizard_step", 1)  # 1,2,3
st.session_state.setdefault("deck_plan", None)
st.session_state.setdefault("report", None)
st.session_state.setdefault("tpl_bytes", None)
st.session_state.setdefault("rfp_names", None)
st.session_state.setdefault("template_info", None)
st.session_state.setdefault("retrieved_context", None)
st.session_state.setdefault("diagrams_generated", False)  # ran generation at least once
st.session_state.setdefault("diagram_images", {})
st.session_state.setdefault("diagram_failures", [])
st.session_state.setdefault("rag_index", None)
st.session_state.setdefault("customer_logo_bytes", None)
st.session_state.setdefault("customer_logo_name", None)

# Corporate template path (no UI upload required). Configure this with
# HCLTECH_TEMPLATE_PATH in .env. It may point to either the official HCLTech
# POTX or a PPTX derived from that POTX.
_configured_template = settings.proposal_template_path
STANDARD_TEMPLATE = (
    _configured_template
    if _configured_template.is_absolute()
    else PROJECT_ROOT / _configured_template
)
_configured_template_cache = settings.template_cache_dir
STANDARD_TEMPLATE_CACHE_DIR = (
    _configured_template_cache
    if _configured_template_cache.is_absolute()
    else PROJECT_ROOT / _configured_template_cache
)


@st.cache_resource(show_spinner=False)
def resolve_standard_template(template_path: str, cache_dir: str) -> Path:
    """Resolve the configured template to a PPTX path python-pptx can read."""
    return resolve_pptx_template(Path(template_path), Path(cache_dir))


@st.cache_resource(show_spinner=False)
def load_persistent_index(index_dir: str):
    """Load the persistent RAG index (e.g. built from SharePoint) from disk.

    Cached so FAISS isn't re-read on every Streamlit rerun. Returns None if no
    index exists at the given location. The directory path is the cache key, so
    rebuilding under a new path (or clearing the cache) picks up changes.
    """
    path = Path(index_dir)
    if not (path / "index.faiss").exists() or not (path / "chunks.json").exists():
        return None
    try:
        rag = load_index(path)
    except Exception as exc:  # corrupt index shouldn't crash the app
        log.warning("Failed to load persistent RAG index from %s: %s", path, exc)
        return None

    # Guard against a dimension mismatch / garbage retrieval: the index is only
    # valid if it was built with the embeddings model that's active now.
    if rag.embeddings_model and rag.embeddings_model != settings.embeddings_model:
        log.warning(
            "Persistent RAG index at %s was built with embeddings model %r but the "
            "active model is %r; ignoring it. Rebuild the index to use it.",
            path,
            rag.embeddings_model,
            settings.embeddings_model,
        )
        return None
    return rag


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Settings")

    st.subheader("Deck Mode")
    deck_mode = st.radio(
        "Select Output Mode",
        options=["Bid Defense (Core Only)", "Full Proposal (Core + Appendix)"],
        index=0,
    )
    st.session_state.deck_mode = deck_mode
    st.caption(f"Selected mode: {deck_mode}")

    deck_plan_specialists = st.checkbox(
        "Specialist-architect planning (parallel Application/Integration/Data architects)",
        value=settings.deck_plan_specialists,
        help="When on, each proposal section is expanded by a role-focused specialist "
        "in parallel instead of one generalist. Overrides the OPENAI_DECK_PLAN_SPECIALISTS "
        "default per run — no redeploy needed to switch.",
    )
    st.session_state.deck_plan_specialists = deck_plan_specialists

    st.write("Models are configured via `.env`.")
    st.code(
        "Reasoning model: {reasoning}\n"
        "Fast model: {fast}\n"
        "Image model: {image}\n"
        "Embeddings: {embeddings}".format(
            reasoning=settings.model_reasoning,
            fast=settings.model_fast,
            image=settings.image_model,
            embeddings=settings.embeddings_model,
        )
    )

    enable_notes = st.checkbox(
        "Generate speaker notes (presenter notes per slide)", value=True
    )
    enable_diagrams = st.checkbox("Enable diagram generation (guarded + approval)", value=True)
    diagram_model = st.text_input("Diagram model", value=settings.image_model)
    diagram_size = st.selectbox(
        "Diagram size",
        options=["auto", "1024x1024", "1024x1536", "1536x1024"],
        index=0,
    )
    diagram_quality = st.selectbox(
        "Diagram quality",
        options=["auto", "low", "medium", "high"],
        index=0,
    )

    build_index = st.checkbox(
        "Build/Update RAG index from uploaded reference text (optional)", value=False
    )
    st.caption(
        "Tip: upload a TXT file of reusable assets/proposal boilerplates " "to build a quick index."
    )

    persistent_index = load_persistent_index(str(settings.rag_index_dir))
    use_persistent_rag = st.checkbox(
        "Use persistent knowledge base (SharePoint index)",
        value=False,
        disabled=persistent_index is None,
        help="Retrieve reusable context from the prebuilt index at "
        f"{settings.rag_index_dir}. Build it with the sharepoint_index CLI.",
    )
    if persistent_index is None:
        index_files_exist = (settings.rag_index_dir / "index.faiss").exists()
        if index_files_exist:
            st.caption(
                f"Persistent index at {settings.rag_index_dir} is unusable "
                f"(corrupt or built with a different embeddings model than "
                f"{settings.embeddings_model}). Rebuild it with the sharepoint_index CLI."
            )
        else:
            st.caption(f"No persistent index found at {settings.rag_index_dir}.")
    else:
        st.caption(
            f"Persistent index available ({len(persistent_index.chunks)} chunks). "
            "An uploaded reference index takes priority when both are present."
        )

    st.divider()
    st.caption("Template")
    try:
        resolved_template = resolve_standard_template(
            str(STANDARD_TEMPLATE), str(STANDARD_TEMPLATE_CACHE_DIR)
        )
        if STANDARD_TEMPLATE.suffix.lower() == ".potx":
            st.success(f"Using HCLTech POTX: {STANDARD_TEMPLATE.name}")
            st.caption(f"Cached PPTX: {resolved_template.name}")
        else:
            st.success(f"Using HCLTech template: {resolved_template.name}")
    except Exception as exc:
        st.error(
            "Corporate template is unavailable. Set HCLTECH_TEMPLATE_PATH in .env "
            "to the official HCLTech .potx or a converted .pptx."
        )
        st.caption(str(exc))

    if st.button("Reset wizard", use_container_width=True):
        keys = [
            "wizard_step",
            "deck_plan",
            "report",
            "tpl_bytes",
            "rfp_names",
            "template_info",
            "retrieved_context",
            "diagrams_generated",
            "diagram_images",
            "diagram_failures",
            "customer_logo_bytes",
            "customer_logo_name",
            "rag_index",
            "render_complete",
            "rfp_step1",
            "clarifications_step1",
            "supporting_step1",
            "ref_step1",
            "customer_logo_step1",
        ]
        for k in keys:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()


# ----------------------------
# Helpers
# ----------------------------
def parse_rfp_source(upload, role: str) -> tuple[SourceDocument, list[ClarificationRecord], dict]:
    """Parse one package source while preserving its role and locators."""
    name = getattr(upload, "name", "rfp")
    document, clarifications = parse_source_document(name, upload.getvalue(), role=role)
    meta = {
        "name": document.name,
        "document_id": document.document_id,
        "document_type": document.document_type,
        "authority": document.authority,
        "issue_date": document.issue_date,
        "locator": document.locator_format,
        **document.metadata,
    }
    if clarifications:
        meta["clarifications"] = len(clarifications)
    return document, clarifications, meta


def parse_rfp_sources(
    uploads: list,
    role: str,
    on_progress=None,
) -> tuple[list[SourceDocument], list[ClarificationRecord], list[dict]]:
    """Parse a group of package sources with a shared upload role."""
    documents: list[SourceDocument] = []
    clarifications: list[ClarificationRecord] = []
    summaries: list[dict] = []
    total = len(uploads or [])
    for index, upload in enumerate(uploads or [], start=1):
        name = getattr(upload, "name", "rfp")
        if on_progress is not None:
            on_progress(f"Parsing {role} document {index} of {total}: {name}")
        document, extracted_clarifications, meta = parse_rfp_source(upload, role)
        documents.append(document)
        clarifications.extend(extracted_clarifications)
        summaries.append(meta)
    return documents, clarifications, summaries


def source_count_label(summary: dict) -> str:
    """Return a compact, format-specific extraction summary."""
    if summary.get("extension") == ".pdf":
        return f'{summary.get("pages", 0)} pages'
    if summary.get("extension") == ".docx":
        return (
            f'{summary.get("paragraphs", 0)} paragraphs, '
            f'{summary.get("tables", 0)} tables'
        )
    if summary.get("extension") == ".xlsx":
        return (
            f'{summary.get("sheets", 0)} sheets, {summary.get("rows", 0)} rows, '
            f'{summary.get("clarifications", 0)} Q&A records'
        )
    return "Parsed"


def upload_fingerprint(uploads_by_role: list[tuple[str, list]]) -> list[dict]:
    """Hash uploaded file bytes without depending on Streamlit object identity."""
    files = []
    for role, uploads in uploads_by_role:
        for upload in uploads or []:
            data = upload.getvalue()
            files.append(
                {
                    "role": role,
                    "name": getattr(upload, "name", "upload"),
                    "size": len(data),
                    "sha256": sha256_bytes(data),
                }
            )
    return files


def pipeline_logic_hash() -> str:
    """Hash prompt/node/schema files that materially affect Step 1 planning."""
    return hash_files(
        [
            PROJECT_ROOT / "rfp2deck" / "agent" / "prompts.py",
            PROJECT_ROOT / "rfp2deck" / "agent" / "nodes.py",
            PROJECT_ROOT / "rfp2deck" / "agent" / "graph.py",
            PROJECT_ROOT / "rfp2deck" / "core" / "schemas.py",
        ]
    )


def build_step1_cache_key(
    *,
    uploads_by_role: list[tuple[str, list]],
    template_info: dict,
    retrieved_context: str | None,
    deck_mode: str | None,
    enable_notes: bool,
    deck_plan_specialists: bool,
    customer_technology_context: dict | None = None,
) -> str:
    return build_plan_cache_key(
        {
            "files": upload_fingerprint(uploads_by_role),
            "template_layouts": template_info.get("slide_layout_names", []),
            "template_placeholders_hash": stable_hash(template_info.get("placeholder_map", {})),
            "retrieved_context_hash": sha256_text(retrieved_context or ""),
            "deck_mode": deck_mode or "",
            "enable_notes": bool(enable_notes),
            "customer_technology_context": customer_technology_context or {},
            "models": {
                "reasoning": settings.model_reasoning,
                "fast": settings.model_fast,
                "deck_plan_effort": settings.reasoning_effort_deck_plan,
                "high_effort": settings.reasoning_effort_high,
            },
            "planning": {
                "chunked": settings.deck_plan_chunked,
                "batch_size": settings.deck_plan_batch_size,
                "prompt_max_chars": settings.deck_plan_prompt_max_chars,
                "specialists": bool(deck_plan_specialists),
            },
            "evidence": {
                "direct_max_chars": settings.understanding_direct_max_chars,
                "chunk_chars": settings.understanding_evidence_chunk_chars,
                "max_chars": settings.understanding_evidence_max_chars,
                "contextual_llm": settings.understanding_contextual_evidence_llm_enabled,
            },
            "logic_hash": pipeline_logic_hash(),
        }
    )


def normalize_models(deck_plan, report):
    """Normalize dict responses into Pydantic models."""
    if isinstance(deck_plan, dict):
        deck_plan = DeckPlan.model_validate(deck_plan)
    if report is not None and isinstance(report, dict):
        report = TraceabilityReport.model_validate(report)
    return deck_plan, report


def count_diagrams(plan: DeckPlan, diagram_images: dict[str, bytes] | None = None):
    """Count total diagrams and approved diagrams in the plan."""
    total = 0
    approved = 0
    diagram_images = diagram_images or {}
    for s in plan.slides:
        prompt = (getattr(s.diagram, "prompt", "") or "").strip() if s.diagram else ""
        if s.diagram and (s.slide_id in diagram_images or (prompt and prompt in diagram_images)):
            total += 1
            if bool(s.diagram.approved):
                approved += 1
    return total, approved


def diagram_prompt_for_slide(slide) -> str:
    diagram = getattr(slide, "diagram", None)
    return (getattr(diagram, "prompt", "") or "").strip() if diagram else ""


def diagram_image_for_slide(slide, diagram_images: dict[str, bytes] | None) -> bytes | None:
    images = diagram_images or {}
    if getattr(slide, "slide_id", None) in images:
        return images[slide.slide_id]
    prompt = diagram_prompt_for_slide(slide)
    if prompt and prompt in images:
        return images[prompt]
    return None


def load_cached_diagram_images(
    plan: DeckPlan,
    current_images: dict[str, bytes] | None,
    *,
    model: str,
    size: str,
    quality: str,
) -> dict[str, bytes]:
    """Reload diagram PNGs from persistent cache after Streamlit/server restart."""
    images = dict(current_images or {})
    if not settings.pipeline_cache:
        return images
    for slide in plan.slides:
        diagram = getattr(slide, "diagram", None)
        if not diagram:
            continue
        prompt = diagram_prompt_for_slide(slide)
        if slide.slide_id in images:
            if prompt:
                images.setdefault(prompt, images[slide.slide_id])
            continue
        if prompt and prompt in images:
            images[slide.slide_id] = images[prompt]
            continue
        cache_key = build_diagram_cache_key(
            prompt=diagram.prompt,
            model=model,
            size=size,
            quality=quality,
        )
        cached = read_bytes_cache(cache_key, ".png")
        if cached is None and prompt:
            cached = read_bytes_cache(build_diagram_prompt_cache_key(prompt=prompt), ".png")
        if cached:
            images[slide.slide_id] = cached
            if prompt:
                images[prompt] = cached
    return images


def wizard_header(step: int):
    """Render the wizard header and navigation controls."""
    labels = ["Upload & Plan", "Diagrams & Approval", "Render & Download"]
    step = max(1, min(3, int(step)))
    idx = step - 1

    chips = []
    for i, name in enumerate(labels, start=1):
        if i < step:
            chips.append(f"✅ **{i}. {name}**")
        elif i == step:
            chips.append(f"🟦 **{i}. {name}**")
        else:
            chips.append(f"⬜ {i}. {name}")
    st.markdown(" | ".join(chips))
    st.progress(idx / 2.0)

    col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 2])
    with col_a:
        if st.button("← Back", disabled=(step == 1), use_container_width=True):
            st.session_state.wizard_step = max(1, step - 1)
            st.rerun()
    with col_b:
        can_go_2 = st.session_state.deck_plan is not None
        if st.button("Go to Step 2", disabled=not can_go_2, use_container_width=True):
            st.session_state.wizard_step = 2
            st.rerun()
    with col_c:
        can_go_3 = st.session_state.deck_plan is not None and st.session_state.tpl_bytes is not None
        if st.button("Go to Step 3", disabled=not can_go_3, use_container_width=True):
            st.session_state.wizard_step = 3
            st.rerun()
    with col_d:
        st.caption("Tip: Streamlit reruns on every click — state is preserved in session_state.")


def render_step_progress(value: float, text: str) -> None:
    """Render a simple progress bar with a caption for the current step."""
    st.progress(max(0.0, min(1.0, value)))
    st.caption(text)


def _slugify(value: str) -> str:
    """Make a filesystem-safe, ASCII-only filename fragment."""
    value = re.sub(r"[^\w\s-]", "", value, flags=re.ASCII).strip().lower()
    value = re.sub(r"[-\s]+", "-", value, flags=re.ASCII)
    return value.strip("-") or "proposal"


def build_output_filename(plan: DeckPlan, rfp_names: list[str] | None) -> str:
    """Generate a descriptive PPTX filename based on the deck plan/RFP."""
    title = (getattr(plan, "deck_title", "") or "").strip()
    if title and "not specified" not in title.lower():
        base = _slugify(title)
    elif rfp_names:
        base = _slugify(Path(rfp_names[0]).stem)
    else:
        base = "proposal"
    return f"{base}.pptx"


def stop_on_error(message: str, status: st.delta_generator.DeltaGenerator | None, exc: Exception):
    """Show error details, mark status as failed, and stop execution."""
    logging.getLogger("rfp2deck.app").error("%s", message, exc_info=exc)
    if status is not None:
        status.update(label=message, state="error")
    st.error(message)
    st.exception(exc)
    st.stop()


# Guard: if plan missing, force step 1
if st.session_state.deck_plan is None and st.session_state.wizard_step != 1:
    st.session_state.wizard_step = 1

wizard_header(st.session_state.wizard_step)
st.divider()

# ----------------------------
# STEP 1
# ----------------------------
if st.session_state.wizard_step == 1:
    st.subheader("Step 1 - Upload RFP Package and Generate Deck Plan")
    step1_progress_slot = st.empty()

    def show_step1_progress(value: float, text: str) -> None:
        with step1_progress_slot.container():
            render_step_progress(value, text)

    show_step1_progress(0.05, "Step 1 progress: waiting for inputs.")

    c1, c2 = st.columns(2)
    with c1:
        rfp_files = st.file_uploader(
            "Primary RFP and requirement annexures",
            type=["pdf", "docx", "xlsx"],
            key="rfp_step1",
            accept_multiple_files=True,
        )
    with c2:
        clarification_files = st.file_uploader(
            "Customer clarifications and addenda",
            type=["pdf", "docx", "xlsx"],
            key="clarifications_step1",
            accept_multiple_files=True,
        )
    customer_logo = st.file_uploader(
        "Optional customer logo",
        type=["png", "jpg", "jpeg"],
        key="customer_logo_step1",
        help="Upload an approved PNG or JPEG logo. It will be added to every generated slide without changing the HCLTech template or master.",
    )
    if customer_logo is not None:
        logo_bytes = customer_logo.getvalue()
        if len(logo_bytes) > 5 * 1024 * 1024:
            st.error("Customer logo must be 5 MB or smaller.")
            st.stop()
        previous_logo = st.session_state.get("customer_logo_bytes")
        st.session_state.customer_logo_bytes = logo_bytes
        st.session_state.customer_logo_name = customer_logo.name
        if previous_logo != logo_bytes:
            st.session_state.render_complete = False
        st.image(logo_bytes, caption=f"Customer logo: {customer_logo.name}", width=160)
    else:
        if st.session_state.get("customer_logo_bytes") is not None:
            st.session_state.render_complete = False
        st.session_state.customer_logo_bytes = None
        st.session_state.customer_logo_name = None
    with st.expander("Optional supporting material", expanded=False):
        supporting_files = st.file_uploader(
            "Supporting reference documents",
            type=["pdf", "docx", "xlsx"],
            key="supporting_step1",
            accept_multiple_files=True,
            help="Supporting material is contextual and cannot create scope unless the RFP or a customer response incorporates it.",
        )
        ref_txt = st.file_uploader(
            "Optional: Reusable content (TXT) for RAG", type=["txt"], key="ref_step1"
        )
        st.caption("Enable 'Build/Update RAG index' in sidebar to index this TXT.")

    with st.expander("Customer technology constraints", expanded=False):
        cloud_platform = st.selectbox(
            "Known hyperscaler or hosting platform",
            ["Not specified", "Microsoft Azure", "Amazon Web Services (AWS)", "Google Cloud Platform", "On-premises / private cloud", "Hybrid", "Other"],
            key="customer_cloud_platform_step1",
        )
        cloud_context_status = st.selectbox(
            "How should this information be treated?",
            ["Customer-preferred", "Customer-mandated", "Existing estate", "Working assumption"],
            key="customer_cloud_status_step1",
        )
        cloud_context_details = st.text_area(
            "Relevant customer standards, approved services, regions, or restrictions",
            key="customer_cloud_details_step1",
            help="This is treated separately from RFP evidence and is carried into technology and architecture decisions.",
        )

    customer_technology_context = {
        "platform": cloud_platform,
        "status": cloud_context_status,
        "details": cloud_context_details.strip(),
    } if cloud_platform != "Not specified" or cloud_context_details.strip() else {}

    if rfp_files:
        show_step1_progress(0.1, "Step 1 progress: ready to generate plan.")

    with st.form("step1_form"):
        submitted = st.form_submit_button(
            "Generate Plan (Step 1)", type="primary", use_container_width=True
        )

    if submitted:
        show_step1_progress(0.15, "Step 1 progress: generating plan...")
        step_progress = st.progress(0.0)
        step_status = st.status(
            "Step 1 - Generate the proposal deck plan",
            expanded=False,
        )
        activity_slot = st.empty()
        activity_state = {"progress": 0.1}

        def update_step1_activity(message: str, progress: float | None = None) -> None:
            if progress is not None:
                # Parallel analysis branches may finish out of display order;
                # never move the user-visible progress bar backwards.
                activity_state["progress"] = max(
                    activity_state["progress"],
                    max(0.0, min(1.0, progress)),
                )
                step_progress.progress(activity_state["progress"])
            activity_slot.info(
                f"{round(activity_state['progress'] * 100)}% complete - {message}"
            )

        step_progress.progress(0.1)

        if not rfp_files:
            st.error("Please upload at least one primary RFP file (PDF, DOCX, or XLSX).")
            st.stop()

        try:
            resolved_template = resolve_standard_template(
                str(STANDARD_TEMPLATE), str(STANDARD_TEMPLATE_CACHE_DIR)
            )
            update_step1_activity("Preparing the proposal template and uploaded package...", 0.15)
            all_uploads = list(rfp_files) + list(clarification_files or []) + list(
                supporting_files or []
            )
            rfp_names = [getattr(f, "name", "rfp") for f in all_uploads]
            st.session_state.rfp_names = rfp_names
            step_progress.progress(0.3)

            tpl_bytes = resolved_template.read_bytes()
            st.session_state.tpl_bytes = tpl_bytes

            primary_documents, primary_clarifications, primary_summaries = parse_rfp_sources(
                rfp_files, "primary", update_step1_activity
            )
            clarification_documents, customer_clarifications, clarification_summaries = (
                parse_rfp_sources(clarification_files or [], "clarification", update_step1_activity)
            )
            supporting_documents, supporting_clarifications, supporting_summaries = (
                parse_rfp_sources(supporting_files or [], "supporting", update_step1_activity)
            )
            source_documents = (
                primary_documents + clarification_documents + supporting_documents
            )
            clarification_records = (
                primary_clarifications + customer_clarifications + supporting_clarifications
            )
            rfp_summaries = (
                primary_summaries + clarification_summaries + supporting_summaries
            )
            rfp_text = render_source_package(source_documents)
            update_step1_activity("Building the reconciled source package...", 0.45)

            if rfp_summaries:
                st.markdown("**RFP package evidence summary**")
                rows = []
                for s in rfp_summaries:
                    rows.append(
                        {
                            "File": s["name"],
                            "Role": s["document_type"].replace("_", " ").title(),
                            "Authority": s["authority"].replace("_", " ").title(),
                            "Issue Date": s.get("issue_date") or "Unknown",
                            "Extracted": source_count_label(s),
                        }
                    )
                st.table(rows)
                unresolved_count = sum(
                    1 for item in clarification_records if not item.customer_response.strip()
                )
                st.caption(
                    f"Sources: {len(source_documents)} | Clarification Q&A records: "
                    f"{len(clarification_records)} | Unanswered questions: {unresolved_count}"
                )
                package_warnings = [
                    f"{document.name}: {warning}"
                    for document in source_documents
                    for warning in document.warnings
                ]
                if package_warnings:
                    st.warning("\n".join(package_warnings[:10]))

            # Analyze template layouts/placeholders
            ti = analyze_pptx_template(tpl_bytes)
            template_info = {
                "slide_layout_names": ti.slide_layout_names,
                "masters": ti.masters,
                "placeholder_map": ti.placeholder_map,
            }
            st.session_state.template_info = template_info
            st.info(f"Template analyzed: {len(ti.slide_layout_names)} layouts found.")
            update_step1_activity("Analyzing presentation layouts and placeholders...", 0.52)

            # Optional RAG (in-memory only)
            retrieved_context = None
            if ref_txt and build_index:
                ref_text = ref_txt.getvalue().decode("utf-8", errors="ignore")
                chunks = chunk_text(ref_text)
                rag = build_faiss_index(chunks)
                st.session_state.rag_index = rag
                st.success(f"Built RAG index with {len(chunks)} chunks.")

            # Pick a retrieval source: the in-memory index built from an
            # uploaded TXT takes priority; otherwise fall back to the persistent
            # knowledge base (e.g. the SharePoint index) when the user opted in.
            rag = st.session_state.get("rag_index")
            rag_source = "in-memory" if rag is not None else None
            if rag is None and use_persistent_rag:
                rag = persistent_index
                rag_source = "persistent"

            if rag is not None:
                query = """
                mandatory proposal sections, required slides,
                governance model, compliance, team structure,
                risk framework, commercial assumptions,
                architecture standards, delivery model
                """
                top = retrieve(rag, query, k=10)
                retrieved_context = "\n\n".join([f"[score={c.score:.3f}]\n{c.text}" for c in top])
                st.caption(f"Retrieved reusable context from {rag_source} RAG index.")

            st.session_state.retrieved_context = retrieved_context
            update_step1_activity("Checking reusable knowledge and the Step 1 cache...", 0.58)

            cache_key = build_step1_cache_key(
                uploads_by_role=[
                    ("primary", list(rfp_files or [])),
                    ("clarification", list(clarification_files or [])),
                    ("supporting", list(supporting_files or [])),
                ],
                template_info=template_info,
                retrieved_context=retrieved_context,
                deck_mode=st.session_state.get("deck_mode"),
                enable_notes=enable_notes,
                deck_plan_specialists=deck_plan_specialists,
                customer_technology_context=customer_technology_context,
            )
            if settings.pipeline_cache:
                cached = read_json_cache(cache_key)
                if cached:
                    log.info("Step 1 pipeline cache HIT: %s", cache_key[:16])
                    deck_plan = DeckPlan.model_validate(cached.get("deck_plan"))
                    report_payload = cached.get("report")
                    report = (
                        TraceabilityReport.model_validate(report_payload)
                        if report_payload
                        else None
                    )
                    st.session_state.deck_plan = deck_plan
                    st.session_state.report = report
                    st.session_state.diagrams_generated = False
                    st.session_state.diagram_images = {}
                    st.session_state.diagram_failures = []
                    st.session_state.render_complete = False
                    step_progress.progress(1.0)
                    activity_slot.success("100% complete - Proposal plan loaded from cache.")
                    step_status.update(label="Step 1 complete from cache.", state="complete")
                    st.success("Step 1 loaded from cache. Moving to next step...")
                    st.session_state.wizard_step = 2 if enable_diagrams else 3
                    st.rerun()
                log.info("Step 1 pipeline cache MISS: %s", cache_key[:16])
                st.caption(f"Step 1 cache miss: {cache_key[:12]}")

            # Run agent
            graph = build_graph()
            state = AgentState(
                rfp_text=rfp_text,
                template_info=template_info,
                retrieved_context=retrieved_context,
                source_documents=source_documents,
                clarification_records=clarification_records,
                customer_technology_context=customer_technology_context,
            )
            state.deck_mode = st.session_state.get("deck_mode")
            state.enable_notes = enable_notes
            state.deck_plan_specialists = deck_plan_specialists

            log.info(
                "Invoking agent pipeline (rfp_chars=%d, sources=%d, clarifications=%d, deck_mode=%s, rag=%s)",
                len(rfp_text or ""),
                len(source_documents),
                len(clarification_records),
                st.session_state.get("deck_mode"),
                retrieved_context is not None,
            )
            node_progress = {
                "reconcile_sources": ("Extracting requirements and traceable evidence...", 0.66),
                "extract_source_evidence": ("Analyzing objectives, scope, constraints, and risks...", 0.71),
                "understand_rfp": ("Determining the proposal storyline and required sections...", 0.75),
                "derive_sections": ("Building the executive narrative and win themes...", 0.79),
                "build_narrative": ("Designing proposal-specific visual briefs...", 0.83),
                "derive_visual_briefs": ("Evaluating suitable technologies and cloud services...", 0.87),
                "derive_technology_recommendations": ("Creating the detailed slide plan...", 0.91),
                "plan_deck": ("Refining slide content for readability...", 0.94),
                "compress_bullets": ("Generating presenter notes...", 0.96),
                "generate_notes": ("Validating coverage, traceability, and proposal quality...", 0.98),
                "qa_and_report": ("Finalizing the proposal plan...", 0.99),
            }
            final_state = state.model_dump()
            update_step1_activity("Reconciling source authority and clarifications...", 0.62)
            for event in graph.stream(state, stream_mode="updates"):
                for node_name, update in event.items():
                    if node_name in node_progress:
                        message, progress_value = node_progress[node_name]
                        update_step1_activity(message, progress_value)
                    if isinstance(update, dict):
                        final_state.update(update)
            log.info("Agent pipeline completed.")

            if isinstance(final_state, dict):
                deck_plan = final_state.get("deck_plan")
                report = final_state.get("report")
            else:
                deck_plan = getattr(final_state, "deck_plan", None)
                report = getattr(final_state, "report", None)

            if not deck_plan:
                st.error("Failed to produce a deck plan.")
                st.stop()

            deck_plan, report = normalize_models(deck_plan, report)
            if settings.pipeline_cache:
                write_json_cache(
                    cache_key,
                    {
                        "cache_key": cache_key,
                        "cache_version": settings.pipeline_cache_version,
                        "deck_plan": deck_plan.model_dump(),
                        "report": report.model_dump() if report else None,
                    },
                )
                log.info("Step 1 pipeline cache SAVED: %s", cache_key[:16])
                st.caption("Saved Step 1 plan to pipeline cache.")

            st.session_state.deck_plan = deck_plan
            st.session_state.report = report
            st.session_state.diagrams_generated = False  # reset for new run
            st.session_state.diagram_images = {}
            st.session_state.diagram_failures = []
            st.session_state.render_complete = False
            step_progress.progress(1.0)
            activity_slot.success("100% complete - Proposal plan and quality checks completed.")
            step_status.update(label="Step 1 complete.", state="complete")

            # Advance
            st.session_state.wizard_step = 2 if enable_diagrams else 3
            st.success("Step 1 complete. Moving to next step…")
            st.rerun()
        except Exception as exc:
            stop_on_error("Step 1 failed. See details below.", step_status, exc)

    if st.session_state.deck_plan:
        with st.expander("Deck Plan JSON (current session)"):
            st.json(st.session_state.deck_plan.model_dump())
        with st.expander("Traceability Report (current session)"):
            rep = st.session_state.report
            st.json(rep.model_dump() if rep else {})

# ----------------------------
# STEP 2
# ----------------------------
if st.session_state.wizard_step == 2:
    st.subheader("Step 2 — Generate Diagrams and Approve (Guarded)")

    # Diagnostics: show diagram prompt coverage on key slides
    plan = st.session_state.deck_plan
    if plan:
        with st.expander("Diagram Coverage (diagnostics)"):
            total = len(plan.slides)
            with_prompt = sum(1 for s in plan.slides if getattr(s, "diagram", None) is not None)
            st.write(f"Slides: {total} | With diagram prompts: {with_prompt}")
            weak = [
                {
                    "slide_id": s.slide_id,
                    "title": s.title,
                    "score": getattr(s.diagram, "grounding_score", 0.0),
                    "warnings": getattr(s.diagram, "grounding_warnings", []),
                }
                for s in plan.slides
                if getattr(s, "diagram", None) is not None
                and getattr(s.diagram, "grounding_score", 0.0) < 0.45
            ]
            if weak:
                st.warning("Some diagram prompts are weakly grounded and will be skipped unless improved.")
                st.json(weak)
            missing = [
                f"{s.slide_id}: {s.title} ({s.archetype})"
                for s in plan.slides
                if getattr(s, "diagram", None) is None
                and (str(s.archetype).lower() in ["architecture", "team"])
            ]
            if missing:
                st.warning("Missing DiagramSpec on key slides (should be fixed in v4.5.5):")
                st.code("\n".join(missing))
            else:
                st.success("All key slides have diagram prompts.")

    if st.session_state.deck_plan is None:
        st.error("No deck plan found. Please complete Step 1.")
        st.stop()

    plan: DeckPlan = st.session_state.deck_plan

    # Surface persistent images immediately after a restart. Approval is a
    # session decision; it must not prevent already-paid-for images from being
    # discovered and reviewed again.
    st.session_state.diagram_images = load_cached_diagram_images(
        plan,
        st.session_state.get("diagram_images"),
        model=diagram_model,
        size=diagram_size,
        quality=diagram_quality,
    )

    if not enable_diagrams:
        st.info("Diagram generation disabled in sidebar. You can go to Step 3.")
        if st.button("Proceed to Step 3", type="primary", use_container_width=True):
            st.session_state.wizard_step = 3
            st.rerun()
        st.stop()

    total_diagrams, approved_diagrams = count_diagrams(
        plan, st.session_state.get("diagram_images")
    )
    if not st.session_state.diagrams_generated:
        render_step_progress(0.2, "Step 2 progress: ready to generate diagrams.")
    elif total_diagrams:
        ratio = approved_diagrams / max(total_diagrams, 1)
        render_step_progress(0.6 + 0.4 * ratio, "Step 2 progress: approvals in progress.")
    else:
        render_step_progress(0.4, "Step 2 progress: diagrams generated.")

    has_diagram_specs = any((s.diagram is not None) for s in plan.slides)
    if not has_diagram_specs:
        st.warning("This plan did not propose any diagrams. You can proceed to Step 3.")
        if st.button("Proceed to Step 3", type="primary", use_container_width=True):
            st.session_state.wizard_step = 3
            st.rerun()
        st.stop()

    colL, colR = st.columns([1, 1])
    with colL:
        gen_clicked = st.button(
            "Generate / Regenerate Diagrams", type="primary", use_container_width=True
        )
    with colR:
        st.caption(
            "You can regenerate after changing diagram model, size, or quality in the sidebar."
        )

    if gen_clicked:
        diagram_images = {}
        diagram_failures = []

        slides_with_diagrams = [s for s in plan.slides if s.diagram]
        eligible_targets = []
        for slide in slides_with_diagrams:
            score = float(getattr(slide.diagram, "grounding_score", 0.0) or 0.0)
            warnings = list(getattr(slide.diagram, "grounding_warnings", []) or [])
            if score < 0.45:
                diagram_failures.append(
                    {
                        "slide_id": slide.slide_id,
                        "title": slide.title,
                        "error": "Diagram prompt blocked because it is not grounded enough for generation.",
                        "grounding_score": score,
                        "warnings": warnings,
                    }
                )
            else:
                eligible_targets.append(slide)
        total_targets = len(eligible_targets)
        progress = st.progress(0)
        status = st.status(
            f"Generating {total_targets} grounded diagram(s)...",
            expanded=False,
        )

        if diagram_failures:
            blocked_names = "; ".join(item["title"] for item in diagram_failures)
            st.warning(
                f"Skipped {len(diagram_failures)} weakly grounded diagram prompt(s): {blocked_names}"
            )

        made = 0
        for idx, s in enumerate(eligible_targets, start=1):
            if not s.diagram:
                continue
            status.update(
                label=f"Generating diagram {idx}/{total_targets}: {s.title}",
                state="running",
            )
            try:
                diagram_cache_key = build_diagram_cache_key(
                    prompt=s.diagram.prompt,
                    model=diagram_model,
                    size=diagram_size,
                    quality=diagram_quality,
                )
                img_bytes = (
                    read_bytes_cache(diagram_cache_key, ".png")
                    if settings.pipeline_cache
                    else None
                )
                if img_bytes:
                    status.update(
                        label=f"Loaded cached diagram {idx}/{total_targets}: {s.title}",
                        state="running",
                    )
                else:
                    img_bytes = generate_diagram_png(
                        s.diagram.prompt,
                        out_path=None,
                        model=diagram_model,
                        size=diagram_size,
                        quality=diagram_quality,
                    )
                    if settings.pipeline_cache:
                        write_bytes_cache(diagram_cache_key, img_bytes, ".png")
                diagram_images[s.slide_id] = img_bytes
                prompt = diagram_prompt_for_slide(s)
                if prompt:
                    diagram_images[prompt] = img_bytes
                    if settings.pipeline_cache:
                        write_bytes_cache(
                            build_diagram_prompt_cache_key(prompt=prompt),
                            img_bytes,
                            ".png",
                        )
                made += 1
                status.update(
                    label=f"Generated {idx}/{total_targets} diagrams",
                    state="running",
                )
            except Exception as exc:
                diagram_failures.append(
                    {
                        "slide_id": s.slide_id,
                        "title": s.title,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                status.update(
                    label=f"Skipped failed diagram {idx}/{total_targets}: {s.title}",
                    state="running",
                )
            finally:
                progress.progress(idx / max(total_targets, 1))

        st.session_state.deck_plan = plan
        st.session_state.diagram_images = diagram_images
        st.session_state.diagram_failures = diagram_failures
        st.session_state.diagrams_generated = True
        if made:
            status.update(label="Diagram generation complete with available images.", state="complete")
            if diagram_failures:
                st.warning(
                    f"Generated {made} diagram(s); {len(diagram_failures)} diagram(s) were blocked or failed."
                )
            else:
                st.success(f"Generated {made} diagram(s). Now approve below.")
            st.rerun()
        else:
            status.update(label="No eligible diagrams were generated.", state="error")
            st.error("No diagrams met the grounding gate or generation failed. Check the details below.")
            st.json(diagram_failures)

    diagram_images = st.session_state.get("diagram_images", {})
    diagram_failures = st.session_state.get("diagram_failures", [])
    if diagram_failures:
        with st.expander("Skipped diagram failures", expanded=False):
            st.json(diagram_failures)
    any_images = any((s.diagram and diagram_image_for_slide(s, diagram_images)) for s in plan.slides)
    if not any_images:
        st.info("No diagram images generated yet. Click **Generate / Regenerate Diagrams** above.")
        st.stop()

    st.markdown("### Diagram Review & Approval")

    with st.form("diagram_approvals_form"):
        for s in plan.slides:
            img_bytes = diagram_image_for_slide(s, diagram_images)
            if not s.diagram or img_bytes is None:
                continue

            st.markdown(f"""**{s.slide_id} — {s.title}**  
Kind: `{s.diagram.kind}`""")

            with st.expander("Grounding details", expanded=False):
                st.metric("Grounding score", f"{getattr(s.diagram, 'grounding_score', 0.0):.2f}")
                st.write("Entities")
                st.json(getattr(s.diagram, "entities", []) or [])
                st.write("Flows")
                st.json(getattr(s.diagram, "flows", []) or [])
                st.write("Controls")
                st.json(getattr(s.diagram, "controls", []) or [])
                st.write("Evidence refs")
                st.json(getattr(s.diagram, "evidence_refs", []) or [])
                warnings = getattr(s.diagram, "grounding_warnings", []) or []
                if warnings:
                    st.warning("Grounding warnings")
                    st.json(warnings)

            st.image(img_bytes, caption=s.diagram.prompt)

            s.diagram.approved = st.checkbox(
                f"Approve diagram for {s.slide_id}",
                value=bool(s.diagram.approved),
                key=f"approve_{s.slide_id}",
            )

        save = st.form_submit_button(
            "Save approvals and continue", type="primary", use_container_width=True
        )

    if save:
        st.session_state.deck_plan = plan
        total, approved = count_diagrams(plan, st.session_state.get("diagram_images"))
        st.success(f"Approvals saved ({approved}/{total}). Moving to Step 3…")
        st.session_state.wizard_step = 3
        st.rerun()

# ----------------------------
# STEP 3
# ----------------------------
if st.session_state.wizard_step == 3:
    st.subheader("Step 3 - Render PPTX and Download Outputs")
    if st.session_state.get("render_complete"):
        render_step_progress(1.0, "Step 3 complete: proposal generation finished.")
    else:
        render_step_progress(0.2, "Step 3 progress: ready to render.")

    if st.session_state.deck_plan is None or st.session_state.tpl_bytes is None:
        st.error("Missing required state. Please complete Step 1 first.")
        st.stop()

    plan: DeckPlan = st.session_state.deck_plan
    tpl_bytes = st.session_state.tpl_bytes
    st.session_state.diagram_images = load_cached_diagram_images(
        plan,
        st.session_state.get("diagram_images"),
        model=diagram_model,
        size=diagram_size,
        quality=diagram_quality,
    )

    diagram_images = st.session_state.get("diagram_images") or {}
    total, approved = count_diagrams(plan, diagram_images)
    final_pages = rendered_slide_count(plan, diagram_images, tpl_bytes)
    cols = st.columns(4)
    cols[0].metric("Planned slides", value=len(plan.slides))
    cols[1].metric("Final PPTX slides", value=final_pages)
    cols[2].metric("Usable diagrams", value=total)
    cols[3].metric("Approved for PPTX", value=approved)

    missing_approved = [
        slide.title
        for slide in plan.slides
        if slide.diagram
        and slide.diagram.approved
        and diagram_image_for_slide(slide, diagram_images) is None
    ]
    if missing_approved:
        st.warning(
            "Approved diagram assets are missing and will render as recovery text: "
            + "; ".join(missing_approved)
            + ". Regenerate only these diagrams before customer delivery."
        )

    if enable_diagrams and total > 0 and approved == 0:
        st.warning("No diagrams are approved. They will NOT be inserted into the PPTX.")

    render_now = st.button("Render PPTX", type="primary", use_container_width=True)

    if render_now:
        render_progress = st.progress(0.2)
        render_status = st.status("Rendering outputs...", expanded=False)
        out_name = build_output_filename(plan, st.session_state.get("rfp_names"))
        try:
            pptx_bytes = render_deck_from_template(
                plan,
                tpl_bytes,
                out_path=None,
                diagram_images=st.session_state.get("diagram_images"),
                customer_logo=st.session_state.get("customer_logo_bytes"),
            )
            render_progress.progress(0.7)

            report_bytes = None
            if st.session_state.report:
                report_bytes = st.session_state.report.model_dump_json(indent=2).encode("utf-8")
            render_progress.progress(1.0)
            render_status.update(label="Render complete.", state="complete")
            st.session_state.render_complete = True

            st.success("Rendered PPTX successfully.")

            st.download_button(
                "Download PPTX",
                data=pptx_bytes,
                file_name=out_name,
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                use_container_width=True,
            )

            if report_bytes is not None:
                st.download_button(
                    "Download Traceability Report (JSON)",
                    data=report_bytes,
                    file_name="traceability.json",
                    mime="application/json",
                    use_container_width=True,
                )
        except Exception as exc:
            stop_on_error("Render failed. See details below.", render_status, exc)

    with st.expander("Deck Plan JSON (current session)"):
        st.json(plan.model_dump())
    rep = st.session_state.report
    with st.expander("Traceability Report (current session)"):
        st.json(rep.model_dump() if rep else {})
