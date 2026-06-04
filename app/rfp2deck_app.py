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
    from rfp2deck.core.schemas import DeckPlan, TraceabilityReport
    from rfp2deck.diagrams.generator import generate_diagram_png
    from rfp2deck.ingestion.deck_analyzer import analyze_pptx_template
    from rfp2deck.ingestion.docx_parser import parse_docx
    from rfp2deck.ingestion.pdf_parser import parse_pdf
    from rfp2deck.rag.indexer import build_faiss_index, chunk_text, load_index
    from rfp2deck.rag.retriever import retrieve
    from rfp2deck.rendering.pptx_renderer import render_deck_from_template
except ModuleNotFoundError:
    # Ensure local package imports work when running via `streamlit run app/rfp2deck_app.py`.
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from rfp2deck.agent.graph import build_graph
    from rfp2deck.agent.state import AgentState
    from rfp2deck.core.config import settings
    from rfp2deck.core.logging import setup_logging
    from rfp2deck.core.schemas import DeckPlan, TraceabilityReport
    from rfp2deck.diagrams.generator import generate_diagram_png
    from rfp2deck.ingestion.deck_analyzer import analyze_pptx_template
    from rfp2deck.ingestion.docx_parser import parse_docx
    from rfp2deck.ingestion.pdf_parser import parse_pdf
    from rfp2deck.rag.indexer import build_faiss_index, chunk_text, load_index
    from rfp2deck.rag.retriever import retrieve
    from rfp2deck.rendering.pptx_renderer import render_deck_from_template

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
st.session_state.setdefault("rag_index", None)

# Embedded standard template path (no UI upload required)
STANDARD_TEMPLATE = PROJECT_ROOT / "templates" / "standard_proposal_template_v1.pptx"


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

    st.write("Models are configured via `.env`.")
    st.code(
        "Reasoning model: {reasoning}\n"
        "Fast model: {fast}\n"
        "Embeddings: {embeddings}".format(
            reasoning=settings.model_reasoning,
            fast=settings.model_fast,
            embeddings=settings.embeddings_model,
        )
    )

    enable_notes = st.checkbox(
        "Generate speaker notes (presenter notes per slide)", value=True
    )
    enable_diagrams = st.checkbox("Enable diagram generation (guarded + approval)", value=True)
    diagram_model = st.text_input("Diagram model", value="gpt-image-1")
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
    if STANDARD_TEMPLATE.exists():
        st.success(f"Using embedded template: {STANDARD_TEMPLATE.name}")
    else:
        st.error(
            "Embedded template missing. Ensure templates/standard_proposal_template_v1.pptx exists."
        )

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
            "rag_index",
            "render_complete",
        ]
        for k in keys:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()


# ----------------------------
# Helpers
# ----------------------------
def parse_rfp(upload) -> tuple[str, dict]:
    """Parse a single RFP upload and return its extracted text and metadata."""
    name = getattr(upload, "name", "rfp")
    suffix = Path(name).suffix.lower()
    data = upload.getvalue()
    if suffix == ".pdf":
        parsed = parse_pdf(data)
        st.success(f"Parsed PDF ({parsed.page_count} pages).")
        return (
            parsed.text,
            {"name": name, "type": "pdf", "pages": parsed.page_count},
        )
    parsed = parse_docx(data)
    st.success(f"Parsed DOCX ({parsed.paragraph_count} paragraphs).")
    return (
        parsed.text,
        {
            "name": name,
            "type": "docx",
            "paragraphs": parsed.paragraph_count,
        },
    )


def parse_rfps(uploads: list) -> tuple[str, list[dict], int, int]:
    """Parse multiple RFPs and return combined text and summary stats."""
    texts: list[str] = []
    summaries: list[dict] = []
    total_pages = 0
    total_paragraphs = 0
    for upload in uploads:
        name = getattr(upload, "name", "rfp")
        st.info(f"Parsing {name}...")
        text, meta = parse_rfp(upload)
        texts.append(text)
        summaries.append(meta)
        if meta.get("type") == "pdf":
            total_pages += int(meta.get("pages", 0))
        else:
            total_paragraphs += int(meta.get("paragraphs", 0))
    return "\n\n".join(texts), summaries, total_pages, total_paragraphs


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
        if s.diagram and (s.slide_id in diagram_images):
            total += 1
            if bool(s.diagram.approved):
                approved += 1
    return total, approved


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
    st.subheader("Step 1 - Upload RFP and Generate Deck Plan (Standard Template)")
    step1_progress_slot = st.empty()

    def show_step1_progress(value: float, text: str) -> None:
        with step1_progress_slot.container():
            render_step_progress(value, text)

    show_step1_progress(0.05, "Step 1 progress: waiting for inputs.")

    c1, c2 = st.columns(2)
    with c1:
        rfp_files = st.file_uploader(
            "Upload RFP file(s) (PDF/DOCX)",
            type=["pdf", "docx"],
            key="rfp_step1",
            accept_multiple_files=True,
        )
    with c2:
        ref_txt = st.file_uploader(
            "Optional: Reusable content (TXT) for RAG", type=["txt"], key="ref_step1"
        )
        st.caption("Enable 'Build/Update RAG index' in sidebar to index this TXT.")

    if rfp_files:
        show_step1_progress(0.1, "Step 1 progress: ready to generate plan.")

    with st.form("step1_form"):
        submitted = st.form_submit_button(
            "Generate Plan (Step 1)", type="primary", use_container_width=True
        )

    if submitted:
        show_step1_progress(0.15, "Step 1 progress: generating plan...")
        step_progress = st.progress(0.0)
        step_status = st.status("Step 1 in progress...", expanded=False)
        step_progress.progress(0.1)
        if not STANDARD_TEMPLATE.exists():
            st.error(
                "Embedded template is missing. Please restore templates/standard_proposal_template_v1.pptx"
            )
            st.stop()

        if not rfp_files:
            st.error("Please upload at least one RFP file (PDF or DOCX).")
            st.stop()

        try:
            rfp_names = [getattr(f, "name", "rfp") for f in rfp_files]
            st.session_state.rfp_names = rfp_names
            step_progress.progress(0.3)

            tpl_bytes = STANDARD_TEMPLATE.read_bytes()
            st.session_state.tpl_bytes = tpl_bytes

            rfp_text, rfp_summaries, total_pages, total_paragraphs = parse_rfps(rfp_files)
            step_progress.progress(0.5)

            if rfp_summaries:
                st.markdown("**RFP upload summary**")
                rows = []
                for s in rfp_summaries:
                    if s.get("type") == "pdf":
                        rows.append(
                            {
                                "File": s["name"],
                                "Type": "PDF",
                                "Count": f'{s.get("pages", 0)} pages',
                            }
                        )
                    else:
                        rows.append(
                            {
                                "File": s["name"],
                                "Type": "DOCX",
                                "Count": f'{s.get("paragraphs", 0)} paragraphs',
                            }
                        )
                st.table(rows)
                st.caption(f"Totals: {total_pages} pages, {total_paragraphs} paragraphs")

            # Analyze template layouts/placeholders
            ti = analyze_pptx_template(tpl_bytes)
            template_info = {
                "slide_layout_names": ti.slide_layout_names,
                "masters": ti.masters,
                "placeholder_map": ti.placeholder_map,
            }
            st.session_state.template_info = template_info
            st.info(f"Template analyzed: {len(ti.slide_layout_names)} layouts found.")
            step_progress.progress(0.7)

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
            step_progress.progress(0.85)

            # Run agent
            graph = build_graph()
            state = AgentState(
                rfp_text=rfp_text,
                template_info=template_info,
                retrieved_context=retrieved_context,
            )
            state.deck_mode = st.session_state.get("deck_mode")
            state.enable_notes = enable_notes

            log.info(
                "Invoking agent pipeline (rfp_chars=%d, deck_mode=%s, rag=%s)",
                len(rfp_text or ""),
                st.session_state.get("deck_mode"),
                retrieved_context is not None,
            )
            final_state = graph.invoke(state)
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

            st.session_state.deck_plan = deck_plan
            st.session_state.report = report
            st.session_state.diagrams_generated = False  # reset for new run
            st.session_state.diagram_images = {}
            st.session_state.render_complete = False
            step_progress.progress(1.0)
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

        slides_with_diagrams = [s for s in plan.slides if s.diagram]
        total_targets = len(slides_with_diagrams)
        progress = st.progress(0)
        status = st.status("Generating diagrams...", expanded=False)

        try:
            made = 0
            for idx, s in enumerate(slides_with_diagrams, start=1):
                if not s.diagram:
                    continue
                img_bytes = generate_diagram_png(
                    s.diagram.prompt,
                    out_path=None,
                    model=diagram_model,
                    size=diagram_size,
                    quality=diagram_quality,
                )
                diagram_images[s.slide_id] = img_bytes
                made += 1
                status.update(
                    label=f"Generated {idx}/{total_targets} diagrams",
                    state="running",
                )
                progress.progress(idx / max(total_targets, 1))

            status.update(label="Diagram generation complete.", state="complete")
            st.session_state.deck_plan = plan
            st.session_state.diagram_images = diagram_images
            st.session_state.diagrams_generated = True
            st.success(f"Generated {made} diagram(s). Now approve below.")
            st.rerun()
        except Exception as exc:
            stop_on_error("Diagram generation failed. See details below.", status, exc)

    diagram_images = st.session_state.get("diagram_images", {})
    any_images = any((s.diagram and s.slide_id in diagram_images) for s in plan.slides)
    if not any_images:
        st.info("No diagram images generated yet. Click **Generate / Regenerate Diagrams** above.")
        st.stop()

    st.markdown("### Diagram Review & Approval")

    with st.form("diagram_approvals_form"):
        for s in plan.slides:
            if not s.diagram or s.slide_id not in diagram_images:
                continue

            st.markdown(f"""**{s.slide_id} — {s.title}**  
Kind: `{s.diagram.kind}`""")

            st.image(diagram_images[s.slide_id], caption=s.diagram.prompt)

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

    total, approved = count_diagrams(plan, st.session_state.get("diagram_images"))
    cols = st.columns(3)
    cols[0].metric("Slides", value=len(plan.slides))
    cols[1].metric("Diagrams generated", value=total)
    cols[2].metric("Diagrams approved", value=approved)

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
