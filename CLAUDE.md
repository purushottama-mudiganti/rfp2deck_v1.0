# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An RFP → PowerPoint proposal generator: Streamlit UI + a LangGraph agent pipeline (OpenAI) that
produces a `DeckPlan`, then a deterministic `python-pptx` renderer turns that plan into a `.pptx`
against a bundled HCLTech corporate template. See `README.md` for the full product description,
environment variables, and the 3-step UI wizard — this file only covers what a coding agent needs
that isn't there.

## Environment (Windows-specific — get this wrong and everything silently fails)

Two virtualenvs exist at the repo root and are **not interchangeable**:
- `.venv-win` — created with native Windows Python. **Use this one** for any command run from
  PowerShell, cmd, or Git Bash acting as a Windows process: `.venv-win\Scripts\python.exe`.
- `.venv` — created under WSL (`/usr/bin/python3.12`, `/mnt/c/...` paths). Only valid inside WSL.

Neither `pytest` nor `black`/`isort` are installed in `.venv-win`, even though `pyproject.toml`
configures `black`/`isort` (line-length 100). Don't assume `pip install -r requirements.txt` pulled
dev tools — it didn't; there's no separate dev-requirements file. Install them yourself if you need
them, or just use the commands below.

## Commands

```powershell
# Install
.venv-win\Scripts\python.exe -m pip install -r requirements.txt

# Run the app
.venv-win\Scripts\python.exe -m streamlit run app/rfp2deck_app.py

# Run the full test suite (pytest is NOT installed — use unittest)
.venv-win\Scripts\python.exe -m unittest tests.test_source_package tests.test_structured_retry

# Run one test case / one test method
.venv-win\Scripts\python.exe -m unittest tests.test_source_package.SourcePackageTests
.venv-win\Scripts\python.exe -m unittest tests.test_source_package.SourcePackageTests.test_renderer_preserves_cards_on_split_pages
```

There is no lint/CI script in the repo; `black`/`isort` config in `pyproject.toml` is aspirational
until those packages are installed.

## Architecture

### The pipeline is a fan-out/fan-in LangGraph, not a straight line

`rfp2deck/agent/graph.py` wires **11 nodes** (`build_graph()` logs this count; trust the code over
any older diagram you find elsewhere). After `understand_rfp`, three analyses that only depend on
`understanding` run in parallel — `derive_sections`, `build_narrative`, `derive_technology_recommendations`
— and merge back together (`derive_sections`+`build_narrative` → `derive_visual_briefs`;
`derive_visual_briefs`+`derive_technology_recommendations` → `plan_deck`) before `plan_deck` runs.
State is one shared `AgentState` (Pydantic, `rfp2deck/agent/state.py`); every node reads/enriches it.
Each node is wrapped by `_logged_node` for START/DONE/duration logging and surfaces the failing node
on error — check `logs/rfp2deck.log` first when a run dies partway through.

### Content vs. render is a hard separation, and post-processing is deliberately not LLM-driven

The LLM produces a `DeckPlan` (`plan_deck`); everything after that — required-slide backfill,
ordering (`order_deck`), text polish, dedup across slides (`_dedupe_across_slides`), diagram
attachment, section-divider insertion (`insert_section_dividers`), the win-theme slide
(`insert_win_theme_slide`) — is pure Python in `rfp2deck/agent/nodes.py`, not another model call.
This is intentional: it keeps deck structure testable and reproducible without mocking an LLM. When
fixing a "the deck looks wrong" bug, check whether it's a content problem (`nodes.py`/`prompts.py`)
or a rendering problem (`rendering/pptx_renderer.py`) before touching either — they fail
independently and look similar from a screenshot.

### The renderer has two template modes; the interesting one is native HCLTech layout selection

`render_deck_from_template` accepts a `.pptx` or `.potx`. A `.potx` can't be opened by `python-pptx`
directly — `rfp2deck/ingestion/template_resolver.py::resolve_pptx_template` rewrites the package's
`[Content_Types].xml` content-type declaration and caches the converted copy under
`TEMPLATE_CACHE_DIR` (`.data/templates/`), keyed by a hash of the source path + size + mtime. If a
template edit doesn't seem to take effect, check for a stale cached copy there.

The bundled `templates/hcltech_expanded_v5.potx` is the official HCLTech "Expanded Version" template
with 200+ slide layouts — this is a fundamentally different, more sophisticated rendering path than
a simple two-layout theme. `_choose_hcltech_layout` in `pptx_renderer.py` maps each `SlideSpec`
(archetype + populated `cards`/`detailed_points`/`bullets`/`diagram`) to one specific native layout by
name/token matching. Invariants worth knowing before touching this function:
- It must **never return `None`** — every branch either resolves a layout or falls through to a
  content-shaped catch-all (single-statement / three-or-four-key-points) at the bottom.
- **Never force an N-box layout when the actual populated content count differs from N** — several
  archetype branches used to do this (e.g. routing an empty "Timeline"/"team" slide with no approved
  diagram straight to a 4-box or org-chart layout) and it rendered with most boxes empty. The fix
  pattern is: only pick a diagram-only/fixed-count layout when there's real content or a diagram to
  put in it; otherwise fall through to the generic catch-all.
- A **layout "variety" pool** (`_pick_varied_layout` + `_TWO_KEY_POINT_LAYOUTS` /
  `_THREE_KEY_POINT_LAYOUTS` / `_FOUR_KEY_POINT_LAYOUTS`) rotates between sibling layouts so a long
  deck doesn't repeat one skin. This must be chosen **once per original slide, before pagination
  splits it into "(1 of N)"/"(2 of N)" pages** (`_lock_continuation_layouts`, called once up front in
  `render_deck_from_template`) — choosing it per rendered page instead makes a slide's own
  continuation pages land on different layouts.
- `_remove_unused_placeholders` deletes any placeholder with no text/table/chart/image — but some
  layouts carry pure decoration (e.g. "Sidebar 1"/"Sidebar 2" accent panels) in a real placeholder
  shape that's never meant to hold text. Those are exempted by name pattern; if a new layout's
  decoration goes missing after render, this is the first place to check.

### Verifying a rendering change without spending an LLM call

`render_deck_from_template(deck_plan, template_pptx, out_path, diagram_images)` is a pure function of
a `DeckPlan` — you don't need to run the agent pipeline to test a renderer fix. Cached plans from real
runs live at `.data/pipeline_cache/<hash>.json` as `{"deck_plan": {...}, ...}`; load one with
`DeckPlan.model_validate(data["deck_plan"])` and render it directly. Cached diagram PNGs live
alongside as `.data/pipeline_cache/<hash>.png`. For a quick visual check, PowerPoint COM automation
works from PowerShell: `New-Object -ComObject PowerPoint.Application`, `Presentations.Open(...)`,
`Slides.Item(n).Export(path, "PNG", w, h)`.

### Content must stay engagement-agnostic

`nodes.py` generates content for arbitrary RFPs — never hardcode a customer name, a specific tech
stack, or an industry assumption into a content-generation function. Existing generic helpers to
reuse rather than duplicate: `_customer_label(understanding)`, `_is_data_platform_engagement(understanding)`
(gates data/analytics-specific content on explicit positive scope signals only), `_cloud_signal(corpus)`.
A prior regression from skipping this: a Kubernetes-migration deck came out with a Fabric/OneLake data
stack because a helper assumed every engagement was a data platform.
