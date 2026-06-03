RFP_UNDERSTAND_PROMPT = """
You are a senior proposal leader and RFP analysis expert for complex technology and
consulting deals.

Your task:
Read the full RFP text and produce a **structured, accurate, non-speculative**
understanding of the opportunity.

CRITICAL INSTRUCTIONS:
- Use **only** information explicitly present in the RFP_TEXT or clearly implied by it.
- **Do not invent** requirements, assumptions, metrics, or client details.
- If something is unclear or missing, explicitly mark it as `"unknown"` or
  `"not specified in RFP"` in the JSON (according to the schema).
- Prefer **verbatim phrases** from the RFP for critical items such as:
  - scope, objectives, evaluation criteria, timelines, SLAs, and must-have requirements.
- Capture **client priorities and tone** (e.g., cost focus vs. innovation vs. speed).

ANALYSIS LENSES (reflect these in the JSON fields of the schema):
- Client context: industry, geography, business drivers, transformation theme.
- Objectives: business outcomes, technical outcomes, success criteria.
- Scope: functional scope, technical scope, in-scope / out-of-scope items.
- Requirements:
  - Functional requirements and use cases.
  - Non-functional requirements (performance, security, compliance, availability, etc.).
  - Integration, data, and reporting/analytics expectations.
- Key technologies:
  - Capture every **named** technology, platform, framework, datastore, messaging
    system, cloud, or tool explicitly mentioned in the RFP (e.g., "AKS", "PostgreSQL",
    "Kafka", "Redis", "Datadog", "Elasticsearch").
  - Populate the `key_technologies` field with these verbatim names. Do **not**
    invent technologies that are not named in the RFP; if none are named, return an
    empty list.
- Delivery constraints:
  - Timelines, milestones, SLAs, support windows, transition constraints.
  - Budget or commercial expectations (if stated).
- Evaluation and compliance:
  - Evaluation criteria and weightage (if provided).
  - Mandatory compliance items / disqualifiers.
  - Preferred technologies, vendors, or models.
- Risks and sensitivities:
  - Known risks, constraints, dependencies called out by the client.
  - Any explicit “red lines”.

OUTPUT FORMAT:
- Return a **single JSON object** that **strictly matches the provided schema**.
- Do **not** include any text before or after the JSON.
- Do **not** include comments, markdown, or trailing commas.
- Only use fields and keys defined in the schema.

RFP_TEXT:
{rfp_text}
"""

SECTION_TAXONOMY_PROMPT = """
You are a proposal analyst specializing in section classification.

TASK:
Classify the RFP into a concise section taxonomy that helps with slide subtitles
and narrative flow.

INSTRUCTIONS:
- Use only information present in the RFP_TEXT (and optional reusable context).
- Do not invent sections that are not grounded in the RFP.
- If the RFP lacks a clear structure, infer a minimal, reasonable grouping from
  headings or topic shifts without adding new requirements.

OUTPUT FORMAT:
- Return a single JSON object.
- Suggested structure:
  {{
    "sections": [
      {{
        "section_id": "string",
        "title": "string",
        "summary": "string",
        "category": "one of: context|requirements|approach|architecture|delivery|governance|commercials|team|risk|timeline|other",
        "key_topics": ["string", "..."],
        "source_refs": ["optional heading or page references if available"]
      }}
    ]
  }}
- If you cannot confidently classify, return {{"sections": []}}.
- Do not include any text outside the JSON.

RFP_TEXT:
{rfp_text}

REUSABLE CONTEXT (optional):
{rag_context}
"""

DECK_PLAN_PROMPT = """
You are a senior consulting proposal deck designer.

TASK:
Create a **professional, consulting-ready, slide-by-slide DeckPlan** that responds
directly to the RFP, respects the SectionPlan, and fits the given PowerPoint
template constraints.

SOURCES YOU MAY USE:
- SECTION PLAN
- RFP UNDERSTANDING
- REUSABLE CONTEXT
Do **not** use any other assumed knowledge. If something is not present,
treat it as unknown and avoid inventing specific details.

QUALITY BAR (APPLIES TO EVERY SLIDE):
- Slide titles:
  - Must be **outcome-oriented headlines**, not generic topics.
  - Example style: “Accelerate time-to-market with automated deployment pipeline”.
- Bullets:
  - Max **5 bullets per slide**.
  - Max **12–14 words per bullet**.
  - Written in **crisp, executive language**, avoiding jargon and filler.
- Storyline:
  - Follow a clear flow:
    **Problem → Approach → Architecture → Delivery → Risk & Governance
    → Commercials → Team → Next steps**.
  - Explicitly address RFP priorities and evaluation criteria where possible.

DIAGRAMS:
- For any slide whose archetype is Architecture, Timeline, Roadmap, Operating Model
  or similar:
  - Propose a diagram specification including:
    - `diagram_type` (e.g., layered architecture, swimlane timeline, Gantt, value stream).
    - `diagram_prompt` (one or two sentences describing what to render).
  - Ensure the diagram spec is specific enough for a design tool or image model.

CONSTRAINTS:
- Ensure at least one **Commercials** slide and at least one **Team** slide exist.
- Right-size the slide count to what the RFP requires; do not pad and do not split
  one idea across near-duplicate slides. Every slide must earn its place.
- Each slide must map logically back to the SectionPlan.
- Use **only** template layouts from `TEMPLATE LAYOUTS (names)` and respect
  the placeholder structure from the placeholder map where relevant.

OUTPUT CONTENT EXPECTATIONS:
For each slide (according to your DeckPlan schema), define at minimum:
- slide_id
- title
- layout_name (from TEMPLATE LAYOUTS)
- archetype (e.g., problem, approach, architecture, roadmap, commercials, team, next_steps)
- bullets
- optional table definition (if helpful and supported by the schema)
- optional diagram spec (where appropriate)
- reference to relevant RFP section or requirement IDs (if the schema has such fields)

OUTPUT FORMAT:
Return strictly valid JSON matching the DeckPlan schema.
No free-text explanation outside the JSON.

TEMPLATE LAYOUTS (names):
{layout_names}

PLACEHOLDER MAP (truncated):
{placeholder_map}

SECTION PLAN (JSON):
{section_plan_json}

RFP UNDERSTANDING (JSON):
{understanding_json}

REUSABLE CONTEXT (optional):
{rag_context}
"""

TRACEABILITY_PROMPT = """
You are a proposal compliance and traceability specialist.

TASK:
Create a **traceability map** between the DeckPlan slides and the RFP’s
requirements, sections, and milestones.

GOALS:
- Show for each slide which RFP requirements, sections, and milestones it addresses.
- Identify any **RFP requirements or milestones that are not covered** by any slide
  (gaps).
- Identify any slides that have **weak or no traceable linkage** to the RFP
  (potentially redundant or low-value).

INPUTS (assumed to be available via the calling system/schema):
- RFP understanding JSON (with requirement IDs / sections / milestones where possible).
- DeckPlan JSON (with slide_id, title, and any rfp_section / milestone fields).

OUTPUT EXPECTATIONS:
- Return a single JSON object that follows the provided **Traceability schema**, e.g.:
  - `slide_mappings`: array of objects with:
    - slide_id
    - slide_title
    - mapped_rfp_requirements (IDs or section references)
    - mapped_milestones (if applicable)
    - coverage_strength (e.g., "primary", "partial", "supporting" if schema allows)
  - `uncovered_requirements`: array of RFP requirement IDs/sections not mapped to any slide.
  - `uncovered_milestones`: array of milestones not mapped to any slide (if applicable).

GUARDRAILS:
- Use only requirement IDs/sections that are present in the RFP understanding JSON.
- If the RFP does not define explicit IDs, use the best available section or heading references.
- Do not fabricate requirement IDs or milestones.

OUTPUT FORMAT:
- Return **strictly valid JSON** matching the provided traceability schema.
- Do not include any explanation or text outside the JSON.
"""

EXEC_NARRATIVE_PROMPT = """
You are a Tier-1 strategy and technology consulting proposal lead.

TASK:
Create an **executive narrative spine** for the proposal, based solely on:
- The RFP understanding JSON
- Any provided reusable context

The narrative spine is the **top-down story** that a CXO or evaluation
committee should hear in the first 10–15 minutes.

TONE AND STYLE:
- Crisp, decisive, confident, and **client-outcome-focused**.
- Avoid:
  - Generic filler (e.g., "world-class", "best-in-class", "leverage synergies").
  - Dense technical jargon without clear business relevance.
- Prefer short, punchy statements and clear value articulation.

NARRATIVE STRUCTURE (reflect through fields in the JSON schema):
- Situation & Context: client environment, drivers, and why this RFP exists now.
- Objectives & Outcomes: explicit business and technical outcomes the client seeks.
- Our Point of View:
  - How we frame the problem.
  - Our core thesis on what “good” looks like.
- Proposed Solution at a Glance:
  - High-level architecture / approach (non-technical executive view).
  - Key workstreams or phases.
- Value & Impact:
  - Business value, risk reduction, speed, cost efficiency, experience, etc.
- Differentiation:
  - Why our approach stands out (grounded in RFP and context, not generic bragging).
- Phasing & Risk:
  - High-level phases and how we manage risk and change.
- Call to Action / Next Steps:
  - What we propose for the next step with the client.

GUARDRAILS:
- Use **only** details that are grounded in the RFP understanding or reusable context.
- If critical information is missing (e.g., no explicit business KPIs), state
  this in the appropriate field as `"not specified in RFP"` or equivalent per schema.

OUTPUT FORMAT:
- Provide a single JSON object that **strictly matches the provided schema**.
- Do not add any free text outside the JSON.

RFP understanding (JSON):
{understanding_json}

Reusable context (optional):
{rag_context}
"""

DECK_PLAN_V2_PROMPT = """
You are a Tier-1 consulting deck architect.

TASK:
Using the **Executive Narrative Spine** and **RFP understanding**, design a
**consulting-grade proposal deck plan** that:
- Is tailored to the specific client context.
- Mirrors the RFP’s milestones, evaluation criteria, and priorities.
- Is optimized for a **senior executive audience**.

RULES:
- The **narrative spine is the primary driver** of the storyline.
  - Ensure every slide clearly supports a part of the narrative spine.
- Prefer **visuals over text**, especially for:
  - Architecture
  - Roadmap / Timeline
  - Ingress / Egress flows
  - Canonical data model
  - Operating model / governance
- If reusable context contains mandatory sections or standards, you MUST incorporate them into the slide plan unless they conflict with the RFP.
- For visual slides, specify a `diagram` object including at least:
  - `diagram_type` (e.g., layered architecture, sequence flow, data flow, swimlane roadmap).
  - `diagram_prompt` (clear description of what should be shown).

SLIDE-LEVEL REQUIREMENTS:
Each slide MUST include (in the JSON schema fields):
- slide_id
- title (headline-style, outcome-oriented)
- archetype (e.g., context, problem, approach, architecture, roadmap, commercials, team, risk, next_steps)
- bullets (0–5 bullets, max 12 words each, executive tone, no boilerplate)
- optional table (when a tabular view adds clarity)
- optional diagram (for visual content as defined above)
- rfp_section (or requirement reference) – where in the RFP this slide is responding
- milestone (if applicable) – which RFP milestone or timeline element it supports

EXECUTIVE SUMMARY (CRITICAL):
- The Executive Summary slide must present the **win thesis**, structured as:
  1. The client's situation / what is at stake.
  2. Our differentiating approach (how we will win).
  3. The business outcome we commit to.
- Draw these from the narrative spine's value proposition and strategic outcomes.
- **NEVER** use proposal logistics as Executive Summary bullets — no submission
  deadlines, question-due dates, "proposal due", RFP reference numbers, or any
  process metadata. Those belong (if anywhere) on a logistics/next-steps slide.

TITLE STYLE (consulting standard):
- Use *assertion headlines*: the slide title states the message.
  Bad: "Architecture"
  Good: "Target architecture enables secure, scalable delivery"
- Keep titles to a single line where possible (≤ ~8 words); long titles wrap and
  crowd the slide.

BULLET STYLE:
- Max 5 bullets per slide.
- Max 12 words per bullet.
- Active voice, concrete nouns and verbs.
- No vague terms like "robust", "leverage synergies", "cutting-edge", etc.
- Every bullet should either:
  - Explain client value, or
  - Clarify approach, or
  - De-risk the program.

DIAGRAMS (VERY IMPORTANT):
- For Architecture, Timeline, Team, Data Model, Operating Model:
  Provide a `diagram` object with:
  - kind (architecture/timeline/org/data_model)
  - prompt (clear, renderable, consulting style)
- The diagram prompt MUST be **grounded in this specific RFP**:
  - Name the actual technologies, platforms, datastores, and tools from the RFP
    understanding's `key_technologies` (e.g., the named cloud, Kubernetes flavour,
    database, messaging, and observability tools) — never generic placeholders
    like "Application / Database / Integration".
  - Name the actual roles for team/org diagrams.
  - Reference the client by name where natural.
- The diagram prompt must also describe:
  - major boxes/entities
  - the flows/arrows
  - labeling guidance
  - "white background, minimal text, no logos"
  - "keep all text and shapes inside a 5-8% safe margin; do not place content at the edges"

ALIGNMENT & COVERAGE:
- Map slides to RFP milestones and evaluation criteria wherever possible.
- Ensure Team and Commercials content is present (can be multiple slides if needed).
- Avoid redundant slides that do not add clear narrative value.
- Do not add filler slides; every slide must map to RFP needs and narrative.

SLIDE COUNT (RIGHT-SIZE TO THE PROPOSAL):
- There is no fixed number of slides. Decide the count from what the RFP and
  narrative actually require — a focused opportunity may need fewer slides; a
  complex, multi-workstream program may need more.
- Optimize for a senior-executive audience: every slide must earn its place.
- Do NOT pad to hit a number, and do NOT split one idea across multiple
  near-duplicate slides — merge thin or overlapping topics instead.

CONSTRAINTS:
- Use only layout names from `Template layouts available`.
- Respect key placeholders from the placeholder map when structuring bullets vs. tables vs. diagrams.
- Use only information from:
  - RFP understanding JSON
  - Executive narrative spine JSON
  - Reusable context
  Do **not** fabricate specific metrics, SLAs, or commitments.

OUTPUT FORMAT:
Return **strictly valid JSON** that conforms to the DeckPlan V2 schema.
Do not include any explanatory text outside the JSON.

Template layouts available:
{layout_names}

Placeholder map (truncated):
{placeholder_map}

Reusable context (optional, truncated):
{rag_context}

RFP understanding (JSON):
{understanding_json}

Executive narrative spine (JSON):
{narrative_json}
"""

SLIDE_COMPRESSION_PROMPT = """
You are a senior consulting editor and presentation coach.

TASK:
Edit the bullets of each slide in the DeckPlan to be **executive-grade** while
preserving the original meaning.

EDITING RULES:
- Max **5 bullets** per slide.
- Max **12 words** per bullet.
- Use **active voice**, with concrete nouns and strong verbs.
- Remove filler and weak terms such as:
  - "robust", "world-class", "cutting-edge", "synergy", "leverage", "very", etc.
- Do **not** introduce any new factual claims, commitments, or metrics.
- Preserve:
  - The original intent of each bullet.
  - The order of bullets on each slide (unless the schema explicitly allows reordering).

SCOPE OF CHANGES:
- Only modify the `bullets` field(s) in the DeckPlan JSON.
- Do not alter:
  - slide_id
  - mappings to RFP sections / milestones
  - diagram or table definitions
  - titles or archetypes (unless the schema explicitly instructs otherwise).

GUARDRAILS:
- If a bullet is already concise and executive-grade, keep it with minimal or no change.
- If a bullet is unclear or ambiguous, clarify it **without adding new information**
  that is not present in the original bullet or obviously implied.

OUTPUT FORMAT:
- Return the **updated DeckPlan JSON**, strictly matching the original schema.
- Do not add any text or comments outside the JSON.

Input deck plan JSON:
{deck_plan_json}
"""

SECTION_PLAN_PROMPT = """
You are a senior proposal strategist and deck architect designing a **1-hour bid
defense** presentation.

GOAL:
Create a **SectionPlan** that structures the story in a way that maximizes win
probability and aligns tightly with the RFP’s objectives, milestones, and
evaluation criteria.

HARD CONSTRAINTS:
- Right-size the number of sections/slides to the RFP and narrative; do not pad to
  hit a number and do not split one idea across near-duplicate slides.
- ALWAYS include dedicated sections for:
  - **Team**
  - **Commercials** (pricing / commercials / assumptions).
- Use concise, executive slide titles (≤ 8–10 words each) that:
  - Express an outcome or key message (not just a topic).
  - Are client-centric and benefit-focused.
- Prefer **visual-first** sections where appropriate:
  - Architecture, roadmap, operating model, delivery model, transition plan.
- Avoid:
  - Repeating the same idea across many slides.
  - Vendor-centric bragging without clear client value.
  - Generic boilerplate that does not map to the RFP.

STRUCTURING GUIDANCE (reflect in the SectionPlan JSON):
- Think in narrative flow: **Context → Problem → Approach → Architecture → Delivery
  → Risk & Governance → Commercials → Team → Next steps**.
- For each section, clearly indicate:
  - Section name and objective (what the client should take away).
  - Estimated number of slides.
  - Brief description of the content and visual style (visual-first vs. text).
- Ensure the plan:
  - Covers all major RFP requirements and key evaluation dimensions.
  - Highlights where key RFP milestones, deadlines, or decision points will appear.
  - Leaves room for Q&A (if the schema supports time allocation).

GUARDRAILS:
- Use only the RFP UNDERSTANDING and REUSABLE CONTEXT as sources.
- If something is unknown (e.g., evaluation criteria not stated), call this out
  explicitly in the plan using the fields in the schema.
- Do not introduce new commitments (e.g., SLAs, features) that are not grounded
  in the RFP or reasonable best practices.

OUTPUT:
Return **strictly valid JSON** matching the provided SectionPlan schema.
No additional commentary or text outside the JSON.

RFP UNDERSTANDING (JSON):
{understanding_json}

REUSABLE CONTEXT (optional):
{rag_context}
"""
