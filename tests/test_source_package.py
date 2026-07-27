from __future__ import annotations

import unittest
import base64
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from docx import Document
from openpyxl import Workbook
from openai import OpenAIError

from rfp2deck.agent.prompts import RFP_UNDERSTAND_PROMPT
from rfp2deck.agent.nodes import _extract_evidence_chunk, extract_source_evidence
from rfp2deck.agent.nodes import (
    enrich_understanding_risks,
    enforce_slide_density,
    order_deck,
    _compact_deck_plan_prompt,
    _fallback_deck_plan,
    _proposal_section_skeleton,
    _chunked_deck_plan,
    _risk_detailed_points,
    _ai_ml_opportunities,
    _sdlc_technology_table,
    _sbom_table,
    _build_diagram_prompt,
    _testing_proposal_points,
    _ams_proposal_points,
    _assumptions_dependency_points,
    _data_domain_points,
    ensure_diagrams_for_key_slides,
    ensure_required_slides,
    consulting_grade_proposal_polish,
    prune_redundant_storyline_slides,
)
from rfp2deck.agent.state import AgentState
from rfp2deck.diagrams import generator as diagram_generator
from rfp2deck.agent.evidence import (
    merge_evidence_batches,
    render_evidence_for_prompt,
    split_source_document,
)
from rfp2deck.core.schemas import (
    BulletPoint,
    DeckPlan,
    DiagramSpec,
    RFPUnderstanding,
    Requirement,
    SlideSpec,
    SourceDocument,
    SourceEvidenceBatch,
)
from rfp2deck.ingestion.docx_parser import parse_docx
from rfp2deck.ingestion.source_package import (
    build_source_reconciliation,
    classify_source,
    parse_source_document,
    render_source_package,
    sort_sources_by_precedence,
)
from rfp2deck.llm.structured import _dereference, _is_proxy_block_error, _make_strict
from rfp2deck.qa.coverage import build_traceability_report
from rfp2deck.rendering.pptx_renderer import (
    _dedupe_render_slides,
    _normalize_singleton_continuation_titles,
    _remove_overlapping_generated_pictures,
    _render_pages_for_slide,
    _repair_title_only_slide,
    _choose_hcltech_layout,
    rendered_slide_count,
)


class SourcePackageTests(unittest.TestCase):
    def test_compact_deck_plan_prompt_is_bounded(self) -> None:
        understanding = RFPUnderstanding(
            summary="A" * 3000,
            project_scope="B" * 3000,
            requirements=[
                Requirement(id=f"R-{i}", text="Requirement text " * 60, priority="must")
                for i in range(80)
            ],
            risks=["Risk text " * 40 for _ in range(20)],
            assumptions=["Assumption text " * 40 for _ in range(20)],
        )

        prompt = _compact_deck_plan_prompt(
            layout_names=[f"Layout {i}" for i in range(100)],
            understanding=understanding,
            narrative=None,
        )

        self.assertLess(len(prompt), 16000)
        self.assertIn("strict JSON", prompt)
        self.assertIn("top_requirements", prompt)

    def test_diagram_generator_uses_explicit_timeout_and_no_sdk_retries(self) -> None:
        png_b64 = base64.b64encode(b"fake-png").decode("ascii")
        client = SimpleNamespace(
            images=SimpleNamespace(
                generate=Mock(
                    return_value=SimpleNamespace(
                        data=[SimpleNamespace(b64_json=png_b64)]
                    )
                )
            )
        )
        settings = SimpleNamespace(
            image_model="gpt-image-2",
            image_timeout_s=321,
            image_retry_attempts=1,
        )

        with patch.object(diagram_generator, "settings", settings), patch.object(
            diagram_generator, "get_client", return_value=client
        ) as get_client:
            png = diagram_generator.generate_diagram_png(
                "draw architecture",
                out_path=None,
                model=None,
            )

        self.assertEqual(png, b"fake-png")
        get_client.assert_called_once_with(timeout=321.0, max_retries=0)
        client.images.generate.assert_called_once_with(
            model="gpt-image-2",
            prompt="draw architecture",
            size="auto",
            quality="auto",
        )

    def test_diagram_generator_retries_cloudflare_520_using_server_delay(self) -> None:
        png_b64 = base64.b64encode(b"retry-png").decode("ascii")
        transient = OpenAIError("Cloudflare 520")
        transient.status_code = 520
        transient.body = {"retryable": True, "retry_after": 60}
        success = SimpleNamespace(data=[SimpleNamespace(b64_json=png_b64)])
        client = SimpleNamespace(
            images=SimpleNamespace(generate=Mock(side_effect=[transient, success]))
        )
        settings = SimpleNamespace(
            image_model="gpt-image-2",
            image_timeout_s=300,
            image_retry_attempts=3,
            openai_retry_base_wait_s=5,
            openai_retry_max_wait_s=90,
        )

        with patch.object(diagram_generator, "settings", settings), patch.object(
            diagram_generator, "get_client", return_value=client
        ), patch.object(diagram_generator.time, "sleep") as sleep:
            png = diagram_generator.generate_diagram_png(
                "draw deployment architecture",
                out_path=None,
            )

        self.assertEqual(png, b"retry-png")
        self.assertEqual(client.images.generate.call_count, 2)
        sleep.assert_called_once_with(60.0)

    def test_risk_detailed_points_have_default_mitigations_when_no_risks_exist(self) -> None:
        understanding = RFPUnderstanding(
            customer_name="Customer",
            opportunity_title="Opportunity",
            summary="Summary",
            project_scope="Scope",
            risks=[],
        )

        points = _risk_detailed_points(understanding)

        self.assertGreaterEqual(len(points), 3)
        self.assertTrue(all(point.sub_points for point in points))

    def test_understanding_risks_are_inferred_from_delivery_signals(self) -> None:
        understanding = RFPUnderstanding(
            customer_name="Customer",
            opportunity_title="Data Hub",
            summary="Build a centralized data hub.",
            project_scope="Integrate source systems, extract data, validate reporting, and deploy to production.",
            in_scope_work=["API integration", "Data quality validation", "Production deployment"],
            risks=[],
        )

        enriched = enrich_understanding_risks(understanding)

        self.assertGreaterEqual(len(enriched.risks), 3)
        self.assertTrue(any("integration" in risk.lower() for risk in enriched.risks))
        self.assertTrue(any("data" in risk.lower() for risk in enriched.risks))
        self.assertTrue(any("deployment" in risk.lower() for risk in enriched.risks))

    def test_fallback_deck_plan_contains_core_storyline(self) -> None:
        understanding = RFPUnderstanding(
            customer_name="SATS",
            opportunity_title="Catering Uplift Data Hub",
            summary="SATS needs a centralized operational data hub.",
            project_scope="Design and build the data hub.",
            requirements=[
                Requirement(id="R-1", text="Automate ELP extraction", priority="must")
            ],
        )

        deck = _fallback_deck_plan(
            title="Catering Uplift Data Hub",
            understanding=understanding,
            narrative=None,
        )
        archetypes = [slide.archetype for slide in deck.slides]

        self.assertIn("Customer Context", archetypes)
        self.assertIn("Architecture", archetypes)
        self.assertIn("Deployment Architecture", archetypes)
        self.assertIn("Next Steps", archetypes)

    def test_adaptive_skeleton_adds_data_integration_and_support_sections(self) -> None:
        understanding = RFPUnderstanding(
            customer_name="SATS",
            opportunity_title="Catering Uplift Data Hub",
            summary="Build a data hub with ELP extraction, SAP and KSMS API integration, AMS support, and ICCMS migration.",
            project_scope="Data hub, reporting, migration, warranty and support.",
        )

        sections = _proposal_section_skeleton(understanding)
        ids = {section["slide_id"] for section in sections}

        self.assertIn("sk_integration", ids)
        self.assertIn("sk_data_model", ids)
        self.assertIn("sk_reporting", ids)
        self.assertIn("sk_ai_opportunities", ids)
        self.assertIn("sk_migration", ids)
        self.assertIn("sk_ams", ids)
        self.assertGreaterEqual(len(sections), 24)

    def test_ai_opportunities_are_grounded_in_data_hub_scope(self) -> None:
        understanding = RFPUnderstanding(
            customer_name="SATS",
            summary="Centralize flight, ELP, productivity, accuracy and SLA data for reporting and AMS support.",
            project_scope="Ingest email and ELP files, validate operational data, report exceptions, and support uplift operations.",
        )

        opportunities = _ai_ml_opportunities(understanding)

        self.assertGreaterEqual(len(opportunities), 3)
        self.assertTrue(any("Anomaly" in item["name"] for item in opportunities))
        self.assertTrue(all("GPU" not in item["approach"] for item in opportunities))

    def test_assumptions_cover_four_proposal_control_categories(self) -> None:
        understanding = RFPUnderstanding(
            customer_name="SATS",
            summary="Azure data hub with interfaces, UAT, cutover, warranty and AMS.",
            assumptions=["Representative source data will be available during mobilisation."],
            requirements=[
                Requirement(id="R-1", text="Customer provides interface access and sample ELP files.", priority="must"),
                Requirement(id="R-2", text="Confirm Azure hosting, identity and security controls.", priority="must"),
                Requirement(id="R-3", text="SATS owns UAT approval and production cutover sign-off.", priority="must"),
            ],
        )

        points = _assumptions_dependency_points(understanding)

        self.assertEqual(len(points), 4)
        self.assertEqual(
            [point.text for point in points],
            [
                "Scope and design baseline to validate",
                "Customer inputs and access dependencies",
                "Platform, security and environment decisions",
                "Acceptance and operational readiness dependencies",
            ],
        )
        self.assertTrue(all(point.sub_points for point in points))

    def test_data_domain_companion_content_is_authored_not_prompt_instructions(self) -> None:
        understanding = RFPUnderstanding(
            customer_name="SATS",
            summary="Operational data domains for the catering uplift hub.",
            requirements=[
                Requirement(id="R-1", text="Ingest FIH flight and SQ SFTP ELP source records.", priority="must"),
                Requirement(id="R-2", text="Track catering uplift productivity, accuracy and SLA milestones.", priority="must"),
                Requirement(id="R-3", text="Validate and reconcile records with complete audit lineage.", priority="must"),
                Requirement(id="R-4", text="Publish governed Power BI operational reports.", priority="must"),
            ],
        )

        points = _data_domain_points(understanding)

        self.assertEqual(len(points), 4)
        self.assertTrue(any("FIH" in " ".join(point.sub_points) for point in points))
        self.assertTrue(any("Power BI" in " ".join(point.sub_points) for point in points))

    def test_required_slides_add_ai_opportunity_and_cost_controls(self) -> None:
        understanding = RFPUnderstanding(
            customer_name="SATS",
            summary="Operational data hub for flights, ELP files, reporting, SLA exceptions, and support.",
            project_scope="Ingest email and files, validate data, support forecasting, reporting, and AMS.",
        )
        deck = DeckPlan(
            deck_title="Test",
            slides=[SlideSpec(slide_id="title", title="Test", archetype="Title")],
        )

        enriched = ensure_required_slides(deck, understanding=understanding)
        ai_slides = [slide for slide in enriched.slides if "ai-assisted" in slide.title.lower()]
        technology = _sdlc_technology_table(understanding)

        self.assertEqual(len(ai_slides), 1)
        self.assertGreaterEqual(len(ai_slides[0].cards), 3)
        self.assertTrue(any(row[0] == "AI-assisted capabilities" for row in technology["rows"]))

    def test_existing_technology_table_receives_ai_row(self) -> None:
        understanding = RFPUnderstanding(
            summary="Data hub for flight files, SLA exceptions, reporting and support.",
            project_scope="Validate operational data, forecast demand, and support analytics.",
        )
        deck = DeckPlan(
            deck_title="Test",
            slides=[
                SlideSpec(
                    slide_id="tech",
                    title="Technology stack",
                    archetype="Software Bill of Materials",
                    table={"headers": ["Phase", "Toolset", "Purpose", "Basis"], "rows": [["Build", "TBC", "Build", "Scope"]]},
                )
            ],
        )

        enriched = ensure_required_slides(deck, understanding=understanding)
        rows = next(slide.table["rows"] for slide in enriched.slides if slide.slide_id == "tech")

        self.assertTrue(any(row[0] == "AI-assisted capabilities" for row in rows))

    def test_azure_data_hub_stack_names_implementation_services(self) -> None:
        understanding = RFPUnderstanding(
            customer_name="SATS",
            summary="Central catering uplift data hub with reporting and SLA analytics.",
            project_scope="Ingest ELP files, validate canonical data, integrate APIs, and serve Power BI.",
            solution_technologies=["Power BI", "Microsoft Entra ID", "Microsoft Sentinel", "SAP Ariba"],
            procurement_or_submission_tools=["SAP Ariba"],
        )

        table = _sdlc_technology_table(understanding)
        text = " ".join(str(cell) for row in table["rows"] for cell in row)

        self.assertIn("Fabric Data Factory pipelines", text)
        self.assertIn("Microsoft Fabric OneLake", text)
        self.assertIn("Azure Functions", text)
        self.assertIn("Azure API Management", text)
        self.assertIn("Microsoft Sentinel", text)
        self.assertNotIn("Ariba", text)
        self.assertEqual(table["headers"][0], "Architecture layer")

    def test_procurement_tools_are_excluded_from_sbom_and_diagrams(self) -> None:
        understanding = RFPUnderstanding(
            customer_name="SATS",
            summary="Azure-hosted operational data hub with Power BI reporting.",
            project_scope="Ingest, validate, store, and report operational data.",
            solution_technologies=["Power BI", "SAP Ariba"],
            procurement_or_submission_tools=["Ariba"],
        )

        sbom_text = " ".join(str(cell) for row in _sbom_table(understanding)["rows"] for cell in row)
        diagram_prompt = _build_diagram_prompt("architecture", understanding)

        self.assertNotIn("Ariba", sbom_text)
        self.assertNotIn("Ariba", diagram_prompt)
        self.assertIn("Fabric Data Factory pipelines", diagram_prompt)

    def test_explicit_aws_signal_does_not_propose_fabric(self) -> None:
        understanding = RFPUnderstanding(
            summary="AWS-hosted operational data platform.",
            project_scope="Ingest files, expose APIs, store curated data, and provide reporting.",
            solution_technologies=["AWS", "Amazon S3"],
        )

        text = " ".join(str(cell) for row in _sdlc_technology_table(understanding)["rows"] for cell in row)

        self.assertIn("AWS Glue", text)
        self.assertIn("Amazon S3", text)
        self.assertNotIn("Microsoft Fabric", text)

    def test_architecture_prompt_places_ai_in_optional_sidecar(self) -> None:
        understanding = RFPUnderstanding(
            customer_name="SATS",
            summary="Data hub for flight, ELP, SLA, reporting and support data.",
            project_scope="Ingest files, validate operational data, report exceptions, and forecast uplift demand.",
        )
        deck = DeckPlan(
            deck_title="Test",
            slides=[
                SlideSpec(
                    slide_id="arch",
                    title="Target architecture",
                    archetype="Architecture",
                    diagram=DiagramSpec(kind="architecture", prompt="Show the data hub."),
                )
            ],
        )

        enriched = ensure_diagrams_for_key_slides(deck, understanding)
        prompt = enriched.slides[0].diagram.prompt.lower()

        self.assertIn("ai-assisted", prompt)
        self.assertIn("deterministic fallback", prompt)
        self.assertIn("no autonomous write-back", prompt)
        self.assertIn("without dedicated gpu", prompt)

    def test_non_data_proposal_does_not_force_ai_slide(self) -> None:
        understanding = RFPUnderstanding(
            customer_name="Client",
            summary="Provide a fixed classroom training schedule.",
            project_scope="Deliver instructor-led training sessions and attendance certificates.",
        )

        self.assertEqual(_ai_ml_opportunities(understanding), [])
        self.assertNotIn(
            "sk_ai_opportunities",
            {section["slide_id"] for section in _proposal_section_skeleton(understanding)},
        )

    def test_chunked_deck_plan_uses_placeholders_when_batch_call_fails(self) -> None:
        understanding = RFPUnderstanding(
            customer_name="SATS",
            opportunity_title="Catering Uplift Data Hub",
            summary="Build a data hub with ELP extraction, SAP and KSMS API integration, AMS support, and ICCMS migration.",
            project_scope="Data hub, reporting, migration, warranty and support.",
        )
        fake_settings = SimpleNamespace(
            deck_plan_batch_size=4,
            reasoning_effort_deck_plan="medium",
            deck_plan_timeout_s=30,
        )

        with patch("rfp2deck.agent.nodes.settings", fake_settings), patch(
            "rfp2deck.agent.nodes.response_as_schema",
            side_effect=RuntimeError("blocked"),
        ):
            deck = _chunked_deck_plan(
                title="Catering Uplift Data Hub",
                understanding=understanding,
                narrative=None,
            )

        self.assertGreaterEqual(len(deck.slides), 24)
        self.assertTrue(any(slide.slide_id == "sk_arch" for slide in deck.slides))
        self.assertTrue(any(slide.slide_id == "sk_risks" for slide in deck.slides))

    def test_storyline_order_keeps_context_before_non_exec_solution(self) -> None:
        deck = DeckPlan(
            deck_title="Test",
            slides=[
                SlideSpec(
                    slide_id="solution",
                    title="The proposed workstreams form one platform",
                    archetype="Solution Overview",
                    bullets=["Solution point"],
                ),
                SlideSpec(
                    slide_id="context",
                    title="Today's operating model creates avoidable effort",
                    archetype="Customer Context",
                    bullets=["Current challenge"],
                ),
                SlideSpec(
                    slide_id="exec",
                    title="Executive Summary",
                    archetype="Solution Overview",
                    bullets=["Executive thesis"],
                ),
            ],
        )

        ordered = order_deck(deck)
        ids = [slide.slide_id for slide in ordered.slides]

        self.assertLess(ids.index("exec"), ids.index("context"))
        self.assertLess(ids.index("context"), ids.index("solution"))

    def test_storyline_order_recognizes_exec_summary_by_title(self) -> None:
        deck = DeckPlan(
            deck_title="Test",
            slides=[
                SlideSpec(slide_id="commercials", title="Commercials Structure", archetype="Commercials"),
                SlideSpec(slide_id="exec", title="Executive Summary", archetype="Content", bullets=["Thesis"]),
                SlideSpec(slide_id="context", title="Current Challenges", archetype="Customer Context"),
            ],
        )

        ordered = order_deck(deck)
        ids = [slide.slide_id for slide in ordered.slides]

        self.assertLess(ids.index("exec"), ids.index("context"))
        self.assertLess(ids.index("exec"), ids.index("commercials"))

    def test_redundant_storyline_slides_are_pruned(self) -> None:
        deck = DeckPlan(
            deck_title="Test",
            slides=[
                SlideSpec(slide_id="title", title="Title", archetype="Title"),
                SlideSpec(slide_id="dep1", title="Deployment architecture", archetype="Deployment Architecture", bullets=["A"]),
                SlideSpec(slide_id="dep2", title="Recovery planning", archetype="High Availability & DR", bullets=["B"]),
                SlideSpec(slide_id="team1", title="Persistent squads", archetype="Team", bullets=["C"]),
                SlideSpec(slide_id="team2", title="Governance should resolve decisions", archetype="Delivery Plan", bullets=["Agile squad backlog sprint"]),
                SlideSpec(slide_id="team3", title="Agile roadmap", archetype="Timeline", bullets=["Sprint release backlog"]),
            ],
        )

        pruned = prune_redundant_storyline_slides(deck)
        ids = [slide.slide_id for slide in pruned.slides]

        self.assertIn("dep1", ids)
        self.assertIn("dep2", ids)
        delivery_count = sum(slide.slide_id.startswith("team") for slide in pruned.slides)
        self.assertGreaterEqual(delivery_count, 1)
        self.assertLessEqual(delivery_count, 2)

    def test_pruning_preserves_risks_technology_and_assumptions(self) -> None:
        deck = DeckPlan(
            deck_title="Test",
            slides=[
                SlideSpec(slide_id="scope", title="Scope, Dependencies, and Open Points", archetype="Requirements"),
                SlideSpec(slide_id="assumptions", title="Assumptions and Dependencies", archetype="Assumptions & Dependencies"),
                SlideSpec(slide_id="risks", title="Key Risks and Mitigations", archetype="Risks", bullets=["Risk"]),
                SlideSpec(slide_id="tech", title="Proposed technologies span the delivery lifecycle", archetype="Software Bill of Materials", table={"headers": ["A"], "rows": [["B"]]}),
                SlideSpec(slide_id="delivery", title="Agile roadmap", archetype="Timeline", bullets=["Sprint backlog release"]),
            ],
        )

        pruned = prune_redundant_storyline_slides(deck)
        ids = [slide.slide_id for slide in pruned.slides]

        self.assertIn("scope", ids)
        self.assertIn("assumptions", ids)
        self.assertIn("risks", ids)
        self.assertIn("tech", ids)

    def test_pruning_keeps_value_deployment_and_ha_dr_as_distinct_proof_objects(self) -> None:
        deck = DeckPlan(
            deck_title="Test",
            slides=[
                SlideSpec(
                    slide_id="value",
                    title="Proposal value and differentiators",
                    archetype="Value & Differentiators",
                    bullets=["Governance and engineering ownership reduce delivery risk."],
                ),
                SlideSpec(
                    slide_id="deployment",
                    title="Deployment topology protects controlled releases",
                    archetype="Deployment Architecture",
                    diagram=DiagramSpec(kind="deployment", prompt="deployment", approved=False),
                ),
                SlideSpec(
                    slide_id="hadr",
                    title="HA and DR protect business continuity",
                    archetype="High Availability & DR",
                    diagram=DiagramSpec(kind="hadr", prompt="recovery", approved=False),
                ),
            ],
        )

        pruned = prune_redundant_storyline_slides(deck)

        self.assertEqual(
            [slide.slide_id for slide in pruned.slides],
            ["value", "deployment", "hadr"],
        )

    def test_consulting_polish_turns_exec_summary_into_cards(self) -> None:
        deck = DeckPlan(
            deck_title="Test",
            slides=[
                SlideSpec(
                    slide_id="exec",
                    title="Executive Summary",
                    archetype="Solution Overview",
                    bullets=["Centralise data", "Reduce manual work", "Improve operational control"],
                )
            ],
        )
        understanding = RFPUnderstanding(
            summary="SATS needs a controlled catering uplift data hub.",
            project_scope="Build a centralized data hub for operational reporting.",
            in_scope_work=["Data ingestion", "Validation", "Reporting"],
        )

        polished = consulting_grade_proposal_polish(deck, understanding=understanding, narrative=None)
        slide = polished.slides[0]

        self.assertGreaterEqual(len(slide.cards), 2)
        self.assertEqual(slide.bullets, [])

    def test_consulting_polish_turns_current_state_into_comparison(self) -> None:
        deck = DeckPlan(
            deck_title="Test",
            slides=[
                SlideSpec(
                    slide_id="context",
                    title="Current challenges define priorities",
                    archetype="Customer Context",
                    bullets=["Manual consolidation delays reporting", "Interface readiness is a delivery dependency"],
                )
            ],
        )

        polished = consulting_grade_proposal_polish(deck)
        slide = polished.slides[0]

        self.assertIsNotNone(slide.comparison)
        self.assertEqual(slide.bullets, [])

    def test_consulting_polish_turns_security_nfr_into_cards(self) -> None:
        deck = DeckPlan(
            deck_title="Test",
            slides=[
                SlideSpec(
                    slide_id="nfr",
                    title="Security, observability and NFR controls are built in",
                    archetype="Architecture",
                    detailed_points=[
                        BulletPoint(
                            text="End-to-end secured communication",
                            sub_points=["Encrypt source-to-hub and hub-to-consumer paths"],
                        )
                    ],
                )
            ],
        )

        polished = consulting_grade_proposal_polish(deck)
        slide = polished.slides[0]

        self.assertEqual(len(slide.cards), 1)
        self.assertEqual(slide.detailed_points, [])

    def test_slide_density_bounds_counts_without_truncating_text(self) -> None:
        long_text = " ".join(f"word{i}" for i in range(60))
        deck = DeckPlan(
            deck_title="Test",
            slides=[
                SlideSpec(
                    slide_id="dense",
                    title="Dense",
                    archetype="Customer Context",
                    detailed_points=[
                        BulletPoint(
                            text="Current environment and constraints are lengthy",
                            sub_points=[long_text, long_text, long_text, long_text],
                        )
                    ],
                )
            ],
        )

        bounded = enforce_slide_density(deck)
        point = bounded.slides[0].detailed_points[0]

        self.assertEqual(point.text, "Current environment and constraints are lengthy")
        self.assertEqual(len(point.sub_points), 3)
        self.assertTrue(all(sub == long_text for sub in point.sub_points))

    def test_renderer_paginates_long_bullets_by_text_capacity(self) -> None:
        slide = SlideSpec(
            slide_id="long_bullets",
            title="Executive Summary",
            archetype="Content",
            bullets=[
                " ".join([f"Proposal point {idx} contains detailed rationale and customer implications"] * 16)
                for idx in range(4)
            ],
        )

        pages = _render_pages_for_slide(slide)

        self.assertGreater(len(pages), 1)
        self.assertTrue(all(len(page.bullets) < len(slide.bullets) for page in pages))
        self.assertTrue(all(page.archetype == "Content" for page in pages))
        self.assertTrue(all(page.layout_hint is None for page in pages))

    def test_renderer_paginates_solution_stack_at_six_rows(self) -> None:
        slide = SlideSpec(
            slide_id="tech",
            title="Proposed solution stack",
            archetype="Software Bill of Materials",
            table={
                "headers": ["Layer", "Technology", "Role", "Basis"],
                "rows": [[f"Layer {idx}", f"Service {idx}", "Role", "Proposed"] for idx in range(11)],
            },
        )

        pages = _render_pages_for_slide(slide)

        self.assertEqual([len(page.table["rows"]) for page in pages], [6, 5])

    def test_renderer_paginates_long_detailed_points_by_text_capacity(self) -> None:
        slide = SlideSpec(
            slide_id="long_detail",
            title="Risks and Mitigations",
            archetype="Risks",
            detailed_points=[
                BulletPoint(
                    text=f"Risk theme {idx}",
                    sub_points=[
                        " ".join(["This mitigation explanation includes evidence, impact, owner action, and acceptance dependency"] * 14)
                    ],
                )
                for idx in range(3)
            ],
        )

        pages = _render_pages_for_slide(slide)

        self.assertGreater(len(pages), 1)
        self.assertTrue(all(page.archetype == "Risks" for page in pages))
        self.assertTrue(all(page.layout_hint is None for page in pages))
        self.assertTrue(all(page.detailed_points for page in pages))
        self.assertTrue(all(not page.bullets for page in pages))

    def test_renderer_keeps_concise_five_step_action_slide_together(self) -> None:
        slide = SlideSpec(
            slide_id="next",
            title="Next Steps",
            archetype="Next Steps",
            bullets=[f"Confirm action {idx} with the accountable SATS owner." for idx in range(5)],
        )

        pages = _render_pages_for_slide(slide)

        self.assertEqual(len(pages), 1)

    def test_renderer_balances_four_detailed_points_across_continuations(self) -> None:
        slide = SlideSpec(
            slide_id="domains",
            title="Data domains",
            archetype="Content",
            detailed_points=[
                BulletPoint(
                    text=f"Domain {idx}",
                    sub_points=[" ".join(["Grounded domain control and source-of-record detail"] * 12)],
                )
                for idx in range(4)
            ],
        )

        pages = _render_pages_for_slide(slide)

        self.assertEqual([len(page.detailed_points) for page in pages], [2, 2])

    def test_renderer_preserves_cards_on_split_pages(self) -> None:
        from rfp2deck.core.schemas import Card

        slide = SlideSpec(
            slide_id="exec",
            title="Executive Summary",
            archetype="Solution Overview",
            cards=[
                Card(
                    heading=f"Executive theme {idx}",
                    body=" ".join(["Detailed customer implication and proposal response"] * 30),
                )
                for idx in range(3)
            ],
        )

        pages = _render_pages_for_slide(slide)

        self.assertGreater(len(pages), 1)
        self.assertEqual(sum(len(page.cards) for page in pages), 3)

    def test_renderer_preserves_native_layout_selection_for_regular_text_slides(self) -> None:
        slide = SlideSpec(
            slide_id="exec",
            title="Executive Summary",
            archetype="Solution Overview",
            bullets=["A complete proposal point.", "A second complete proposal point."],
        )

        pages = _render_pages_for_slide(slide)

        self.assertEqual(len(pages), 1)
        self.assertIsNone(pages[0].layout_hint)

    def test_renderer_adds_explanation_after_approved_diagram(self) -> None:
        slide = SlideSpec(
            slide_id="arch",
            title="Architecture",
            archetype="Architecture",
            bullets=["Explain source to target flow", "Explain controls and operations"],
            diagram=DiagramSpec(kind="architecture", prompt="draw architecture", approved=True),
        )

        pages = _render_pages_for_slide(slide, diagram_images={"arch": b"png"})

        self.assertEqual(len(pages), 2)
        self.assertIsNotNone(pages[0].diagram)
        self.assertEqual(pages[0].bullets, [])
        self.assertIsNone(pages[1].diagram)
        self.assertIsNone(pages[1].layout_hint)
        self.assertEqual(
            pages[1].bullets,
            ["Explain source to target flow", "Explain controls and operations"],
        )

    def test_hcltech_diagram_uses_native_diagram_layout_not_title_only(self) -> None:
        from pptx import Presentation

        template = Path(".data/outputs/latest_user_deck.pptx")
        if not template.exists():
            self.skipTest("Full HCLTech template fixture is not available")
        prs = Presentation(template)
        slide = SlideSpec(
            slide_id="arch",
            title="Architecture",
            archetype="Architecture",
            diagram=DiagramSpec(kind="architecture", prompt="draw architecture", approved=True),
        )

        layout = _choose_hcltech_layout(prs, slide)

        self.assertIn("diagram", layout.name.lower())
        self.assertNotEqual(layout.name.lower(), "title only")

    def test_rendered_slide_count_includes_diagram_companion(self) -> None:
        deck = DeckPlan(
            deck_title="Test",
            slides=[
                SlideSpec(slide_id="title", title="Test", archetype="Title"),
                SlideSpec(
                    slide_id="arch",
                    title="Architecture",
                    archetype="Architecture",
                    bullets=["Explain the design decision."],
                    diagram=DiagramSpec(kind="architecture", prompt="draw architecture", approved=True),
                ),
            ],
        )

        self.assertEqual(rendered_slide_count(deck, {"arch": b"png"}), 3)

    def test_renderer_finds_approved_diagram_by_prompt_cache_key(self) -> None:
        slide = SlideSpec(
            slide_id="arch_v2",
            title="Architecture",
            archetype="Architecture",
            bullets=["Explain source to target flow"],
            diagram=DiagramSpec(kind="architecture", prompt="draw architecture", approved=True),
        )

        pages = _render_pages_for_slide(slide, diagram_images={"draw architecture": b"png"})

        self.assertEqual(len(pages), 2)
        self.assertIsNotNone(pages[0].diagram)
        self.assertEqual(pages[0].bullets, [])
        self.assertIsNone(pages[1].diagram)

    def test_density_preserves_diagram_explanation_for_companion_slide(self) -> None:
        slide = SlideSpec(
            slide_id="arch",
            title="Architecture",
            archetype="Architecture",
            bullets=[
                "Source systems feed a governed ingestion layer.",
                "Controls protect every interface.",
            ],
            diagram=DiagramSpec(kind="architecture", prompt="draw architecture", approved=True),
        )

        bounded = enforce_slide_density(DeckPlan(deck_title="Test", slides=[slide]))

        self.assertEqual(len(bounded.slides[0].bullets), 2)

    def test_renderer_preserves_diagram_picture_during_overlap_cleanup(self) -> None:
        from pptx import Presentation
        from pptx.util import Inches

        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1)).text = "Title"
        picture_like = slide.shapes.add_picture(
            BytesIO(base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
            )),
            Inches(0.8),
            Inches(0.8),
            width=Inches(5),
            height=Inches(2),
        )
        spec = SlideSpec(
            slide_id="arch",
            title="Architecture",
            archetype="Architecture",
            diagram=DiagramSpec(kind="architecture", prompt="draw architecture", approved=True),
        )

        removed = _remove_overlapping_generated_pictures(slide, spec)

        self.assertEqual(removed, 0)
        self.assertIn(picture_like, list(slide.shapes))

    def test_renderer_removes_stale_singleton_continuation_title(self) -> None:
        slide = SlideSpec(
            slide_id="exec_page_1",
            title="Executive Summary (1 of 2)",
            archetype="Content",
            bullets=["Executive point"],
        )

        normalized = _normalize_singleton_continuation_titles([slide])

        self.assertEqual(normalized[0].title, "Executive Summary")

    def test_renderer_drops_exact_duplicate_render_slides(self) -> None:
        first = SlideSpec(
            slide_id="exec_1",
            title="Executive Summary",
            archetype="Solution Overview",
            bullets=["Centralise catering uplift data across ICC1 and ICC2."],
        )
        duplicate = SlideSpec(
            slide_id="exec_2",
            title="Executive Summary",
            archetype="Solution Overview",
            bullets=["Centralise catering uplift data across ICC1 and ICC2."],
        )

        slides = _dedupe_render_slides([first, duplicate])

        self.assertEqual([slide.slide_id for slide in slides], ["exec_1"])

    def test_renderer_drops_adjacent_prefix_duplicate_render_slides(self) -> None:
        first = SlideSpec(
            slide_id="scope_1",
            title="Scope & Boundaries",
            archetype="Requirements",
            bullets=[
                "Centralized data hub for catering uplift operations.",
                "Integration and reporting controls for operational users.",
            ],
        )
        prefix_duplicate = SlideSpec(
            slide_id="scope_2",
            title="Scope & Boundaries",
            archetype="Requirements",
            bullets=["Centralized data hub for catering uplift operations."],
        )

        slides = _dedupe_render_slides([first, prefix_duplicate])

        self.assertEqual([slide.slide_id for slide in slides], ["scope_1"])

    def test_renderer_falls_back_to_text_when_approved_diagram_image_is_missing(self) -> None:
        slide = SlideSpec(
            slide_id="solution",
            title="Solution overview",
            archetype="Solution Overview",
            bullets=["Diagram explanation: Create a left-to-right architecture view. Left: source systems. Right: reporting users."],
            diagram=DiagramSpec(kind="generic", prompt="draw solution", approved=True),
        )

        pages = _render_pages_for_slide(slide, diagram_images={})

        self.assertEqual(len(pages), 1)
        self.assertIsNone(pages[0].diagram)
        self.assertIsNone(pages[0].layout_hint)
        self.assertNotIn("Diagram explanation:", " ".join(pages[0].bullets))
        self.assertNotIn("left-to-right architecture view", " ".join(pages[0].bullets).lower())
        self.assertIn("asset is unavailable", pages[0].bullets[0].lower())

    def test_renderer_repairs_title_only_slide_with_diagram_prompt(self) -> None:
        from pptx import Presentation

        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        spec = SlideSpec(
            slide_id="sk_solution",
            title="Solution overview",
            archetype="Solution Overview",
            diagram=DiagramSpec(kind="generic", prompt="Show source systems and data hub.", approved=True),
        )

        repaired = _repair_title_only_slide(slide, spec)

        self.assertTrue(repaired)
        self.assertTrue(any("data hub" in (getattr(shape, "text", "") or "").lower() for shape in slide.shapes))

    def test_delivery_testing_and_ams_diagrams_are_not_generic_squad_diagrams(self) -> None:
        deck = DeckPlan(
            deck_title="Test",
            slides=[
                SlideSpec(slide_id="test", title="Testing and acceptance create delivery evidence", archetype="Delivery Plan"),
                SlideSpec(slide_id="ams", title="Warranty and AMS sustain the live service", archetype="Delivery Plan"),
            ],
        )

        planned = ensure_diagrams_for_key_slides(deck, understanding=None)

        self.assertEqual(planned.slides[0].diagram.kind, "testing")
        self.assertIn("testing and acceptance evidence map", planned.slides[0].diagram.prompt.lower())
        self.assertIn("do not show a textbook test pyramid", planned.slides[0].diagram.prompt.lower())
        self.assertEqual(planned.slides[1].diagram.kind, "ams")
        self.assertIn("warranty and ams service map", planned.slides[1].diagram.prompt.lower())
        self.assertIn("do not use a generic l1/l2/l3 pyramid", planned.slides[1].diagram.prompt.lower())

    def test_testing_and_ams_are_tailored_to_named_solution_requirements(self) -> None:
        understanding = RFPUnderstanding(
            customer_name="SATS",
            opportunity_title="Catering Uplift Data Hub",
            summary="Replace ICCMS with a secure data hub and provide warranty and AMS support.",
            project_scope="Integrate operational sources, reconcile uplift data, cut over safely, and support the live service.",
            requirements=[
                Requirement(id="R-1", text="Integrate FIH and GP4 feeds, SAP and KSMS APIs, and SQ SFTP ELP files.", priority="must"),
                Requirement(id="R-2", text="Replace ICCMS while preserving operational processes and functional outcomes.", priority="must"),
                Requirement(id="R-3", text="Provide data validation, reconciliation, security audit evidence, and performance assurance.", priority="must"),
                Requirement(id="R-4", text="Provide warranty and AMS support for production integrations, reporting, and incident resolution.", priority="must"),
            ],
        )

        testing_points = _testing_proposal_points(understanding)
        ams_points = _ams_proposal_points(understanding)
        testing_prompt = _build_diagram_prompt("testing", understanding)
        ams_prompt = _build_diagram_prompt("ams", understanding)

        self.assertTrue(any("FIH" in " ".join(point.sub_points) or "GP4" in " ".join(point.sub_points) for point in testing_points))
        self.assertTrue(any("ICCMS" in " ".join(point.sub_points) for point in testing_points))
        self.assertIn("FIH", testing_prompt)
        self.assertIn("ICCMS", testing_prompt)
        self.assertIn("acceptance owner", testing_prompt.lower())
        self.assertIn("do not show a textbook test pyramid", testing_prompt.lower())
        self.assertTrue(any("SAP" in " ".join(point.sub_points) or "KSMS" in " ".join(point.sub_points) for point in ams_points))
        self.assertIn("business-flow observability", ams_prompt.lower())
        self.assertIn("correction/replay", ams_prompt.lower())
        self.assertIn("to be agreed", ams_prompt.lower())
        self.assertIn("do not use a generic l1/l2/l3 pyramid", ams_prompt.lower())

    def test_corporate_proxy_block_page_is_detected_deep_in_error_body(self) -> None:
        html = (
            "<!DOCTYPE html><html><head><title>Network Error</title></head>"
            "<body>"
            + ("x" * 5000)
            + "ThreatPulse Symantec BlueCoat Access Restricted Website "
            + "Category: Generative AI "
            + "URL: https:&#x2F;&#x2F;api.openai.com&#x2F;v1&#x2F;responses"
            + "</body></html>"
        )
        exc = SimpleNamespace(response=SimpleNamespace(text=html), status_code=500)

        self.assertTrue(_is_proxy_block_error(exc))

    def test_cloudflare_openai_520_is_not_detected_as_proxy_block(self) -> None:
        body = {
            "title": "Error 520: Web server is returning an unknown error",
            "status": 520,
            "zone": "api.openai.com",
            "cloudflare_error": True,
            "retryable": True,
        }
        exc = SimpleNamespace(response=SimpleNamespace(text=str(body)), status_code=520)

        self.assertFalse(_is_proxy_block_error(exc))

    def test_evidence_chunk_results_are_cached_for_downstream_retries(self) -> None:
        document = SourceDocument(
            document_id="doc-cache",
            name="RFP.pdf",
            document_type="base_rfp",
            authority="authoritative",
            text="--- PAGE 1 ---\nA mandatory requirement.",
        )
        chunk = split_source_document(document, 4000)[0]
        with TemporaryDirectory() as temp_dir:
            fake_settings = SimpleNamespace(
                understanding_evidence_cache=True,
                data_dir=Path(temp_dir),
                model_fast="test-fast-model",
                reasoning_effort_low="low",
                understanding_evidence_timeout_s=30,
            )
            with patch("rfp2deck.agent.nodes.settings", fake_settings), patch(
                "rfp2deck.agent.nodes.response_as_schema",
                return_value=SourceEvidenceBatch(source_document_id="", chunk_id=""),
            ) as mocked_response:
                first = _extract_evidence_chunk(chunk)
                second = _extract_evidence_chunk(chunk)

        self.assertEqual(mocked_response.call_count, 1)
        self.assertEqual(first.chunk_id, second.chunk_id)

    def test_large_package_node_uses_chunk_extraction_before_understanding(self) -> None:
        document = SourceDocument(
            document_id="doc-large",
            name="Large RFP.pdf",
            document_type="base_rfp",
            authority="authoritative",
            text="--- PAGE 1 ---\n" + ("Requirement text with locator evidence.\n" * 14500),
        )
        state = AgentState(
            rfp_text=document.text,
            template_info={},
            source_documents=[document],
        )
        fake_settings = SimpleNamespace(
            understanding_direct_max_chars=180000,
            understanding_evidence_chunk_chars=55000,
            understanding_evidence_max_chars=180000,
            understanding_evidence_workers=2,
            understanding_evidence_timeout_s=30,
            model_fast="test-fast-model",
            reasoning_effort_low="low",
        )

        def fake_response(prompt, schema, **kwargs):
            return SourceEvidenceBatch(source_document_id="", chunk_id="")

        with patch("rfp2deck.agent.nodes.settings", fake_settings), patch(
            "rfp2deck.agent.nodes.response_as_schema", side_effect=fake_response
        ) as mocked_response:
            result = extract_source_evidence(state)

        self.assertGreaterEqual(len(document.text), 533817)
        self.assertGreaterEqual(mocked_response.call_count, 9)
        self.assertLessEqual(mocked_response.call_count, 11)
        self.assertTrue(result["source_evidence"])
        self.assertLessEqual(len(result["evidence_text"]), 180000)

    def test_large_source_is_split_into_bounded_locator_aware_chunks(self) -> None:
        text = "--- PAGE 1 ---\n" + ("Requirement A must be retained.\n" * 350)
        document = SourceDocument(
            document_id="doc-large",
            name="Large RFP.pdf",
            document_type="base_rfp",
            authority="authoritative",
            text=text,
        )

        chunks = split_source_document(document, 4000)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 4000 for chunk in chunks))
        self.assertTrue(all(chunk.document.document_id == "doc-large" for chunk in chunks))
        self.assertIn("LOCATOR CONTEXT", chunks[1].text)

    def test_chunk_evidence_merge_deduplicates_overlap_and_preserves_requirements(self) -> None:
        requirement = Requirement(
            id="R-1",
            text="Integrate all nominated source systems",
            source_ref="doc-rfp / page 20",
            source_document_ids=["doc-rfp"],
        )
        batches = [
            SourceEvidenceBatch(
                source_document_id="doc-rfp",
                chunk_id="chunk-1",
                summary_points=["Integration is in scope"],
                requirements=[requirement],
            ),
            SourceEvidenceBatch(
                source_document_id="doc-rfp",
                chunk_id="chunk-2",
                summary_points=["Integration is in scope"],
                requirements=[requirement.model_copy(deep=True)],
            ),
        ]

        merged = merge_evidence_batches(batches)
        rendered = render_evidence_for_prompt(merged, 40000)

        self.assertEqual(len(merged.requirements), 1)
        self.assertEqual(merged.summary_points, ["Integration is in scope"])
        self.assertIn("doc-rfp / page 20", rendered)

    def test_large_requirement_register_uses_lossless_indexed_compaction(self) -> None:
        requirements = [
            Requirement(
                id=f"REQ-{index:04d}",
                text=(
                    f"Requirement {index} requires validated processing, audit evidence, "
                    "operational monitoring, exception handling, and retained traceability."
                ),
                source_ref=f"doc-rfp / sheet Requirements / row {index + 2}",
                source_refs=[f"doc-rfp / sheet Requirements / row {index + 2}"],
                source_document_ids=["doc-rfp"],
            )
            for index in range(700)
        ]
        evidence = SourceEvidenceBatch(
            source_document_id="rfp-package",
            chunk_id="merged-evidence",
            requirements=requirements,
        )

        rendered = render_evidence_for_prompt(evidence, 180000)

        self.assertLessEqual(len(rendered), 180000)
        self.assertIn("indexed-rfp-evidence-v1", rendered)
        self.assertIn("Requirement 699", rendered)
        self.assertIn("row 701", rendered)

    def test_oversized_requirement_register_uses_budgeted_indexed_compaction(self) -> None:
        requirements = [
            Requirement(
                id=f"REQ-{index:04d}",
                text=(
                    f"Requirement {index} requires validated processing, audit evidence, "
                    "operational monitoring, exception handling, retained traceability, "
                    "implementation controls, support procedures, and reporting."
                ),
                priority="must" if index < 10 else "should",
                source_ref=f"doc-rfp / sheet Requirements / row {index + 2}",
                source_refs=[f"doc-rfp / sheet Requirements / row {index + 2}"],
                source_document_ids=["doc-rfp"],
            )
            for index in range(1200)
        ]
        evidence = SourceEvidenceBatch(
            source_document_id="rfp-package",
            chunk_id="merged-evidence",
            requirements=requirements,
        )

        rendered = render_evidence_for_prompt(evidence, 40000)

        self.assertLessEqual(len(rendered), 40000)
        self.assertIn('"budgeted":true', rendered)
        self.assertIn('"omitted_requirements":', rendered)
        self.assertIn("Requirement 0", rendered)

    def test_base_rfp_is_not_reclassified_by_procurement_wording(self) -> None:
        document_type, authority = classify_source(
            "Request for Tender.pdf",
            "Vendors may submit clarification questions before the deadline. Scope of Work follows.",
            "primary",
        )

        self.assertEqual(document_type, "base_rfp")
        self.assertEqual(authority, "authoritative")

    def test_understanding_schema_can_be_fully_inlined_for_structured_output(self) -> None:
        def contains_ref(value: object) -> bool:
            if isinstance(value, dict):
                return "$ref" in value or any(contains_ref(item) for item in value.values())
            if isinstance(value, list):
                return any(contains_ref(item) for item in value)
            return False

        for model in (RFPUnderstanding, SourceEvidenceBatch):
            schema = model.model_json_schema()
            defs = schema.get("$defs", {})
            inlined = _dereference(schema, defs)
            inlined.pop("$defs", None)
            strict = _make_strict(inlined)
            self.assertFalse(contains_ref(strict), model.__name__)

    def test_understanding_prompt_accepts_source_reconciliation(self) -> None:
        prompt = RFP_UNDERSTAND_PROMPT.format(
            rfp_text="=== SOURCE DOCUMENT base ===\nRequirement text",
            rfp_focus_guide="Scope section detected",
            source_reconciliation='{"precedence_summary": []}',
        )

        self.assertIn("SOURCE_RECONCILIATION", prompt)
        self.assertIn("vendor question is context only", prompt.lower())

    def test_xlsx_clarifications_preserve_rows_and_answer_authority(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Clarifications"
        sheet.append(["Question ID", "Bidder's Query", "SATS Response"])
        sheet.append(["Q-01", "Must the solution use Ariba?", "No. Ariba is submission-only."])
        sheet.append(["Q-02", "What is the required RTO?", ""])
        stream = BytesIO()
        workbook.save(stream)

        document, clarifications = parse_source_document(
            "Customer Clarifications 2026-07-15.xlsx",
            stream.getvalue(),
            role="clarification",
        )

        self.assertEqual(document.document_type, "customer_clarification")
        self.assertEqual(document.issue_date, "2026-07-15")
        self.assertEqual(document.locator_format, "sheet/row")
        self.assertIn('[SHEET "Clarifications"][ROW 2]', document.text)
        self.assertEqual(len(clarifications), 2)
        self.assertEqual(clarifications[0].authority, "authoritative")
        self.assertEqual(clarifications[0].status, "active")
        self.assertEqual(clarifications[1].authority, "non_authoritative")
        self.assertEqual(clarifications[1].status, "unresolved")

    def test_docx_tables_are_available_as_row_level_evidence(self) -> None:
        document = Document()
        document.add_heading("Scope", level=1)
        document.add_paragraph("Create a centralized data repository.")
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Requirement"
        table.cell(0, 1).text = "Priority"
        table.cell(1, 0).text = "Integrate source systems"
        table.cell(1, 1).text = "Mandatory"
        stream = BytesIO()
        document.save(stream)

        parsed = parse_docx(stream.getvalue())

        self.assertEqual(parsed.table_count, 1)
        self.assertIn("[PARAGRAPH 2]", parsed.text)
        self.assertIn("[TABLE 1][ROW 2]", parsed.text)
        self.assertIn("Integrate source systems", parsed.text)

    def test_source_precedence_orders_addenda_before_base_and_supporting(self) -> None:
        sources = [
            SourceDocument(
                document_id="support",
                name="Reference.docx",
                document_type="supporting_reference",
                authority="contextual",
                text="Reference",
            ),
            SourceDocument(
                document_id="base",
                name="RFP.pdf",
                document_type="base_rfp",
                authority="authoritative",
                text="Base",
            ),
            SourceDocument(
                document_id="addendum-old",
                name="Addendum 1.pdf",
                document_type="customer_addendum",
                authority="binding",
                issue_date="2026-07-01",
                text="Old amendment",
            ),
            SourceDocument(
                document_id="addendum-new",
                name="Addendum 2.pdf",
                document_type="customer_addendum",
                authority="binding",
                issue_date="2026-07-15",
                text="New amendment",
            ),
        ]

        ordered = sort_sources_by_precedence(sources)

        self.assertEqual(
            [source.document_id for source in ordered],
            ["addendum-new", "addendum-old", "base", "support"],
        )
        rendered = render_source_package(sources)
        self.assertLess(rendered.index("addendum-new"), rendered.index("base"))
        self.assertIn("vendor questions alone are not requirements", rendered.lower())

    def test_reconciliation_flags_unanswered_questions(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["ID", "Question", "Response"])
        sheet.append(["Q1", "Required hosting region?", ""])
        stream = BytesIO()
        workbook.save(stream)
        document, clarifications = parse_source_document(
            "Clarifications.xlsx", stream.getvalue(), role="clarification"
        )

        reconciliation = build_source_reconciliation([document], clarifications)

        self.assertEqual(reconciliation.unresolved_questions, [clarifications[0].source_ref])
        self.assertIn("vendor question", " ".join(reconciliation.precedence_summary).lower())

    def test_requirement_schema_remains_backward_compatible(self) -> None:
        requirement = Requirement(id="R-1", text="Retain data for seven years")

        self.assertEqual(requirement.status, "active")
        self.assertEqual(requirement.source_refs, [])
        self.assertEqual(requirement.confidence, 1.0)

    def test_traceability_report_retains_source_provenance(self) -> None:
        requirement = Requirement(
            id="R-1",
            text="Retain data for seven years",
            priority="must",
            source_refs=["doc-rfp / page 12"],
            source_document_ids=["doc-rfp"],
            authority="binding",
            status="clarified",
        )
        understanding = RFPUnderstanding(summary="Retention requirement", requirements=[requirement])
        deck = DeckPlan(
            deck_title="Proposal",
            slides=[SlideSpec(slide_id="requirements", title="Retention", rfps=["R-1"])],
        )

        report = build_traceability_report(understanding, deck)

        self.assertEqual(report.coverage[0].source_refs, ["doc-rfp / page 12"])
        self.assertEqual(report.coverage[0].source_document_ids, ["doc-rfp"])
        self.assertEqual(report.coverage[0].status, "clarified")


if __name__ == "__main__":
    unittest.main()
