from __future__ import annotations

import unittest
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from openai import APIConnectionError, APITimeoutError, InternalServerError
from pydantic import BaseModel

from rfp2deck.agent.nodes import compress_bullets, derive_technology_recommendations
from rfp2deck.agent.graph import build_graph
from rfp2deck.agent.state import AgentState
from rfp2deck.core.schemas import (
    BulletCompressionSet,
    DeckPlan,
    DiagramSpec,
    RFPUnderstanding,
    SlideBulletEdit,
    SlideSpec,
    TechnologyRecommendationSet,
)
from rfp2deck.llm.structured import (
    StructuredLLMError,
    _poll_background_response,
    _responses_create_with_backoff,
    response_as_schema,
)


class _Payload(BaseModel):
    value: str


def _settings(**overrides):
    values = {
        "openai_retry_attempts": 3,
        "openai_retry_base_wait_s": 0.01,
        "openai_retry_max_wait_s": 0.05,
        "openai_retry_jitter_ratio": 0.0,
        "openai_structured_streaming": False,
        "openai_structured_background_enabled": True,
        "openai_structured_background_all": False,
        "openai_structured_background_min_chars": 30000,
        "openai_structured_background_poll_s": 0.0,
        "openai_structured_background_grace_s": 300.0,
        "openai_timeout_s": 120.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class StructuredRetryTests(unittest.TestCase):
    def test_graph_fans_out_independent_analysis_nodes(self) -> None:
        graph = build_graph().get_graph()
        edges = {(edge.source, edge.target) for edge in graph.edges}

        self.assertIn(("understand_rfp", "derive_sections"), edges)
        self.assertIn(("understand_rfp", "build_narrative"), edges)
        self.assertIn(("understand_rfp", "derive_technology_recommendations"), edges)
        self.assertIn(("derive_sections", "derive_visual_briefs"), edges)
        self.assertIn(("build_narrative", "derive_visual_briefs"), edges)
        self.assertNotIn(("derive_sections", "build_narrative"), edges)

    def test_technology_recommendation_node_uses_background_transport(self) -> None:
        state = AgentState(
            rfp_text="",
            template_info={},
            understanding=RFPUnderstanding(
                summary="Build a catalogue platform with integrations and governed data."
            ),
            customer_technology_context={"platform": "Microsoft Azure"},
        )
        with patch(
            "rfp2deck.agent.nodes.response_as_schema",
            return_value=TechnologyRecommendationSet(),
        ) as structured_call:
            derive_technology_recommendations(state)

        self.assertIs(structured_call.call_args.kwargs["background"], True)
        prompt = structured_call.call_args.args[0]
        self.assertIn("customer_technology_context", prompt)
        self.assertNotIn('"narrative"', prompt)

    def test_compress_bullets_node_uses_shared_transport_policy(self) -> None:
        original = DeckPlan(
            deck_title="Proposal",
            slides=[
                SlideSpec(
                    slide_id="s1",
                    title="Summary",
                    bullets=["Original"],
                    notes="NOT_SENT_TO_COMPRESSION",
                    diagram=DiagramSpec(prompt="DIAGRAM_NOT_SENT_TO_COMPRESSION"),
                )
            ],
        )
        compressed = BulletCompressionSet(
            slides=[SlideBulletEdit(slide_id="s1", bullets=["Tighter"])],
        )
        state = AgentState(rfp_text="", template_info={}, deck_plan=original)

        with patch(
            "rfp2deck.agent.nodes.response_as_schema", return_value=compressed
        ) as structured_call:
            result = compress_bullets(state)

        self.assertEqual(result["deck_plan"].slides[0].bullets, ["Tighter"])
        self.assertIs(structured_call.call_args.kwargs["background"], False)
        prompt = structured_call.call_args.args[0]
        self.assertNotIn("NOT_SENT_TO_COMPRESSION", prompt)
        self.assertNotIn("DIAGRAM_NOT_SENT_TO_COMPRESSION", prompt)
        self.assertIs(structured_call.call_args.args[1], BulletCompressionSet)

    def test_timeout_is_retried_before_success(self) -> None:
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        completed = SimpleNamespace(status="completed", output_text='{"value":"ok"}')
        client = SimpleNamespace(
            responses=SimpleNamespace(
                create=Mock(
                    side_effect=[
                        APITimeoutError(request),
                        completed,
                    ]
                )
            )
        )

        with (
            patch("rfp2deck.llm.structured.settings", _settings()),
            patch("rfp2deck.llm.structured.time.sleep") as sleep,
        ):
            result = _responses_create_with_backoff(
                client,
                {"model": "gpt-test"},
                schema_name="Payload",
                model="gpt-test",
                reasoning_effort="medium",
            )

        self.assertIs(result, completed)
        self.assertEqual(client.responses.create.call_count, 2)
        sleep.assert_called_once()

    def test_connection_error_is_retried_before_success(self) -> None:
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        completed = SimpleNamespace(status="completed", output_text='{"value":"ok"}')
        client = SimpleNamespace(
            responses=SimpleNamespace(
                create=Mock(
                    side_effect=[
                        APIConnectionError(request=request),
                        completed,
                    ]
                )
            )
        )

        with (
            patch("rfp2deck.llm.structured.settings", _settings()),
            patch("rfp2deck.llm.structured.time.sleep"),
        ):
            result = _responses_create_with_backoff(
                client,
                {"model": "gpt-test"},
                schema_name="Payload",
                model="gpt-test",
                reasoning_effort="medium",
            )

        self.assertIs(result, completed)
        self.assertEqual(client.responses.create.call_count, 2)

    def test_explicit_proxy_policy_page_is_not_retried(self) -> None:
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        response = httpx.Response(
            403,
            request=request,
            text=(
                "<html>BlueCoat access restricted website: Generative AI "
                "https://api.openai.com/v1/responses</html>"
            ),
        )
        error = InternalServerError("blocked", response=response, body=None)
        client = SimpleNamespace(responses=SimpleNamespace(create=Mock(side_effect=error)))

        with (
            patch("rfp2deck.llm.structured.settings", _settings()),
            self.assertRaises(StructuredLLMError) as raised,
        ):
            _responses_create_with_backoff(
                client,
                {"model": "gpt-test"},
                schema_name="Payload",
                model="gpt-test",
                reasoning_effort="medium",
            )

        self.assertIn("corporate network proxy", str(raised.exception))
        self.assertEqual(client.responses.create.call_count, 1)

    def test_transient_proxy_503_is_retried_before_success(self) -> None:
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        response = httpx.Response(
            503,
            request=request,
            text=(
                "<html><title>Network Error</title>BlueCoat Generative AI "
                "api.openai.com/v1/responses Request could not be handled</html>"
            ),
        )
        error = InternalServerError("proxy gateway", response=response, body=None)
        completed = SimpleNamespace(status="completed", output_text='{"value":"ok"}')
        client = SimpleNamespace(
            responses=SimpleNamespace(create=Mock(side_effect=[error, completed]))
        )

        with (
            patch("rfp2deck.llm.structured.settings", _settings()),
            patch("rfp2deck.llm.structured.time.sleep") as sleep,
        ):
            result = _responses_create_with_backoff(
                client,
                {"model": "gpt-test"},
                schema_name="Payload",
                model="gpt-test",
                reasoning_effort="medium",
            )

        self.assertIs(result, completed)
        self.assertEqual(client.responses.create.call_count, 2)
        sleep.assert_called_once()

    def test_long_structured_call_uses_background_polling(self) -> None:
        queued = SimpleNamespace(id="resp_123", status="queued", output_text="")
        in_progress = SimpleNamespace(id="resp_123", status="in_progress", output_text="")
        completed = SimpleNamespace(id="resp_123", status="completed", output_text='{"value":"ok"}')
        responses = SimpleNamespace(
            create=Mock(return_value=queued),
            retrieve=Mock(side_effect=[in_progress, completed]),
        )
        client = SimpleNamespace(responses=responses)

        with (
            patch(
                "rfp2deck.llm.structured.settings",
                _settings(
                    openai_structured_background_min_chars=10,
                ),
            ),
            patch("rfp2deck.llm.structured.get_client", return_value=client),
        ):
            result = response_as_schema(
                "x" * 20,
                _Payload,
                model="gpt-test",
                timeout_seconds=30,
            )

        self.assertEqual(result.value, "ok")
        self.assertTrue(responses.create.call_args.kwargs["background"])
        self.assertEqual(responses.retrieve.call_count, 2)

    def test_active_background_job_continues_through_grace_period(self) -> None:
        in_progress = SimpleNamespace(id="resp_grace", status="in_progress", output_text="")
        completed = SimpleNamespace(
            id="resp_grace", status="completed", output_text='{"value":"ok"}'
        )
        responses = SimpleNamespace(retrieve=Mock(return_value=completed))
        client = SimpleNamespace(responses=responses)

        with patch(
            "rfp2deck.llm.structured.settings",
            _settings(openai_structured_background_grace_s=30.0),
        ):
            result = _poll_background_response(
                client,
                in_progress,
                schema_name="Payload",
                model="gpt-test",
                reasoning_effort="medium",
                deadline=time.perf_counter() - 0.01,
            )

        self.assertIs(result, completed)
        responses.retrieve.assert_called_once_with("resp_grace")

    def test_background_job_fails_after_grace_period_expires(self) -> None:
        in_progress = SimpleNamespace(id="resp_expired", status="in_progress", output_text="")
        cancelled = SimpleNamespace(id="resp_expired", status="cancelled", output_text="")
        responses = SimpleNamespace(retrieve=Mock(), cancel=Mock(return_value=cancelled))
        client = SimpleNamespace(responses=responses)

        with (
            patch(
                "rfp2deck.llm.structured.settings",
                _settings(openai_structured_background_grace_s=10.0),
            ),
            self.assertRaises(StructuredLLMError) as raised,
        ):
            _poll_background_response(
                client,
                in_progress,
                schema_name="Payload",
                model="gpt-test",
                reasoning_effort="medium",
                deadline=time.perf_counter() - 11.0,
            )

        self.assertIn("grace period", str(raised.exception))
        responses.retrieve.assert_not_called()
        responses.cancel.assert_called_once_with("resp_expired")

    def test_background_job_honors_per_call_grace_override(self) -> None:
        in_progress = SimpleNamespace(
            id="resp_contextual", status="in_progress", output_text=""
        )
        cancelled = SimpleNamespace(id="resp_contextual", status="cancelled", output_text="")
        responses = SimpleNamespace(retrieve=Mock(), cancel=Mock(return_value=cancelled))
        client = SimpleNamespace(responses=responses)

        with (
            patch(
                "rfp2deck.llm.structured.settings",
                _settings(openai_structured_background_grace_s=300.0),
            ),
            self.assertRaises(StructuredLLMError) as raised,
        ):
            _poll_background_response(
                client,
                in_progress,
                schema_name="Payload",
                model="gpt-test",
                reasoning_effort="medium",
                deadline=time.perf_counter() - 61.0,
                grace_seconds=60.0,
            )

        self.assertIn("grace_s=60", str(raised.exception))
        responses.retrieve.assert_not_called()
        responses.cancel.assert_called_once_with("resp_contextual")

    def test_recoverable_background_failure_is_logged_as_warning(self) -> None:
        queued = SimpleNamespace(
            id="resp_recoverable", status="in_progress", output_text=""
        )
        responses = SimpleNamespace(create=Mock(return_value=queued))
        client = SimpleNamespace(responses=responses)

        with (
            patch(
                "rfp2deck.llm.structured.settings",
                _settings(openai_structured_background_all=True),
            ),
            patch("rfp2deck.llm.structured.get_client", return_value=client),
            patch(
                "rfp2deck.llm.structured._poll_background_response",
                side_effect=StructuredLLMError("application deadline exceeded"),
            ),
            self.assertLogs("rfp2deck.llm.structured", level="WARNING") as captured,
            self.assertRaises(StructuredLLMError),
        ):
            response_as_schema(
                "contextual reference",
                _Payload,
                model="gpt-test",
                timeout_seconds=60,
                background_grace_seconds=30,
                recoverable_failure=True,
            )

        self.assertTrue(captured.records)
        self.assertTrue(all(record.levelname == "WARNING" for record in captured.records))
        self.assertIn("caller will apply its fallback", captured.output[-1])

    def test_all_structured_calls_use_background_polling_by_default(self) -> None:
        completed = SimpleNamespace(id="resp_all", status="completed", output_text='{"value":"ok"}')
        responses = SimpleNamespace(create=Mock(return_value=completed))
        client = SimpleNamespace(responses=responses)

        with (
            patch(
                "rfp2deck.llm.structured.settings",
                _settings(
                    openai_structured_background_all=True,
                    openai_structured_background_min_chars=10000,
                ),
            ),
            patch("rfp2deck.llm.structured.get_client", return_value=client),
        ):
            result = response_as_schema(
                "short",
                _Payload,
                model="gpt-test",
                timeout_seconds=30,
            )

        self.assertEqual(result.value, "ok")
        self.assertTrue(responses.create.call_args.kwargs["background"])

    def test_short_structured_call_remains_synchronous(self) -> None:
        completed = SimpleNamespace(status="completed", output_text='{"value":"ok"}')
        responses = SimpleNamespace(create=Mock(return_value=completed))
        client = SimpleNamespace(responses=responses)

        with (
            patch(
                "rfp2deck.llm.structured.settings",
                _settings(
                    openai_structured_background_min_chars=100,
                ),
            ),
            patch("rfp2deck.llm.structured.get_client", return_value=client),
        ):
            result = response_as_schema(
                "short",
                _Payload,
                model="gpt-test",
                timeout_seconds=30,
            )

        self.assertEqual(result.value, "ok")
        self.assertNotIn("background", responses.create.call_args.kwargs)

    def test_explicit_background_overrides_prompt_threshold(self) -> None:
        completed = SimpleNamespace(
            id="resp_forced", status="completed", output_text='{"value":"ok"}'
        )
        responses = SimpleNamespace(create=Mock(return_value=completed))
        client = SimpleNamespace(responses=responses)

        with (
            patch(
                "rfp2deck.llm.structured.settings",
                _settings(
                    openai_structured_background_min_chars=10000,
                ),
            ),
            patch("rfp2deck.llm.structured.get_client", return_value=client),
        ):
            result = response_as_schema(
                "short",
                _Payload,
                model="gpt-test",
                timeout_seconds=30,
                background=True,
            )

        self.assertEqual(result.value, "ok")
        self.assertTrue(responses.create.call_args.kwargs["background"])

    def test_background_can_be_disabled_for_retention_policy(self) -> None:
        completed = SimpleNamespace(status="completed", output_text='{"value":"ok"}')
        responses = SimpleNamespace(create=Mock(return_value=completed))
        client = SimpleNamespace(responses=responses)

        with (
            patch(
                "rfp2deck.llm.structured.settings",
                _settings(
                    openai_structured_background_enabled=False,
                ),
            ),
            patch("rfp2deck.llm.structured.get_client", return_value=client),
        ):
            result = response_as_schema(
                "x" * 40000,
                _Payload,
                model="gpt-test",
                timeout_seconds=30,
                background=True,
            )

        self.assertEqual(result.value, "ok")
        self.assertNotIn("background", responses.create.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
