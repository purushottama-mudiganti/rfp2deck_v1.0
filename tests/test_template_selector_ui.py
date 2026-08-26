from __future__ import annotations

import os
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class TemplateSelectorUiTests(unittest.TestCase):
    def test_template_selector_defaults_to_expanded_and_resets_stale_plan(self) -> None:
        original_password = os.environ.get("APP_PASSWORD")
        original_template = os.environ.get("HCLTECH_TEMPLATE_PATH")
        os.environ["APP_PASSWORD"] = "template-ui-test"
        os.environ["HCLTECH_TEMPLATE_PATH"] = "templates/hcltech_expanded_v5.potx"
        try:
            app = AppTest.from_file("app/rfp2deck_app.py", default_timeout=30).run()
            app.text_input[0].input("template-ui-test").run()

            template = next(item for item in app.selectbox if item.label == "Template")
            self.assertEqual(Path(template.value).name, "hcltech_expanded_v5.potx")
            self.assertTrue(
                {
                    "hcltech_expanded_v5.potx",
                    "hcltech_modern_template_16x9.pptx",
                    "standard_proposal_template_v1.pptx",
                }.issubset(set(template.options))
            )
            self.assertNotIn("hcltech_expanded_v5.potx.bak", template.options)

            app.session_state["deck_plan"] = "stale-plan"
            app.session_state["tpl_bytes"] = b"stale-template"
            selected_path = str(Path("templates/hcltech_modern_template_16x9.pptx").resolve())
            template.set_value(selected_path).run()

            self.assertEqual(app.session_state["selected_template_path"], selected_path)
            self.assertIsNone(app.session_state["deck_plan"])
            self.assertIsNone(app.session_state["tpl_bytes"])
            self.assertEqual(app.session_state["wizard_step"], 1)
            self.assertEqual(list(app.exception), [])
        finally:
            if original_password is None:
                os.environ.pop("APP_PASSWORD", None)
            else:
                os.environ["APP_PASSWORD"] = original_password
            if original_template is None:
                os.environ.pop("HCLTECH_TEMPLATE_PATH", None)
            else:
                os.environ["HCLTECH_TEMPLATE_PATH"] = original_template


if __name__ == "__main__":
    unittest.main()
