from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rfp2deck.ingestion.template_resolver import discover_presentation_templates


class TemplateDiscoveryTests(unittest.TestCase):
    def test_discovers_only_supported_templates_in_top_level_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            template_dir = Path(temp_dir)
            (template_dir / "B-template.POTX").touch()
            (template_dir / "a-template.pptx").touch()
            (template_dir / "ignored.potx.bak").touch()
            nested = template_dir / "nested"
            nested.mkdir()
            (nested / "nested-template.pptx").touch()

            discovered = discover_presentation_templates(template_dir)

            self.assertEqual(
                [path.name for path in discovered],
                ["a-template.pptx", "B-template.POTX"],
            )

    def test_missing_template_directory_returns_empty_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing"

            self.assertEqual(discover_presentation_templates(missing), [])


if __name__ == "__main__":
    unittest.main()
