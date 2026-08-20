import sys
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import call1


class Call1LanguagePromptTests(unittest.TestCase):
    def test_selects_language_specific_templates(self):
        prompts = {
            "languages": {
                "en": {"common_header": "English", "strategies": {}},
                "fr": {"common_header": "Français", "strategies": {}},
            }
        }

        selected = call1.prompt_templates_for_language(prompts, "fr")

        self.assertEqual(selected["common_header"], "Français")

    def test_missing_language_has_clear_error(self):
        prompts = {"languages": {"en": {}}}

        with self.assertRaisesRegex(SystemExit, "language 'fr'"):
            call1.prompt_templates_for_language(prompts, "fr")

    def test_french_prompt_uses_french_empty_value(self):
        prompts = {
            "empty_value": "aucune",
            "labels": {"question": "Question"},
            "common_header": "Hypothèses globales :\n$global_assumptions",
            "strategies": {
                "strictly_sequential": (
                    "Hypothèses locales :\n$current_assumptions\n"
                    "Question $current_question_number :\n$current_question"
                )
            },
        }
        exercise = {"assumption_global": []}
        subquestion = {"question": "Calculer la valeur.", "assumptions": []}

        prompt = call1.build_prompt(
            exercise,
            Path("pc0_q0.yaml"),
            subquestion,
            [subquestion],
            1,
            "strictly_sequential",
            [],
            prompts,
            [],
        )

        self.assertIn("Hypothèses globales :\naucune", prompt)
        self.assertIn("Hypothèses locales :\naucune", prompt)


if __name__ == "__main__":
    unittest.main()
