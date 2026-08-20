import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from call2 import (  # noqa: E402
    output_root,
    prompt_templates_for_language,
    split_question_blocks,
)


class Call2QuestionSplitTest(unittest.TestCase):
    def test_splits_script_labeled_question_blocks(self):
        source = """Exercise: pc2_q1

Question 1:
First question.

Answer:
First answer.

Question 2:
Second question.

Answer:
Second answer.
"""

        blocks = split_question_blocks(source)

        self.assertEqual(len(blocks), 2)
        self.assertIn("Question 1:", blocks[0])
        self.assertIn("First answer.", blocks[0])
        self.assertIn("Question 2:", blocks[1])
        self.assertIn("Second answer.", blocks[1])


class Call2LanguagePromptTest(unittest.TestCase):
    def test_applies_french_overrides(self):
        prompts = {
            "conversion": "English conversion",
            "per_question_conversion": "English per question",
            "repair": "English repair",
            "languages": {
                "fr": {
                    "conversion": "Conversion francaise",
                    "repair": "Reparation francaise",
                }
            },
        }

        selected = prompt_templates_for_language(prompts, "fr")

        self.assertEqual(selected["conversion"], "Conversion francaise")
        self.assertEqual(selected["repair"], "Reparation francaise")
        self.assertEqual(selected["per_question_conversion"], "English per question")

    def test_uses_default_templates_for_english(self):
        prompts = {
            "conversion": "English conversion",
            "repair": "English repair",
            "languages": {"fr": {"conversion": "Conversion francaise"}},
        }

        selected = prompt_templates_for_language(prompts, "en")

        self.assertEqual(selected["conversion"], "English conversion")
        self.assertEqual(selected["repair"], "English repair")


class Call2OutputRootTest(unittest.TestCase):
    def test_defaults_to_per_question_output_root(self):
        root = output_root({})
        self.assertEqual(root, PROJECT_ROOT / "outputs" / "call2_per_question")

    def test_defaults_to_zeroshot_per_question_output_root(self):
        root = output_root(
            {
                "experiment": {"name": "financial_math_call2_zeroshot_v1"},
                "conversion": {"mode": "per_question"},
            }
        )
        self.assertEqual(root, PROJECT_ROOT / "outputs" / "call2_zeroshot_per_question")

    def test_explicit_output_root_overrides_mode_default(self):
        root = output_root(
            {
                "conversion": {"mode": "per_question"},
                "output": {"root_directory": "outputs/custom"},
            }
        )
        self.assertEqual(root, PROJECT_ROOT / "outputs" / "custom")


if __name__ == "__main__":
    unittest.main()
