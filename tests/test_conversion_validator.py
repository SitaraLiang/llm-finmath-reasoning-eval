import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conversion_validator import (  # noqa: E402
    repair_converted_exercise,
    strip_yaml_fence,
    validate_converted_exercise,
    validate_single_question_conversion,
)


class ConversionValidatorRepairTest(unittest.TestCase):
    def test_repairs_split_atom_fields(self):
        data = {
            "subquestions": [
                {
                    "atoms": [
                        {"preconditions": ["P"]},
                        {"arguments": ["Calculation"]},
                        {"outcomes": ["O"]},
                    ]
                }
            ]
        }

        repaired = repair_converted_exercise(data)

        self.assertEqual(
            repaired["subquestions"][0]["atoms"],
            [{"preconditions": ["P"], "arguments": ["Calculation"], "outcomes": ["O"]}],
        )
        _, error = validate_converted_exercise(repaired, expected_subquestions=1)
        self.assertIsNone(error)

    def test_leaves_complete_atom_unchanged(self):
        data = {
            "subquestions": [
                {
                    "atoms": [
                        {
                            "preconditions": ["P"],
                            "arguments": ["A"],
                            "outcomes": ["O"],
                        }
                    ]
                }
            ]
        }

        repaired = repair_converted_exercise(data)

        self.assertEqual(repaired, data)
        _, error = validate_converted_exercise(repaired, expected_subquestions=1)
        self.assertIsNone(error)

    def test_accepts_single_question_atoms_payload(self):
        data = {
            "atoms": [
                {
                    "preconditions": ["P"],
                    "arguments": ["A"],
                    "outcomes": ["O"],
                }
            ]
        }

        atoms, error = validate_single_question_conversion(data)

        self.assertIsNone(error)
        self.assertEqual(atoms, data["atoms"])

    def test_accepts_single_question_subquestions_wrapper(self):
        data = {
            "subquestions": [
                {
                    "atoms": [
                        {
                            "preconditions": ["P"],
                            "arguments": ["A"],
                            "outcomes": ["O"],
                        }
                    ]
                }
            ]
        }

        atoms, error = validate_single_question_conversion(data)

        self.assertIsNone(error)
        self.assertEqual(atoms, data["subquestions"][0]["atoms"])

    def test_strips_thinking_block_before_yaml(self):
        raw_response = """<think>
I should reason here, but this is not part of the YAML answer.
</think>
```yml
subquestions:
- atoms:
  - preconditions:
    - P
    arguments:
    - Calculation
    outcomes:
    - O
```
"""

        yaml_text = strip_yaml_fence(raw_response)

        self.assertTrue(yaml_text.startswith("subquestions:"))
        self.assertNotIn("<think>", yaml_text)
        self.assertNotIn("I should reason here", yaml_text)


if __name__ == "__main__":
    unittest.main()
