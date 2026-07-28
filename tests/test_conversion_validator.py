import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conversion_validator import repair_converted_exercise, validate_converted_exercise  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
