import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = types.ModuleType("yaml")
    yaml.FullLoader = object
    yaml.YAMLError = ValueError
    yaml.load = lambda text, Loader=None: json.loads(text)
    yaml.safe_dump = lambda data, **kwargs: json.dumps(data, indent=4)
    sys.modules["yaml"] = yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    "/Users/sitaraliang/Downloads/stage/cmap/llm-finmath-reasoning-eval/src",
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from select_responses import run_selection  # noqa: E402


def valid_conversion(outcome: str, argument: str = "Calculation") -> dict:
    return {
        "subquestions": [
            {
                "atoms": [
                    {
                        "preconditions": ["W is a Brownian motion"],
                        "arguments": [argument],
                        "outcomes": [outcome],
                    }
                ]
            }
        ]
    }


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.call1 = self.root / "call1"
        self.call2 = self.root / "call2"
        self.selected = self.root / "selected"
        source_path = self.call1 / "model-a" / "en" / "baseline" / "pc2_q1_seq.txt"
        source_path.parent.mkdir(parents=True)
        source_path.write_text(
            "Question 1:\nShow the result.\n\nAnswer:\n"
            "Since W is a Brownian motion, calculate $E[W_t]=0$.\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def config(self):
        return {
            "input": {
                "call1_root": str(self.call1),
                "call2_root": str(self.call2),
            },
            "output": {
                "root_directory": str(self.selected),
                "synchronize_outputs": True,
            },
            "filters": {},
        }

    def write_candidate(self, model: str, data: dict):
        path = self.call2 / "model-a" / model / "en" / "baseline" / "pc2_q1_seq.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=4), encoding="utf-8")
        return path

    def test_selects_more_source_faithful_valid_candidate(self):
        faithful = self.write_candidate("model-faithful", valid_conversion("E[W_t]=0"))
        self.write_candidate(
            "model-hallucinated",
            valid_conversion("The stock price equals one million euros", "Unrelated theorem"),
        )

        report = run_selection(self.config())

        selection = report["selections"][0]
        self.assertEqual(selection["selected_call2_model"], "model-faithful")
        output = self.selected / "model-a" / "en" / "baseline" / "pc2_q1_seq.yaml"
        self.assertEqual(output.read_text(encoding="utf-8"), faithful.read_text(encoding="utf-8"))

    def test_excludes_structurally_invalid_candidate(self):
        invalid = valid_conversion("E[W_t]=0")
        invalid["subquestions"][0]["atoms"][0]["arguments"] = []
        self.write_candidate("invalid-model", invalid)
        self.write_candidate("valid-model", valid_conversion("E[W_t]=0"))

        report = run_selection(self.config())

        candidates = report["selections"][0]["candidates"]
        invalid_result = next(item for item in candidates if item["call2_model"] == "invalid-model")
        self.assertFalse(invalid_result["valid"])
        self.assertIn("arguments must contain", invalid_result["error"])
        self.assertEqual(report["selections"][0]["selected_call2_model"], "valid-model")

    def test_all_failed_is_reported_and_stale_output_removed(self):
        invalid = valid_conversion("E[W_t]=0")
        del invalid["subquestions"][0]["atoms"][0]["outcomes"]
        self.write_candidate("invalid-model", invalid)
        stale = self.selected / "model-a" / "en" / "baseline" / "pc2_q1_seq.yaml"
        stale.parent.mkdir(parents=True)
        stale.write_text("stale: true\n", encoding="utf-8")

        report = run_selection(self.config())

        self.assertEqual(report["summary"]["all_candidates_failed"], 1)
        self.assertEqual(report["summary"]["stale_outputs_removed"], 1)
        self.assertEqual(report["selections"][0]["status"], "all_candidates_failed")
        self.assertFalse(stale.exists())

    def test_existing_selected_yaml_is_not_overwritten_by_default(self):
        self.write_candidate("valid-model", valid_conversion("E[W_t]=0"))
        destination = self.selected / "model-a" / "en" / "baseline" / "pc2_q1_seq.yaml"
        destination.parent.mkdir(parents=True)
        destination.write_text("existing: true\n", encoding="utf-8")

        report = run_selection(self.config())

        self.assertEqual(destination.read_text(encoding="utf-8"), "existing: true\n")
        self.assertEqual(report["summary"]["selected_files_written"], 0)
        self.assertEqual(report["summary"]["selected_files_skipped"], 1)
        self.assertEqual(report["selections"][0]["file_status"], "skipped_existing")


if __name__ == "__main__":
    unittest.main()
