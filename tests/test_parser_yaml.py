import unittest
from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parser import AnnotationParseError, parse_solution_structure  # noqa: E402


def atom(label: str) -> str:
    return f"""%@ATOM
%@PRECOND
P{label}
%@ARGUMENT
A{label}
%@OUTCOME
O{label}
%@ATOM_END"""


class ParserYamlRepresentationTest(unittest.TestCase):
    def parse_atoms(self, text: str):
        return parse_solution_structure(text, "test.tex")["atoms"]

    def test_one_atom(self):
        atoms = self.parse_atoms(atom("1"))
        self.assertIsInstance(atoms, list)
        self.assertEqual(atoms[0]["preconditions"], ["P1"])
        self.assertEqual(atoms[0]["outcomes"], ["O1"])

    def test_ordered_list_containing_atoms(self):
        atoms = self.parse_atoms(f"%@LIST_START\n{atom('1')}\n{atom('2')}\n%@LIST_END")
        self.assertIsInstance(atoms, list)
        self.assertEqual([item["outcomes"][0] for item in atoms], ["O1", "O2"])

    def test_unordered_set_containing_atoms(self):
        atoms = self.parse_atoms(f"%@SET_START\n{atom('1')}\n{atom('2')}\n%@SET_END")
        self.assertIsInstance(atoms, tuple)
        self.assertEqual({item["outcomes"][0] for item in atoms}, {"O1", "O2"})

    def test_set_containing_list(self):
        atoms = self.parse_atoms(f"%@SET_START\n%@LIST_START\n{atom('1')}\n%@LIST_END\n%@SET_END")
        self.assertIsInstance(atoms, tuple)
        self.assertIsInstance(atoms[0], list)

    def test_list_containing_set(self):
        atoms = self.parse_atoms(f"%@LIST_START\n%@SET_START\n{atom('1')}\n%@SET_END\n%@LIST_END")
        self.assertIsInstance(atoms, list)
        self.assertIsInstance(atoms[0], tuple)

    def test_set_containing_set(self):
        atoms = self.parse_atoms(f"%@SET_START\n%@SET_START\n{atom('1')}\n%@SET_END\n%@SET_END")
        self.assertIsInstance(atoms, tuple)
        self.assertIsInstance(atoms[0], tuple)

    def test_list_containing_list(self):
        atoms = self.parse_atoms(f"%@LIST_START\n%@LIST_START\n{atom('1')}\n%@LIST_END\n%@LIST_END")
        self.assertIsInstance(atoms, list)
        self.assertIsInstance(atoms[0], list)

    def test_several_levels_of_mixed_nesting(self):
        atoms = self.parse_atoms(
            f"%@SET_START\n%@LIST_START\n%@SET_START\n%@LIST_START\n{atom('1')}\n%@LIST_END\n%@SET_END\n%@LIST_END\n%@SET_END"
        )
        self.assertIsInstance(atoms, tuple)
        self.assertIsInstance(atoms[0], list)
        self.assertIsInstance(atoms[0][0], tuple)
        self.assertIsInstance(atoms[0][0][0], list)

    def test_atom_with_several_preconditions(self):
        atoms = self.parse_atoms("""%@ATOM
%@PRECOND
P1
%@PRECOND
P2
%@ARGUMENT
A
%@OUTCOME
O
%@ATOM_END""")
        self.assertEqual(atoms[0]["preconditions"], ["P1", "P2"])

    def test_atom_with_several_outcomes(self):
        atoms = self.parse_atoms("""%@ATOM
%@ARGUMENT
A
%@OUTCOME
O1
%@OUTCOME
O2
%@ATOM_END""")
        self.assertEqual(atoms[0]["outcomes"], ["O1", "O2"])

    def test_atom_with_valid_strength(self):
        atoms = self.parse_atoms("""%@ATOM
%@STRENGTH: 0.5
%@ARGUMENT
A
%@OUTCOME
O
%@ATOM_END""")
        self.assertEqual(atoms[0]["strength"], 0.5)

    def test_calculation_argument_uses_fixed_label(self):
        atoms = self.parse_atoms("""%@ATOM
%@ARGUMENT:CALCUL
\\[
1 + 1 = 2
\\]
%@OUTCOME
O
%@ATOM_END""")
        self.assertEqual(atoms[0]["arguments"], ["Calculation"])

    def test_ignores_comment_text_on_tag_lines(self):
        atoms = self.parse_atoms("""%@ATOM
%@PRECOND <-- note only, not content
P
%@ARGUMENT <-- this is alternative argument
A
%@OUTCOME <-- note only, not content
O
%@ATOM_END""")
        self.assertEqual(atoms[0]["preconditions"], ["P"])
        self.assertEqual(atoms[0]["arguments"], ["A"])
        self.assertEqual(atoms[0]["outcomes"], ["O"])

    def test_ignores_plain_tex_comments_inside_tag_content(self):
        atoms = self.parse_atoms("""%@ATOM
%@PRECOND
P
% this comment should not become a precondition
%@ARGUMENT
A % this comment should be stripped
%@OUTCOME
O
%@ATOM_END""")
        self.assertEqual(atoms[0]["preconditions"], ["P"])
        self.assertEqual(atoms[0]["arguments"], ["A"])

    def test_invalid_strength(self):
        with self.assertRaisesRegex(AnnotationParseError, "Invalid @STRENGTH"):
            self.parse_atoms("""%@ATOM
%@STRENGTH: strong
%@ARGUMENT
A
%@OUTCOME
O
%@ATOM_END""")

    def test_unclosed_annotation_block(self):
        with self.assertRaisesRegex(AnnotationParseError, "Unclosed @SET_START"):
            self.parse_atoms(f"%@SET_START\n{atom('1')}")

    def test_deterministic_output_across_two_executions(self):
        atoms = self.parse_atoms(f"%@SET_START\n{atom('1')}\n{atom('1')}\n{atom('2')}\n%@SET_END")
        first = yaml.dump(atoms, Dumper=yaml.Dumper, allow_unicode=True, sort_keys=False, default_flow_style=False)
        second = yaml.dump(atoms, Dumper=yaml.Dumper, allow_unicode=True, sort_keys=False, default_flow_style=False)
        self.assertEqual(first, second)

    def test_yaml_round_trip_preserves_tuple_and_list(self):
        atoms = self.parse_atoms(f"%@SET_START\n%@LIST_START\n{atom('1')}\n%@LIST_END\n%@SET_END")
        dumped = yaml.dump(atoms, Dumper=yaml.Dumper, allow_unicode=True, sort_keys=False, default_flow_style=False)
        loaded = yaml.load(dumped, Loader=yaml.FullLoader)
        self.assertIsInstance(loaded, tuple)
        self.assertIsInstance(loaded[0], list)


if __name__ == "__main__":
    unittest.main()
