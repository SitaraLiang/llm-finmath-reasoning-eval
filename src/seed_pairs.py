import argparse
import csv
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

THEOREM_PROPERTY_KEYWORDS = (
    "theorem",
    "lemma",
    "definition",
    "property",
    "martingale",
    "brownian",
    "gaussian",
    "increment",
    "independent",
    "independence",
    "stationary",
    "covariance",
    "continuity",
    "continuous",
    "kolmogorov",
    "optional stopping",
    "doob",
    "law",
    "centered",
    "variance",
)
FORMULA_KEYWORDS = (
    "cov",
    "var",
    "lim",
    "mathbb",
    "mathrm",
    "operatorname",
    "min",
    "inf",
    "tau",
)
LABELS = ("equivalent", "unrelated")
CONCEPT_RULES = (
    (
        "optional_stopping_theorem",
        ("optional stopping", "doob"),
        "Doob optional stopping theorem bounded stopping time martingale",
    ),
    (
        "kolmogorov_continuity_theorem",
        ("kolmogorov", "continuous version", "continuous modification"),
        "Kolmogorov continuity theorem continuous modification stochastic process",
    ),
    (
        "martingale_property",
        ("martingale",),
        "Brownian motion martingale property",
    ),
    (
        "bounded_stopping_time",
        ("bounded stopping time",),
        "bounded stopping time definition probability",
    ),
    (
        "brownian_motion",
        ("$w_t$ is a brownian motion", "is a brownian motion", "brownian motion."),
        "Brownian motion definition stochastic process",
    ),
    (
        "gaussian_linear_combination",
        ("linear combination of independent gaussian", "linear combination of the \\(w_{1/t_i}\\)"),
        "linear combination of independent Gaussian random variables is Gaussian",
    ),
    (
        "gaussian_increments",
        ("gaussian increments", "gaussian increment"),
        "Brownian motion Gaussian increments",
    ),
    (
        "independent_increments",
        (
            "independent increments",
            "disjoint increments",
            "uncorrelated, hence independent",
            "independence of",
        ),
        "Brownian motion independent increments disjoint intervals",
    ),
    (
        "stationary_increments",
        ("stationary", "law depends only on"),
        "Brownian motion stationary increments",
    ),
    (
        "centered_gaussian_process",
        ("centered gaussian", "mean zero gaussian", "gaussianity:"),
        "centered Gaussian process definition",
    ),
    (
        "increment_variance",
        ("variance of an increment",),
        "Brownian motion increment variance",
    ),
    (
        "gaussian_increment_law",
        ("law of increments", "variance $t-s", "normal distribution"),
        "Brownian increments normal distribution variance t-s",
    ),
    (
        "brownian_motion_definition",
        ("classical definition of brownian motion", "definition of a brownian motion"),
        "classical definition of Brownian motion",
    ),
    (
        "continuity_of_paths",
        ("continuity of paths", "continuous sample paths"),
        "stochastic process continuous sample paths",
    ),
    (
        "brownian_law_of_large_numbers",
        ("law of large numbers for brownian motion",),
        "law of large numbers for Brownian motion W_t over t",
    ),
    (
        "brownian_time_inversion",
        ("time inversion", "w_{1/t}", "w 1 over t"),
        "Brownian motion time inversion t W 1 over t",
    ),
    (
        "finite_dimensional_gaussian_vector",
        ("for any arbitrary \\(t_1", "vector \\((x_{t_1}"),
        "Gaussian process finite-dimensional distributions Gaussian vector",
    ),
)


def project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def dedupe_key(text: str) -> str:
    normalized = normalize_text(text).lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip(" .,:;")
    return normalized


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_\\]+", text))


def keyword_score(text: str) -> int:
    lowered = text.lower()
    return sum(1 for keyword in THEOREM_PROPERTY_KEYWORDS if keyword in lowered)


def infer_concept(text: str) -> tuple[str, str]:
    lowered = text.lower()
    for concept, patterns, search_hint in CONCEPT_RULES:
        if any(pattern in lowered for pattern in patterns):
            return concept, search_hint
    return "TODO", "TODO"


def is_formula_heavy(text: str) -> bool:
    if not text:
        return False
    formula_chars = sum(1 for char in text if char in "\\{}_^=$[]()")
    return formula_chars / max(len(text), 1) > 0.28


def looks_formula_like(text: str, field: str = "") -> bool:
    lowered = text.lower()
    math_markers = ("\\", "$", "=", "^", "_", "{", "}", "[", "]")
    return (
        any(marker in text for marker in math_markers)
        and (field == "outcomes" or is_formula_heavy(text) or any(k in lowered for k in FORMULA_KEYWORDS))
    )


def looks_semantic_outcome(text: str, field: str = "") -> bool:
    return field == "outcomes" and not looks_formula_like(text, field)


def candidate_score(row: dict) -> int:
    text = normalize_text(row["text"])
    field = row["field"]

    if row.get("statement_type") != "semantic_text":
        return -100
    if not text or text == "Calculation":
        return -100
    if word_count(text) < 3:
        return -20

    score = keyword_score(text) * 4
    if field == "arguments":
        score += 5
    elif looks_semantic_outcome(text, field):
        score += 8
    elif field in {"preconditions", "assumptions", "assumption_global"}:
        score += 3
    elif field == "outcomes":
        score += 1

    if is_formula_heavy(text):
        score -= 4
        if field != "outcomes":
            score -= 4
    if len(text) > 220:
        score -= 2
    return score


def formula_candidate_score(row: dict) -> int:
    text = normalize_text(row["text"])
    field = row["field"]
    if row.get("statement_type") != "formula":
        return -100
    if not text or text == "Calculation":
        return -100
    if not looks_formula_like(text, field):
        return -100
    score = 8
    if field == "outcomes":
        score += 6
    elif field in {"preconditions", "assumptions", "assumption_global"}:
        score += 3
    if "=" in text:
        score += 3
    if len(text) > 180:
        score -= 2
    return score


def read_statement_rows(input_path: Path) -> list[dict]:
    if not input_path.exists():
        raise SystemExit(f"Error: Input CSV '{input_path}' does not exist.")

    with input_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit(f"Error: Input CSV '{input_path}' is empty.")
    if "text" not in rows[0] or "field" not in rows[0]:
        raise SystemExit(
            f"Error: Input CSV '{input_path}' must contain at least 'field' and 'text' columns."
        )
    if "statement_type" not in rows[0]:
        raise SystemExit(
            f"Error: Input CSV '{input_path}' must contain a 'statement_type' column. "
            "Regenerate it with `python src/extract_statements.py`."
        )
    return rows


def selected_enough(selected: list[dict], limit: int) -> bool:
    return limit > 0 and len(selected) >= limit


def select_candidates(rows: list[dict], limit: int, fields: set[str]) -> list[dict]:
    best_by_text = {}
    for row in rows:
        if fields and row.get("field") not in fields:
            continue
        text = normalize_text(row.get("text", ""))
        key = dedupe_key(text)
        score = candidate_score(row)
        if score <= 0:
            continue
        candidate = {**row, "text": text, "selection_score": score}
        if key not in best_by_text or score > best_by_text[key]["selection_score"]:
            best_by_text[key] = candidate

    ranked = sorted(
        best_by_text.values(),
        key=lambda item: (
            -item["selection_score"],
            item.get("field", ""),
            item["text"].lower(),
        ),
    )
    selected = []
    seen_concepts = set()
    for candidate in ranked:
        concept, _ = infer_concept(candidate["text"])
        if concept in seen_concepts:
            continue
        selected.append(candidate)
        seen_concepts.add(concept)
        if selected_enough(selected, limit):
            return selected

    for candidate in ranked:
        if candidate in selected:
            continue
        selected.append(candidate)
        if selected_enough(selected, limit):
            break
    return selected


def select_formula_candidates(rows: list[dict], limit: int, fields: set[str]) -> list[dict]:
    best_by_text = {}
    for row in rows:
        if fields and row.get("field") not in fields:
            continue
        text = normalize_text(row.get("text", ""))
        key = dedupe_key(text)
        score = formula_candidate_score(row)
        if score <= 0:
            continue
        candidate = {**row, "text": text, "selection_score": score}
        if key not in best_by_text or score > best_by_text[key]["selection_score"]:
            best_by_text[key] = candidate

    candidates = sorted(
        best_by_text.values(),
        key=lambda item: (
            -item["selection_score"],
            item.get("field", ""),
            item["text"].lower(),
        ),
    )
    return candidates[:limit]


def candidate_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index:04d}"


def text_b_variant_templates(label: str, equivalent_variants: int) -> list[str]:
    count = equivalent_variants if label == "equivalent" else 1
    return ["TODO" for _ in range(count)]


def add_candidates(
    selected: list[dict],
    pair_templates: list[dict],
    candidates: list[dict],
    prefix: str,
    statement_type: str,
    equivalent_variants: int,
) -> None:
    for index, candidate in enumerate(candidates, start=1):
        statement_id = candidate_id(prefix, index)
        concept, search_hint = infer_concept(candidate["text"])
        selected.append(
            {
                "id": statement_id,
                "type": statement_type,
                "concept": concept,
                "search_hint": search_hint,
                "field": candidate.get("field", ""),
                "source_file": candidate.get("source_file", ""),
                "subquestion_index": candidate.get("subquestion_index", ""),
                "atom_path": candidate.get("atom_path", ""),
                "selection_score": candidate["selection_score"],
                "text": candidate["text"],
            }
        )

        for label in LABELS:
            pair_templates.append(
                {
                    "id": f"{statement_id}_{label}",
                    "type": statement_type,
                    "label": label,
                    "concept": concept,
                    "search_hint": search_hint,
                    "source_statement_id": statement_id,
                    "text_a": candidate["text"],
                    "text_b_variants": text_b_variant_templates(label, equivalent_variants),
                }
            )


def build_output(
    semantic_candidates: list[dict],
    formula_candidates: list[dict],
    input_path: Path,
    equivalent_variants: int,
) -> dict:
    selected = []
    pair_templates = []
    try:
        generated_from = str(input_path.relative_to(PROJECT_ROOT))
    except ValueError:
        generated_from = str(input_path)

    add_candidates(
        selected,
        pair_templates,
        semantic_candidates,
        "sem",
        "semantic_text",
        equivalent_variants,
    )
    add_candidates(
        selected,
        pair_templates,
        formula_candidates,
        "form",
        "formula",
        equivalent_variants,
    )

    return {
        "metadata": {
            "description": (
                "Starter file for semantic embedding cosine-similarity calibration."
            ),
            "generated_from": generated_from,
            "labels": list(LABELS),
            "types": {
                "semantic_text": "theorem/property/assumption/outcome statements compared with sentence embeddings",
            },
            "note": (
                "This file calibrates semantic-text thresholds only. Formula/result statements "
                "and generic arguments such as Calculation are handled by rule-based comparisons "
                "in eval_embeddings.py/evaluate.py, so they are intentionally omitted here."
            ),
        },
        "selected_statements": selected,
        "pair_templates": pair_templates,
        "pairs": [],
    }


def write_yaml(output_path: Path, data: dict) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "Error: PyYAML is required. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.dump(
            data,
            file,
            Dumper=yaml.Dumper,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            width=100,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a manually curated starter formulation_pairs.yaml from "
            "data/evaluation/ground_truth_statements.csv."
        )
    )
    parser.add_argument(
        "--input",
        default="data/evaluation/ground_truth_statements.csv",
        help="CSV produced by src/extract_statements.py.",
    )
    parser.add_argument(
        "--output",
        default="data/evaluation/formulation_pairs.yaml",
        help="Starter YAML file to write.",
    )
    parser.add_argument(
        "--semantic-limit",
        type=int,
        default=0,
        help="Maximum number of semantic source statements to select. Use 0 to select all.",
    )
    parser.add_argument(
        "--formula-limit",
        type=int,
        default=0,
        help=(
            "Maximum number of formula/result source statements to select. Keep this at 0 for "
            "the current strategy because formulas are not calibrated with embeddings."
        ),
    )
    parser.add_argument(
        "--equivalent-variants",
        type=int,
        default=2,
        help="Number of text_b TODO slots to create for each equivalent template.",
    )
    parser.add_argument(
        "--fields",
        nargs="*",
        default=["arguments", "preconditions", "assumptions", "assumption_global", "outcomes"],
        help="Statement fields to consider.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = project_path(args.input)
    output_path = project_path(args.output)

    rows = read_statement_rows(input_path)
    fields = set(args.fields)
    semantic_candidates = select_candidates(rows, args.semantic_limit, fields)
    formula_candidates = select_formula_candidates(rows, args.formula_limit, fields)
    if not semantic_candidates and not formula_candidates:
        print("Error: No candidate statements were selected.", file=sys.stderr)
        sys.exit(1)

    data = build_output(
        semantic_candidates,
        formula_candidates,
        input_path,
        args.equivalent_variants,
    )
    write_yaml(output_path, data)

    print("Formulation-pair seed creation complete.")
    print(f"Input statements: {len(rows)}")
    print(f"Semantic statements: {len(semantic_candidates)}")
    print(f"Formula statements: {len(formula_candidates)}")
    print(f"Selected statements: {len(data['selected_statements'])}")
    print(f"Pair templates: {len(data['pair_templates'])}")
    print(f"Output YAML: {output_path}")


if __name__ == "__main__":
    main()
