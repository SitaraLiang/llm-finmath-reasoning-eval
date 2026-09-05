# LLM FinMath Reasoning Eval

This repository was developed as part of my research internship at CMAP, École Polytechnique, under the supervision of Charles-Albert Lehalle.

The project introduces a structured evaluation framework for analyzing and diagnosing LLM reasoning in quantitative finance and financial mathematics. The framework decomposes human-annotated LaTeX solutions and model-generated reasoning into a common proof-atom representation, enabling step-level evaluation of logical alignment, reasoning chains, and sequential error propagation. It supports multiple reasoning protocols for studying how intermediate context affects downstream performance and combines calibrated embedding-based alignment with selective LLM-as-a-judge adjudication for ambiguous cases.


## Table of Contents

- [Pipeline](#pipeline)
  - [LaTeX annotation reference](#latex-annotation-reference)
  - [Evaluation dimensions](#evaluation-dimensions)
- [Setup](#setup)
- [Usage](#usage)
  - [Step 1: Import and parse exercises](#step-1-import-and-parse-exercises)
  - [Step 2: Refresh calibration data](#step-2-refresh-the-calibration-data-when-necessary)
  - [Step 3: Generate Call 1 answers](#step-3-generate-call-1-answers)
  - [Step 4: Convert and select answers](#step-4-convert-and-select-plain-text-answers)
  - [Step 5: Evaluate](#step-5-evaluate)
- [Detailed configuration reference](#detailed-configuration-reference)
  - [Adding a Call 1 experiment](#adding-a-call-1-experiment)
  - [Adding a downstream experiment](#adding-a-downstream-experiment)
- [Notes](#notes)


## Pipeline

1. **Import LaTeX exercises**
   - Overleaf is treated as the single source of truth.
   - `src/import.py` copies downloaded `.tex` files into `data/raw_tex/{lang}/`.
   - Imported filenames are normalized from `pc{n}_q{m}_{lang}.tex` to `pc{n}_q{m}.tex`.

2. **Parse annotated LaTeX to ground truth**
   - `src/parser.py` reads tagged LaTeX from `data/raw_tex/{lang}/`.
   - It writes ground-truth YAML to `data/ground_truth/{lang}/`.
   - YAML uses Python lists for ordered proof lists and `!!python/tuple` for unordered mathematical sets.

3. **Call 1: generate model answers**
   - `src/call1.py` prompts Ollama models to solve each exercise.
   - Plain-text outputs are stored under `outputs/call1/plain_text/{model}/{lang}/{variation}/`.
   - Direct proof-atom YAML outputs are stored under `outputs/call1/yaml/{model}/{lang}/{variation}/` and bypass Call 2.
   - Call 1 configurations are composed from shared settings, an output mode, and one multilingual variation under `config/call1/`.
   - Each strategy gets its own file:
     - `pc2_q1_seq.txt`: strictly sequential
     - `pc2_q1_acc.txt`: prompt accumulation
     - `pc2_q1_gtf.txt`: ground-truth forcing
     - `pc2_q1_self.txt`: self history
   - Strategy meanings:
     - **strictly sequential (`seq`)**: the model answers only the current subquestion, with its local assumptions.
     - **prompt accumulation (`acc`)**: the model sees all questions up to the current one, but answers only the current subquestion.
     - **ground-truth forcing (`gtf`)**: the model sees previous questions together with their ground-truth solutions, then answers the current subquestion.
     - **self history (`self`)**: the model sees previous questions together with its own previous answers, then answers the current subquestion.
   - Failed generations are written separately under the corresponding `outputs/call1/{mode}/` directory.

4. **Call 2: convert answers to proof-atom YAML**
   - `src/call2.py` converts Call 1 text answers into the structured YAML format.
   - Current recommended mode is `per_question`: each `Question N:` block is converted separately, then Python assembles the final `subquestions` list.
   - Validation and repair helpers live in `src/conversion_validator.py`.
   - Successful conversions write only `.yaml`; raw model text is saved as `.raw.txt` only for failed conversions.
   - The current zero-shot per-question experiment writes to `outputs/call2/`.
   - Failed conversions are summarized in `{output_root}/error_files.yaml`.

5. **Select Call 2 conversions**
   - `src/select_responses.py` validates and ranks all available Call 2
     conversions for each Call 1 response.
   - Ranking compares candidates only with the Call 1 source; it never reads the
     ground truth.
   - Selected YAML files are stored unchanged under `outputs/selected_responses/`.
   - `outputs/selected_responses/selection_report.yaml` records candidate scores,
     validation failures, close decisions, and cases where every converter failed.

6. **Evaluation**
   - `src/extract_statements.py` extracts statements separately to `data/evaluation/{lang}/ground_truth_statements.csv`.
   - `data/evaluation/{lang}/formulation_pairs.yaml` stores independently curated formulation pairs for each language.
   - `src/eval_embeddings.py` scores formulation pairs and writes calibration outputs under `outputs/evaluation/`.
   - `src/evaluate.py` builds D1/D2/D3 alignment tables, a D4 order report, and aggregate summaries.
   - Evaluation can run in embedding-only mode or with an optional second-stage LLM judge for ambiguous embedding matches.
   - Downstream stages use composable `base.yaml` and `experiments/` configurations, matching the Call 1 structure.

### LaTeX Annotation Reference

Tags are LaTeX comments written as either `% @TAG` or `%@TAG`. Tag content is
ordinary, uncommented LaTeX; unrelated comments and trailing comments on tag
lines are ignored by the parser.

| Tag | Meaning |
|---|---|
| `@CONTEXT` | Exercise title or shared context. |
| `@ASSUMPTION_GLOBAL` | Assumption available to every subquestion. Repeat the tag for multiple assumptions. |
| `@QUESTION` / `@QUESTION_END` | Start and end of a subquestion statement. |
| `@ASSUMPTION` / `@ASSUMPTION_END` | Local assumption inside the current question. |
| `@LIST_START` / `@LIST_END` | Ordered sequence of atoms or nested containers. |
| `@SET_START` / `@SET_END` | Unordered mathematical set of atoms or nested containers. |
| `@ATOM` / `@ATOM_END` | One proof step containing preconditions, arguments, and outcomes. |
| `@PRECOND` / `@PRECOND_END` | Required input of an atom. Repeat `@PRECOND` for multiple values. |
| `@ARGUMENT` / `@ARGUMENT_END` | The theorem, property, lemma, or method used by an atom. |
| `@ARGUMENT:CALCUL` | Store the standardized argument `Calculation`. |
| `@OUTCOME` / `@OUTCOME_END` | Result produced by an atom. Repeat `@OUTCOME` for multiple values. |

There is no generic `@END` tag. An atom must contain at least one argument; its
preconditions and outcomes may be empty or contain several entries. Lists and
sets may be nested to any depth. YAML lists preserve proof order, while sets are
serialized as deterministic Python tuples with `!!python/tuple`; tuple order has
no mathematical meaning, and exact duplicate set children are removed.

```latex
\part
% @QUESTION
Show that $X_t$ is a martingale.
% @ASSUMPTION
$X_t$ is integrable.
% @ASSUMPTION_END
% @QUESTION_END

\begin{xsolution}
% @LIST_START
% @ATOM
% @PRECOND
$X_t$ is integrable.
% @PRECOND_END
% @ARGUMENT
Conditional expectation property.
% @ARGUMENT_END
% @OUTCOME
$\mathbb{E}[X_t\mid\mathcal F_s]=X_s$.
% @OUTCOME_END
% @STRENGTH: 1.0
% @ATOM_END
% @LIST_END
\end{xsolution}
```

### Evaluation Dimensions

- **D1 - Assumption coverage:** measures whether the prediction's preconditions cover the global assumptions, local assumptions, required ground-truth preconditions, and outcomes established by preceding subquestions.
- **D2 - Argument identification:** measures whether the prediction uses the expected theorem, lemma, property, or calculation method.
- **D3 - Mathematical correctness:** measures whether the prediction's intermediate and final outcomes match the ground truth.
- **D4 - Reasoning Order:** measures whether matched proof atoms respect the ordering constraints induced by ordered lists in the ground truth; elements inside tuples are treated as unordered alternatives.

The aggregate `overall_mean_score` averages D1, D2, and D3. The aggregate
`overall_with_D4_score` additionally includes the strict D4 order score.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install and run Ollama, then pull the models you want to test:

```bash
ollama pull tinyllama
ollama pull llama3.2:1b
ollama pull deepseek-r1:1.5b
ollama pull qwen2.5-coder:0.5b
ollama pull qwen3:0.6b
ollama pull llama3.1:8b
ollama pull deepseek-r1:7b
ollama pull mistral:7b
ollama pull qwen3:8b
ollama pull gemma3:12b
```

For Call 2, `llama3.1:8b` is currently the strongest baseline observed for strict YAML formatting. Smaller models are useful as baselines, but often fail on schema or YAML syntax.

## Usage

Run commands from the repository root. A complete experiment follows the steps
below; do not evaluate new data before checking whether the embedding calibration
dataset also needs to be refreshed.

### Step 1: Import and parse exercises

Import a downloaded Overleaf directory. The Overleaf project remains the source
of truth and is not stored in this repository:

```bash
python src/import.py \
  --source ../overleaf/exercices_en_changed \
  --destination data/raw_tex
```

Convert all annotated LaTeX exercises into ground-truth YAML:

```bash
python src/parser.py --input data/raw_tex --output data/ground_truth
```

This produces one file per language and exercise under
`data/ground_truth/{lang}/pc{n}_q{m}.yaml`.

### Step 2: Refresh the calibration data when necessary

This step is required after adding exercises that introduce new mathematical
concepts or substantially new formulations. If the added exercises contain only
concepts already represented in `formulation_pairs.yaml`, the existing calibrated
threshold may be reused, although regenerating the statement inventory is still
recommended.

1. Extract all assumptions, preconditions, arguments, and outcomes from the
   current ground truth:

   ```bash
   python src/extract_statements.py
   ```

   This writes `data/evaluation/{lang}/ground_truth_statements.csv`.

2. Review the statement inventories. If new concepts should participate in
   calibration, increase `--concept-count` and update the seed without
   discarding completed variants. For example, to expand an existing 12-concept
   file to 16 concepts:

   ```bash
   python src/seed_pairs.py \
     --input-root data/evaluation \
     --output-root data/evaluation \
     --concept-count 16 \
     --overwrite
   ```

   Use `--overwrite --reselect` only when you intentionally want to replace the
   existing concept selection. Otherwise, completed concepts and variants are
   retained.

3. Manually complete every `TODO` in each
   `data/evaluation/{lang}/formulation_pairs.yaml`. The variants must be written
   in the same language as `text_a` and classified using the configured labels,
   such as `equivalent`, `related_but_not_equivalent`, and `unrelated`.

4. Calibrate the embedding threshold for every available language:

   ```bash
   python src/eval_embeddings.py pairs \
     --input data/evaluation \
     --output-dir outputs/evaluation/threshold
   ```

   The generated registry is stored at
   `outputs/evaluation/threshold/calibration.yaml`. `evaluate.py` reads the
   corresponding model/language threshold automatically.

To update only one language, pass `--language` to statement extraction and pair
seeding, for example:

```bash
python src/extract_statements.py --language fr
python src/seed_pairs.py --language fr --concept-count 12
```

The calibration registry is keyed by embedding model and language. By default,
the generated judge interval is symmetric around the recommended embedding
threshold: `threshold - 0.10` to `threshold + 0.10`, clamped to `[0, 1]`.
The margins can be overridden with `--judge-low-margin` and
`--judge-high-margin`; `--judge-high-threshold` remains available as an explicit
upper-bound override. Command-line evaluation thresholds take priority over the
registry; missing registry entries fall back to `config/evaluation/base.yaml`.

### Step 3: Generate Call 1 answers

For the plain-text pipeline:

```bash
python src/call1.py \
  --config config/call1/experiments/baseline_plain_text.yaml
```

For the direct-YAML experiment, which bypasses Call 2 and response selection:

```bash
python src/call1.py \
  --config config/call1/experiments/baseline_direct_yaml.yaml
```

### Step 4: Convert and select plain-text answers

For example, process the English baseline outputs with Call 2 and select the
best valid conversion for each Call 1 response:

```bash
python src/call2.py --config config/call2/experiments/baseline_en.yaml
python src/select_responses.py --config config/selection/experiments/baseline_en.yaml
```

Call 2 converts one question block at a time. The selector excludes missing or
invalid conversions and ranks the remaining candidates without consulting the
ground truth.

### Step 5: Evaluate

Evaluate selected plain-text responses:

```bash
python src/evaluate.py \
  --config config/evaluation/experiments/baseline_plain_text_en.yaml
```

Evaluate direct-YAML responses:

```bash
python src/evaluate.py \
  --config config/evaluation/experiments/baseline_direct_yaml_en.yaml
```

Evaluation produces per-case matrices and aggregate reports by model and
strategy. Depending on the configuration, ambiguous embedding matches can be
reviewed by an LLM judge. Existing output groups are skipped when
`output.overwrite_existing` is `false`.

## Detailed Configuration Reference

### Adding a Call 1 Experiment

Call 1 configurations are assembled recursively with `extends`:

```text
config/call1/
├── base.yaml                 # models, languages, paths, retries, strategies
├── modes/
│   ├── plain_text.yaml       # localized plain-text prompts
│   └── yaml.yaml             # localized direct-YAML prompts
├── variations/              # one multilingual experimental change per file
└── experiments/             # small runnable manifests
```

To add a variation, first create `config/call1/variations/role_researcher.yaml`:

```yaml
variation:
  id: "role_researcher"
  instruction:
    en: "You are a researcher in quantitative finance."
    fr: "Vous êtes chercheur en finance quantitative."
```

Then create a runnable manifest, for example
`config/call1/experiments/researcher_plain_text.yaml`:

```yaml
extends:
  - "../base.yaml"
  - "../modes/plain_text.yaml"
  - "../variations/role_researcher.yaml"

experiment:
  name: "call1_researcher_plain_text"
  version: 1
```

Run it with:

```bash
python src/call1.py \
  --config config/call1/experiments/researcher_plain_text.yaml
```

For direct YAML, create the same manifest with `../modes/yaml.yaml`. The
variation identifier becomes the output directory name, for example
`outputs/call1/plain_text/{model}/fr/role_researcher/`. Shared settings should
be changed in `base.yaml`; mode-specific prompt rules belong in `modes/`; only
the experimental instruction belongs in `variations/`.

To run an experiment for only one language, override the inherited language
filter in its manifest:

```yaml
input:
  filters:
    languages:
      - "fr"
```

When adding a new language, add its ground-truth directory, include it in
`base.yaml`, provide localized prompts in both mode files, and add the localized
instruction to every variation that will run in that language.

### Adding a Downstream Experiment

Call 2, selection, and evaluation use the same inheritance convention as Call
1. Shared prompts, models, paths, and scoring settings belong in `base.yaml` or
`modes/`; a file under `experiments/` should contain only the language and
variation being tested.

```text
config/call2/
├── base.yaml
└── experiments/

config/selection/
├── base.yaml
└── experiments/

config/evaluation/
├── base.yaml
├── modes/
└── experiments/
```

For a new English variation named `role_researcher`, create matching thin
manifests in all three stages:

```yaml
# config/call2/experiments/researcher_en.yaml
extends:
  - "../base.yaml"

input:
  filters:
    languages: ["en"]
    variations: ["role_researcher"]
```

```yaml
# config/selection/experiments/researcher_en.yaml
extends:
  - "../base.yaml"

filters:
  languages: ["en"]
  variations: ["role_researcher"]
```

```yaml
# config/evaluation/experiments/researcher_plain_text_en.yaml
extends:
  - "../base.yaml"
  - "../modes/plain_text.yaml"

input:
  filters:
    languages: ["en"]
    variations: ["role_researcher"]
```

Evaluation filters are real input filters. Files from other languages or
variations are not scored, while the expected model and strategy lists still
make missing outputs visible in coverage reports.

The current setup uses all-MiniLM embeddings and `llama3.1:8b` as the judge.
Language-specific thresholds are read from the generated calibration registry.

Judge-enabled parsed-response results are written to
`outputs/evaluation/parsed_responses/{lang}/{variation}/with_judge/`. The shared
file `judge_cache.yaml`
stores past judge decisions so repeated evaluations do not call Ollama again for
the same ambiguous pair.

Evaluation outputs are grouped by prediction source and scoring method:

```text
outputs/evaluation/
  parsed_responses/
    {lang}/{variation}/
      embedding_only/
      with_judge/
  direct_yaml/
    {lang}/{variation}/
      embedding_only/
      with_judge/
  threshold/
```

The configured `output.root_directory` (or `--output`) is the source-level base
directory. `evaluate.py` automatically partitions results by language and
variation, then appends `embedding_only` when `judge.enabled` is false or
`with_judge` when it is true.

With `output.overwrite_existing: false`, an existing non-empty
language/variation/method directory is left untouched, while missing groups are
still evaluated.

## Notes

- The Overleaf project itself should not be committed to this repository.
- YAML files containing `!!python/tuple` should only be loaded with a trusted PyYAML loader when they are benchmark-generated local files.
- Generated outputs can be large and are usually not meant to be committed.
- If a Call 2 run fails, inspect `{output_root}/error_files.yaml` and the corresponding `.raw.txt` failure sidecar.
- Evaluation aggregate files report both conditional scores over available
  selected responses and coverage-adjusted `end_to_end_*` scores. A response for
  which every Call 2 model failed remains explicitly missing rather than being
  silently excluded.
