# LLM FinMath Reasoning Eval

This repository was developed as part of my research internship at CMAP, École Polytechnique, under the supervision of Charles-Albert Lehalle.

The project introduces a structured evaluation framework for analyzing and diagnosing LLM reasoning in quantitative finance and financial mathematics. The framework decomposes human-annotated LaTeX solutions and model-generated reasoning into a common proof-atom representation, enabling step-level evaluation of logical alignment, reasoning chains, and sequential error propagation. It supports multiple reasoning protocols for studying how intermediate context affects downstream performance and combines calibrated embedding-based alignment with selective LLM-as-a-judge adjudication for ambiguous cases.


## Table of Contents

- [Data Format and Metrics](#data-format-and-metrics)
  - [LaTeX annotation reference](#latex-annotation-reference)
  - [Evaluation dimensions](#evaluation-dimensions)
- [Setup](#setup)
  - [Native Ollama](#native-ollama-recommended-for-macos)
  - [Docker Ollama](#docker-ollama-linux-with-nvidia-gpus)
  - [Models](#models)
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


## Data Format and Metrics

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

Create the Python environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The scripts call the Ollama API at
`http://localhost:11434/api/generate` by default. Choose either the native or
Docker setup below; do not start both servers on port `11434` at the same time.

### Native Ollama (recommended for macOS)

Install [Ollama](https://ollama.com/download), start the application or server,
and verify that its API is available:

```bash
ollama serve
curl http://localhost:11434/api/tags
```

On macOS, the Ollama application may already run the server, in which case
`ollama serve` is unnecessary. Pull only the models required by the experiment
configuration you plan to run, for example:

```bash
ollama pull llama3.1:8b
```

### Docker Ollama (Linux with NVIDIA GPUs)

This setup requires Linux, an NVIDIA GPU, NVIDIA Container Toolkit, and a
running Docker service. First choose where Ollama will store its downloaded
models, then start the container:

```bash
export OLLAMA_DATA_PATH="/path/to/ollama_data"
mkdir -p "$OLLAMA_DATA_PATH"
docker compose up -d
```

Optionally verify the container and API:

```bash
docker compose ps
curl http://localhost:11434/api/tags
```

Pull and list models inside the container:

```bash
docker exec ollama ollama pull llama3.1:8b
docker exec ollama ollama list
```

Stop Ollama when it is no longer needed:

```bash
docker compose down
```

The Compose file uses GPU 0 by default; change `CUDA_VISIBLE_DEVICES` if needed.

### Models

The complete set of models referenced by the provided experiments is listed
below. With native Ollama, use these commands directly. With Docker, prefix each
pull with `docker exec ollama ollama` as shown above.

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

For Call 2, `llama3.1:8b` is currently the strongest baseline observed for
strict YAML formatting. Smaller models are useful as baselines, but often fail
on schema or YAML syntax.

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

### Step 2: Refresh the calibration data (when necessary)

The existing calibrated threshold may be reused when new exercises remain in the
same language and mathematical domain, even if they introduce a few concepts not
currently represented in `formulation_pairs.yaml`. Update and version the calibration only after a substantial domain expansion. 

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
Embedding cosine similarities are also clipped to `[0, 1]`: negative cosine
values are treated as zero because they indicate no useful semantic match for
the benchmark's coverage-oriented metrics.
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

Call 1 tests four strategies; each produces its own file (for example,
`pc2_q1_seq.txt`):

| Strategy | Context provided for the current subquestion |
|---|---|
| Strictly sequential (`seq`) | Only the current subquestion and its local assumptions. |
| Prompt accumulation (`acc`) | All questions so far, without previous answers. |
| Ground-truth forcing (`gtf`) | Previous questions and their ground-truth answers. |
| Self history (`self`) | Previous questions and the model's own answers. |

Outputs are stored under `outputs/call1/{plain_text,yaml}/{model}/{lang}/{variation}/`;
generation failures are recorded in each mode's `error_files.yaml`.

Without further configuration, the plain text experiment file processes all three supported
languages. To customize the languages, models, modes, or other settings, see
[Detailed Configuration Reference](#detailed-configuration-reference).

### Step 4: Convert and select plain-text answers

For example, process the English baseline outputs with Call 2 and select the
best valid conversion for each Call 1 response:

```bash
python src/call2.py --config config/call2/experiments/baseline_en.yaml
python src/select_responses.py --config config/selection/experiments/baseline_en.yaml
```

Call 2 converts one question block at a time. The selector excludes missing or
invalid conversions and ranks the remaining candidates without consulting the
ground truth. Call 2 validation and repair live in `src/conversion_validator.py`;
failed conversions are listed in `outputs/call2/error_files.yaml`. Selection
results and close decisions are recorded in
`outputs/selected_responses/selection_report.yaml`.

### Step 5: Evaluate

Evaluate selected plain-text responses:

```bash
python src/evaluate.py \
  --config config/evaluation/experiments/baseline_plain_text_en.yaml \
  --no-judge
```

Evaluate direct-YAML responses:

```bash
python src/evaluate.py \
  --config config/evaluation/experiments/baseline_direct_yaml_en.yaml
```

Evaluation produces per-case matrices and aggregate reports by model and
strategy.

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
