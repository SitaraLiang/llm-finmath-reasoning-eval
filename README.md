# LLM FinMath Reasoning Eval

A structured evaluation framework for analyzing and diagnosing LLM reasoning in quantitative finance and financial mathematics. The framework decomposes human-annotated LaTeX solutions and model-generated reasoning into a common proof-atom representation, enabling step-level evaluation of logical alignment, reasoning chains, and sequential error propagation. It supports multiple reasoning protocols for studying how intermediate context affects downstream performance and combines calibrated embedding-based alignment with selective LLM-as-a-judge adjudication for ambiguous cases.

This repository was developed as part of my research internship at CMAP, École Polytechnique, under the supervision of Charles-Albert Lehalle.


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
   - `src/evaluate.py` builds D1/D3/D4 alignment tables, an experimental D2 order report, and aggregate summaries.
   - Evaluation can run in embedding-only mode or with an optional second-stage LLM judge for ambiguous embedding matches.
   - Downstream stages use composable `base.yaml` and `experiments/` configurations, matching the Call 1 structure.

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

Import downloaded Overleaf files:

```bash
python src/import.py --source ../overleaf/exercices_en_changed --destination data/raw_tex
```

Parse annotated LaTeX into ground-truth YAML:

```bash
python src/parser.py --input data/raw_tex --output data/ground_truth
```

Run Call 1:

```bash
python src/call1.py \
  --config config/call1/experiments/baseline_plain_text.yaml
```

Run the baseline direct-YAML pipeline:

```bash
python src/call1.py \
  --config config/call1/experiments/baseline_direct_yaml.yaml
```

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

Run the English baseline plain-text pipeline after Call 1:

```bash
python src/call2.py --config config/call2/experiments/baseline_en.yaml
python src/select_responses.py --config config/selection/experiments/baseline_en.yaml
python src/evaluate.py --config config/evaluation/experiments/baseline_plain_text_en.yaml
```

Run the English quantitative-expert plain-text pipeline:

```bash
python src/call2.py --config config/call2/experiments/quant_expert_en.yaml
python src/select_responses.py --config config/selection/experiments/quant_expert_en.yaml
python src/evaluate.py --config config/evaluation/experiments/quant_expert_plain_text_en.yaml
```

Run the Chinese baseline plain-text pipeline:

```bash
python src/call1.py --config config/call1/experiments/baseline_plain_text_ch.yaml
python src/call2.py --config config/call2/experiments/baseline_ch.yaml
python src/select_responses.py --config config/selection/experiments/baseline_ch.yaml
python src/evaluate.py --config config/evaluation/experiments/baseline_plain_text_ch.yaml
```

The Chinese evaluation currently uses the calibrated all-MiniLM threshold
`0.58` and the LLM-judge interval `0.40` to `1.00`.

Evaluate the English baseline direct-YAML experiment, which bypasses Call 2
and selection:

```bash
python src/evaluate.py --config config/evaluation/experiments/baseline_direct_yaml_en.yaml
```

The selector first excludes missing or structurally invalid candidates. It then
ranks valid candidates using source coverage, formula fidelity, a hallucination
proxy, atom completeness, and duplicate detection. Close rankings are marked
with `needs_review: true` in the selection report.

Extract statements for embedding calibration:

```bash
python src/extract_statements.py
```

This reads every `data/ground_truth/{lang}/` directory and writes one statement
inventory per language. To extract only French, use:

```bash
python src/extract_statements.py --language fr
```

Seed 12 formulation-pair concepts for every available language:

```bash
python src/seed_pairs.py \
  --input-root data/evaluation \
  --output-root data/evaluation \
  --concept-count 12
```

To seed only English:

```bash
python src/seed_pairs.py --language en --concept-count 12
```

Existing `formulation_pairs.yaml` files are preserved by default. Use
`--overwrite` to update a seed while retaining completed variants, or combine
`--overwrite --reselect` to discard the previous statement selection and draw
a new balanced sample:

```bash
python src/seed_pairs.py --language en --concept-count 12 --overwrite --reselect
```

After seeding, curate `formulation_pairs.yaml` separately for each language. Pair
variants must be written in the same language as `text_a`; an English threshold
must not be reused as a French calibration result without testing it.

Calibrate embedding similarity thresholds:

```bash
python src/eval_embeddings.py pairs \
  --input data/evaluation \
  --output-dir outputs/evaluation/threshold 
```

This writes one folder per embedding model, for example:

```text
outputs/evaluation/threshold/en/all-minilm-l6-v2/
outputs/evaluation/threshold/fr/all-minilm-l6-v2/
```

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

The current judge setup uses all-MiniLM embeddings with an LLM judge for ambiguous cases:

```text
embedding thresholds: en=0.375, fr=0.45
judge bands: en=0.30 to 1.0, fr=0.35 to 1.0
judge model: llama3.1:8b
```

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

## Main Directories

- `data/raw_tex/{lang}/`: imported annotated LaTeX exercises.
- `data/ground_truth/{lang}/`: parsed ground-truth YAML.
- `data/evaluation/{lang}/`: language-specific statement inventories and formulation-pair calibration datasets.
- `outputs/call1/`: model-generated exercise answers.
- `outputs/call2/`: zero-shot per-question Call 2 conversions.
- `outputs/selected_responses/`: curated Call 2 responses selected for evaluation.
- `outputs/evaluation/`: evaluation results grouped by prediction source and scoring method, plus embedding-threshold calibration outputs.
- `config/call1/`: Call 1 experiment configurations.
- `config/call2/`: shared Call 2 settings and thin experiment manifests.
- `config/selection/`: shared candidate-selection settings and experiment manifests.
- `config/evaluation/`: shared scoring settings, prediction modes, and experiment manifests.

## Notes

- The Overleaf project itself should not be committed to this repository.
- YAML files containing `!!python/tuple` should only be loaded with a trusted PyYAML loader when they are benchmark-generated local files.
- Generated outputs can be large and are usually not meant to be committed.
- If a Call 2 run fails, inspect `{output_root}/error_files.yaml` and the corresponding `.raw.txt` failure sidecar.
- Evaluation aggregate files report both conditional scores over available
  selected responses and coverage-adjusted `end_to_end_*` scores. A response for
  which every Call 2 model failed remains explicitly missing rather than being
  silently excluded.
