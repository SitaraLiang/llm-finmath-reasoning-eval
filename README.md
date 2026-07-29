# LLM FinMath Reasoning Eval

A lightweight framework for evaluating how language models solve quantitative finance and financial mathematics exercises. The project turns human-annotated LaTeX solutions into structured ground-truth YAML, asks local Ollama models to solve the same exercises, then converts model answers into the same proof-atom representation for later evaluation.

## Current Pipeline

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
   - Outputs are stored under `outputs/call1/plain_text/{model}/{lang}/{variation}/`.
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
   - Failed generations are written to `outputs/call1/error_files.yaml`.

4. **Call 2: convert answers to proof-atom YAML**
   - `src/call2.py` converts Call 1 text answers into the structured YAML format.
   - Current recommended mode is `per_question`: each `Question N:` block is converted separately, then Python assembles the final `subquestions` list.
   - Validation and repair helpers live in `src/conversion_validator.py`.
   - Successful conversions write only `.yaml`; raw model text is saved as `.raw.txt` only for failed conversions.
   - Few-shot outputs default to `outputs/call2/`.
   - Zero-shot per-question outputs default to `outputs/call2_zeroshot_per_question/`.
   - Failed conversions are summarized in `{output_root}/error_files.yaml`.

5. **Evaluation**
   - In progress: select or validate the best Call 2 conversion for each Call 1 answer and store parsed responses under `outputs/parsed_responses/`.

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
python src/call1.py --config config/call1/experiment_v1.yaml
```

Run Call 2 few-shot:

```bash
python src/call2.py --config config/call2/experiment_v1.yaml
```

Run Call 2 zero-shot per-question conversion:

```bash
python src/call2.py --config config/call2/experiment_v1_zeroshot.yaml
```

## Main Directories

- `data/raw_tex/{lang}/`: imported annotated LaTeX exercises.
- `data/ground_truth/{lang}/`: parsed ground-truth YAML.
- `outputs/call1/`: model-generated exercise answers.
- `outputs/call2/`: few-shot Call 2 conversions.
- `outputs/call2_zeroshot_per_question/`: zero-shot per-question Call 2 conversions.
- `outputs/parsed_responses/`: planned curated/selected parsed model responses.
- `config/call1/`: Call 1 experiment configurations.
- `config/call2/`: Call 2 conversion configurations.
- `tests/`: parser and conversion validation tests.

## Notes

- The Overleaf project itself should not be committed to this repository.
- YAML files containing `!!python/tuple` should only be loaded with a trusted PyYAML loader when they are benchmark-generated local files.
- Generated outputs can be large and are usually not meant to be committed.
- If a Call 2 run fails, inspect `{output_root}/error_files.yaml` and the corresponding `.raw.txt` failure sidecar.
