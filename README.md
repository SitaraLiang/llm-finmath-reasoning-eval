# LLM FinMath Reasoning Eval

A lightweight framework for evaluating how language models solve quantitative finance and financial mathematics exercises. The project converts human-annotated LaTeX solutions into structured ground-truth YAML, asks models to solve the same exercises, and converts model answers into the same proof-atom representation for later evaluation.

## Pipeline

1. **Import LaTeX exercises**
   - Overleaf is treated as the single source of truth.
   - Downloaded `.tex` files are copied into `data/raw_tex/{lang}/`.

2. **Parse annotated solutions**
   - `src/parser.py` reads tagged LaTeX files from `data/raw_tex/{lang}/`.
   - It writes ground-truth YAML files to `data/ground_truth/{lang}/`.
   - YAML uses Python tuples (`!!python/tuple`) to represent unordered mathematical sets and lists to represent ordered proof steps.

3. **Call 1: generate model answers**
   - `src/call1.py` prompts Ollama models to solve each exercise.
   - Outputs are stored under `outputs/call1/plain_text/{model}/{lang}/{variation}/`.
   - Each prompt strategy gets its own file, e.g. `pc2_q1_seq.txt`, `pc2_q1_acc.txt`, `pc2_q1_gtf.txt`, `pc2_q1_self.txt`.

4. **Call 2: convert model answers**
   - `src/call2.py` converts Call 1 text answers into the structured YAML proof-atom format.
   - Outputs are stored under `outputs/call2/{call1_model}/{call2_model}/{lang}/{variation}/`.
   - Failed conversions are summarized in `outputs/call2/error_files.yaml`.

5. **Evaluation**

To be continued.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For local model inference, install and run Ollama, then pull the models you want to test:

```bash
ollama pull tinyllama
ollama pull llama3.2:1b
ollama pull deepseek-r1:1.5b
ollama pull qwen2.5-coder:0.5b
ollama pull qwen3:0.6b
ollama pull deepseek-r1:7b
ollama pull mistral:7b
ollama pull llama3.1:8b
```

## Usage

Import downloaded Overleaf files:

```bash
python src/import.py --source ../overleaf/exercices_en_changed --destination data/raw_tex
```

Parse annotated LaTeX into ground-truth YAML:

```bash
python src/parser.py --input data/raw_tex/en --output data/ground_truth/en
```

Run Call 1:

```bash
python src/call1.py --config config/call1/experiment_v0.yaml
```

Run Call 2:

```bash
python src/call2.py --config config/call2/experiment_v0.yaml
```

## Main Directories

- `data/raw_tex/{lang}/`: imported annotated LaTeX exercises.
- `data/ground_truth/{lang}/`: parsed ground-truth YAML.
- `outputs/call1/`: model-generated exercise answers.
- `outputs/call2/`: converted proof-atom YAML for model answers.
- `config/call1/`: Call 1 experiment configurations.
- `config/call2/`: Call 2 conversion configurations.
- `tests/`: parser tests.

## Notes

- The Overleaf project itself should not be committed to this repository.
- YAML files containing `!!python/tuple` should only be loaded with a trusted PyYAML loader when they are benchmark-generated local files.
- Generated outputs can be large and are usually not meant to be committed.
