# CodeMEM

## About

CodeMEM is a memory management system designed for repository-level iterative code generation tasks. It dynamically updates repository context through code context memory, constructs session memory centred on code modifications via code session memory, and detects and mitigates forgetting phenomena.

## Installation

```bash
cd codemem
conda create --name codemem
conda activate codemem
pip install -r requirements.txt
```

## Running Pipelines

**Note:** Please configure the LLM API parameters in `retriever.py` and `llm_factory.py`.

### CodeIF Bench
```bash
python codemem/run_codeif_bench.py \
  --jsonl_path /path/to/CodeIFBenchL2.jsonl \
  --root_path /path/to/Source_Code \
  --output_path /tmp/codeif_results.jsonl \
  --model deepseek-v3.2 \
  --max_workers 16
```
### Codereval
```bash
python codemem/run_codereval.py \
  --json_path codemem/data/CoderEval4Python.json \
  --root_path /path/to/codereval/repos \
  --output_path /tmp/codereval_results.json \
  --repo_graph repo_graph_v2.json \
  --project_separator --- \
  --model deepseek-v3.2
```
- Use `--jsonl` when the input is JSON Lines.
- `--project_separator` mirrors the Codereval convention that replaces `/` with `---` in folder names.

