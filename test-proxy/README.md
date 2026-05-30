# Test‑Proxy

This directory contains the test harness for the **LLM‑Proxy** service. It provides:

- **Datasets** (`*.jsonl`) – sample input texts in English, Ukrainian and Russian.
- **Test runner** (`test_runner.py`) – orchestrates the execution of the proxy against the datasets.
- **Test suites** (`test_*.py`) – PyTest based tests that verify filtering, alias handling and result correctness.
- **Result snapshots** (`results/`) – JSON files with expected outputs for regression checks.
- **Demo scripts** (`demo_test.json`, `demo_dataset.jsonl`) – quick examples for manual run.

## Quick Start

### Dataset file structure
Each dataset file is a **JSON Lines** (`.jsonl`) file where each line is a JSON object with the following keys:
- `tokens`: list of token strings (the original text split into tokens).
- `labels`: list of NER tags aligned with `tokens`. Use `"O"` for non‑entity tokens or the entity label (e.g. `"PERSON"`, `"EMAIL"`).
- `source` (optional): identifier of the data source.

Example line:
```json
{"tokens": ["John", "Doe", "works", "at", "Acme"], "labels": ["PERSON", "PERSON", "O", "O", "ORG"], "source": "example"}
```

### Running the filtering test script
```bash
# Ensure the proxy service is running (default http://localhost:8080 or as configured)
python test_filtering_test.py path/to/dataset.jsonl -o my_results
```
- `dataset` defaults to `demo_dataset.jsonl` if omitted.
- Use `-o`/`--output` to set the JSON results filename (without extension).
```bash
# Install dependencies (if not already installed)
pip install -r ../llm-proxy/requirements.txt

# Run all tests
pytest -q

# Run a single test suite, e.g. filtering tests
pytest test_filtering_test.py
```

## Running the Proxy Manually
The proxy itself lives in `../llm-proxy`. To start it locally:
```bash
cd ../llm-proxy
python -m uvicorn app.main:app --reload
```
Then execute the runner against the running service:
```bash
python test_runner.py --host http://localhost:8000
```


## Repository Layout
- `test_proxy/` – this directory (tests, data, results).
- `llm-proxy/` – the actual proxy service (FastAPI app, config, providers).

For more details on the service itself, see the README in the `llm-proxy` folder.
