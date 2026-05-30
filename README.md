# LLM-Sanitizing-Proxy

## Overview
This repository contains two main components:

1. **`llm-proxy/`** – the FastAPI gateway that routes OpenAI‑compatible chat requests to multiple LLM providers and can apply request‑filtering (regex, Presidio, or LLM‑based PII masking).
2. **`test-proxy/`** – a test harness with sample datasets, PyTest suites, and result snapshots used to verify the proxy’s filtering and routing behaviour.

Each component ships its own detailed README (`llm-proxy/README.md` and `test-proxy/README.md`) describing installation, configuration and usage.

## Repository Layout
```
NLP_Project/
├─ llm-proxy/          # LLM‑Proxy service source code
│   ├─ app/            # FastAPI app, providers, interceptor, config
│   ├─ config.example.yaml   # Example configuration file
│   ├─ requirements.txt      # Python dependencies
│   └─ README.md               # Service‑specific documentation
├─ test-proxy/         # Test harness for the proxy
│   ├─ *.jsonl         # Sample input datasets (English, Ukrainian, Russian)
│   ├─ results/        # Expected JSON snapshots for regression testing
│   ├─ test_*.py       # PyTest suites (filtering, alias handling, etc.)
│   ├─ test_runner.py  # Utility to run datasets against a running proxy
│   └─ README.md        # Instructions for running the tests
└─ README.md           # **You are here** – high‑level guide to the whole project
```

## Getting Started
1. **Install the proxy** – follow the steps in `llm-proxy/README.md` to set up a virtual environment, install dependencies and launch the server.
2. **Run the tests** – once the proxy is running, consult `test-proxy/README.md` for commands to execute the test suites against the live service.

Both READMEs contain full command examples, configuration details and troubleshooting tips.

