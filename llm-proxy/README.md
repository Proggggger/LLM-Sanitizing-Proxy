# LLM‑Proxy

This directory contains the **LLM‑Proxy** service – a FastAPI‑based gateway that forwards OpenAI‑compatible chat‑completion requests to multiple underlying LLM providers and optionally applies request‑filtering (regex, Presidio, or LLM‑based PII masking).

Unified LLM proxy server with support for multiple providers, streaming, rate limiting, and caching.

## Features

- **Multi-Provider Support**: OpenAI, Anthropic, Google, Ollama
- **Streaming**: Full support for streaming responses (SSE)
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API
- **Smart Routing**: Route requests based on model patterns
- **Load Balancing**: Round-robin, random, least-requests strategies
- **Rate Limiting**: Token bucket rate limiter
- **Caching**: LRU cache with TTL for responses
- **Health Checks**: Built-in health and readiness endpoints

## Installation

```bash
# Activate virtual environment
cd llm-proxy
pip install -r requirements.txt

# Or install dependencies
pip install fastapi uvicorn httpx pydantic pyyaml structlog
```

## Configuration

1. Copy `config.example.yaml` to `config.yaml`
2. Set your API keys via environment variables or in the config:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```
### Filtering configuration
In **filter** section of config file you can enable request interception for filtering/ THrer are three **type** options available: *regex, presidio, llm* also **hybrid** configuration option present, which additionaly enables regexp filter after main one.

**Pay attention** LLM filter requires external llm server, url for accessing it is also stored in config. By default it's adopted for __*KoboldCPP*__ https://github.com/LostRuins/koboldcpp as download and run solution. Proxy was tested with gemma-4-E4B model(https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF).

## Usage

### Start the server

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or using the CLI:

```bash
python main.py config.yaml
```

### API Endpoints

API documentation is available when proxy is running on http://127.0.0.1:8088/docs

#### Chat Completions (OpenAI-compatible)

```bash
POST /v1/chat/completions
POST /chat/completions
```

Request:
```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "stream": false
}
```

#### Streaming Response

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

#### List Models

```bash
GET /v1/models
GET /models
```

#### Health Check

## Repository Layout

- `app/` – core FastAPI application, provider implementations, routing, interceptor and configuration.
- `config.example.yaml` – example configuration file.
- `requirements.txt` – Python dependencies.
- `tests/` – (optional) pytest suite for the proxy itself.
- `../test-proxy/` – separate test harness with datasets, test runner and expected result snapshots.

---

#### Health Check

```bash
GET /health
GET /health/ready
```

## Example Client

```python
import httpx

# Non-streaming request
response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello!"}],
    }
)
print(response.json())

# Streaming request
with httpx.stream(
    "POST",
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": True
    }
) as response:
    for line in response.iter_lines():
        if line.startswith("data: "):
            print(line[6:])
```

## Testing

```bash
pytest tests/ -v
pytest tests/ -v --cov=app
```

## License

MIT
