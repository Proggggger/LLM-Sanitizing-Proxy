"""LLM Proxy - Main FastAPI application."""
import json
import time
import uuid
from typing import AsyncGenerator, Dict, Optional

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from app.config import Config, load_config, ProviderConfig
from app.providers import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    OllamaProvider,
    NVIDIAProvider,
    ChatRequest,
    Message,
    MessageRole,
    StreamChunk,
)
from app.router import Router
from app.middleware import RateLimiter, ResponseCache

logger = structlog.get_logger()

# Global state
config: Optional[Config] = None
router: Optional[Router] = None
rate_limiter: Optional[RateLimiter] = None
cache: Optional[ResponseCache] = None


def create_provider(name: str, provider_config: ProviderConfig) -> LLMProvider:
    """Create a provider instance from configuration."""
    # Check if this is NVIDIA based on base_url
    is_nvidia = "nvidia" in provider_config.base_url.lower()
    
    provider_class = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "ollama": OllamaProvider,
        "nvidia": NVIDIAProvider if is_nvidia else None,
    }.get(name)

    # Fallback to NVIDIA if base_url contains nvidia
    if not provider_class and is_nvidia:
        provider_class = NVIDIAProvider
    elif not provider_class:
        logger.warning(f"Unknown provider: {name}")
        return None

    return provider_class(
        name=name,
        api_key=provider_config.api_key,
        base_url=provider_config.base_url,
        models=provider_config.models,
        timeout=provider_config.timeout,
        retry_count=provider_config.retry_count,
    )


def create_app(cfg: Config) -> FastAPI:
    """Create and configure the FastAPI application."""
    global config, router, rate_limiter, cache
    config = cfg

    app = FastAPI(
        title="LLM Proxy",
        description="Unified LLM proxy with support for multiple providers",
        version="1.0.0",
    )

    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize providers
    providers: Dict[str, LLMProvider] = {}
    for name, provider_config in config.providers.items():
        if provider_config.enabled:
            provider = create_provider(name, provider_config)
            if provider:
                providers[name] = provider
                logger.info(f"Initialized provider: {name}")

    # Initialize router
    if config.routing:
        router_obj = Router(
            providers=providers,
            routing_rules=config.routing,
            load_balancing_config=config.load_balancing,
        )
    else:
        # Default routing
        from app.config import RoutingRule
        default_rules = []
        for name in providers.keys():
            default_rules.append(RoutingRule(
                name=f"default-{name}",
                condition={"default": True},
                provider=name,
                priority=1,
            ))
        router_obj = Router(
            providers=providers,
            routing_rules=default_rules,
            load_balancing_config=config.load_balancing,
        )

    router = router_obj

    # Initialize middleware
    rate_limiter = RateLimiter(config.rate_limiting)
    cache = ResponseCache(config.caching)

    # Register routes
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """OpenAI-compatible chat completions endpoint."""
        return await handle_chat_completions(request)

    @app.post("/chat/completions")
    async def chat_completions_compat(request: Request):
        """Alternative chat completions endpoint."""
        return await handle_chat_completions(request)

    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        return JSONResponse({
            "object": "list",
            "data": [
                {"id": model, "object": "model"}
                for model in router.get_available_models()
            ]
        })

    @app.get("/models")
    async def list_models_compat():
        """List available models (compatibility)."""
        return await list_models()

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": time.time()}

    @app.get("/health/ready")
    async def readiness_check():
        """Readiness check endpoint."""
        providers_status = {}
        for name in router.providers.keys():
            stats = router.stats.get(name, {})
            providers_status[name] = {
                "healthy": getattr(stats, 'is_healthy', True),
                "total_requests": getattr(stats, 'total_requests', 0),
            }

        return {
            "status": "ready",
            "providers": providers_status,
            "timestamp": time.time()
        }

    return app


async def handle_chat_completions(request: Request):
    """Handle chat completion requests."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Extract parameters
    model = body.get("model", "gpt-3.5-turbo")
    messages_data = body.get("messages", [])
    stream = body.get("stream", False)
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens")
    top_p = body.get("top_p", 1.0)
    stop = body.get("stop")

    # Convert messages
    messages = []
    for msg in messages_data:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        messages.append(Message(role=MessageRole(role), content=content))

    # Rate limiting
    user_id = request.headers.get("X-User-ID", "default")
    allowed, wait_time = rate_limiter.acquire(user_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {wait_time:.2f}s",
            headers={"Retry-After": str(wait_time)},
        )

    # Check cache for non-streaming
    if not stream:
        cached = cache.get(model, messages, temperature=temperature)
        if cached:
            logger.info("Cache hit", model=model)
            return cached

    # Route request
    provider = router.get_provider_for_model(model, user_id)
    if not provider:
        raise HTTPException(
            status_code=400,
            detail=f"No provider available for model: {model}"
        )

    # Create chat request
    chat_request = ChatRequest(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=stream,
        stop=stop,
    )

    try:
        if stream:
            return StreamingResponse(
                stream_generator(provider, chat_request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            response = await provider.chat(chat_request)

            # Cache the response
            if not stream:
                cache.set(
                    model,
                    messages,
                    {
                        "id": response.id,
                        "model": response.model,
                        "choices": [{
                            "message": {
                                "role": response.role,
                                "content": response.content,
                            },
                            "finish_reason": response.finish_reason,
                        }],
                        "usage": response.usage,
                    },
                    temperature=temperature,
                )

            return {
                "id": response.id,
                "model": response.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": response.role,
                            "content": response.content,
                        },
                        "finish_reason": response.finish_reason,
                    }
                ],
                "usage": response.usage,
                "created": response.created_at,
            }

    except Exception as e:
        logger.error(f"Provider error: {e}", model=model, provider=provider.name)
        router.record_request(provider.name, success=False)
        raise HTTPException(status_code=500, detail=str(e))


async def stream_generator(
    provider: LLMProvider,
    request: ChatRequest,
) -> AsyncGenerator[str, None]:
    """Generate streaming responses."""
    try:
        async for chunk in provider.chat_stream(request):
            data = {
                "id": str(uuid.uuid4()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk.content},
                        "finish_reason": chunk.finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"

            if chunk.finish_reason:
                yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
        yield "data: [DONE]\n\n"


def run_server(cfg_path: str = "config.yaml"):
    """Run the LLM proxy server."""
    import uvicorn

    global config
    config = load_config(cfg_path)

    app = create_app(config)

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
    )


if __name__ == "__main__":
    run_server()
