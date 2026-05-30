"""LLM Proxy - Main FastAPI application."""
import json
import time
import uuid
import logging
from typing import AsyncGenerator, Dict, Optional

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.responses import RedirectResponse

from contextlib import asynccontextmanager

from app.config import Config, load_config, ProviderConfig, LoggingConfig
from app.providers import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    OllamaProvider,
    NVIDIAProvider,
    DummyProvider,
    ChatRequest,
    Message,
    MessageRole,
    StreamChunk,
)
from app.router import Router
from app.middleware import RateLimiter, ResponseCache
from app.interceptor import intercept_request


logger = structlog.get_logger()

# Global state
config: Optional[Config] = None
router: Optional[Router] = None
rate_limiter: Optional[RateLimiter] = None
cache: Optional[ResponseCache] = None


def create_provider(name: str, provider_config: ProviderConfig) -> LLMProvider:
    """Create a provider instance from configuration."""
    # Check if this is NVIDIA based on base_url
    is_nvidia = "nvidia" in (provider_config.base_url or "").lower()
    
    # Define which providers require special handling due to API differences
    special_providers = {
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "ollama": OllamaProvider,
        "dummy": DummyProvider,
    }
    provider_class = special_providers.get(name, OpenAIProvider)
    
    # Dummy provider does not require config parameters
    if name == "dummy":
        return DummyProvider(
            name="dummy",
            models=["dummy-model"],
        )
    
    if not provider_class:
        logger.warning(f"Unknown provider: {name}")
        return None

    return provider_class(
        name=name,
        api_key=provider_config.api_key,
        base_url=provider_config.base_url,
        models=provider_config.models,
        model_prefix=provider_config.model_prefix,
        timeout=provider_config.timeout,
        retry_count=provider_config.retry_count,
    )


def create_app(cfg: Config) -> FastAPI:
    """Create and configure the FastAPI application."""
    global config, router, rate_limiter, cache
    config = cfg
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: Fetch models if enabled
        for name, provider in providers.items():
            config_entry = config.providers.get(name)
            if config_entry and config_entry.fetch_models:
                try:
                    logger.info(f"Fetching models for {name}...")
                    models = await provider.fetch_available_models()
                    # Apply prefix if configured
                    if provider.model_prefix:
                        models = [f"{provider.model_prefix}{m}" for m in models]
                    provider.models = models # Update the provider's model list
                    logger.info(f"Updated models for {name}: {len(models)} found")
                except Exception as e:
                    logger.error(f"Failed to fetch models for {name}: {e}")
        yield

    app = FastAPI(
        title="LLM Proxy",
        description="Unified LLM proxy with support for multiple providers",
        version="1.0.0",
        lifespan=lifespan
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

    # Ensure DummyProvider is always available
    if "dummy" not in providers:
        providers["dummy"] = DummyProvider(name="dummy", models=["dummy-model"])
        logger.info("Initialized provider: dummy (auto)")

    # Initialize router
    # Ensure dummy routing rule exists
    if "dummy" in providers:
        from app.config import RoutingRule
        dummy_rule_exists = any(r.provider == "dummy" for r in config.routing)
        if not dummy_rule_exists:
            config.routing.append(RoutingRule(
                name="dummy-rule",
                condition={"model_pattern": "dummy.*"},
                provider="dummy",
                priority=20,
            ))
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
    
    @app.get("/", include_in_schema=False)
    async def redirect_to_docs():
        """Redirect user to the root of the documentation"""
        return RedirectResponse(url="/docs")

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
    
    # Intercept and process messages
    intercepted = intercept_request("filter", messages_data, config.filter)
    print(intercepted)
    if isinstance(intercepted, dict) and intercepted.get("status") == "success":
        messages_data = intercepted.get("processed_data", messages_data)
    else:
        logger.warning("Request interception failed or returned error, proceeding with original data")

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

    # except Exception as e:
    #     logger.error(f"Provider error: {e}", model=model, provider=provider.name)
    #     router.record_request(provider.name, success=False)
    #     raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Provider error: {e}", model=model, provider=provider.name)
        router.record_request(provider.name, success=False)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": 500
                }
            }
        )


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
        # logger.error(f"Streaming error: {e}")
        # yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
        # yield "data: [DONE]\n\n"
        logger.error(f"Streaming error: {e}")
        error_payload = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": 500
            }
        }
        yield f"data: {json.dumps(error_payload)}\n\n"
        yield "data: [DONE]\n\n"


def configure_logging(logging_config: LoggingConfig):
    """Configure structlog globally based on config."""
    
    # Map configuration string level to logging level
    level = getattr(logging, logging_config.level.upper(), logging.INFO)

    # Configure processors based on format
    processors = [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if logging_config.format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

def run_server(cfg_path: str = "config.yaml"):
    """Run the LLM proxy server."""
    import uvicorn

    global config
    config = load_config(cfg_path)

    configure_logging(config.logging)

    app = create_app(config)

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
    )


if __name__ == "__main__":
    run_server()
