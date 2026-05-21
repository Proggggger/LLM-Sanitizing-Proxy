"""Tests for LLM Proxy."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.providers import (
    ChatRequest,
    Message,
    MessageRole,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    DummyProvider,
)
from app.router import Router, ProviderStats
from app.middleware import RateLimiter, ResponseCache
from app.config import (
    Config,
    ProviderConfig,
    RoutingRule,
    LoadBalancingConfig,
    RateLimitingConfig,
    CachingConfig,
)


class TestProviders:
    """Test provider implementations."""

    @pytest.mark.asyncio
    async def test_openai_request_conversion(self):
        """Test OpenAI request conversion."""
        provider = OpenAIProvider(
            name="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            models=["gpt-3.5-turbo"],
        )

        request = ChatRequest(
            model="gpt-3.5-turbo",
            messages=[
                Message(role=MessageRole.SYSTEM, content="You are helpful"),
                Message(role=MessageRole.USER, content="Hello"),
            ],
            temperature=0.7,
            max_tokens=100,
        )

        converted = provider._convert_request(request)

        assert "model" in converted
        assert "messages" in converted
        assert converted["temperature"] == 0.7
        assert converted["max_tokens"] == 100

    def test_anthropic_headers(self):
        """Test Anthropic headers."""
        provider = AnthropicProvider(
            name="anthropic",
            api_key="test-key",
            base_url="https://api.anthropic.com",
            models=["claude-3-opus-20240229"],
        )

        headers = provider._build_headers()

        assert "X-API-Key" in headers
        assert "anthropic-version" in headers

    def test_model_support(self):
        """Test model support checking."""
        provider = OpenAIProvider(
            name="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            models=["gpt-3.5-turbo", "gpt-4"],
        )

        assert provider.is_model_supported("gpt-3.5-turbo")
        assert provider.is_model_supported("gpt-4")
        assert not provider.is_model_supported("claude-3")

    @pytest.mark.asyncio
    async def test_dummy_provider_chat(self):
        """Test DummyProvider chat and chat_stream."""
        provider = DummyProvider(
            name="dummy",
            models=["dummy-model"],
        )

        request = ChatRequest(
            model="dummy-model",
            messages=[
                Message(role=MessageRole.USER, content="Hello World Test"),
            ],
        )

        # Test non-streaming chat
        response = await provider.chat(request)
        assert response.content == "Hello World Test"
        assert response.model == "dummy-model"

        # Test streaming chat
        chunks = []
        async for chunk in provider.chat_stream(request):
            chunks.append(chunk)

        # Check content chunks reconstructed
        combined = "".join(c.content for c in chunks if c.content)
        assert combined == "Hello World Test"
        assert chunks[-1].finish_reason == "stop"


class TestRouter:
    """Test routing functionality."""

    def test_router_selects_provider(self):
        """Test router selects correct provider."""
        providers = {
            "openai": MagicMock(name="openai"),
            "anthropic": MagicMock(name="anthropic"),
        }

        rules = [
            RoutingRule(
                name="openai-gpt",
                condition={"model_pattern": "gpt-.*"},
                provider="openai",
                priority=10,
            ),
            RoutingRule(
                name="anthropic-claude",
                condition={"model_pattern": "claude-.*"},
                provider="anthropic",
                priority=10,
            ),
        ]

        router = Router(
            providers=providers,
            routing_rules=rules,
            load_balancing_config=LoadBalancingConfig(),
        )

        provider = router.get_provider_for_model("gpt-3.5-turbo")
        assert provider == providers["openai"]

        provider = router.get_provider_for_model("claude-3-opus")
        assert provider == providers["anthropic"]

    def test_router_get_available_models(self):
        """Test getting available models."""
        openai_mock = MagicMock()
        openai_mock.models = ["gpt-3.5-turbo", "gpt-4"]

        anthropic_mock = MagicMock()
        anthropic_mock.models = ["claude-3-opus", "claude-3-sonnet"]

        router = Router(
            providers={"openai": openai_mock, "anthropic": anthropic_mock},
            routing_rules=[],
            load_balancing_config=LoadBalancingConfig(),
        )

        models = router.get_available_models()
        assert len(models) == 4


class TestRateLimiter:
    """Test rate limiting."""

    def test_rate_limiter_allows_requests(self):
        """Test rate limiter allows requests within limit."""
        config = RateLimitingConfig(
            enabled=True,
            requests_per_minute=60,
            burst=10,
        )
        limiter = RateLimiter(config)

        allowed, _ = limiter.acquire("test-user")
        assert allowed

    def test_rate_limiter_blocks_excess(self):
        """Test rate limiter blocks excess requests."""
        config = RateLimitingConfig(
            enabled=True,
            requests_per_minute=60,
            burst=2,
        )
        limiter = RateLimiter(config)

        # Use up burst
        limiter.acquire("test-user")
        limiter.acquire("test-user")

        # This should be blocked
        allowed, wait_time = limiter.acquire("test-user")
        assert not allowed
        assert wait_time > 0


class TestCache:
    """Test caching."""

    def test_cache_stores_and_retrieves(self):
        """Test cache stores and retrieves values."""
        config = CachingConfig(enabled=True, ttl=3600, max_size=100)
        cache = ResponseCache(config)

        cache.set("gpt-3.5", [Message(MessageRole.USER, "test")], {"result": "value"})

        result = cache.get("gpt-3.5", [Message(MessageRole.USER, "test")])
        assert result == {"result": "value"}

    def test_cache_miss(self):
        """Test cache miss returns None."""
        config = CachingConfig(enabled=True, ttl=3600, max_size=100)
        cache = ResponseCache(config)

        result = cache.get("gpt-3.5", [Message(MessageRole.USER, "test")])
        assert result is None

    def test_cache_clear(self):
        """Test cache clear."""
        config = CachingConfig(enabled=True, ttl=3600, max_size=100)
        cache = ResponseCache(config)

        cache.set("gpt-3.5", [Message(MessageRole.USER, "test")], {"result": "value"})
        cache.clear()

        result = cache.get("gpt-3.5", [Message(MessageRole.USER, "test")])
        assert result is None


class TestLoadBalancing:
    """Test load balancing strategies."""

    def test_round_robin(self):
        """Test round-robin load balancing."""
        config = LoadBalancingConfig(strategy="round-robin")
        providers = [MagicMock(name=f"provider-{i}") for i in range(3)]
        stats = {}

        lb = Router(
            providers={p.name: p for p in providers},
            routing_rules=[],
            load_balancing_config=config,
        )

        # Should cycle through providers
        selected = []
        for _ in range(3):
            provider = lb.load_balancer.select_provider(providers, stats, "test-model")
            selected.append(provider)

        # All providers should be selected once
        assert len(set(selected)) == 3

    def test_random(self):
        """Test random load balancing."""
        config = LoadBalancingConfig(strategy="random")
        providers = [MagicMock(name=f"provider-{i}") for i in range(3)]
        stats = {}

        lb = Router(
            providers={p.name: p for p in providers},
            routing_rules=[],
            load_balancing_config=config,
        )

        # Should select a provider
        provider = lb.load_balancer.select_provider(providers, stats, "test-model")
        assert provider in providers


@pytest.mark.asyncio
async def test_chat_request_creation():
    """Test chat request creation."""
    request = ChatRequest(
        model="test-model",
        messages=[
            Message(role=MessageRole.USER, content="Hello"),
        ],
        temperature=0.5,
    )

    assert request.model == "test-model"
    assert len(request.messages) == 1
    assert request.temperature == 0.5
