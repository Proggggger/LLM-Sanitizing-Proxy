"""Router module for LLM provider selection and load balancing."""
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import structlog

from app.providers import LLMProvider
from app.config import RoutingRule, LoadBalancingConfig

logger = structlog.get_logger()


@dataclass
class ProviderStats:
    """Statistics for a provider."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_request_time: float = 0
    is_healthy: bool = True


class LoadBalancer:
    """Load balancer for distributing requests across providers."""

    def __init__(self, config: LoadBalancingConfig):
        self.strategy = config.strategy
        self.health_check_interval = config.health_check_interval
        self._counters: Dict[str, int] = defaultdict(int)
        self._current_index: Dict[str, int] = defaultdict(int)

    def select_provider(
        self,
        providers: List[LLMProvider],
        stats: Dict[str, ProviderStats],
        model: str,
    ) -> LLMProvider:
        """Select a provider based on the configured strategy."""
        if not providers:
            raise ValueError("No providers available")

        # Filter healthy providers
        healthy_providers = [
            p for p in providers
            if stats.get(p.name, ProviderStats()).is_healthy
        ]

        # If all unhealthy, use all providers
        if not healthy_providers:
            healthy_providers = providers

        if self.strategy == "random":
            import random
            return random.choice(healthy_providers)

        elif self.strategy == "least-requests":
            min_requests = float('inf')
            selected = healthy_providers[0]
            for provider in healthy_providers:
                p_stats = stats.get(provider.name, ProviderStats())
                # Calculate active requests
                active = p_stats.total_requests - p_stats.failed_requests
                if active < min_requests:
                    min_requests = active
                    selected = provider
            return selected

        else:  # round-robin (default)
            key = f"round_robin_{model}"
            index = self._current_index[key] % len(healthy_providers)
            self._current_index[key] += 1
            return healthy_providers[index]


class Router:
    """Routes requests to appropriate LLM providers."""

    def __init__(
        self,
        providers: Dict[str, LLMProvider],
        routing_rules: List[RoutingRule],
        load_balancing_config: LoadBalancingConfig,
    ):
        self.providers = providers
        self.rules = sorted(routing_rules, key=lambda x: x.priority, reverse=True)
        self.load_balancer = LoadBalancer(load_balancing_config)
        self.stats: Dict[str, ProviderStats] = {}

    def get_provider_for_model(
        self,
        model: str,
        user_id: Optional[str] = None,
    ) -> Optional[LLMProvider]:
        """Find the appropriate provider for a given model."""
        matching_providers = []

        # Find matching rules
        for rule in self.rules:
            condition = rule.condition

            # Check model pattern
            if "model_pattern" in condition:
                pattern = condition["model_pattern"]
                if re.match(pattern, model):
                    provider_name = rule.provider
                    if provider_name in self.providers:
                        matching_providers.append(self.providers[provider_name])

            # Check default rule
            elif condition.get("default"):
                provider_name = rule.provider
                if provider_name in self.providers:
                    matching_providers.append(self.providers[provider_name])

        if not matching_providers:
            # Fallback: try to find provider by model name
            for provider_name, provider in self.providers.items():
                if provider.is_model_supported(model):
                    matching_providers.append(provider)

        if not matching_providers:
            return None

        # Use load balancer to select from matching providers
        return self.load_balancer.select_provider(
            matching_providers,
            self.stats,
            model,
        )

    def get_available_models(self) -> List[str]:
        """Get list of all available models."""
        models = []
        for provider in self.providers.values():
            models.extend(provider.models)
        return sorted(set(models))

    def record_request(self, provider_name: str, success: bool):
        """Record request statistics."""
        if provider_name not in self.stats:
            self.stats[provider_name] = ProviderStats()

        stats = self.stats[provider_name]
        stats.total_requests += 1
        if success:
            stats.successful_requests += 1
        else:
            stats.failed_requests += 1

        import time
        stats.last_request_time = time.time()
