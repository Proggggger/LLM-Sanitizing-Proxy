"""Rate limiting and caching middleware."""
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Tuple

import structlog

from app.config import RateLimitingConfig, CachingConfig

logger = structlog.get_logger()


@dataclass
class RateLimitState:
    """State for rate limiting."""
    tokens: float
    last_update: float = field(default_factory=time.time)


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, config: RateLimitingConfig):
        self.enabled = config.enabled
        self.requests_per_minute = config.requests_per_minute
        self.burst = config.burst
        self._buckets: Dict[str, RateLimitState] = {}
        self._lock = defaultdict(bool)

        if self.enabled:
            # Calculate tokens per second
            self.tokens_per_second = self.requests_per_minute / 60.0
            logger.info(
                "Rate limiter enabled",
                requests_per_minute=self.requests_per_minute,
                burst=self.burst,
            )
        else:
            self.tokens_per_second = 0
            logger.info("Rate limiter disabled")

    def _get_bucket(self, key: str) -> RateLimitState:
        """Get or create a rate limit bucket."""
        if key not in self._buckets:
            self._buckets[key] = RateLimitState(
                tokens=self.burst,
                last_update=time.time(),
            )
        return self._buckets[key]

    def acquire(self, key: str = "default") -> Tuple[bool, float]:
        """
        Try to acquire a token.
        Returns (success, wait_time).
        """
        if not self.enabled:
            return True, 0.0

        bucket = self._get_bucket(key)
        now = time.time()

        # Refill tokens based on elapsed time
        elapsed = now - bucket.last_update
        bucket.tokens = min(
            self.burst,
            bucket.tokens + elapsed * self.tokens_per_second
        )
        bucket.last_update = now

        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True, 0.0
        else:
            # Calculate wait time
            wait_time = (1.0 - bucket.tokens) / self.tokens_per_second
            return False, wait_time

    def get_remaining(self, key: str = "default") -> int:
        """Get remaining tokens."""
        if not self.enabled:
            return self.burst
        bucket = self._get_bucket(key)
        return int(bucket.tokens)


@dataclass
class CacheEntry:
    """Cache entry with TTL."""
    value: Any
    expires_at: float
    hits: int = 0


class ResponseCache:
    """LRU cache with TTL for responses."""

    def __init__(self, config: CachingConfig):
        self.enabled = config.enabled
        self.ttl = config.ttl
        self.max_size = config.max_size
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: list = []

        if self.enabled:
            logger.info(
                "Cache enabled",
                ttl=self.ttl,
                max_size=self.max_size,
            )
        else:
            logger.info("Cache disabled")

    def _generate_key(self, model: str, messages: list, **kwargs) -> str:
        """Generate cache key from request."""
        key_data = f"{model}:{str(messages)}:{str(kwargs)}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, model: str, messages: list, **kwargs) -> Optional[Any]:
        """Get cached response."""
        if not self.enabled:
            return None

        key = self._generate_key(model, messages, **kwargs)
        entry = self._cache.get(key)

        if entry:
            if time.time() < entry.expires_at:
                entry.hits += 1
                # Update access order
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                return entry.value
            else:
                # Expired
                del self._cache[key]
                self._access_order.remove(key)

        return None

    def set(self, model: str, messages: list, value: Any, **kwargs):
        """Cache a response."""
        if not self.enabled:
            return

        key = self._generate_key(model, messages, **kwargs)

        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                del self._cache[oldest_key]

        self._cache[key] = CacheEntry(
            value=value,
            expires_at=time.time() + self.ttl,
        )
        self._access_order.append(key)

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(entry.hits for entry in self._cache.values())
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "total_hits": total_hits,
            "enabled": self.enabled,
        }
