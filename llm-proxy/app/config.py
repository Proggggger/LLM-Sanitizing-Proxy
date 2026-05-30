"""Configuration management for LLM Proxy."""
import os
import re
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""
    enabled: bool = True
    api_key: Optional[str] = None
    base_url: str
    models: List[str] = Field(default_factory=list)
    model_prefix: Optional[str] = None  # Add this
    fetch_models: bool = False  #Option for fetching model list from  provider
    timeout: int = 60
    retry_count: int = 3


class RoutingRule(BaseModel):
    """Routing rule configuration."""
    name: str
    condition: Dict[str, Any]
    provider: str
    priority: int = 1


class LoadBalancingConfig(BaseModel):
    """Load balancing configuration."""
    strategy: str = "round-robin"  # round-robin, random, least-requests
    health_check_interval: int = 30


class RateLimitingConfig(BaseModel):
    """Rate limiting configuration."""
    enabled: bool = False
    requests_per_minute: int = 60
    burst: int = 10


class CachingConfig(BaseModel):
    """Caching configuration."""
    enabled: bool = False
    ttl: int = 3600
    max_size: int = 1000


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"
    include_body: bool = True
    include_headers: bool = False


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


class FilterConfig(BaseModel):
    """Configuration for Request Filtering."""
    enabled: bool = False
    type: str = "regexp" # regexp, presidio, llm
    regexphybrid: bool = False
    regexp_rules: List[Dict[str, str]] = Field(default_factory=list)
    llm_url: str = ""
    llm_prompt: str = ""


class Config(BaseModel):
    """Main configuration class."""
    server: ServerConfig = Field(default_factory=ServerConfig)
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)
    routing: List[RoutingRule] = Field(default_factory=list)
    load_balancing: LoadBalancingConfig = Field(default_factory=LoadBalancingConfig)
    rate_limiting: RateLimitingConfig = Field(default_factory=RateLimitingConfig)
    caching: CachingConfig = Field(default_factory=CachingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file with environment variable substitution."""
        with open(path, "r") as f:
            content = f.read()

        # Substitute environment variables
        def replace_env(match):
            env_var = match.group(1)
            return os.environ.get(env_var, "")

        content = re.sub(r"\$\{([^}]+)\}", replace_env, content)

        data = yaml.safe_load(content)
        return cls(**data)


def load_config(path: str = "config.yaml") -> Config:
    """Load configuration from file."""
    return Config.from_yaml(path)
