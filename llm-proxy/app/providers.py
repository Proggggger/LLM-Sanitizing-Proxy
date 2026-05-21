"""LLM Providers - Base classes and implementations for different LLM providers."""
import abc
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional
from enum import Enum

import httpx
import structlog

logger = structlog.get_logger()


class MessageRole(str, Enum):
    """Message role enumeration."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Chat message structure."""
    role: MessageRole
    content: str


@dataclass
class ChatRequest:
    """Chat request structure."""
    model: str
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stream: bool = False
    stop: Optional[List[str]] = None
    extra_body: Optional[Dict[str, Any]] = None


@dataclass
class ChatResponse:
    """Chat response structure."""
    id: str
    model: str
    content: str
    role: str = "assistant"
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    created_at: int = field(default_factory=lambda: int(time.time()))


@dataclass
class StreamChunk:
    """Streaming chunk structure."""
    content: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


class LLMProvider(abc.ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        name: str,
        api_key: Optional[str],
        base_url: str,
        models: List[str],
        timeout: int = 60,
        retry_count: int = 3,
        model_prefix: str = ""
    ):
        
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        #self.models = models
        self.model_prefix = model_prefix
        # Prepend prefix to configured models
        if self.model_prefix:
            self.models = [f"{self.model_prefix}{m}" for m in models]
        else:
            self.models = models
        self.timeout = timeout
        self.retry_count = retry_count
        self._client: Optional[httpx.AsyncClient] = None

    def strip_prefix(self, model_name: str) -> str:
        """Strip the prefix from a model name if present."""
        if self.model_prefix and model_name.startswith(self.model_prefix):
            return model_name[len(self.model_prefix):]
        return model_name
    
    async def fetch_available_models(self) -> List[str]:
        """Fetch models from the provider API."""
        raise NotImplementedError("fetch_available_models not implemented for this provider")
    

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=100),
            )
        return self._client

    @abc.abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Non-streaming chat completion."""
        pass

    @abc.abstractmethod
    async def chat_stream(self, request: ChatRequest) -> AsyncGenerator[StreamChunk, None]:
        """Streaming chat completion."""
        pass

    @abc.abstractmethod
    def _build_headers(self) -> Dict[str, str]:
        """Build headers for API requests."""
        pass

    def is_model_supported(self, model: str) -> bool:
        """Check if model is supported by this provider."""
        # return model in self.models or any(
        #     model.startswith(m.split("-")[0]) for m in self.models if m
        # )
        return model in self.models

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()


class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        return headers
    
    async def fetch_available_models(self) -> List[str]:
        """Fetch models from OpenAI-compatible /v1/models endpoint."""
        url = f"{self.base_url}/models"
        # async with self.client.get(url, headers=self._build_headers()) as response:
        #     response.raise_for_status()
        #     data = response.json()
        #     # Assuming standard OpenAI schema: {"data": [{"id": "model1"}, ...]}
        #     return [model["id"] for model in data.get("data", [])]
        response = await self.client.get(url, headers=self._build_headers())
        response.raise_for_status()
        data = response.json()
    
        # Assuming standard OpenAI schema: {"data": [{"id": "model1"}, ...]}
        return [model["id"] for model in data.get("data", [])]

    def _convert_request(self, request: ChatRequest) -> Dict[str, Any]:
        """Convert internal request to OpenAI format."""
        messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in request.messages
        ]

        payload = {
            "model": self.strip_prefix(request.model),
            "messages": messages,
        }

        # Add optional parameters only if they have valid values
        if request.temperature is not None and request.temperature >= 0:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None and request.max_tokens > 0:
            payload["max_tokens"] = request.max_tokens
        if request.top_p is not None and 0 < request.top_p <= 1:
            payload["top_p"] = request.top_p
        if request.frequency_penalty is not None and request.frequency_penalty != 0:
            payload["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None and request.presence_penalty != 0:
            payload["presence_penalty"] = request.presence_penalty
        if request.stream is not None:
            payload["stream"] = request.stream
        if request.stop:
            payload["stop"] = request.stop
        if request.extra_body:
            payload.update(request.extra_body)

        return payload

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Non-streaming chat completion."""
        payload = self._convert_request(request)
        url = f"{self.base_url}/chat/completions"

        logger.debug("LLM API Request", provider=self.name, method="chat", url=url, payload=payload)
        response = await self.client.post(
            url,
            json=payload,
            headers=self._build_headers(),
        )
        response.raise_for_status()
        data = response.json()
        logger.debug("LLM API Response", provider=self.name, method="chat", status=response.status_code, response=data)

        return ChatResponse(
            id=data.get("id", ""),
            model=data.get("model", request.model),
            content=data["choices"][0]["message"]["content"],
            role=data["choices"][0]["message"].get("role", "assistant"),
            finish_reason=data["choices"][0].get("finish_reason"),
            usage=data.get("usage"),
        )

    async def chat_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[StreamChunk, None]:
        """Streaming chat completion."""
        request.stream = True
        payload = self._convert_request(request)
        url = f"{self.base_url}/chat/completions"

        logger.debug("LLM API Request", provider=self.name, method="chat_stream", url=url, payload=payload)
        async with self.client.stream(
            "POST",
            url,
            json=payload,
            headers=self._build_headers(),
        ) as response:
            response.raise_for_status()
            logger.debug("LLM API Response", provider=self.name, method="chat_stream", status=response.status_code)
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:] # Remove "data: " prefix
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data)
                        choices = chunk_data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            finish_reason = choices[0].get("finish_reason")
                            if content or finish_reason:
                                yield StreamChunk(
                                    content=content,
                                    finish_reason=finish_reason,
                                )
                    except json.JSONDecodeError:
                        continue

class AnthropicProvider(LLMProvider):
    """Anthropic API provider implementation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        return headers

    def _convert_request(self, request: ChatRequest) -> Dict[str, Any]:
        """Convert internal request to Anthropic format."""
        # Anthropic uses system as a separate field
        messages = []
        system_content = None

        for msg in request.messages:
            if msg.role == MessageRole.SYSTEM:
                system_content = msg.content
            else:
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })

        payload = {
            "model": self.strip_prefix(request.model),
            "messages": messages,
        }

        # Add optional parameters only if they have valid values
        if request.temperature is not None and request.temperature >= 0:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None and request.max_tokens > 0:
            payload["max_tokens"] = request.max_tokens
        else:
            payload["max_tokens"] = 1024  # Anthropic requires max_tokens
        if request.top_p is not None and 0 < request.top_p <= 1:
            payload["top_p"] = request.top_p
        if system_content:
            payload["system"] = system_content
        if request.stop:
            payload["stop_sequences"] = request.stop

        return payload

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Non-streaming chat completion."""
        payload = self._convert_request(request)
        url = f"{self.base_url}/v1/messages"

        logger.debug("LLM API Request", provider=self.name, method="chat", url=url, payload=payload)
        response = await self.client.post(
            url,
            json=payload,
            headers=self._build_headers(),
        )
        response.raise_for_status()
        data = response.json()
        logger.debug("LLM API Response", provider=self.name, method="chat", status=response.status_code, response=data)

        return ChatResponse(
            id=data.get("id", ""),
            model=request.model,
            content=data["content"][0]["text"],
            role="assistant",
            finish_reason=data.get("stop_reason"),
            usage={
                "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
            },
        )

    async def chat_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[StreamChunk, None]:
        """Streaming chat completion."""
        payload = self._convert_request(request)
        payload["stream"] = True
        url = f"{self.base_url}/v1/messages"

        logger.debug("LLM API Request", provider=self.name, method="chat_stream", url=url, payload=payload)
        async with self.client.stream(
            "POST",
            url,
            json=payload,
            headers=self._build_headers(),
        ) as response:
            response.raise_for_status()
            logger.debug("LLM API Response", provider=self.name, method="chat_stream", status=response.status_code)
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    try:
                        chunk_data = json.loads(data)
                        if chunk_data.get("type") == "content_block_delta":
                            content = chunk_data.get("delta", {}).get("text", "")
                            yield StreamChunk(content=content)
                        elif chunk_data.get("type") == "message_stop":
                            yield StreamChunk(content="", finish_reason="stop")
                    except json.JSONDecodeError:
                        continue

class GoogleProvider(LLMProvider):
    """Google Generative AI provider implementation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}

    def _convert_request(self, request: ChatRequest) -> Dict[str, Any]:
        """Convert internal request to Google format."""
        contents = []
        for msg in request.messages:
            contents.append({
                "role": "user" if msg.role == MessageRole.USER else "model",
                "parts": [{"text": msg.content}],
            })

        payload = {
            #"model": self.strip_prefix(request.model),
            "contents": contents,
            "generationConfig": {},
        }

        # Add optional parameters only if they have valid values
        if request.temperature is not None and request.temperature >= 0:
            payload["generationConfig"]["temperature"] = request.temperature
        if request.max_tokens is not None and request.max_tokens > 0:
            payload["generationConfig"]["maxOutputTokens"] = request.max_tokens
        if request.top_p is not None and 0 < request.top_p <= 1:
            payload["generationConfig"]["topP"] = request.top_p
        if request.stop:
            payload["generationConfig"]["stopSequences"] = request.stop

        return payload

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Non-streaming chat completion."""
        payload = self._convert_request(request)
        url = f"{self.base_url}/models/{request.model}:generateContent"

        logger.debug("LLM API Request", provider=self.name, method="chat", url=url, payload=payload)
        response = await self.client.post(
            f"{url}?key={self.api_key}",
            json=payload,
            headers=self._build_headers(),
        )
        response.raise_for_status()
        data = response.json()
        logger.debug("LLM API Response", provider=self.name, method="chat", status=response.status_code, response=data)

        content = ""
        if "candidates" in data and data["candidates"]:
            content = data["candidates"][0]["content"]["parts"][0]["text"]

        return ChatResponse(
            id="",
            model=request.model,
            content=content,
            role="assistant",
            finish_reason=data.get("candidates", [{}])[0].get("finishReason"),
        )

    async def chat_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[StreamChunk, None]:
        """Streaming chat completion."""
        payload = self._convert_request(request)
        url = f"{self.base_url}/models/{request.model}:streamGenerateContent"

        logger.debug("LLM API Request", provider=self.name, method="chat_stream", url=url, payload=payload)
        response = await self.client.post(
            f"{url}?key={self.api_key}&alt=sse",
            json=payload,
            headers=self._build_headers(),
        )
        response.raise_for_status()
        logger.debug("LLM API Response", provider=self.name, method="chat_stream", status=response.status_code)
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]
                try:
                    chunk_data = json.loads(data)
                    if "candidates" in chunk_data:
                        content = chunk_data["candidates"][0]["content"]["parts"][0]["text"]
                        yield StreamChunk(content=content)
                except (json.JSONDecodeError, KeyError):
                    continue

class NVIDIAProvider(LLMProvider):
    """NVIDIA NIM API provider implementation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        return headers

    def _convert_request(self, request: ChatRequest) -> Dict[str, Any]:
        """Convert internal request to NVIDIA format."""
        messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in request.messages
        ]

        payload = {
            "model": self.strip_prefix(request.model),
            "messages": messages,
        }

        # Add optional parameters only if they have valid values
        if request.temperature is not None and request.temperature >= 0:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None and request.max_tokens > 0:
            payload["max_tokens"] = request.max_tokens
        if request.top_p is not None and 0 < request.top_p <= 1:
            payload["top_p"] = request.top_p
        if request.stream is not None:
            payload["stream"] = request.stream
        if request.stop:
            payload["stop"] = request.stop
        if request.extra_body:
            payload.update(request.extra_body)

        return payload

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Non-streaming chat completion."""
        payload = self._convert_request(request)
        url = f"{self.base_url}/chat/completions"
        logger.debug("LLM API Request Headers", provider=self.name, method="chat", url=url, headers=self._build_headers())
        logger.debug("LLM API Request", provider=self.name, method="chat", url=url, payload=payload)
        response = await self.client.post(
            url,
            json=payload,
            headers=self._build_headers(),
        )
        response.raise_for_status()
        data = response.json()
        logger.debug("LLM API Response", provider=self.name, method="chat", status=response.status_code, response=data)

        return ChatResponse(
            id=data.get("id", ""),
            model=data.get("model", request.model),
            content=data["choices"][0]["message"]["content"],
            role=data["choices"][0]["message"].get("role", "assistant"),
            finish_reason=data["choices"][0].get("finish_reason"),
            usage=data.get("usage"),
        )

    async def chat_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[StreamChunk, None]:
        """Streaming chat completion."""
        request.stream = True
        payload = self._convert_request(request)
        url = f"{self.base_url}/chat/completions"
        logger.debug("LLM API Request Headers", provider=self.name, method="chat_stream", url=url, headers=self._build_headers())
        logger.debug("LLM API Request", provider=self.name, method="chat_stream", url=url, payload=payload)
        async with self.client.stream(
            "POST",
            url,
            json=payload,
            headers=self._build_headers(),
        ) as response:
            response.raise_for_status()
            logger.debug("LLM API Response", provider=self.name, method="chat_stream", status=response.status_code)
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data)
                        choices = chunk_data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            finish_reason = choices[0].get("finish_reason")
                            if content or finish_reason:
                                yield StreamChunk(
                                    content=content,
                                    finish_reason=finish_reason,
                                )
                    except json.JSONDecodeError:
                        continue

class OllamaProvider(LLMProvider):
    """Ollama provider implementation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}

    def _convert_request(self, request: ChatRequest) -> Dict[str, Any]:
        """Convert internal request to Ollama format."""
        messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in request.messages
        ]

        payload = {
            "model": self.strip_prefix(request.model),
            "messages": messages,
            "stream": request.stream,
            "options": {},
        }

        # Add optional parameters only if they have valid values
        if request.temperature is not None and request.temperature >= 0:
            payload["options"]["temperature"] = request.temperature
        if request.max_tokens is not None and request.max_tokens > 0:
            payload["options"]["num_predict"] = request.max_tokens
        if request.top_p is not None and 0 < request.top_p <= 1:
            payload["options"]["top_p"] = request.top_p

        return payload

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Non-streaming chat completion."""
        payload = self._convert_request(request)
        payload["stream"] = False
        url = f"{self.base_url}/api/chat"

        logger.debug("LLM API Request", provider=self.name, method="chat", url=url, payload=payload)
        response = await self.client.post(
            url,
            json=payload,
            headers=self._build_headers(),
        )
        response.raise_for_status()
        data = response.json()
        logger.debug("LLM API Response", provider=self.name, method="chat", status=response.status_code, response=data)

        return ChatResponse(
            id="",
            model=request.model,
            content=data["message"]["content"],
            role="assistant",
            finish_reason="stop" if data.get("done") else None,
        )

    async def chat_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[StreamChunk, None]:
        """Streaming chat completion."""
        payload = self._convert_request(request)
        payload["stream"] = True
        url = f"{self.base_url}/api/chat"

        logger.debug("LLM API Request", provider=self.name, method="chat_stream", url=url, payload=payload)
        async with self.client.stream(
            "POST",
            url,
            json=payload,
            headers=self._build_headers(),
        ) as response:
            response.raise_for_status()
            logger.debug("LLM API Response", provider=self.name, method="chat_stream", status=response.status_code)
            async for line in response.aiter_lines():
                try:
                    chunk_data = json.loads(line)
                    if "message" in chunk_data:
                        content = chunk_data["message"].get("content", "")
                        done = chunk_data.get("done", False)
                        yield StreamChunk(
                            content=content,
                            finish_reason="stop" if done else None,
                        )
                except json.JSONDecodeError:
                    continue


class DummyProvider(LLMProvider):
    """Dummy provider that returns the prompt as the response."""

    def __init__(self, **kwargs):
        if "base_url" not in kwargs or not kwargs["base_url"]:
            kwargs["base_url"] = "http://dummy"
        if "api_key" not in kwargs:
            kwargs["api_key"] = "dummy-key"
        super().__init__(**kwargs)

    def _build_headers(self) -> Dict[str, str]:
        return {}

    async def fetch_available_models(self) -> List[str]:
        return self.models

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Non-streaming chat completion returning the last prompt."""
        prompt = ""
        if request.messages:
            prompt = request.messages[-1].content

        return ChatResponse(
            id=f"dummy-{int(time.time())}",
            model=request.model,
            content=prompt,
            role="assistant",
            finish_reason="stop",
            usage={
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": len(prompt) // 4,
                "total_tokens": (len(prompt) // 4) * 2,
            },
        )

    async def chat_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[StreamChunk, None]:
        """Streaming chat completion returning the last prompt."""
        prompt = ""
        if request.messages:
            prompt = request.messages[-1].content

        if prompt:
            words = prompt.split(" ")
            for i, word in enumerate(words):
                space = " " if i > 0 else ""
                yield StreamChunk(
                    content=space + word,
                    finish_reason=None,
                )
        yield StreamChunk(
            content="",
            finish_reason="stop",
        )

