from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from config.settings import LLMRoleSettings, Settings, get_anthropic_client


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient(Protocol):
    def complete(self, *, system: str, prompt: str, json_mode: bool = False) -> LLMResponse:
        ...


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    candidates = [text.strip(), cleaned]
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise LLMError("Model response is not a valid JSON object")


class OpenAICompatibleLLMClient:
    def __init__(self, config: LLMRoleSettings) -> None:
        if not config.base_url:
            raise LLMError("OpenAI-compatible provider requires base_url")
        self.config = config
        self._client = httpx.Client(timeout=config.timeout_seconds)

    def complete(self, *, system: str, prompt: str, json_mode: bool = False) -> LLMResponse:
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = self._client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        body = response.json()
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Invalid OpenAI-compatible response") from exc
        usage = body.get("usage") or {}
        return LLMResponse(
            text=str(text),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )


class AnthropicLLMClient:
    def __init__(self, config: LLMRoleSettings, settings: Settings) -> None:
        self.config = config
        self._client = get_anthropic_client(
            settings,
            api_key=config.api_key,
            base_url=config.base_url,
            timeout_seconds=config.timeout_seconds,
        )

    def complete(self, *, system: str, prompt: str, json_mode: bool = False) -> LLMResponse:
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )
        text = "\n".join(str(block.text) for block in response.content if getattr(block, "text", None)).strip()
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=text,
            prompt_tokens=int(getattr(usage, "input_tokens", 0)),
            completion_tokens=int(getattr(usage, "output_tokens", 0)),
        )


def create_llm_client(config: LLMRoleSettings, settings: Settings) -> LLMClient:
    if config.provider in {"openai", "vllm", "tgi"}:
        return OpenAICompatibleLLMClient(config)
    if config.provider == "anthropic":
        return AnthropicLLMClient(config, settings)
    raise LLMError(f"Unsupported LLM provider: {config.provider}")


__all__ = ["LLMClient", "LLMError", "LLMResponse", "create_llm_client", "parse_json_object"]
