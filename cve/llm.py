"""Provider-agnostic LLM client.

The framework depends only on the small `LLMClient` interface below, so any
backend (OpenAI, Anthropic, a local model) can be plugged in. A deterministic
`MockLLM` lets the whole pipeline and test suite run without an API key.

IMPORTANT: `MockLLM` exists only to exercise the plumbing. It is NOT a model of
how a real LLM behaves and must NOT be used to produce benchmark numbers for the
paper. Real results require a real model (see experiments/run_benchmark.py).
"""
from __future__ import annotations

import os
from typing import Protocol


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str:
        ...


class MockLLM:
    """Deterministic stub. Echoes provided context; invents nothing.

    Because it never computes or guesses numbers, it cannot exhibit
    hallucination -- which is exactly why it cannot stand in for a real model
    when measuring hallucination.
    """
    name = "mock"

    def complete(self, system: str, user: str) -> str:
        # Return a fixed, grounded sentence set. Downstream code that needs
        # numbers should read them from the deterministic layer, not from here.
        return (
            "Based on the provided computed results, the treatment appears to "
            "move the metric relative to control. See the computed figures for "
            "the exact values; no numbers are invented here."
        )


class OpenAILLM:
    """Adapter for OpenAI-compatible chat models. Requires `openai` + API key."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        from openai import OpenAI  # lazy import

        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.name = f"openai:{model}"

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


class AnthropicLLM:
    """Adapter for Anthropic messages API. Requires `anthropic` + API key."""

    def __init__(self, model: str = "claude-3-5-sonnet-latest", temperature: float = 0.0):
        import anthropic  # lazy import

        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.name = f"anthropic:{model}"

    def complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text"))


class BedrockLLM:
    """Adapter for Amazon Bedrock via the Converse API. Requires `boto3` and
    AWS credentials in the environment (use a PERSONAL AWS account).

    The credentials come from the standard boto3 chain (env vars, ~/.aws, etc.),
    so configure a personal profile — never a work/internal account.
    """

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0",
        region: str = None,
        temperature: float = 0.0,
    ):
        import boto3  # lazy import

        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region or os.environ.get("AWS_REGION", "us-east-1"),
        )
        self.model_id = model_id
        self.temperature = temperature
        self.name = f"bedrock:{model_id}"

    def complete(self, system: str, user: str) -> str:
        resp = self._client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"temperature": self.temperature, "maxTokens": 1024},
        )
        return resp["output"]["message"]["content"][0]["text"]


def get_client(spec: str = "mock") -> LLMClient:
    """Factory. spec examples: 'mock', 'openai:gpt-4o-mini',
    'anthropic:claude-3-5-sonnet-latest', 'bedrock:anthropic.claude-3-5-sonnet-20240620-v1:0'."""
    if spec == "mock":
        return MockLLM()
    if spec.startswith("openai:"):
        return OpenAILLM(model=spec.split(":", 1)[1])
    if spec.startswith("anthropic:"):
        return AnthropicLLM(model=spec.split(":", 1)[1])
    if spec.startswith("bedrock:"):
        return BedrockLLM(model_id=spec.split(":", 1)[1])
    raise ValueError(f"Unknown LLM spec: {spec}")
