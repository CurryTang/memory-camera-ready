"""
LLM provider abstraction.

Decouples agents and operations from any specific LLM SDK.
Agents call BaseLLMProvider.chat() — the provider handles auth,
retries, and SDK-specific serialization.

Two concrete providers are planned:
  - OpenAICompatibleProvider  (OpenAI, local vLLM, Gemini via compat endpoint)
  - AnthropicProvider         (native Claude API with tool_use blocks)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class Message:
    """
    A single message in a chat conversation.

    role: "system" | "user" | "assistant" | "tool"
    content: Text content (may be None for tool-result messages).
    tool_calls: Populated when role="assistant" and LLM invoked tools.
    tool_call_id: Populated when role="tool" (correlates with a tool_call).
    name: Tool name, used when role="tool".
    """

    role: str
    content: Optional[str] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

@dataclass
class LLMResponse:
    """
    Parsed response from an LLM call.

    Attributes:
        content: Text output (None if the model only made tool calls).
        tool_calls: Structured tool calls requested by the model.
        raw: The raw SDK response object for debugging.
        usage: Token usage dict (prompt_tokens, completion_tokens, total_tokens).
    """

    content: Optional[str]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: Any = None
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def is_final(self) -> bool:
        """True if the model returned a text answer with no tool calls."""
        return self.content is not None and not self.has_tool_calls

class BaseLLMProvider(ABC):
    """
    Abstract LLM provider.

    All agents and operations use this interface — never import an SDK directly
    inside agent or operation code.
    """

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a chat request and return a structured response.

        Args:
            messages: Conversation history in provider-agnostic format.
            tools: Optional list of tool schemas (OpenAI function-calling format).
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.

        Returns:
            LLMResponse with parsed content and/or tool_calls.
        """

    @property
    @abstractmethod
    def model(self) -> str:
        """The model identifier being used."""
