"""LLM adapters behind one async method.

Conversation state uses the Anthropic Messages content-block format
(text / tool_use / tool_result blocks), which maps 1:1 onto MCP tool calls.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class Turn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    def as_content(self) -> list[dict]:
        blocks: list[dict] = [{"type": "text", "text": self.text}] if self.text else []
        blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.args} for c in self.tool_calls]
        return blocks


@dataclass
class Request:
    agent: str  # "planner" | "executor:<step_id>" | "synthesizer" | "critic"
    system: str
    messages: list[dict]
    tools: list[dict]


class LLM(Protocol):
    async def __call__(self, req: Request) -> Turn: ...


class AnthropicLLM:
    def __init__(self, model: str | None = None, max_tokens: int = 2048):
        import anthropic

        self.client = anthropic.AsyncAnthropic()
        self.model = model or os.getenv("AGENTFLOW_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens

    async def __call__(self, req: Request) -> Turn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": req.system,
            "messages": req.messages,
        }
        if req.tools:
            kwargs["tools"] = req.tools
        resp = await self.client.messages.create(**kwargs)
        turn = Turn(input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens)
        for block in resp.content:
            if block.type == "text":
                turn.text += block.text
            elif block.type == "tool_use":
                turn.tool_calls.append(ToolCall(block.id, block.name, dict(block.input)))
        return turn


class ScriptedLLM:
    """Deterministic LLM for tests: routes each request to a per-agent queue of turns.

    A queued item may be a Turn or a function Request -> Turn, so a script can
    react to what the agent actually saw (e.g. tool results).
    """

    def __init__(self, script: dict[str, list[Turn | Callable[[Request], Turn]]]):
        self.script = {k: list(v) for k, v in script.items()}
        self.requests: list[Request] = []

    async def __call__(self, req: Request) -> Turn:
        self.requests.append(req)
        queue = self.script.get(req.agent) or self.script.get(req.agent.split(":")[0])
        if not queue:
            raise RuntimeError(f"ScriptedLLM has no turns left for {req.agent!r}")
        item = queue.pop(0)
        return item(req) if callable(item) else item


def call(name: str, _id: str | None = None, **args) -> ToolCall:
    """Shorthand for scripts: call('knowledge__search_docs', query='x')."""
    return ToolCall(_id or f"call_{name}_{abs(hash(str(args))) % 10_000}", name, args)
