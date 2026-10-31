"""ToolHub: one MCP client per server, a single namespaced tool catalogue, and a policy gate.

- Tools are exposed to the model as `<server>__<tool>` so two servers can both
  have a `search` tool without colliding.
- Every call passes through a Policy: read-only tools run, destructive tools
  (per the server's own ToolAnnotations) need an approval callback, and
  allow/deny lists can pin anything else.
- A shared budget caps total tool calls across concurrently running agents.
- Results are truncated before they reach the model, and errors come back as
  tool results (not exceptions) so the model can recover.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

import anyio
from mcp import Client, StdioServerParameters
from mcp.server.mcpserver import MCPServer
from typing_extensions import Self

from .trace import Tracer

SEP = "__"
ApprovalFn = Callable[[str, dict], Awaitable[bool]]


async def deny_all(tool: str, args: dict) -> bool:
    return False


@dataclass
class ToolSpec:
    server: str
    name: str
    description: str
    input_schema: dict
    read_only: bool
    destructive: bool

    @property
    def qualified(self) -> str:
        return f"{self.server}{SEP}{self.name}"

    def to_llm(self) -> dict:
        tag = "[read-only] " if self.read_only else "[needs approval] " if self.destructive else ""
        return {
            "name": self.qualified,
            "description": tag + (self.description or "").strip(),
            "input_schema": self.input_schema,
        }


@dataclass
class Policy:
    approve: ApprovalFn = deny_all
    allow: set[str] = field(default_factory=set)  # qualified names always allowed
    deny: set[str] = field(default_factory=set)  # qualified names never allowed
    max_tool_calls: int = 40
    max_result_chars: int = 4000
    call_timeout_s: float = 30.0


@dataclass
class ToolOutcome:
    ok: bool
    content: str
    ms: float


class BudgetExceeded(RuntimeError):
    pass


class ToolHub:
    def __init__(
        self,
        servers: dict[str, MCPServer | StdioServerParameters],
        policy: Policy,
        tracer: Tracer | None = None,
    ):
        self._targets = servers
        self.policy = policy
        self.tracer = tracer or Tracer()
        self.clients: dict[str, Client] = {}
        self.tools: dict[str, ToolSpec] = {}
        self.instructions: dict[str, str] = {}
        self.calls_made = 0
        self._lock = anyio.Lock()
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> Self:
        for name, target in self._targets.items():
            if SEP in name:
                raise ValueError(f"server name {name!r} may not contain {SEP!r}")
            client = await self._stack.enter_async_context(Client(target))
            self.clients[name] = client
            self.instructions[name] = client.instructions or ""
            for t in (await client.list_tools()).tools:
                ann = t.annotations
                spec = ToolSpec(
                    server=name,
                    name=t.name,
                    description=t.description or "",
                    input_schema=t.input_schema,
                    read_only=bool(ann and ann.read_only_hint),
                    destructive=bool(ann and ann.destructive_hint),
                )
                self.tools[spec.qualified] = spec
        self.tracer.event("hub.ready", servers=list(self.clients), tools=sorted(self.tools))
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    def llm_tools(self, only: set[str] | None = None) -> list[dict]:
        return [s.to_llm() for q, s in sorted(self.tools.items()) if only is None or q in only]

    def server_instructions(self) -> str:
        return "\n".join(f"- {n}: {i}" for n, i in self.instructions.items() if i)

    async def _reserve(self) -> None:
        async with self._lock:
            if self.calls_made >= self.policy.max_tool_calls:
                raise BudgetExceeded(f"tool-call budget of {self.policy.max_tool_calls} exhausted")
            self.calls_made += 1

    async def call(self, qualified: str, args: dict[str, Any], *, agent: str = "?") -> ToolOutcome:
        start = time.perf_counter()
        spec = self.tools.get(qualified)
        if spec is None:
            return self._done(agent, qualified, args, False, f"unknown tool {qualified!r}", start)
        if qualified in self.policy.deny:
            return self._done(agent, qualified, args, False, "blocked by policy (deny list)", start)
        needs_approval = qualified not in self.policy.allow and not spec.read_only and spec.destructive
        if needs_approval:
            approved = await self.policy.approve(qualified, args)
            self.tracer.event("policy.approval", agent=agent, tool=qualified, approved=approved)
            if not approved:
                return self._done(
                    agent,
                    qualified,
                    args,
                    False,
                    "denied: a human did not approve this destructive action",
                    start,
                )
        await self._reserve()
        try:
            with anyio.fail_after(self.policy.call_timeout_s):
                res = await self.clients[spec.server].call_tool(spec.name, args)
        except TimeoutError:
            return self._done(
                agent, qualified, args, False, f"timed out after {self.policy.call_timeout_s}s", start
            )
        except Exception as e:  # noqa: BLE001 - any transport failure becomes a tool error
            return self._done(agent, qualified, args, False, f"transport error: {e}", start)
        content = _render(res)
        return self._done(agent, qualified, args, not res.is_error, content, start)

    def _done(self, agent, tool, args, ok, content, start) -> ToolOutcome:
        limit = self.policy.max_result_chars
        if len(content) > limit:
            content = content[:limit] + f"\n...[truncated {len(content) - limit} chars]"
        ms = (time.perf_counter() - start) * 1000
        self.tracer.event(
            "tool.call", agent=agent, tool=tool, args=args, ok=ok, ms=round(ms, 1), preview=content[:160]
        )
        return ToolOutcome(ok, content, ms)


def _render(res) -> str:
    if getattr(res, "structured_content", None) is not None:
        sc = res.structured_content
        value = sc.get("result", sc) if isinstance(sc, dict) and len(sc) == 1 else sc
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return "\n".join(getattr(c, "text", str(c)) for c in res.content)
