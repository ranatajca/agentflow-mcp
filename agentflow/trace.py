"""Structured run tracing: every LLM turn, tool call, approval and state change is an event."""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Tracer:
    events: list[dict[str, Any]] = field(default_factory=list)
    sink: Path | None = None
    t0: float = field(default_factory=time.perf_counter)

    def event(self, kind: str, **data: Any) -> None:
        ev = {"t_ms": round((time.perf_counter() - self.t0) * 1000, 1), "kind": kind, **data}
        self.events.append(ev)
        if self.sink:
            with self.sink.open("a") as f:
                f.write(json.dumps(ev, default=str) + "\n")

    def of(self, kind: str) -> list[dict]:
        return [e for e in self.events if e["kind"] == kind]

    def summary(self) -> dict[str, Any]:
        tools = self.of("tool.call")
        llm = self.of("llm.turn")
        return {
            "wall_ms": self.events[-1]["t_ms"] if self.events else 0,
            "llm_turns": len(llm),
            "input_tokens": sum(e.get("input_tokens", 0) for e in llm),
            "output_tokens": sum(e.get("output_tokens", 0) for e in llm),
            "tool_calls": len(tools),
            "tool_errors": sum(1 for e in tools if not e["ok"]),
            "by_tool": dict(Counter(e["tool"] for e in tools)),
            "approvals": [(e["tool"], e["approved"]) for e in self.of("policy.approval")],
            "revisions": len(self.of("critic.revise")),
        }
