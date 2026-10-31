"""Plan schema: a validated DAG of steps, executed in dependency waves."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError, model_validator


class Step(BaseModel):
    id: str = Field(pattern=r"^s\d{1,2}$")
    goal: str = Field(min_length=5, max_length=400)
    depends_on: list[str] = Field(default_factory=list)
    tools: list[str] | None = Field(
        None, description="optional allow-list of qualified tool names for this step"
    )


class Plan(BaseModel):
    reasoning: str = Field("", max_length=1200)
    steps: list[Step] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def _check_dag(self) -> Plan:
        ids = [s.id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate step ids: {ids}")
        known = set(ids)
        for s in self.steps:
            missing = set(s.depends_on) - known
            if missing:
                raise ValueError(f"step {s.id} depends on unknown steps {sorted(missing)}")
            if s.id in s.depends_on:
                raise ValueError(f"step {s.id} depends on itself")
        self.waves()  # raises on cycles
        return self

    def waves(self) -> list[list[Step]]:
        """Kahn's algorithm, grouped by depth: each wave only depends on earlier waves."""
        remaining = {s.id: set(s.depends_on) for s in self.steps}
        by_id = {s.id: s for s in self.steps}
        done: set[str] = set()
        waves = []
        while remaining:
            ready = sorted(i for i, deps in remaining.items() if deps <= done)
            if not ready:
                raise ValueError(f"dependency cycle among steps {sorted(remaining)}")
            waves.append([by_id[i] for i in ready])
            done.update(ready)
            for i in ready:
                del remaining[i]
        return waves


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_object(text: str) -> dict:
    text = _FENCE.sub("", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a == -1 or b <= a:
            raise
        return json.loads(text[a : b + 1])


def describe_error(err: Exception) -> str:
    if isinstance(err, ValidationError):
        return "\n".join(f"- {'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}" for e in err.errors())
    return f"- not valid JSON: {err}"
