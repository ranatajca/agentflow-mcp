"""The four agents. Each is a small function over (LLM, ToolHub), not a class hierarchy.

planner      question -> validated Plan (DAG), with error-guided repair
executor     one Step -> StepResult, via a bounded tool-use loop over MCP
synthesizer  question + step results -> cited answer
critic       answer + evidence -> pass / revise, plus a code-level grounding check
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

import anyio
from pydantic import BaseModel, Field, ValidationError

from .hub import BudgetExceeded, ToolHub
from .llm import LLM, Request, Turn
from .plan import Plan, Step, describe_error, parse_json_object

# ----------------------------------------------------------------------------- planner

PLANNER_SYSTEM = """You are the planner in a multi-agent system. Break the user's question into
the smallest set of steps (1-8) that tool-using executor agents can each complete on their own.
Steps with no dependency between them run in parallel, so only add depends_on when a step
truly needs another step's output. Prefer one step per independent fact to look up, then a
final step that combines them. Use the compute tools for every number or date.

Available tools:
{tools}

Server guidance:
{guidance}

Reply with JSON only:
{{"reasoning": "<one short paragraph>",
  "steps": [{{"id": "s1", "goal": "...", "depends_on": [], "tools": ["server__tool", ...] or null}}]}}"""


async def plan(question: str, llm: LLM, hub: ToolHub, max_repairs: int = 2) -> Plan:
    catalogue = "\n".join(f"- {t['name']}: {t['description']}" for t in hub.llm_tools())
    system = PLANNER_SYSTEM.format(tools=catalogue, guidance=hub.server_instructions() or "-")
    messages: list[dict] = [{"role": "user", "content": question}]
    last_err = ""
    for attempt in range(max_repairs + 1):
        turn = await llm(Request("planner", system, messages, []))
        hub.tracer.event(
            "llm.turn",
            agent="planner",
            attempt=attempt,
            input_tokens=turn.input_tokens,
            output_tokens=turn.output_tokens,
        )
        try:
            result = Plan.model_validate(parse_json_object(turn.text))
            unknown = {t for s in result.steps for t in (s.tools or [])} - set(hub.tools)
            if unknown:
                raise ValueError(f"unknown tools in plan: {sorted(unknown)}")
            hub.tracer.event(
                "plan.ready",
                steps=[s.model_dump() for s in result.steps],
                waves=[[s.id for s in w] for w in result.waves()],
            )
            return result
        except ValueError as e:  # JSONDecodeError and ValidationError are ValueErrors too
            if isinstance(e, (json.JSONDecodeError, ValidationError)):
                last_err = describe_error(e)
            else:
                last_err = f"- {e}"
            hub.tracer.event("plan.repair", attempt=attempt, error=last_err)
            messages = [
                *messages,
                {"role": "assistant", "content": turn.text},
                {
                    "role": "user",
                    "content": f"That plan is invalid:\n{last_err}\nReturn corrected JSON only.",
                },
            ]
    raise PlanningFailed(last_err)


class PlanningFailed(RuntimeError):
    pass


# ----------------------------------------------------------------------------- executor

EXECUTOR_SYSTEM = """You are executor agent {step_id}. Complete exactly this step using the tools:

STEP: {goal}

Rules: call tools to get facts, never guess. You may call several tools at once when they are
independent. When done, reply with no tool calls and a short factual result that cites sources
as [doc_id#section]. If a tool errors, read the error and adapt. If a tool needs approval and is
denied, continue without it and say so."""


@dataclass
class Evidence:
    tool: str
    args: dict
    ok: bool
    content: str


@dataclass
class StepResult:
    step_id: str
    status: Literal["done", "partial", "failed"]
    output: str
    evidence: list[Evidence] = field(default_factory=list)
    iterations: int = 0


async def execute_step(
    step: Step, deps: dict[str, StepResult], llm: LLM, hub: ToolHub, max_iters: int = 6
) -> StepResult:
    agent = f"executor:{step.id}"
    tools = hub.llm_tools(set(step.tools) if step.tools else None)
    system = EXECUTOR_SYSTEM.format(step_id=step.id, goal=step.goal)
    context = "\n\n".join(f"Result of {d}:\n{r.output}" for d, r in deps.items())
    messages: list[dict] = [
        {"role": "user", "content": (context + "\n\n" if context else "") + f"Do step {step.id} now."}
    ]
    evidence: list[Evidence] = []
    for i in range(1, max_iters + 1):
        turn = await llm(Request(agent, system, messages, tools))
        _log_turn(hub, agent, turn, i)
        if not turn.tool_calls:
            return StepResult(step.id, "done", turn.text.strip(), evidence, i)
        messages.append({"role": "assistant", "content": turn.as_content()})
        try:
            results = await _run_parallel(turn, hub, agent)
        except BudgetExceeded as e:
            hub.tracer.event("budget.exhausted", agent=agent, detail=str(e))
            return StepResult(step.id, "partial", f"Stopped early: {e}", evidence, i)
        blocks = []
        for call_, out in zip(turn.tool_calls, results):
            evidence.append(Evidence(call_.name, call_.args, out.ok, out.content))
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call_.id,
                    "content": out.content,
                    "is_error": not out.ok,
                }
            )
        messages.append({"role": "user", "content": blocks})
    # Iteration cap hit: force a tool-free summary of what was found.
    messages.append({"role": "user", "content": "Iteration limit reached. Summarise what you found."})
    turn = await llm(Request(agent, system, messages, []))
    _log_turn(hub, agent, turn, max_iters + 1)
    return StepResult(step.id, "partial", turn.text.strip(), evidence, max_iters)


async def _run_parallel(turn: Turn, hub: ToolHub, agent: str):
    results: list = [None] * len(turn.tool_calls)
    budget_err: list[BudgetExceeded] = []

    async def one(idx, c):
        try:
            results[idx] = await hub.call(c.name, c.args, agent=agent)
        except BudgetExceeded as e:
            budget_err.append(e)

    async with anyio.create_task_group() as tg:
        for idx, c in enumerate(turn.tool_calls):
            tg.start_soon(one, idx, c)
    if budget_err:
        raise budget_err[0]
    return results


def _log_turn(hub: ToolHub, agent: str, turn: Turn, i: int) -> None:
    hub.tracer.event(
        "llm.turn",
        agent=agent,
        iteration=i,
        tool_calls=[c.name for c in turn.tool_calls],
        input_tokens=turn.input_tokens,
        output_tokens=turn.output_tokens,
    )


# ----------------------------------------------------------------------------- synthesizer

SYNTH_SYSTEM = """Write the final answer to the user's question from the step results only.
Be direct, show the key numbers, and cite every factual claim as [doc_id#section].
If a step was partial or failed, say what is unknown instead of filling the gap."""


async def synthesize(
    question: str,
    results: dict[str, StepResult],
    llm: LLM,
    hub: ToolHub,
    issues: list[str] | None = None,
    previous: str | None = None,
) -> str:
    body = "\n\n".join(f"[{r.step_id} | {r.status}]\n{r.output}" for r in results.values())
    content = f"Question: {question}\n\nStep results:\n{body}"
    if issues:
        content += (
            f"\n\nYour previous answer:\n{previous}\n\nA reviewer found these problems, fix "
            "all of them:\n" + "\n".join(f"- {x}" for x in issues)
        )
    turn = await llm(Request("synthesizer", SYNTH_SYSTEM, [{"role": "user", "content": content}], []))
    _log_turn(hub, "synthesizer", turn, 1)
    return turn.text.strip()


# ----------------------------------------------------------------------------- critic

CRITIC_SYSTEM = """You review an answer produced by other agents. Check it against the raw tool
evidence only. Flag: claims not supported by evidence, arithmetic that does not match a
compute tool result, missing parts of the question, and missing citations.
Reply with JSON only: {"verdict": "pass" | "revise", "issues": ["..."]}"""

_CITE = re.compile(r"\[([a-z0-9_/\-]+)#([^\]]+)\]", re.IGNORECASE)


class Review(BaseModel):
    verdict: Literal["pass", "revise"]
    issues: list[str] = Field(default_factory=list)


def grounding_issues(answer: str, evidence: list[Evidence]) -> list[str]:
    """Deterministic check that runs before the LLM critic: every [doc#section] cited in the
    answer must have been returned by a successful search or read during this run."""
    seen_text = "\n".join(e.content for e in evidence if e.ok).lower()
    issues = []
    for doc, section in _CITE.findall(answer):
        if doc.lower() not in seen_text:
            issues.append(f"citation [{doc}#{section}] refers to a document no agent retrieved")
    if not _CITE.search(answer) and any(e.tool.endswith(("search_docs", "read_doc")) for e in evidence):
        issues.append("answer has no [doc_id#section] citations")
    return issues


async def critique(question: str, answer: str, evidence: list[Evidence], llm: LLM, hub: ToolHub) -> Review:
    hard = grounding_issues(answer, evidence)
    ev_text = "\n".join(
        f"- {e.tool}({json.dumps(e.args)}) -> {'OK' if e.ok else 'ERROR'}: {e.content[:600]}"
        for e in evidence
    )
    content = f"Question: {question}\n\nAnswer:\n{answer}\n\nEvidence:\n{ev_text}"
    turn = await llm(Request("critic", CRITIC_SYSTEM, [{"role": "user", "content": content}], []))
    _log_turn(hub, "critic", turn, 1)
    try:
        review = Review.model_validate(parse_json_object(turn.text))
    except (json.JSONDecodeError, ValidationError):
        review = Review(verdict="pass", issues=[])  # an unparseable critic never blocks an answer
        hub.tracer.event("critic.unparseable", text=turn.text[:200])
    if hard:
        review = Review(verdict="revise", issues=[*hard, *review.issues])
    return review
