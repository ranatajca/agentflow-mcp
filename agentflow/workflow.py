"""Orchestrator: plan -> parallel DAG execution -> synthesis -> critic/revise loop.

question
   │
planner ──(invalid JSON / bad DAG)──> repair (≤2)
   │ Plan
wave 1: [s1 ‖ s2 ‖ s3]   executors run concurrently, each a tool-use loop over MCP
wave 2: [s4]             sees outputs of its dependencies only
   │
synthesizer ──> critic ──(revise)──> synthesizer (≤ max_revisions)
   │
Result(answer, plan, steps, review, trace)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import anyio

from .agents import (
    Evidence,
    Review,
    StepResult,
    critique,
    execute_step,
    plan,
    synthesize,
)
from .hub import ToolHub
from .llm import LLM
from .plan import Plan


@dataclass
class Config:
    max_step_iterations: int = 6
    max_parallel_steps: int = 4
    max_revisions: int = 2
    timeout_s: float = 300.0


@dataclass
class Result:
    question: str
    answer: str
    plan: Plan
    steps: dict[str, StepResult]
    review: Review
    revisions: int
    summary: dict = field(default_factory=dict)

    @property
    def evidence(self) -> list[Evidence]:
        return [e for r in self.steps.values() for e in r.evidence]


async def run(question: str, llm: LLM, hub: ToolHub, config: Config | None = None) -> Result:
    cfg = config or Config()
    tracer = hub.tracer
    tracer.event("run.start", question=question)
    with anyio.fail_after(cfg.timeout_s):
        the_plan = await plan(question, llm, hub)
        results: dict[str, StepResult] = {}
        limiter = anyio.CapacityLimiter(cfg.max_parallel_steps)

        for wave_no, wave in enumerate(the_plan.waves(), start=1):
            tracer.event("wave.start", wave=wave_no, steps=[s.id for s in wave])

            async def run_step(step):
                async with limiter:
                    deps = {d: results[d] for d in step.depends_on}
                    if any(deps[d].status == "failed" for d in deps):
                        results[step.id] = StepResult(step.id, "failed", "skipped: a dependency failed")
                        return
                    try:
                        results[step.id] = await execute_step(step, deps, llm, hub, cfg.max_step_iterations)
                    except Exception as e:  # noqa: BLE001 - one bad step must not kill the run
                        tracer.event("step.error", step=step.id, error=repr(e))
                        results[step.id] = StepResult(step.id, "failed", f"error: {e}")
                    tracer.event("step.done", step=step.id, status=results[step.id].status)

            async with anyio.create_task_group() as tg:
                for step in wave:
                    tg.start_soon(run_step, step)

        ordered = {s.id: results[s.id] for s in the_plan.steps}
        evidence = [e for r in ordered.values() for e in r.evidence]
        answer = await synthesize(question, ordered, llm, hub)
        review = await critique(question, answer, evidence, llm, hub)
        revisions = 0
        while review.verdict == "revise" and revisions < cfg.max_revisions:
            revisions += 1
            tracer.event("critic.revise", round=revisions, issues=review.issues)
            answer = await synthesize(question, ordered, llm, hub, review.issues, answer)
            review = await critique(question, answer, evidence, llm, hub)

    tracer.event("run.end", verdict=review.verdict, revisions=revisions)
    return Result(question, answer, the_plan, ordered, review, revisions, tracer.summary())
