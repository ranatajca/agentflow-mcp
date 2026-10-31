"""End-to-end tests: the real MCP servers, the real orchestrator, a scripted LLM."""

import json

import pytest

from agentflow import Config, Policy, ScriptedLLM, ToolHub, Tracer, Turn, run
from agentflow.agents import Evidence, grounding_issues
from agentflow.llm import call
from agentflow.plan import Plan
from agentflow.servers import build_compute_server, build_knowledge_server

pytestmark = pytest.mark.anyio

Q = (
    "An X200 was delivered on 2026-03-14 and its LiDAR failed on 2026-10-20. Is it under "
    "warranty, and what would a Loyalty member pay out of warranty?"
)

PLAN = {
    "reasoning": "Look up warranty terms and part price in parallel, then do the math.",
    "steps": [
        {
            "id": "s1",
            "goal": "Find the X200 warranty period and out-of-warranty pricing rules",
            "tools": ["knowledge__search_docs", "knowledge__read_doc"],
        },
        {"id": "s2", "goal": "Find the price of the X200 LiDAR sensor", "tools": ["knowledge__search_docs"]},
        {
            "id": "s3",
            "goal": "Compute days since delivery and the Loyalty repair cost",
            "depends_on": ["s1", "s2"],
            "tools": ["compute__date_diff", "compute__calculate"],
        },
    ],
}

ANSWER = (
    "Yes: 220 days have passed and the X200 has a 24 month warranty "
    "[policies/warranty#Coverage periods]. Out of warranty a Loyalty member would pay "
    "408.6 USD (79 diagnostic + 412 LiDAR less 20%) [policies/warranty#Out-of-warranty "
    "repairs] [products/x200#Common replacement parts]."
)


def servers(kb):
    return {"knowledge": build_knowledge_server(kb), "compute": build_compute_server()}


def happy_script():
    return {
        "planner": [Turn(text=json.dumps(PLAN))],
        "executor:s1": [
            Turn(tool_calls=[call("knowledge__search_docs", "a", query="X200 warranty")]),
            Turn(tool_calls=[call("knowledge__read_doc", "b", doc_id="policies/warranty")]),
            Turn(
                text="24 months; out of warranty = 79 USD + parts, Loyalty 20% off parts "
                "[policies/warranty#Coverage periods]"
            ),
        ],
        "executor:s2": [
            Turn(tool_calls=[call("knowledge__search_docs", "c", query="X200 LiDAR sensor price")]),
            Turn(text="LiDAR sensor 412 USD [products/x200#Common replacement parts]"),
        ],
        "executor:s3": [
            # Two independent tool calls in one turn run concurrently.
            Turn(
                tool_calls=[
                    call("compute__date_diff", "d", start="2026-03-14", end="2026-10-20"),
                    call("compute__calculate", "e", expression="79 + 412 * 0.8"),
                ]
            ),
            Turn(text="220 days elapsed; repair cost 408.6 USD"),
        ],
        "synthesizer": [Turn(text=ANSWER)],
        "critic": [Turn(text='{"verdict": "pass", "issues": []}')],
    }


async def test_full_run_plans_parallelises_and_answers(kb_path):
    llm = ScriptedLLM(happy_script())
    async with ToolHub(servers(kb_path), Policy(), Tracer()) as hub:
        result = await run(Q, llm, hub)

    assert [[s.id for s in w] for w in result.plan.waves()] == [["s1", "s2"], ["s3"]]
    assert all(r.status == "done" for r in result.steps.values())
    assert result.review.verdict == "pass" and result.revisions == 0
    assert "408.6" in result.answer

    calc = next(e for e in result.evidence if e.tool == "compute__calculate")
    assert calc.ok and float(calc.content) == pytest.approx(408.6)
    days = next(e for e in result.evidence if e.tool == "compute__date_diff")
    assert json.loads(days.content)["days"] == 220

    # s3 only saw its dependencies' outputs, and only its allowed tools.
    s3_req = next(r for r in llm.requests if r.agent == "executor:s3")
    assert "Result of s1" in s3_req.messages[0]["content"]
    assert {t["name"] for t in s3_req.tools} == {"compute__date_diff", "compute__calculate"}
    assert result.summary["tool_calls"] == 5 and result.summary["tool_errors"] == 0


async def test_planner_repairs_an_invalid_dag(kb_path):
    cyclic = {
        "steps": [
            {"id": "s1", "goal": "look things up", "depends_on": ["s2"]},
            {"id": "s2", "goal": "do the math please", "depends_on": ["s1"]},
        ]
    }
    script = happy_script()
    script["planner"] = [Turn(text="```json\n" + json.dumps(cyclic) + "\n```"), Turn(text=json.dumps(PLAN))]
    llm = ScriptedLLM(script)
    async with ToolHub(servers(kb_path), Policy(), Tracer()) as hub:
        result = await run(Q, llm, hub)
    repair_msg = [r for r in llm.requests if r.agent == "planner"][1].messages[-1]["content"]
    assert "cycle" in repair_msg
    assert result.review.verdict == "pass"


async def test_critic_grounding_forces_revision(kb_path):
    script = happy_script()
    script["synthesizer"] = [
        Turn(text="Yes, it is covered [policies/extended-care#Terms]. Cost 408.6 USD."),  # fake cite
        Turn(text=ANSWER),
    ]
    script["critic"] = [Turn(text='{"verdict":"pass","issues":[]}')] * 2
    llm = ScriptedLLM(script)
    async with ToolHub(servers(kb_path), Policy(), Tracer()) as hub:
        result = await run(Q, llm, hub)
    assert result.revisions == 1 and result.review.verdict == "pass"
    second = [r for r in llm.requests if r.agent == "synthesizer"][1].messages[0]["content"]
    assert "policies/extended-care" in second and "no agent retrieved" in second


async def test_destructive_tool_needs_approval(kb_path):
    asked = []

    async def deny(tool, args):
        asked.append(tool)
        return False

    plan = {"steps": [{"id": "s1", "goal": "save then delete a scratch note"}]}
    script = {
        "planner": [Turn(text=json.dumps(plan))],
        "executor:s1": [
            Turn(tool_calls=[call("knowledge__save_note", "a", note_id="tmp", content="x")]),
            Turn(tool_calls=[call("knowledge__delete_note", "b", note_id="tmp")]),
            lambda req: Turn(
                text="delete was " + ("denied" if "denied" in json.dumps(req.messages[-1]) else "allowed")
            ),
        ],
        "synthesizer": [Turn(text="The note was saved; deletion was denied.")],
        "critic": [Turn(text='{"verdict":"pass"}')],
    }
    async with ToolHub(servers(kb_path), Policy(approve=deny), Tracer()) as hub:
        result = await run("tidy notes", ScriptedLLM(script), hub)
        # save_note is not destructive, so it ran without asking; the note still exists.
        assert (await hub.clients["knowledge"].read_resource("notes://tmp")).contents[0].text == "x"
    assert asked == ["knowledge__delete_note"]
    assert result.steps["s1"].output == "delete was denied"
    assert result.summary["approvals"] == [("knowledge__delete_note", False)]


async def test_shared_tool_budget_stops_runaway_agents(kb_path):
    plan = {"steps": [{"id": "s1", "goal": "search forever and ever"}]}
    loop = [Turn(tool_calls=[call("knowledge__search_docs", f"x{i}", query=f"q{i}")]) for i in range(10)]
    script = {
        "planner": [Turn(text=json.dumps(plan))],
        "executor:s1": loop,
        "synthesizer": [Turn(text="Partial: budget ran out.")],
        "critic": [Turn(text='{"verdict":"pass"}')],
    }
    async with ToolHub(servers(kb_path), Policy(max_tool_calls=3), Tracer()) as hub:
        result = await run("q", ScriptedLLM(script), hub, Config(max_revisions=0))
    assert result.steps["s1"].status == "partial"
    assert result.summary["tool_calls"] == 3


async def test_iteration_cap_forces_summary(kb_path):
    plan = {"steps": [{"id": "s1", "goal": "keep looking things up"}]}
    loop = [Turn(tool_calls=[call("knowledge__search_docs", f"y{i}", query="returns")]) for i in range(2)]
    script = {
        "planner": [Turn(text=json.dumps(plan))],
        "executor:s1": [*loop, lambda req: Turn(text=f"summary, tools offered={len(req.tools)}")],
        "synthesizer": [Turn(text="ok [policies/returns#Return window]")],
        "critic": [Turn(text='{"verdict":"pass"}')],
    }
    async with ToolHub(servers(kb_path), Policy(), Tracer()) as hub:
        result = await run("q", ScriptedLLM(script), hub, Config(max_step_iterations=2))
    assert result.steps["s1"].status == "partial"
    assert result.steps["s1"].output == "summary, tools offered=0"


async def test_failed_dependency_skips_downstream(kb_path):
    plan = {
        "steps": [
            {"id": "s1", "goal": "this one will crash"},
            {"id": "s2", "goal": "needs s1's output", "depends_on": ["s1"]},
        ]
    }

    def boom(req):
        raise RuntimeError("model API unavailable")

    script = {
        "planner": [Turn(text=json.dumps(plan))],
        "executor:s1": [boom],
        "synthesizer": [Turn(text="Could not answer.")],
        "critic": [Turn(text='{"verdict":"pass"}')],
    }
    async with ToolHub(servers(kb_path), Policy(), Tracer()) as hub:
        result = await run("q", ScriptedLLM(script), hub)
    assert result.steps["s1"].status == "failed"
    assert result.steps["s2"].output.startswith("skipped")


def test_plan_validation():
    with pytest.raises(ValueError, match="unknown steps"):
        Plan.model_validate({"steps": [{"id": "s1", "goal": "abcdef", "depends_on": ["s9"]}]})
    with pytest.raises(ValueError, match="duplicate"):
        Plan.model_validate({"steps": [{"id": "s1", "goal": "abcdef"}, {"id": "s1", "goal": "ghijkl"}]})
    p = Plan.model_validate(
        {
            "steps": [
                {"id": "s1", "goal": "abcdef"},
                {"id": "s2", "goal": "abcdef"},
                {"id": "s3", "goal": "abcdef", "depends_on": ["s1"]},
                {"id": "s4", "goal": "abcdef", "depends_on": ["s2", "s3"]},
            ]
        }
    )
    assert [[s.id for s in w] for w in p.waves()] == [["s1", "s2"], ["s3"], ["s4"]]


def test_grounding_check_unit():
    ev = [Evidence("knowledge__read_doc", {}, True, "doc policies/warranty ...")]
    assert grounding_issues("ok [policies/warranty#Coverage periods]", ev) == []
    assert grounding_issues("no citations here", ev) == ["answer has no [doc_id#section] citations"]
