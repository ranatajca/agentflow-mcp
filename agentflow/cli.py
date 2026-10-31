"""agentflow ask "question" [--stdio] [--trace run.jsonl] [--yes]
agentflow eval examples/tasks.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anyio
from mcp import StdioServerParameters

from .hub import Policy, ToolHub
from .llm import AnthropicLLM
from .servers import build_compute_server, build_knowledge_server
from .trace import Tracer
from .workflow import Config, run


def make_servers(docs: str, stdio: bool) -> dict:
    if stdio:  # real subprocess transport, same as Claude Desktop / any MCP host would use
        py = sys.executable
        return {
            "knowledge": StdioServerParameters(
                command=py, args=["-m", "agentflow.servers.knowledge", "--docs", docs]
            ),
            "compute": StdioServerParameters(command=py, args=["-m", "agentflow.servers.compute"]),
        }
    return {"knowledge": build_knowledge_server(docs), "compute": build_compute_server()}


def make_approver(auto_yes: bool):
    async def approve(tool: str, args: dict) -> bool:
        if auto_yes:
            return True
        prompt = f"\n[approval] {tool}({json.dumps(args)}) — allow? [y/N] "
        answer = await anyio.to_thread.run_sync(lambda: input(prompt))
        return answer.strip().lower() in {"y", "yes"}

    return approve


async def ask(args) -> None:
    tracer = Tracer(sink=Path(args.trace) if args.trace else None)
    policy = Policy(approve=make_approver(args.yes), max_tool_calls=args.max_tool_calls)
    async with ToolHub(make_servers(args.docs, args.stdio), policy, tracer) as hub:
        result = await run(args.question, AnthropicLLM(args.model), hub, Config())
    print("\nPLAN")
    for wave_no, wave in enumerate(result.plan.waves(), 1):
        print(f"  wave {wave_no}: " + " ‖ ".join(f"{s.id} {s.goal}" for s in wave))
    print(f"\nANSWER  (critic: {result.review.verdict}, revisions: {result.revisions})\n")
    print(result.answer)
    print("\nRUN", json.dumps(result.summary, indent=2))


async def evaluate(args) -> None:
    tasks = [json.loads(x) for x in Path(args.tasks).read_text().splitlines() if x.strip()]
    llm = AnthropicLLM(args.model)
    rows = []
    for t in tasks:
        tracer = Tracer()
        async with ToolHub(make_servers(args.docs, False), Policy(), tracer) as hub:
            r = await run(t["question"], llm, hub)
        text = r.answer.lower()
        facts = [f for f in t["must_contain"] if f.lower() in text]
        used = set(r.summary["by_tool"])
        rows.append(
            {
                "id": t["id"],
                "facts": f"{len(facts)}/{len(t['must_contain'])}",
                "tools_ok": set(t.get("must_use", [])) <= used,
                "verdict": r.review.verdict,
                "tool_calls": r.summary["tool_calls"],
                "tokens": r.summary["input_tokens"] + r.summary["output_tokens"],
            }
        )
    print("| task | facts | required tools | critic | tool calls | tokens |\n|---|---|---|---|---|---|")
    for x in rows:
        print(
            f"| {x['id']} | {x['facts']} | {'yes' if x['tools_ok'] else 'no'} | "
            f"{x['verdict']} | {x['tool_calls']} | {x['tokens']} |"
        )


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="agentflow")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--yes", action="store_true", help="auto-approve destructive tools")
    a.add_argument("--max-tool-calls", type=int, default=40)
    a.add_argument("--trace")
    e = sub.add_parser("eval")
    e.add_argument("tasks", nargs="?", default="examples/tasks.jsonl")
    for sp in (a, e):
        sp.add_argument("--docs", default="examples/kb")
        sp.add_argument("--model")
        sp.add_argument("--stdio", action="store_true", help="run MCP servers as subprocesses")
    args = p.parse_args(argv)
    anyio.run(ask if args.cmd == "ask" else evaluate, args)


if __name__ == "__main__":
    main()
