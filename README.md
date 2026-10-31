# agentflow-mcp

A multi-agent workflow built on the **Model Context Protocol**: a planner breaks a question into a dependency graph, executor agents run the independent steps **in parallel** with tools served by MCP servers, a synthesizer writes a cited answer, and a critic (plus a code-level grounding check) sends it back for revision until it's supported by evidence.

It ships two MCP servers, a multi-server client hub with a human-approval gate for destructive tools, budgets and timeouts, structured tracing, and a 23-test suite that runs the whole system offline.

```
question
   │
planner ── invalid JSON / unknown deps / cycle ──► repair with exact errors (≤2)
   │ Plan (validated DAG)
   ▼
wave 1:  executor s1  ‖  executor s2        ← concurrent tool-use loops
wave 2:  executor s3 (sees s1, s2 output)   ← only its dependencies, only its allowed tools
   │            │
   │       ToolHub ── policy: read-only ✓ · destructive → human approval · shared call budget
   │            ├── knowledge MCP server   search_docs · read_doc · save_note · delete_note
   │            └── compute   MCP server   calculate · date_diff · add_days
   ▼
synthesizer ──► grounding check + critic ── revise ──► synthesizer (≤ max_revisions)
   │
Result(answer, plan, steps, review, trace)
```

## Quickstart

```bash
pip install -e ".[dev]"
pytest -q                     # 23 tests, offline, includes a real stdio-transport test

export ANTHROPIC_API_KEY=...
agentflow ask "An X200 was delivered on 2026-03-14 and its LiDAR failed on 2026-10-20. \
Is it under warranty, and what would a Loyalty member pay out of warranty?" --trace run.jsonl

agentflow ask "..." --stdio   # run both MCP servers as subprocesses over stdio
agentflow eval                # score examples/tasks.jsonl: facts found, tools used, tokens
```

The example knowledge base in `examples/kb/` is the documentation of a fictional robotics company. It's small enough to audit, and the questions need multi-hop lookup plus exact arithmetic and date math.

## Use the servers from any MCP host

Both servers are standard MCP servers, so they work with Claude Desktop, IDE agents, or any other host:

```json
{
  "mcpServers": {
    "knowledge": { "command": "agentflow-knowledge", "args": ["--docs", "/path/to/markdown"] },
    "compute":   { "command": "agentflow-compute" }
  }
}
```

The `knowledge` server uses all three MCP primitives: **tools** (with `readOnlyHint` / `destructiveHint` annotations), **resources** (`docs://index`, `notes://{note_id}`), and a **prompt** (`research_brief`).

## Design decisions

**Plans are validated data, not prose.** The planner must return JSON that parses into a Pydantic `Plan`: unique step ids, known dependencies, no cycles (Kahn's algorithm), and only tools that exist. Failures go back to the model with the exact error ("dependency cycle among steps ['s1', 's2']") instead of a blind retry.

**Parallelism comes from the DAG.** Steps are grouped into dependency waves, and each wave runs concurrently under an `anyio.CapacityLimiter`. Inside a step, multiple tool calls in one model turn also run concurrently. Each executor only sees its dependencies' outputs and its allow-listed tools, so context stays small and failures stay local.

**Safety lives in the client, driven by server metadata.** `ToolHub` reads each tool's MCP annotations. Read-only tools run; destructive ones go through an async `approve(tool, args)` callback (a terminal prompt in the CLI). A denial becomes a tool result the model can reason about, not a crash. Allow and deny lists override per tool.

**Hard limits everywhere.** There's a shared tool-call budget across all concurrent agents (lock-protected), a per-step iteration cap that forces a tool-free summary, a per-call timeout, result truncation, and a whole-run timeout. A crashing step is recorded as `failed` and its dependents are skipped. The rest of the run continues.

**Two-layer verification.** Before the LLM critic runs, `grounding_issues()` checks deterministically that every `[doc_id#section]` citation refers to a document an agent actually retrieved. A hallucinated citation forces a revision even if the critic would have passed it. An unparseable critic reply never blocks an answer.

**Errors are observations.** Tool errors are written to help the model recover (`unknown doc_id 'warranty'. Did you mean: ['policies/warranty']`) and are returned as `is_error` tool results.

**Everything is traced.** Every LLM turn (with token counts), tool call (args, latency, ok, preview), approval, plan repair, wave and revision is an event. You can stream it to JSONL with `--trace`, or read `result.summary`.

## Testing an agent system without an API key

`ScriptedLLM` routes each request by agent (`planner`, `executor:s2`, `critic`, …) to a queue of turns, which can also be functions of the request. So the tests drive the **real** orchestrator against the **real** MCP servers and assert on behaviour:

| test | what it proves |
|---|---|
| `test_full_run_plans_parallelises_and_answers` | wave grouping, dependency context, per-step tool allow-lists, concurrent tool calls, exact compute results |
| `test_planner_repairs_an_invalid_dag` | a cyclic plan in a code fence is rejected and repaired using the cycle error |
| `test_critic_grounding_forces_revision` | a fabricated citation triggers revision even when the LLM critic says "pass" |
| `test_destructive_tool_needs_approval` | non-destructive writes run, destructive ones ask, denial is handled gracefully |
| `test_shared_tool_budget_stops_runaway_agents` | the budget halts a looping agent at exactly N calls |
| `test_iteration_cap_forces_summary` | the final forced turn is offered zero tools |
| `test_failed_dependency_skips_downstream` | an exception in one step doesn't take down the run |
| `test_servers_work_over_real_stdio_transport` | the servers run as real subprocesses over stdio |
| `test_safe_eval_rejects_unsafe` | the calculator refuses imports, attribute access, lambdas and huge exponents |

## Layout

```
agentflow/
  servers/knowledge.py  MCP server: section-level BM25 search, docs, notes, resources, prompt
  servers/compute.py    MCP server: AST-whitelisted calculator, date math
  hub.py                multi-server MCP client, namespacing, policy/approval, budget, timeouts
  plan.py               Plan/Step schema, DAG validation, wave scheduling
  agents.py             planner, executor loop, synthesizer, critic, grounding check
  workflow.py           orchestrator
  llm.py                Anthropic adapter + ScriptedLLM for tests
  trace.py              event log and run summary
  cli.py                `agentflow ask` / `agentflow eval`
examples/kb/            sample knowledge base
examples/tasks.jsonl    eval tasks with required facts and tools
tests/
```

## Extending

- **Add a server:** pass another `MCPServer` or `StdioServerParameters` to `ToolHub({...})`. Its tools show up namespaced as `name__tool`, and the planner sees them.
- **Swap the model:** anything with `async __call__(Request) -> Turn` works as an `LLM`.
- **Remote servers:** the MCP client also supports streamable HTTP. Point the hub at a URL instead of a subprocess.

MIT licensed.
