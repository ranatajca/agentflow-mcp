import json
import sys

import pytest
from mcp import Client, StdioServerParameters

from agentflow.servers import build_compute_server, build_knowledge_server
from agentflow.servers.compute import safe_eval

pytestmark = pytest.mark.anyio


async def test_knowledge_search_finds_the_right_section(kb_path):
    async with Client(build_knowledge_server(kb_path)) as c:
        res = await c.call_tool("search_docs", {"query": "warranty period professional X200"})
        hits = res.structured_content["result"]
        assert hits[0]["doc_id"] == "policies/warranty"
        assert hits[0]["section"] == "Coverage periods"


async def test_read_doc_errors_are_helpful(kb_path):
    async with Client(build_knowledge_server(kb_path)) as c:
        res = await c.call_tool("read_doc", {"doc_id": "warranty"})
        assert res.is_error and "policies/warranty" in res.content[0].text
        res = await c.call_tool("read_doc", {"doc_id": "products/x200", "section": "nope"})
        assert res.is_error and "Pricing" in res.content[0].text


async def test_resources_prompts_and_annotations(kb_path):
    async with Client(build_knowledge_server(kb_path)) as c:
        index = json.loads((await c.read_resource("docs://index")).contents[0].text)
        assert "Pricing" in index["products/x200"]
        await c.call_tool("save_note", {"note_id": "n1", "content": "hello"})
        assert (await c.read_resource("notes://n1")).contents[0].text == "hello"
        prompt = await c.get_prompt("research_brief", {"question": "q?"})
        assert "cite" in prompt.messages[0].content.text
        tools = {t.name: t.annotations for t in (await c.list_tools()).tools}
        assert tools["delete_note"].destructive_hint and tools["search_docs"].read_only_hint


@pytest.mark.parametrize(
    "expr,value", [("412 * 0.8 + 79", 408.6), ("round(2349 * 0.15, 2)", 352.35), ("-2 ** 3", -8)]
)
def test_safe_eval(expr, value):
    assert safe_eval(expr) == pytest.approx(value)


@pytest.mark.parametrize(
    "expr", ["__import__('os')", "open('x')", "(1).real", "9 ** 99999", "lambda: 1", "a + 1"]
)
def test_safe_eval_rejects_unsafe(expr):
    with pytest.raises((ValueError, SyntaxError)):
        safe_eval(expr)


async def test_compute_tools():
    async with Client(build_compute_server()) as c:
        diff = await c.call_tool("date_diff", {"start": "2026-03-14", "end": "2026-10-20"})
        assert json.loads(diff.content[0].text)["days"] == 220
        due = await c.call_tool("add_days", {"start": "2026-11-02", "days": 30})
        assert due.structured_content["result"] == "2026-12-02"
        bad = await c.call_tool("calculate", {"expression": "__import__('os')"})
        assert bad.is_error


async def test_servers_work_over_real_stdio_transport(kb_path):
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "agentflow.servers.knowledge", "--docs", str(kb_path)]
    )
    async with Client(params) as c:
        names = {t.name for t in (await c.list_tools()).tools}
        assert {"search_docs", "read_doc", "save_note", "delete_note"} <= names
        res = await c.call_tool("search_docs", {"query": "restocking fee"})
        assert res.structured_content["result"][0]["doc_id"] == "policies/returns"
