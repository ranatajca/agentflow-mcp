"""`compute` MCP server: exact arithmetic and calendar math, so agents never do it in their head.

The expression evaluator walks a whitelisted Python AST: no eval(), no names,
no attribute access, no calls except a fixed set of math functions.
"""

from __future__ import annotations

import ast
import math
import operator
from datetime import date, timedelta

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "round": round,
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "ceil": math.ceil,
    "floor": math.floor,
}
MAX_POW = 1000


def safe_eval(expr: str) -> float:
    def ev(node: ast.AST):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > MAX_POW:
                raise ValueError("exponent too large")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = _FUNCS.get(node.func.id)
            if fn and not node.keywords:
                return fn(*[ev(a) for a in node.args])
        raise ValueError(f"unsupported syntax: {ast.dump(node)[:60]}")

    if len(expr) > 200:
        raise ValueError("expression too long")
    return ev(ast.parse(expr, mode="eval"))


def build_server() -> MCPServer:
    server = MCPServer("compute", instructions="Use these tools for every number and date.")

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
    def calculate(expression: str) -> float:
        """Evaluate an arithmetic expression, e.g. '(1299 * 0.85) - 50' or 'round(sqrt(2), 3)'."""
        try:
            return safe_eval(expression)
        except (ValueError, SyntaxError, ZeroDivisionError, TypeError, OverflowError) as e:
            raise ToolError(f"cannot evaluate {expression!r}: {e}") from e

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
    def date_diff(start: str, end: str) -> dict:
        """Days between two ISO dates (YYYY-MM-DD). Positive if end is after start."""
        try:
            a, b = date.fromisoformat(start), date.fromisoformat(end)
        except ValueError as e:
            raise ToolError(str(e)) from e
        return {"days": (b - a).days, "weeks": round((b - a).days / 7, 2)}

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
    def add_days(start: str, days: int) -> str:
        """ISO date that is `days` after `start` (negative goes back)."""
        try:
            return (date.fromisoformat(start) + timedelta(days=days)).isoformat()
        except ValueError as e:
            raise ToolError(str(e)) from e

    return server


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()
