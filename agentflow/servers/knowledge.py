"""`knowledge` MCP server: search and read a folder of Markdown docs, keep notes.

Exposes all three MCP primitives:
  tools      search_docs, read_doc, save_note, delete_note (destructive)
  resources  docs://index, notes://{note_id}
  prompts    research_brief

Run standalone over stdio:  python -m agentflow.servers.knowledge --docs examples/kb
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "with",
    ]
)


def _tok(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.lower()):
        if t in _STOP:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]  # cheap plural folding: "returns" matches "return"
        out.append(t)
    return out


class DocIndex:
    """BM25 over Markdown sections, so results point at the relevant part of a doc."""

    def __init__(self, root: Path, k1: float = 1.4, b: float = 0.75):
        self.root, self.k1, self.b = root, k1, b
        self.docs: dict[str, str] = {}
        self.sections: list[tuple[str, str, str]] = []  # (doc_id, heading, text)
        for path in sorted(root.glob("**/*.md")):
            doc_id = path.relative_to(root).with_suffix("").as_posix()
            text = path.read_text()
            self.docs[doc_id] = text
            for heading, body in _split_sections(text):
                self.sections.append((doc_id, heading, body))
        self.tfs = [Counter(_tok(h + " " + t)) for _, h, t in self.sections]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = sum(self.lens) / max(1, len(self.lens))
        df = Counter(term for tf in self.tfs for term in tf)
        n = len(self.tfs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int) -> list[dict]:
        terms = set(_tok(query))
        scored = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in terms & tf.keys():
                norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
                s += self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm)
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        out = []
        for s, i in scored[:k]:
            doc_id, heading, body = self.sections[i]
            out.append({"doc_id": doc_id, "section": heading, "score": round(s, 3), "snippet": body[:240]})
        return out


def _split_sections(text: str) -> list[tuple[str, str]]:
    parts, heading, buf = [], "intro", []
    for line in text.splitlines():
        if line.startswith("#"):
            if buf:
                parts.append((heading, "\n".join(buf).strip()))
            heading, buf = line.lstrip("#").strip(), []
        else:
            buf.append(line)
    if buf:
        parts.append((heading, "\n".join(buf).strip()))
    return [(h, b) for h, b in parts if b]


def build_server(docs_dir: str | Path) -> MCPServer:
    index = DocIndex(Path(docs_dir))
    notes: dict[str, str] = {}
    server = MCPServer(
        "knowledge",
        instructions=(
            "Search before you read. Cite doc_id and section for every fact. "
            "Notes are scratch space shared by all agents in a run."
        ),
    )

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
    def search_docs(query: str, k: int = 5) -> list[dict]:
        """BM25 search over documentation sections. Returns doc_id, section, score, snippet."""
        if not query.strip():
            raise ToolError("query must not be empty")
        return index.search(query, max(1, min(k, 20)))

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
    def read_doc(doc_id: str, section: str | None = None) -> str:
        """Read a full document, or only one section of it by heading."""
        if doc_id not in index.docs:
            close = [d for d in index.docs if doc_id.lower() in d.lower()]
            raise ToolError(f"unknown doc_id {doc_id!r}. Did you mean: {close or list(index.docs)[:5]}")
        if section is None:
            return index.docs[doc_id]
        for d, h, body in index.sections:
            if d == doc_id and h.lower() == section.lower():
                return f"## {h}\n{body}"
        heads = [h for d, h, _ in index.sections if d == doc_id]
        raise ToolError(f"no section {section!r} in {doc_id}. Sections: {heads}")

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False))
    def save_note(note_id: str, content: str) -> str:
        """Save or overwrite a scratch note other agents can read via notes://{note_id}."""
        if not re.fullmatch(r"[a-z0-9_-]{1,40}", note_id):
            raise ToolError("note_id must match [a-z0-9_-]{1,40}")
        notes[note_id] = content
        return f"saved {note_id} ({len(content)} chars)"

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True))
    def delete_note(note_id: str) -> str:
        """Permanently delete a note. Destructive: clients should require approval."""
        if notes.pop(note_id, None) is None:
            raise ToolError(f"no note {note_id!r}")
        return f"deleted {note_id}"

    @server.resource("docs://index", mime_type="application/json")
    def docs_index() -> dict:
        """Every doc_id with its section headings."""
        out: dict[str, list[str]] = {}
        for d, h, _ in index.sections:
            out.setdefault(d, []).append(h)
        return out

    @server.resource("notes://{note_id}")
    def read_note(note_id: str) -> str:
        """Contents of a saved note."""
        if note_id not in notes:
            raise ToolError(f"no note {note_id!r}")
        return notes[note_id]

    @server.prompt()
    def research_brief(question: str) -> str:
        """Prompt template for a cited research answer."""
        return (
            f"Answer the question using only the knowledge base.\nQuestion: {question}\n"
            "Search first, read the most relevant sections, and cite [doc_id#section] for every claim."
        )

    return server


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default="examples/kb")
    build_server(p.parse_args().docs).run("stdio")


if __name__ == "__main__":
    main()
