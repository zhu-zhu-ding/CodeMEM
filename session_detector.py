"""Session-level AST helpers for block extraction and diffs."""

from __future__ import annotations

import ast
import io
import textwrap
import tokenize
from typing import Any, Dict, List, Tuple


class SessionDetector:
    """Provides AST-based slicing and diff utilities for session memory."""

    KEY_NODES = {
        "FunctionDef",
        "AsyncFunctionDef",
        "ClassDef",
        "If",
        "For",
        "AsyncFor",
        "While",
        "With",
        "AsyncWith",
        "Try",
        "Assign",
        "AnnAssign",
        "Return",
        "Raise",
        "Import",
        "ImportFrom",
    }

    @staticmethod
    def extract_blocks(code: str) -> List[Dict[str, Any]]:
        """Slice a source string into structural blocks."""
        try:
            tree = ast.parse(code)
        except Exception:
            return []

        code_lines = code.splitlines()

        def slice_code(lines: List[int]) -> str:
            if not lines:
                return ""
            return "\n".join(code_lines[idx - 1] for idx in lines)

        def calc_span(node: ast.AST):
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if start is None or end is None:
                return None, None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = getattr(node, "body", [])
                if body:
                    first_body_line = getattr(body[0], "lineno", start)
                    end = max(start, first_body_line - 1)
            return start, end

        nodes: List[Dict[str, Any]] = []

        class BlockVisitor(ast.NodeVisitor):
            def visit(self, node):
                node_type = type(node).__name__
                if node_type in SessionDetector.KEY_NODES and hasattr(node, "lineno") and hasattr(
                    node, "end_lineno"
                ):
                    start, end = calc_span(node)
                    if start is not None and end is not None and end >= start:
                        nodes.append({"type": node_type, "start": start, "end": end})
                super().visit(node)

        BlockVisitor().visit(tree)
        nodes.sort(key=lambda n: (n["start"], n["end"]))

        merged_ranges: List[Dict[str, Any]] = []
        for node in nodes:
            node_len = node["end"] - node["start"] + 1
            if not merged_ranges or node["start"] > merged_ranges[-1]["end"]:
                merged_ranges.append(
                    {
                        "start": node["start"],
                        "end": node["end"],
                        "types": {node["type"]},
                        "best_range": (node["start"], node["end"]),
                        "best_len": node_len,
                    }
                )
            else:
                entry = merged_ranges[-1]
                entry["end"] = max(entry["end"], node["end"])
                entry["types"].add(node["type"])
                if node_len > entry["best_len"]:
                    entry["best_len"] = node_len
                    entry["best_range"] = (node["start"], node["end"])

        blocks: List[Dict[str, Any]] = []
        covered_lines = set()
        for entry in merged_ranges:
            best_start, best_end = entry["best_range"]
            full_range = range(entry["start"], entry["end"] + 1)
            covered_lines.update(full_range)
            block_code = slice_code(range(best_start, best_end + 1))
            if block_code.strip():
                blocks.append({"type": "+".join(sorted(entry["types"])), "block": block_code})

        remaining = [
            line
            for line in range(1, len(code_lines) + 1)
            if line not in covered_lines
            and code_lines[line - 1].strip()
            and not code_lines[line - 1].lstrip().startswith("#")
        ]
        if remaining:
            blocks.append({"type": "Code", "block": slice_code(remaining)})

        return blocks

    @staticmethod
    def _strip_comments(text: str) -> str:
        dedented = textwrap.dedent(text)
        sio = io.StringIO(dedented)
        result: List[str] = []
        last_line, last_col = 1, 0

        for tok_type, tok_str, (sline, scol), (eline, ecol), line in tokenize.generate_tokens(
            sio.readline
        ):
            if tok_type in (tokenize.COMMENT, tokenize.NL):
                continue
            if tok_type == tokenize.STRING and line.strip().startswith(("'''", '"""')) and scol == 0:
                continue
            if sline > last_line:
                result.append("\n" * (sline - last_line))
                last_col = 0
            if scol > last_col:
                result.append(" " * (scol - last_col))
            result.append(tok_str)
            last_line, last_col = eline, ecol

        cleaned = "".join(result)
        cleaned_lines = [line for line in cleaned.splitlines() if line.strip()]
        return "\n".join(cleaned_lines).strip()

    @classmethod
    def diff_blocks(cls, previous: str, current: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return AST-level differences between two snippets, ignoring comments."""
        blocks_a = cls.extract_blocks(previous)
        blocks_b = cls.extract_blocks(current)
        for blk in blocks_a:
            blk["block"] = cls._strip_comments(blk["block"])
        for blk in blocks_b:
            blk["block"] = cls._strip_comments(blk["block"])
        set_a = {(blk["type"], blk["block"]) for blk in blocks_a}
        set_b = {(blk["type"], blk["block"]) for blk in blocks_b}

        removed = [
            {"type": blk["type"], "block": blk["block"]}
            for blk in blocks_a
            if (blk["type"], blk["block"]) not in set_b
        ]
        added = [
            {"type": blk["type"], "block": blk["block"]}
            for blk in blocks_b
            if (blk["type"], blk["block"]) not in set_a
        ]
        return added, removed


if __name__ == "__main__":
    baseline = """
def foo(value):
    # returns double
    return value * 2
"""
    updated = """
def foo(value):
    total = value * 2
    return total
"""
    print("Blocks:", SessionDetector.extract_blocks(baseline))
    added, removed = SessionDetector.diff_blocks(baseline, updated)
    print("Added:", added)
    print("Removed:", removed)
