
from __future__ import annotations

import ast
import builtins
import textwrap
from typing import Dict, Iterable, List, Optional, Set

class ExternalAPICallExtractor(ast.NodeVisitor):
    def __init__(self, source_code):
        self.calls = []
        self.defined_functions = set()
        self.defined_classes = set()
        self.builtins = set(dir(builtins))

        tree = ast.parse(source_code)
        self._collect_definitions(tree)
        self.visit(tree)

    def _collect_definitions(self, tree):
        """Collect function and class definitions inside the file."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self.defined_functions.add(node.name)
            elif isinstance(node, ast.ClassDef):
                self.defined_classes.add(node.name)

    def visit_Call(self, node):
        func_name = self._resolve_call_name(node.func)

        if func_name is not None:
            root = func_name.split(".")[0]
            if root not in self.builtins and root not in self.defined_functions:
                self.calls.append(func_name)

        self.generic_visit(node)

    def _resolve_call_name(self, node):
        """Resolve full call name including attribute chains."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._attr_to_str(node)
        return None

    def _attr_to_str(self, node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))


class ContextSelector:
    """Encapsulates helpers for selecting relevant repo contexts."""

    @staticmethod
    def extract_external_api_calls(source_code: str) -> List[str]:
        extractor = ExternalAPICallExtractor(source_code)
        return extractor.calls

    @staticmethod
    def match_external_api(namespace: str, api_list: Iterable[str]) -> bool:
        last_segment = namespace.split(".")[-1]
        for api in api_list:
            if "." not in api:
                if last_segment == api:
                    return True
            else:
                if api in namespace:
                    return True
        return False

    @classmethod
    def select_repo_contexts_from_result(
        cls,
        full_repo_contexts: Dict[str, Dict[str, Any]],
        repo_contexts_info: Iterable[Dict[str, Any]],
        contexts: List[str],
        generated_code: str,
    ) -> List[Dict[str, Any]]:
        """
        Filter repo contexts according to the APIs touched by the generated code.
        """
        repo_contexts = list(repo_contexts_info)
        if not generated_code.strip():
            return repo_contexts

        usage = cls.analyze_code_apis(generated_code)
        generated_calls: Set[str] = set(usage["external_calls"])
        if not generated_calls:
            return repo_contexts

        contexts_set: Optional[Set[str]] = {ctx for ctx in contexts if isinstance(ctx, str)} if contexts else None

        selected: List[Dict[str, Any]] = []
        cache: Dict[str, Dict[str, List[str]]] = {}
        for info in repo_contexts:
            if not isinstance(info, dict):
                continue
            namespace = info.get("namespace")
            if not namespace:
                continue
            node = full_repo_contexts.get(namespace)
            if not isinstance(node, dict):
                continue
            code = node.get("code")
            if not isinstance(code, str) or not code.strip():
                continue
            if contexts_set is not None and code not in contexts_set:
                continue
            if code not in cache:
                cache[code] = cls.analyze_code_apis(code)
            ctx_usage = cache[code]
            ctx_apis = set(ctx_usage["provided_apis"] + ctx_usage["external_calls"])
            if not ctx_apis:
                continue
            if generated_calls & ctx_apis:
                selected.append(info)
        return selected

    @staticmethod
    def analyze_code_apis(source: str) -> Dict[str, List[str]]:
        """
        Parse source code and return:
          - 'provided_apis': exported APIs (ClassName, ClassName.method, function_name)
          - 'external_calls': external API calls discovered in code.
        """
        try:
            tree = ast.parse(textwrap.dedent(source))
        except Exception:
            return {"provided_apis": [], "external_calls": []}

        provided_apis: List[str] = []
        defined_functions: Set[str] = set()
        defined_classes: Set[str] = set()

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                defined_functions.add(node.name)
                provided_apis.append(node.name)
            elif isinstance(node, ast.ClassDef):
                defined_classes.add(node.name)
                provided_apis.append(node.name)
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        provided_apis.append(f"{node.name}.{item.name}")

        import_map: Dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    asname = alias.asname or alias.name.split(".")[0]
                    import_map[asname] = alias.name
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ""
                for alias in node.names:
                    local = alias.asname or alias.name
                    if module:
                        import_map[local] = f"{module}.{alias.name}"
                    else:
                        import_map[local] = alias.name

        builtin_names = set(dir(builtins))

        def resolve_attr(node: ast.AST) -> str:
            parts: List[str] = []
            cur = node
            while True:
                if isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                    continue
                elif isinstance(cur, ast.Name):
                    parts.append(cur.id)
                elif isinstance(cur, ast.Call):
                    break
                else:
                    break
                break

            parts = list(reversed(parts))
            if not parts:
                return ""

            root = parts[0]
            if root in import_map:
                fq = import_map[root]
                fq_parts = fq.split(".")
                out_parts = fq_parts + parts[1:]
                return ".".join(out_parts)
            else:
                return ".".join(parts)

        external_calls_set: Set[str] = set()

        class CallVisitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call):
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                    if name in import_map:
                        callee_name = import_map[name]
                    else:
                        callee_name = name
                elif isinstance(node.func, ast.Attribute):
                    callee_name = resolve_attr(node.func) or _attr_to_str_fallback(node.func)
                else:
                    callee_name = _attr_to_str_fallback(node.func)

                if callee_name:
                    root = callee_name.split(".")[0]
                    if root in builtin_names:
                        pass
                    elif root in defined_functions or root in defined_classes:
                        pass
                    else:
                        external_calls_set.add(callee_name)

                self.generic_visit(node)

        def _attr_to_str_fallback(n: ast.AST) -> str:
            if isinstance(n, ast.Attribute):
                parts = []
                cur = n
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                parts = list(reversed(parts))
                return ".".join(parts)
            elif isinstance(n, ast.Name):
                return n.id
            else:
                return type(n).__name__

        CallVisitor().visit(tree)
        external_calls = sorted(external_calls_set)
        return {"provided_apis": sorted(provided_apis), "external_calls": external_calls}


if __name__ == "__main__":
    sample_repo = {
        "utils.json_tools": {
            "code": "import json\n\ndef to_json(data):\n    return json.dumps(data)\n"
        },
        "utils.math_tools": {"code": "def add(a, b):\n    return a + b\n"},
        "utils.yaml_tools": {
            "code": "import yaml\n\ndef to_yaml(data):\n    return yaml.dump(data)\n"
        },
    }
    repo_info = [
        {"namespace": "utils.json_tools"},
        {"namespace": "utils.math_tools"},
        {"namespace": "utils.yaml_tools"},
    ]
    contexts = [node["code"] for node in sample_repo.values()]
    generated_code = """
import yaml

def serialize(data):
    return yaml.dump(data)
"""

    result = ContextSelector.select_repo_contexts_from_result(
        sample_repo,
        repo_info,
        contexts,
        generated_code,
    )
    print("Demo selected namespaces:", [ctx["namespace"] for ctx in result])
