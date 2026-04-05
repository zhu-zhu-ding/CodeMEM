"""Primary interface for Codemem's repository and session memory orchestration."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import json_repair

from context_selector import ContextSelector
from prompts import get_prompt
from retriever import EmbeddingFactory, Retriever
from session_detector import SessionDetector
from utils import extract_json, extract_python


def _repair_json(llm, prompt: str, max_retries: int = 2) -> Optional[Dict[str, Any]]:
    """Call the LLM and coerce the response into JSON."""
    current_prompt = prompt
    for attempt in range(max_retries):
        response = llm.generate([{"role": "user", "content": current_prompt}])
        raw_json = extract_json(response)
        try:
            return json_repair.loads(raw_json)
        except Exception:
            if attempt + 1 < max_retries:
                current_prompt = (
                    f"{current_prompt}\n\nYour previous output was invalid JSON. "
                    "Please re-output strictly valid JSON only!"
                )
            else:
                return None
    return None


@dataclass
class SessionMemoryEntry:
    """Stores one turn of session memory."""

    id: int
    instruction: str
    code: str
    diff: str
    ast_nodes: List[Dict[str, Any]]
    diff_nodes: Dict[str, List[Dict[str, Any]]]
    pre_instruction: List[str] = field(default_factory=list)
    state_links: List[int] = field(default_factory=list)
    insight: Optional[str] = None


@dataclass
class SessionMemorySequence:
    """Tracks memory entries for a single namespace."""

    namespace: str
    memory: List[SessionMemoryEntry] = field(default_factory=list)
    last_entry_id: int = 0

    def latest(self) -> Optional[SessionMemoryEntry]:
        if not self.memory:
            return None
        return self.memory[self.last_entry_id]

    def append(self, entry: SessionMemoryEntry) -> SessionMemoryEntry:
        self.memory.append(entry)
        self.last_entry_id = entry.id
        return entry


class ContextMemory:
    """Manages repository-level retrieval and pruning."""

    def __init__(self, llm, full_repo_contexts: Dict[str, Dict[str, Any]]) -> None:
        self.llm = llm
        self.full_repo_contexts = full_repo_contexts
        self.repo_contexts_info = [
            {k: v for k, v in (node.get("memory") or {}).items() if k != "called_funcs"}
            for node in full_repo_contexts.values()
        ]
        self.repo_mem_info: List[Dict[str, Any]] = []
        repo_codes = [ctx.get("code", "") for ctx in full_repo_contexts.values()]
        self.repo_retriever = Retriever(repo_codes, k_bm25=20, k_rerank=20)
        self.retry = 0

    def refresh(self, instruction: str, instructions_history: List[str]) -> List[str]:
        """Use the retriever to refresh context memory when needed."""
        prompt = get_prompt("if_repo_mem_update").format(
            instructions="\n".join(instructions_history),
            repo_context_info=json.dumps(self.repo_mem_info, ensure_ascii=False),
        )
        decision = _repair_json(self.llm, prompt) or {}
        should_add = decision.get("mode") == "ADD" and self.retry < 5
        if should_add:
            top_indices = self.repo_retriever.retrieve(instruction)
            start = self.retry * 5
            for idx in top_indices[start : start + 5]:
                if 0 <= idx < len(self.repo_contexts_info):
                    self.repo_mem_info.append(self.repo_contexts_info[idx])
            self.retry += 1
        return self._current_context_blocks()

    def update(self, generated_code: str) -> None:
        """Prune repo contexts based on the APIs touched by the generated code."""
        contexts = self._current_context_blocks()
        filtered = ContextSelector.select_repo_contexts_from_result(
            self.full_repo_contexts,
            self.repo_mem_info,
            contexts,
            generated_code,
        )
        self.repo_mem_info = filtered

    def _current_context_blocks(self) -> List[str]:
        return [
            self.full_repo_contexts[data["namespace"]]["code"]
            for data in self.repo_mem_info
            if isinstance(data, dict) and data.get("namespace") in self.full_repo_contexts
        ]


class SessionMemory:
    """Tracks turn-by-turn history for namespaces."""

    def __init__(self, llm) -> None:
        self.llm = llm
        self.embedding_llm = EmbeddingFactory("closed", model="text-embedding-3-small")
        self.memory_blocks: Dict[str, SessionMemorySequence] = {}
        self.last_namespace = ""

    def record(self, instruction: str, answer: str, next_instruction: Optional[str]) -> None:
        namespace = self._extract_namespace(instruction, answer)
        self.last_namespace = namespace
        ast_nodes = SessionDetector.extract_blocks(answer)

        namespace_memory = self.memory_blocks.setdefault(namespace, SessionMemorySequence(namespace))
        last_entry = namespace_memory.latest()
        if not last_entry:
            entry = SessionMemoryEntry(
                id=0,
                instruction=instruction,
                code=answer,
                diff="",
                ast_nodes=ast_nodes,
                diff_nodes={"added": [], "removed": []},
            )
            namespace_memory.append(entry)
        else:
            added_nodes, removed_nodes = SessionDetector.diff_blocks(last_entry.code, answer)
            diff = self._compute_code_diff(last_entry.code, answer)
            block_id = len(namespace_memory.memory)
            entry = SessionMemoryEntry(
                id=block_id,
                instruction=instruction,
                code=answer,
                diff=diff,
                ast_nodes=ast_nodes,
                diff_nodes={"added": added_nodes, "removed": removed_nodes},
                pre_instruction=[],
                state_links=[],
            )
            for existing in namespace_memory.memory:
                try:
                    similarity = self.embedding_llm.similarity(instruction, existing.instruction)
                except Exception:
                    similarity = 0.0
                if similarity >= 0.95:
                    existing.state_links.append(block_id)
                    entry.state_links.append(existing.id)
                else:
                    entry.pre_instruction.append(existing.instruction)
            namespace_memory.append(entry)

        insight = self._generate_turn_memory(instruction, answer, next_instruction)
        if insight:
            namespace_memory.latest().insight = insight

    def render(self) -> str:
        if not self.memory_blocks:
            return ""
        namespace = self.last_namespace or next(iter(self.memory_blocks))
        namespace_memory = self.memory_blocks.get(namespace)
        if not namespace_memory:
            return ""
        last_entry = namespace_memory.latest()
        if not last_entry:
            return ""
        references = [
            f"##Reference Code for {mem.instruction}##\n{mem.code}\n##Experience##\n{mem.insight or ''}"
            for mem in namespace_memory.memory
            if mem.id in last_entry.state_links
        ]
        references.append(
            f"##Reference Code for All History Instruction##\n{last_entry.code}"
            f"##Experience##\n{last_entry.insight or ''}"
        )
        history = last_entry.pre_instruction or []
        sections: List[str] = []
        if references:
            sections.append("##Related Code Experiences##:\n" + "\n".join(references))
        if history:
            sections.append("##History Instructions##:\n" + "\n".join(history))
        return "\n\n".join(sections)

    def _extract_namespace(self, instruction: str, answer: str) -> str:
        namespace = self._extract_namespace_from_code(answer)
        if namespace:
            return namespace
        match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", instruction or "")
        if match:
            return f"{match.group(1)}(...)"
        if self.last_namespace:
            return self.last_namespace
        fallback = hashlib.md5((instruction or answer).encode("utf-8")).hexdigest()[:8]
        return f"block_{fallback}"

    def _extract_namespace_from_code(self, code_text: str) -> Optional[str]:
        try:
            tree = ast.parse(code_text)
        except Exception:
            return None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                args = [arg.arg for arg in node.args.args]
                return f"{node.name}({', '.join(args)})"
            if isinstance(node, ast.ClassDef):
                return f"class {node.name}"
        return None

    def _generate_turn_memory(self, instruction: str, code: str, next_instruction: Optional[str]) -> str:
        if not instruction or not code:
            return ""
        if next_instruction:
            messages = [
                {"role": "system", "content": get_prompt("memory_update_system")},
                {
                    "role": "user",
                    "content": get_prompt("memory_update_user").format(
                        instruction=instruction,
                        answer=code,
                        new_instruction=next_instruction,
                    ),
                },
            ]
        else:
            messages = [
                {"role": "system", "content": get_prompt("memory_update_system_without_next")},
                {
                    "role": "user",
                    "content": get_prompt("memory_update_user_without_next").format(
                        instruction=instruction,
                        answer=code,
                    ),
                },
            ]
        response = self.llm.generate(messages)
        try:
            parsed = json.loads(extract_json(response))
            return parsed.get("memory", "").strip()
        except Exception:
            return response.strip()

    def _compute_code_diff(self, previous: str, current: str) -> str:
        diff = difflib.unified_diff(
            previous.splitlines(),
            current.splitlines(),
            fromfile="previous_answer",
            tofile="current_answer",
            lineterm="",
        )
        diff_text = "\n".join(diff).strip()
        return diff_text or "(No code changes)"

    @staticmethod
    def _dict_list_intersects(list1: List[Dict[str, Any]], list2: List[Dict[str, Any]]) -> bool:
        if not list1 or not list2:
            return False
        set1 = {json.dumps(item, sort_keys=True) for item in list1}
        set2 = {json.dumps(item, sort_keys=True) for item in list2}
        return bool(set1 & set2)

    def _resolve_conflict_links(
        self,
        namespace_memory: SessionMemorySequence,
        added_nodes: List[Dict[str, Any]],
        removed_nodes: List[Dict[str, Any]],
    ) -> List[int]:
        links: List[int] = []
        for entry in namespace_memory.memory:
            entry_added = entry.diff_nodes.get("added", []) if entry.diff_nodes else []
            entry_removed = entry.diff_nodes.get("removed", []) if entry.diff_nodes else []
            if self._dict_list_intersects(added_nodes, entry_removed) or self._dict_list_intersects(
                removed_nodes,
                entry_added,
            ):
                links.append(entry.id)
        return links

    def _format_conflict_memory(self, references: List[SessionMemoryEntry], latest: SessionMemoryEntry) -> str:
        pre_instructions = list(latest.pre_instruction or [])
        if latest.instruction:
            pre_instructions.append(latest.instruction)
        history_block = "\n".join(instr.strip() for instr in pre_instructions if instr.strip())

        formatted_reference: List[str] = []
        for ref in references:
            ref_code = (ref.code or "").strip()
            ref_insight = (ref.insight or "").strip()
            formatted_reference.append(
                f"##Reference Code for {ref.instruction}##\n{ref_code}\n##Experience##\n{ref_insight}"
            )
        formatted_reference.append(
            f"##Reference Code for All History Instruction##\n{latest.code}"
            f"##Experience##\n{latest.insight or ''}"
        )
        reference_block = "\n".join(formatted_reference)

        sections: List[str] = []
        if reference_block:
            sections.append(f"##Related Code Experiences##:\n{reference_block}")
        if history_block:
            sections.append(f"##History Instructions##:\n{history_block}")
        return "\n\n".join(sections).strip()

    def _rewrite_answer_with_conflicts(
        self,
        instruction: str,
        current_answer: str,
        conflicts: List[SessionMemoryEntry],
        pre_instructions: List[str],
    ) -> str:
        if not conflicts or not current_answer:
            return current_answer
        conflict_sections: List[str] = []
        for idx, conflict in enumerate(conflicts, 1):
            conflict_sections.append(
                f"###Conflicting Instruction {idx}###\n"
                f"{conflict.instruction}\n"
                f"###Conflicting Answer {idx}###\n"
                f"{conflict.code or conflict.diff}\n"
            )
        conflict_prompt = "\n".join(conflict_sections)
        current_block_prompt = (
            "###Current Instruction###\n"
            f"{instruction}\n"
            "###Current Answer###\n"
            f"{current_answer}\n"
        )
        history = "\n".join(instr.strip() for instr in pre_instructions if instr.strip())
        history_prompt = f"###Instruction History###\n{history}\n" if history else ""
        user_prompt = (
            "The current answer may conflict with previous instruction-answer pairs. "
            "Regenerate the answer so it satisfies the current instruction while handling conflicting requirements. "
            "Only output the corrected answer/code.\n\n"
            f"{conflict_prompt}\n"
            f"{history_prompt}"
            f"{current_block_prompt}"
        )
        messages = [{"role": "user", "content": user_prompt}]
        try:
            response = self.llm.generate(messages)
        except Exception:
            return current_answer
        updated_answer = extract_python(response).strip()
        if not updated_answer:
            updated_answer = response.strip()
        return updated_answer or current_answer

    def handle_conflicts(
        self,
        instruction: str,
        repo_contexts: List[str],
        prompt: str,
        generation: str,
        prompt_builder: Callable[[str, List[str], Optional[str]], str],
    ) -> Tuple[str, str]:
        if not self.memory_blocks:
            return prompt, generation
        namespace = self.last_namespace or next(iter(self.memory_blocks))
        namespace_memory = self.memory_blocks.get(namespace)
        if not namespace_memory:
            return prompt, generation
        latest_entry = namespace_memory.latest()
        if not latest_entry or not latest_entry.code:
            return prompt, generation
        added_nodes, removed_nodes = SessionDetector.diff_blocks(latest_entry.code, generation)
        conflict_links = self._resolve_conflict_links(namespace_memory, added_nodes, removed_nodes)
        if not conflict_links:
            return prompt, generation

        print("------------------Resolving memory conflicts---------------------")
        print(latest_entry.code)
        print(generation)

        conflict_map = {entry.id: entry for entry in namespace_memory.memory if entry.id in conflict_links}
        reference_entries = [
            entry for entry in namespace_memory.memory if entry.id in (latest_entry.state_links or [])
        ]
        for entry in reference_entries:
            conflict_map.setdefault(entry.id, entry)
        conflicting_entries = list(conflict_map.values())
        print(conflicting_entries)

        conflict_memory = self._format_conflict_memory(conflicting_entries, latest_entry)
        prompt = prompt_builder(instruction, repo_contexts, memory_override=conflict_memory)
        response = self.llm.generate([{"role": "user", "content": prompt}])
        generation = extract_python(response)

        pre_instructions = list(latest_entry.pre_instruction or [])
        if latest_entry.instruction:
            pre_instructions.append(latest_entry.instruction)
        generation = self._rewrite_answer_with_conflicts(
            instruction,
            generation,
            conflicting_entries,
            pre_instructions,
        )
        return prompt, generation


class Codemem:
    """Keeps track of repository contexts and session memories used when coding."""

    def __init__(
        self,
        full_repo_contexts: Dict[str, Dict[str, Any]],
        llm,
    ) -> None:
        self.llm = llm
        self.context_memory = ContextMemory(self.llm, full_repo_contexts)
        self.session_memory = SessionMemory(self.llm)
        self.instruction_history: List[str] = []

    def forward(self, instruction: str) -> Tuple[str, str]:
        """Build the meta prompt and request a generation."""
        self.instruction_history.append(instruction)
        repo_contexts = self.context_memory.refresh(instruction, self.instruction_history)
        prompt = self._build_structured_prompt(instruction, repo_contexts)
        response = self.llm.generate([{"role": "user", "content": prompt}])
        generation = extract_python(response)
        prompt, generation = self.session_memory.handle_conflicts(
            instruction,
            repo_contexts,
            prompt,
            generation,
            self._build_structured_prompt,
        )
        return prompt, generation

    def backward(self, instruction: str, answer: str, next_instruction: Optional[str] = None) -> None:
        """Update repository, experience, and session memories after a generation."""
        self.context_memory.update(answer)
        self.session_memory.record(instruction, answer, next_instruction)

    def _build_structured_prompt(
        self,
        instruction: str,
        repo_contexts: List[str],
        memory_override: Optional[str] = None,
    ) -> str:
        memory_blocks = memory_override if memory_override is not None else self.session_memory.render()
        repo_context_text = "\n".join(
            f"###Context {idx}###\n{ctx}\n" for idx, ctx in enumerate(repo_contexts, start=1)
        )
        return get_prompt("meta_prompt_memory").format(
            memory_blocks=memory_blocks,
            repo_context=repo_context_text,
            instruction=instruction,
        )
