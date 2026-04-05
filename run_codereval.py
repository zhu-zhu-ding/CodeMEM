"""Command-line runner for Codereval-style datasets using Codemem."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Set, Tuple

from tqdm import tqdm

from codemem import Codemem
from llm_factory import LLMFactory
from utils import read_json, save_json


def _load_existing(output_path: str) -> Dict[str, Dict[str, Any]]:
    """Load existing results so the pipeline can resume."""
    if not os.path.exists(output_path):
        return {}
    loaded_results = read_json(output_path, True)
    if not isinstance(loaded_results, list):
        return {}
    existing: Dict[str, Dict[str, Any]] = {}
    for item in loaded_results:
        if not isinstance(item, dict):
            continue
        sample_id = (
            item.get("_id")
            or item.get("question_id")
            or item.get("project_path")
            or item.get("project")
        )
        if sample_id:
            existing[str(sample_id)] = item
    return existing


def _instantiate_repo_memory(full_contexts, model_type: str) -> Tuple[Codemem, Any]:
    """Instantiate Codemem and expose the driving LLM client."""
    llm_client = LLMFactory(llm_type="closed", model=model_type)
    repo_mem = Codemem(full_contexts, llm_client)
    return repo_mem, llm_client


def _build_instruction(sample: Dict[str, Any]) -> str:
    """Create a human-readable instruction for the current Codereval item."""
    base = sample.get("input")
    if isinstance(base, str) and base.strip():
        return base.strip()

    pieces: List[str] = []
    name = sample.get("name")
    if isinstance(name, str) and name.strip():
        pieces.append(f"Please implement the Python function `{name.strip()}`.")

    label = sample.get("human_label")
    if isinstance(label, str) and label.strip():
        pieces.append(f"Task description:\n{label.strip()}")

    doc = sample.get("docstring")
    if isinstance(doc, str) and doc.strip():
        pieces.append(f"Docstring:\n{doc.strip()}")

    oracle = sample.get("oracle_context")
    if isinstance(oracle, str) and oracle.strip():
        pieces.append(f"Available APIs and context:\n{oracle.strip()}")

    return "\n\n".join(pieces).strip() or "Please implement the requested functionality."


def _project_folder_name(project_value: str, separator: str) -> str:
    """Normalize repo folder names (Codereval replaces '/' with '---')."""
    return project_value.replace("/", separator)


def process_single_sample(
    idx: int,
    raw_sample: Dict[str, Any],
    root_path: str,
    repo_graph_name: str,
    project_separator: str,
    model_type: str,
) -> Dict[str, Any]:
    """Process a single Codereval sample."""
    sample = json.loads(json.dumps(raw_sample))
    start_time = time.time()
    try:
        sample["tokens"] = {"session": [], "pipeline": []}
        sample["result"] = []
        prompts: List[str] = []

        project_value = (
            sample.get("project")
            or sample.get("project_path")
            or sample.get("repo_name")
            or sample.get("repo")
        )
        if not isinstance(project_value, str) or not project_value.strip():
            raise ValueError("Sample is missing a valid project identifier.")
        project_folder = _project_folder_name(project_value.strip(), project_separator)
        repo_path = os.path.join(root_path, project_folder)
        graph_path = os.path.join(repo_path, repo_graph_name)
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Repository graph not found: {graph_path}")

        full_contexts = read_json(graph_path, True)
        if not isinstance(full_contexts, dict):
            raise ValueError(f"Unable to load repository graph from {graph_path}")

        prompt = _build_instruction(sample)
        prompts.append(prompt)

        repo_mem, llm_client = _instantiate_repo_memory(full_contexts, model_type)

        meta_prompt, generation = repo_mem.forward(prompt)
        session_messages = [{"role": "user", "content": meta_prompt}]
        session_messages.append({"role": "assistant", "content": generation})

        repo_mem.backward(prompt, generation)
        sample["tokens"]["session"].append(llm_client.last_usage)
        sample["tokens"]["pipeline"].append(llm_client.last_usage)
        sample["session_messages"] = session_messages
        sample["generated_code"] = generation

        extra_rounds = sample.get("follow_up_prompts") or sample.get("multi-turn")
        if isinstance(extra_rounds, list):
            for follow_up in extra_rounds:
                if isinstance(follow_up, dict):
                    follow_prompt = (
                        follow_up.get("requirement")
                        or follow_up.get("prompt")
                        or follow_up.get("text")
                    )
                else:
                    follow_prompt = follow_up
                if not isinstance(follow_prompt, str) or not follow_prompt.strip():
                    continue
                follow_prompt = follow_prompt.strip()
                prompts.append(follow_prompt)
                meta_prompt, generation = repo_mem.forward(follow_prompt)
                session_messages.append({"role": "user", "content": meta_prompt})
                session_messages.append({"role": "assistant", "content": generation})
                sample["session_messages"] = session_messages
                repo_mem.backward(follow_prompt, generation)

        sample["all_session_tokens"] = llm_client.total_usage
        sample["all_target_tokens"] = llm_client.total_usage
        sample["time"] = time.time() - start_time
        return sample
    except Exception:
        sample["error"] = traceback.format_exc()
        sample["time"] = time.time() - start_time
        return sample


def run_pipeline(
    json_path: str,
    json_is_jsonl: bool,
    root_path: str,
    output_path: str,
    repo_graph_name: str,
    project_separator: str,
    model_type: str,
    max_workers: int = 8,
) -> None:
    """Execute the Codereval pipeline with optional retry support."""
    data = read_json(json_path, not json_is_jsonl) or []
    if not isinstance(data, list):
        raise ValueError(f"Input file {json_path} did not contain a list of samples.")

    existing_by_id = _load_existing(output_path)
    results_map: Dict[int, Dict[str, Any]] = {}
    attempt_counts: Dict[int, int] = defaultdict(int)
    pending_indices: Set[int] = set()

    for idx, sample in enumerate(data):
        sample_id = (
            sample.get("_id")
            or sample.get("question_id")
            or sample.get("project")
            or sample.get("project_path")
            or str(idx)
        )
        existing = existing_by_id.get(str(sample_id))
        if existing and isinstance(existing.get("session_messages"), list) and existing["session_messages"]:
            results_map[idx] = existing
        else:
            pending_indices.add(idx)

    progress = tqdm(total=len(data), desc="Processing Codereval samples")
    if results_map:
        progress.update(len(results_map))

    while pending_indices:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    process_single_sample,
                    idx,
                    data[idx],
                    root_path,
                    repo_graph_name,
                    project_separator,
                    model_type,
                ): idx
                for idx in list(pending_indices)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                attempt_counts[idx] += 1
                try:
                    sample_result = future.result()
                except Exception:
                    sample_result = {"index": idx, "error": traceback.format_exc()}
                has_session = (
                    isinstance(sample_result, dict)
                    and isinstance(sample_result.get("session_messages"), list)
                    and sample_result["session_messages"]
                )
                if has_session:
                    results_map[idx] = sample_result
                    pending_indices.discard(idx)
                    progress.update(1)
                elif attempt_counts[idx] >= 2:
                    pending_indices.discard(idx)
                    results_map[idx] = sample_result
                    progress.update(1)
        if pending_indices:
            print(f"Retrying {len(pending_indices)} samples...")

    final_results = [results_map[i] for i in sorted(results_map.keys())]
    save_json(output_path, final_results, True)
    progress.close()


def parse_args() -> argparse.Namespace:
    """Configure the CLI for Codereval execution."""
    parser = argparse.ArgumentParser(description="Run the Codemem pipeline on Codereval data.")
    parser.add_argument("--json_path", required=True, help="Path to the Codereval JSON/JSONL file.")
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Treat the input file as JSON Lines instead of a JSON list.",
    )
    parser.add_argument("--root_path", required=True, help="Directory containing repository graphs.")
    parser.add_argument("--output_path", required=True, help="Destination for saving pipeline results.")
    parser.add_argument(
        "--repo_graph",
        default="repo_graph_v2.json",
        help="File name of the repository graph inside each project folder.",
    )
    parser.add_argument(
        "--project_separator",
        default="---",
        help="Character(s) that replace '/' when mapping project names to folders.",
    )
    parser.add_argument(
        "--model",
        default="deepseek-v3.2",
        help="Model name for all Codemem LLM calls.",
    )
    parser.add_argument("--max_workers", type=int, default=8, help="Thread pool size.")
    return parser.parse_args()


def main() -> None:
    """Entry point for CLI execution."""
    args = parse_args()
    run_pipeline(
        json_path=args.json_path,
        json_is_jsonl=args.jsonl,
        root_path=args.root_path,
        output_path=args.output_path,
        repo_graph_name=args.repo_graph,
        project_separator=args.project_separator,
        model_type=args.model,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
