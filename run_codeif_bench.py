"""Command-line entry point for the refactored Codemem pipeline."""

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
    """Load existing results so we can resume from an interrupted run."""
    if not os.path.exists(output_path):
        return {}
    loaded_results = read_json(output_path, False)
    if not isinstance(loaded_results, list):
        return {}
    existing: Dict[str, Dict[str, Any]] = {}
    for item in loaded_results:
        if not isinstance(item, dict):
            continue
        sample_id = item.get("_id") or item.get("question_id") or item.get("project_path")
        if sample_id:
            existing[sample_id] = item
    return existing


def _instantiate_repo_memory(full_contexts, model_type: str) -> Tuple[Codemem, Any]:
    """Instantiate the repository memory manager and expose the driving LLM."""
    llm_client = LLMFactory(llm_type="closed", model=model_type)
    repo_mem = Codemem(full_contexts, llm_client)
    return repo_mem, llm_client


def process_single_sample(
    idx: int,
    raw_sample: Dict[str, Any],
    root_path: str,
    model_type: str,
) -> Dict[str, Any]:
    """Process a single problem sample end-to-end."""
    sample = json.loads(json.dumps(raw_sample))
    start_time = time.time()
    try:
        sample["tokens"] = {"session": [], "pipeline": []}
        sample["result"] = []
        prompts: List[str] = []
        project_path = sample["project_path"]
        repo_path = os.path.join(root_path, project_path)
        function_name = sample["namespace"].split(".")[-1]
        functionality = sample["requirement"]["Functionality"]
        params = sample["requirement"]["Arguments"]
        prompt = f"Please write a python function called '{function_name}'. The functionality of the function: {functionality} The parameters of the function: {params}."
        prompts.append(prompt)

        full_contexts = read_json(os.path.join(repo_path, "repo_graph_v1.json"), True)

        repo_mem, llm_client = _instantiate_repo_memory(full_contexts, model_type)

        meta_prompt, generation = repo_mem.forward(prompt)
        session_messages = [{"role": "user", "content": meta_prompt}]
        session_messages.append({"role": "assistant", "content": generation})

        repo_mem.backward(prompt, generation)
        sample["tokens"]["session"].append(llm_client.last_usage)
        sample["tokens"]["pipeline"].append(llm_client.last_usage)
        sample["session_messages"] = session_messages

        multi_turn = sample.get("multi-turn", [])[:-1]
        for key in multi_turn:
            regen_prompt = sample["requirements"][key]["requirement"]
            prompts.append(regen_prompt)
            meta_prompt, generation = repo_mem.forward(regen_prompt)
            session_messages.append({"role": "user", "content": meta_prompt})
            session_messages.append({"role": "assistant", "content": generation})
            sample["session_messages"] = session_messages
            repo_mem.backward(regen_prompt, generation)

        sample["all_session_tokens"] = llm_client.total_usage
        sample["all_target_tokens"] = llm_client.total_usage
        sample["time"] = time.time() - start_time
        return sample
    except Exception:
        sample["error"] = traceback.format_exc()
        sample["time"] = time.time() - start_time
        return sample


def run_pipeline(
    jsonl_path: str,
    root_path: str,
    output_path: str,
    model_type: str,
    max_workers: int = 8,
) -> None:
    """Execute the full Codemem workflow."""
    data = read_json(jsonl_path, False) or []
    # data = data[:1]
    existing_by_id = _load_existing(output_path)

    results_map: Dict[int, Dict[str, Any]] = {}
    attempt_counts: Dict[int, int] = defaultdict(int)
    pending_indices: Set[int] = set()

    for idx, sample in enumerate(data):
        sample_id = (
            sample.get("_id")
            or sample.get("question_id")
            or sample.get("project_path")
            or str(idx)
        )
        existing = existing_by_id.get(sample_id)
        if existing and isinstance(existing.get("session_messages"), list) and existing["session_messages"]:
            results_map[idx] = existing
        else:
            pending_indices.add(idx)

    progress = tqdm(total=len(data), desc="Processing samples")
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
    save_json(output_path, final_results, False)
    progress.close()


def parse_args() -> argparse.Namespace:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(description="Run the Codemem repository memory pipeline.")
    parser.add_argument("--jsonl_path", required=True, help="Path to the task jsonl file.")
    parser.add_argument("--root_path", required=True, help="Root folder that stores repository graphs.")
    parser.add_argument("--output_path", required=True, help="Where to store pipeline results.")
    parser.add_argument(
        "--model",
        default="deepseek-v3.2",
        help="Model name for both session and pipeline LLM usage.",
    )
    parser.add_argument("--max_workers", type=int, default=40, help="Thread pool size.")
    return parser.parse_args()


def main() -> None:
    """Entry point when running the module as a script."""
    args = parse_args()
    run_pipeline(
        jsonl_path=args.jsonl_path,
        root_path=args.root_path,
        output_path=args.output_path,
        model_type=args.model,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
