# A centralized repository for all prompt templates used by the agents.
# This makes it easy to view, manage, and evolve the "DNA" of the system.
from loguru import logger

_PROMPTS = {
    # ======= Context Memory Prompts =======
    "if_repo_mem_update":"""You are an expert repository memory manager for repository code generation tasks.
Your goal is to decide **whether the current repository memory needs updating** to support all the user's programming instructions.
### Your Decision Objective
Decide if you need to modify the repository memory (###Existing Repository Context###) based on how well it already covers the entities mentioned in the user instruction.
### Modes (Mutually Exclusive)
- **#KEEP#** — Use this when the existing repository context already contains all relevant classes/functions to understand or execute the instruction.  
- **#ADD#** — Use this mode when Existing Repository Context lacks code context related to user instructions.

### Input
**User Instruction:**
{instructions}
**Existing Repository Context (names only):**
{repo_context_info}

### Output Format (strict JSON)
{{
  "mode": "<ADD | KEEP>",
  "action": "<short, specific description of what to update or not update>",
  "target_context": "<list of relevant namespaces or []>"
}}
""",    "memory_update_system": """You are a code expert. Your task is to generate memory for the given user–LLM interaction rounds in coding tasks. The input contains the previous ###Instruction###, the corresponding ###Answer###, and the user's ###Next Instruction###.""",
    "memory_update_user": """###Instruction###
{instruction}
###Answer###
{answer}
###Next Instruction###
{new_instruction}
Requirements
1. If ###Next Instruction### is independent and has no dependency on ###Instruction###, you must respond based solely on ###Instruction### and ###Answer###, ignoring ###Next Instruction###.
2. If ###Next Instruction### builds upon or references ###Instruction###, you must generate a perspective that considers all three elements: ###Instruction###, ###Answer###, and ###Next Instruction###.
3. The memory should be concise, precise, and actionable. For example: This answer did not follow the instruction; This answer is incorrect and should be avoided.
### Output (JSON only)
{{
  "memory": "<one short sentence>",
}}
""",
"memory_update_system_without_next": """You are a code expert. Your task is to generate memory for the given user–LLM interaction rounds in coding tasks. The input contains the previous ###Instruction###, the corresponding ###Answer###, and the user's ###Next Instruction###.""",
"memory_update_user_without_next": """###Instruction###
{instruction}
###Answer###
{answer}
Requirements
The memory should be concise, precise, and actionable. For example: This answer did not follow the instruction; This answer is incorrect and should be avoided.

### Output (JSON only)
{{
  "memory": "<one short sentence>",
}}
""",
    "meta_prompt_memory": """**Repo Context**
---
{repo_context}
---
**Memory Blocks**
---
{memory_blocks}
---
**Current Instruction**
---
{instruction}
---
Please output the correct function correctly.
"""
}


def get_prompt(name: str) -> str:
    """
    Retrieves the raw prompt template for a given name from the central repository.

    Args:
        name: The key for the desired prompt.

    Returns:
        The prompt template string, or an empty string if not found.
    """
    prompt = _PROMPTS.get(name, "")
    if not prompt:
        logger.warning(f"Prompt template '{name}' not found.")
    return prompt
