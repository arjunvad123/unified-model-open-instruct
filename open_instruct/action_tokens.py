# Single source of truth for the action tokens used by the unified
# agentic model. The trained set is exactly these four tokens (per
# Arjun, 2026-05-07): GEN, RET, THINK, STOP. Training code (open_instruct/unified_finetune.py)
# and benchmark/eval scripts (benchmarks/*) MUST import from here so the
# tokenizer and the evaluation harness can never drift apart.

ACTION_TOKENS: list[str] = [
    "<ACT:THINK>",   # Internal reasoning step
    "<ACT:RET>",     # Trigger retrieval action
    "<ACT:GEN>",     # Generate final response
    "<ACT:STOP>",    # Terminate generation
]

ROUTING_TOKENS: list[str] = ["<ACT:THINK>", "<ACT:RET>", "<ACT:GEN>", "<ACT:STOP>"]


def all_action_tokens() -> list[str]:
    return list(ACTION_TOKENS)
