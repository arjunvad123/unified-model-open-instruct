# Single source of truth for the action / control tokens used by the
# unified agentic model. Training code (open_instruct/unified_finetune.py)
# and benchmark/eval scripts (benchmarks/*) MUST import from here so the
# tokenizer and the evaluation harness can never drift apart.

# Named constants are the preferred entry point: import these by name
# (e.g. `from open_instruct.action_tokens import ACT_RET`) when you need
# a specific token. That way a future renaming surfaces as a static
# import error rather than silently drifting against a hardcoded string
# literal somewhere downstream.
ACT_THINK: str = "<ACT:THINK>"    # Internal reasoning step
ACT_RET: str = "<ACT:RET>"        # Trigger retrieval action
ACT_GEN: str = "<ACT:GEN>"        # Generate final response
ACT_STOP: str = "<ACT:STOP>"      # Terminate generation
WAIT: str = "<WAIT>"              # Pause for external input
RET_RESULT: str = "<RET_RESULT>"  # Marks retrieved content injection

# List views are derived from the named constants so the two cannot
# drift. Existing code that imports ACTION_TOKENS / ROUTING_TOKENS /
# CONTROL_TOKENS keeps working unchanged.
ACTION_TOKENS: list[str] = [ACT_THINK, ACT_RET, ACT_GEN, ACT_STOP, WAIT, RET_RESULT]
ROUTING_TOKENS: list[str] = [ACT_THINK, ACT_RET, ACT_GEN, ACT_STOP]
CONTROL_TOKENS: list[str] = [WAIT, RET_RESULT]


def all_action_tokens() -> list[str]:
    return list(ACTION_TOKENS)
