# Reinforcement Learning Research for Unified Agentic Model

**Date:** January 19, 2026
**Frameworks:** TRL (HuggingFace) & VERL (ByteDance/Volcengine)
**Goal:** Use RL to improve agentic routing (action token usage)

---

## Executive Summary

Our unified agentic model has good **generation** and **retrieval** capabilities, but the **agentic routing** (using action tokens like `<ACT:THINK>`, `<ACT:RET>`, `<ACT:GEN>`, `<ACT:STOP>`) is not working after 500 steps of SFT.

**RL can solve this** by:
1. Rewarding correct action token usage
2. Training on multi-turn interactions with tool feedback
3. Using outcome-based rewards (task success) to learn routing

---

## Framework Comparison

### TRL (HuggingFace)

**Repository:** https://github.com/huggingface/trl

| Feature | Details |
|---------|---------|
| Maintainer | HuggingFace |
| Integration | Deep Transformers/PEFT/Accelerate integration |
| Algorithms | SFT, GRPO, DPO, PPO, RLOO, RewardTrainer |
| Scalability | Single GPU to multi-node via Accelerate |
| QLoRA Support | ✅ Yes |
| CLI Tools | `trl sft`, `trl dpo`, `trl grpo` |
| Best For | Quick experiments, HF ecosystem users |

**Key Advantages:**
- Simple API, familiar to HuggingFace users
- GRPO trainer (memory-efficient, no critic model needed)
- Direct integration with our existing training setup
- Good documentation and examples

**Example GRPO Setup:**
```python
from trl import GRPOTrainer, GRPOConfig

config = GRPOConfig(
    output_dir="./grpo_output",
    num_generations=4,  # samples per prompt
    max_new_tokens=256,
)

trainer = GRPOTrainer(
    model=model,
    config=config,
    train_dataset=dataset,
    reward_funcs=reward_function,  # custom reward
)
trainer.train()
```

---

### VERL (ByteDance/Volcengine)

**Repository:** https://github.com/volcengine/verl

| Feature | Details |
|---------|---------|
| Maintainer | ByteDance (Volcengine) |
| Integration | FSDP, Megatron-LM, vLLM, SGLang |
| Algorithms | PPO, GRPO, DAPO, ReMax, RLOO, PRIME, GSPO |
| Scalability | Up to 70B models, hundreds of GPUs |
| Multi-turn | ✅ Native support |
| Tool Use | ✅ VerlTool framework |
| Best For | Production, complex agentic training |

**Key Advantages:**
- **Multi-turn agentic RL** - Native support for tool interactions
- **VerlTool** - Dedicated framework for agentic RL with tools
- **3D-HybridEngine** - Efficient distributed training
- **DAPO** - SOTA algorithm (50 points on AIME 2024)
- Server-based architecture with async tool calls

**Example Multi-Turn Setup:**
```yaml
# verl config for multi-turn agentic RL
rollout:
  multi_turn: True
  name: "sglang"

tools:
  - name: "retrieval_tool"
    class: "CustomRetrievalTool"
```

---

## Brainstormed Ideas for Our Model

### Idea 1: Action Token Reward Model

**Goal:** Train the model to emit correct action tokens

**Approach:**
1. Create a reward function that checks if action tokens are used appropriately:
   - `<ACT:RET>` when information retrieval is needed
   - `<ACT:THINK>` when reasoning is needed
   - `<ACT:GEN>` when generation is needed
   - `<ACT:STOP>` when task is complete

2. Use GRPO to train with this reward

**Reward Function:**
```python
def action_token_reward(prompt, response, expected_action):
    """Reward correct action token usage."""
    action_tokens = {
        "retrieve": "<ACT:RET>",
        "think": "<ACT:THINK>",
        "generate": "<ACT:GEN>",
        "stop": "<ACT:STOP>"
    }

    # Check if expected action token is present
    expected_token = action_tokens[expected_action]
    if expected_token in response:
        return 1.0  # Correct action

    # Partial reward if any action token is used
    for token in action_tokens.values():
        if token in response:
            return 0.3  # Wrong but tried

    return 0.0  # No action token
```

**Framework:** TRL (simpler for single-turn)

---

### Idea 2: Multi-Turn Tool Interaction Training

**Goal:** Train the model to use tools (retrieval) in multi-turn dialogue

**Approach:**
1. Create a retrieval tool that the model can call
2. Set up multi-turn environment:
   - User asks question
   - Model decides: answer directly OR retrieve first
   - If retrieve: tool returns results, model generates answer
   - Reward based on final answer quality

**Environment Design:**
```
Turn 1: User query
Turn 2: Model outputs <ACT:RET> query
Turn 3: System returns <RET_RESULT> documents
Turn 4: Model outputs <ACT:GEN> answer
Turn 5: Model outputs <ACT:STOP>
```

**Reward:** Based on answer correctness (verifiable) or helpfulness (LLM judge)

**Framework:** VERL (native multi-turn support)

---

### Idea 3: Outcome-Based GRPO for Routing

**Goal:** Let the model learn routing through task success

**Approach:**
1. Create diverse tasks:
   - Factual questions (need retrieval)
   - Reasoning problems (need thinking)
   - Creative writing (direct generation)
   - Conversation endings (need stop)

2. Sample multiple trajectories per task
3. Reward based on task completion quality
4. GRPO automatically learns which actions lead to success

**Dataset Structure:**
```json
{
  "prompt": "What is the capital of France?",
  "task_type": "factual",
  "ground_truth": "Paris",
  "ideal_trajectory": ["<ACT:RET>", "<RET_RESULT>", "<ACT:GEN>", "<ACT:STOP>"]
}
```

**Framework:** TRL or VERL (GRPO available in both)

---

### Idea 4: Agent-R1 Style Training

**Goal:** End-to-end RL for complex agentic behavior

**Reference:** [Agent-R1 Paper](https://arxiv.org/html/2511.14460v1)

**Key Insights from Agent-R1:**
- Loss masking on agent-generated tokens is crucial
- GRPO outperforms PPO for agentic training (0.3877 vs lower)
- Multi-turn decision making needs explicit RL training

**Approach:**
1. Create agentic tasks (web search, code execution, etc.)
2. Train with GRPO + loss masking
3. Focus gradients only on agent action tokens

**Framework:** VERL (better for complex agentic tasks)

---

### Idea 5: Embedding + Routing Joint Training

**Goal:** Improve both embedding quality and routing simultaneously

**Approach:**
1. Combine rewards:
   - Retrieval quality (embedding-based)
   - Action correctness (routing-based)
   - Final answer quality (generation-based)

2. Multi-objective GRPO:
```python
total_reward = (
    0.3 * embedding_reward +   # retrieval precision
    0.3 * action_reward +      # correct action tokens
    0.4 * answer_reward        # final answer quality
)
```

**Framework:** TRL with custom reward function

---

## Recommended Starting Point

### Phase 1: Simple Action Token Training (TRL)

**Why:** Quick to implement, test if RL helps with action tokens

**Steps:**
1. Create dataset with labeled expected actions
2. Implement simple reward function
3. Train with TRL's GRPOTrainer
4. Evaluate action token usage

**Estimated Time:** 1-2 days setup, 1 day training

---

### Phase 2: Multi-Turn Tool Training (VERL)

**Why:** More realistic agentic behavior

**Steps:**
1. Set up VERL with multi-turn support
2. Implement retrieval tool
3. Create multi-turn dialogue dataset
4. Train with GRPO/DAPO
5. Evaluate end-to-end task performance

**Estimated Time:** 3-5 days setup, 2-3 days training

---

## Fork Links

Please fork these repositories:

1. **TRL:** https://github.com/huggingface/trl
   - Fork to: `github.com/arjunvad123/trl`

2. **VERL:** https://github.com/volcengine/verl
   - Fork to: `github.com/arjunvad123/verl`

---

## Key Papers to Read

1. **GRPO:** [Group Relative Policy Optimization](https://cameronrwolfe.substack.com/p/grpo)
2. **Agent-R1:** [Training Powerful LLM Agents with End-to-End RL](https://arxiv.org/html/2511.14460v1)
3. **VerlTool:** [Holistic Agentic RL with Tool Use](https://arxiv.org/abs/2509.01055)
4. **Tree-GRPO:** [Tree Search for LLM Agent RL](https://arxiv.org/pdf/2509.21240)
5. **DAPO:** [Decoupled Advantage Policy Optimization](https://verl.readthedocs.io/)

---

## Next Steps

1. [ ] Fork TRL and VERL repos
2. [ ] Set up TRL with our checkpoint-500 model
3. [ ] Create action token dataset with labels
4. [ ] Implement reward function
5. [ ] Run small-scale GRPO experiment
6. [ ] Evaluate action token usage improvement
7. [ ] If successful, scale up with VERL for multi-turn

---

*Research compiled for Unified Agentic Model RL training.*

**Sources:**
- [TRL GitHub](https://github.com/huggingface/trl)
- [VERL GitHub](https://github.com/volcengine/verl)
- [VERL Multi-turn Docs](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)
- [Agent-R1 Paper](https://arxiv.org/html/2511.14460v1)
- [VerlTool Paper](https://arxiv.org/abs/2509.01055)
