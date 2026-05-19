"""GRPO training package for the Student model.

Pipeline (see README in this directory):
    1. collect_teacher_traces.py  -> teacher_traces.jsonl
    2. sft_warmstart.py           -> checkpoints/sft/
    3. grpo_trainer.py            -> checkpoints/grpo/

Modules:
    rewards.py     reward functions (answer / tool_seq / plan_sim / format)
    rollout.py     multi-turn rollout with tool execution
    data.py        dataset loading utilities
"""