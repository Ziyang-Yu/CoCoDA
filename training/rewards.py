"""Reward functions for GRPO student training.

A *trajectory* is a dict produced by ``training.rollout.rollout_one``:

    {
        "thoughts":   list[str],         # text *outside* <tool_call> blocks
        "tool_calls": list[dict],        # parsed tool calls in order
        "answer":     str,               # final assistant text
        "n_format_errors": int,          # malformed <tool_call> blocks
        "n_tool_errors":   int,          # tool name not in library / exec errors
        "hit_max_steps":   bool,
    }

The combined reward is

    r = w_answer * r_answer
      + w_tool_seq * r_tool_seq
      + w_plan_sim * r_plan_sim
      + w_composite * r_composite
      - w_format  * format_penalty

with the convention that ``r_tool_seq`` and ``r_plan_sim`` are *only* non-zero
when ``r_answer == 1`` AND a teacher reference exists.  This prevents the
student from being rewarded for parroting the teacher on problems it failed.

``r_composite`` (fraction of tool calls that are composite) is *not* gated
on correctness so the model learns to prefer composite tools even while
still learning to produce correct answers.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

# is_correct lives in the project root main.py
from main import is_correct


# ---------------------------------------------------------------------------
# Individual reward components
# ---------------------------------------------------------------------------
def reward_answer(prediction: str, gold: str) -> float:
    """1.0 if *prediction* matches the GSM8K *gold*, else 0.0."""
    return 1.0 if is_correct(prediction, gold) else 0.0


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[len(b)]


def reward_tool_seq(student_seq: list[str], teacher_seq: list[str]) -> float:
    """LCS-based similarity in [0, 1] between two ordered tool-name lists."""
    if not student_seq or not teacher_seq:
        return 0.0
    lcs = _lcs_len(student_seq, teacher_seq)
    denom = max(len(student_seq), len(teacher_seq))
    return lcs / denom


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _f1(a_tokens: list[str], b_tokens: list[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    common = a_set & b_set
    if not common:
        return 0.0
    p = len(common) / len(a_set)
    r = len(common) / len(b_set)
    return 2 * p * r / (p + r)


def reward_plan_sim(
    student_thoughts: list[str], teacher_plan: list[str]
) -> float:
    """Token-F1 similarity between student thought blocks and teacher plan steps.

    Each student thought is greedily aligned to its best-matching teacher
    plan step; the final score is the mean F1 across the *teacher* plan
    (not the student), which prevents reward inflation from extra-verbose
    student traces.
    """
    if not student_thoughts or not teacher_plan:
        return 0.0
    student_tokens = [_tokenize(t) for t in student_thoughts]
    if not any(student_tokens):
        return 0.0

    scores: list[float] = []
    for plan_step in teacher_plan:
        plan_tokens = _tokenize(plan_step)
        if not plan_tokens:
            continue
        best = max(_f1(plan_tokens, st) for st in student_tokens)
        scores.append(best)
    return sum(scores) / len(scores) if scores else 0.0


def reward_composite_ratio(
    student_tool_names: list[str], composite_names: set[str]
) -> float:
    """Fraction of the student's tool calls that are composite tools.

    Returns a value in [0, 1].  If the student made no tool calls, returns 0.
    """
    if not student_tool_names:
        return 0.0
    n_composite = sum(1 for n in student_tool_names if n in composite_names)
    return n_composite / len(student_tool_names)


def format_penalty(trajectory: dict[str, Any]) -> float:
    """Sum of format / tool-error penalties.  Always non-negative."""
    pen = 0.0
    pen += 0.5 * trajectory.get("n_format_errors", 0)
    pen += 0.5 * trajectory.get("n_tool_errors", 0)
    if trajectory.get("hit_max_steps", False):
        pen += 0.25
    return pen


# ---------------------------------------------------------------------------
# Combined reward
# ---------------------------------------------------------------------------
@dataclass
class RewardWeights:
    answer:    float = 1.0
    tool_seq:  float = 0.3
    plan_sim:  float = 0.3
    composite: float = 0.2
    format:    float = 0.2


def combined_reward(
    trajectory: dict[str, Any],
    gold_answer: str,
    teacher_tool_seq: list[str] | None = None,
    teacher_plan: list[str] | None = None,
    composite_names: set[str] | None = None,
    weights: RewardWeights | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute the scalar reward + a breakdown dict for logging."""
    weights = weights or RewardWeights()

    r_ans = reward_answer(trajectory.get("answer", ""), gold_answer)

    student_seq = [tc.get("name", "") for tc in trajectory.get("tool_calls", [])]

    # Plan / tool-seq rewards are gated on correctness — see module docstring.
    if r_ans > 0 and teacher_tool_seq:
        r_tools = reward_tool_seq(student_seq, teacher_tool_seq)
    else:
        r_tools = 0.0

    if r_ans > 0 and teacher_plan:
        r_plan = reward_plan_sim(trajectory.get("thoughts", []), teacher_plan)
    else:
        r_plan = 0.0

    # Composite-usage reward: always active (not gated on correctness) so the
    # model learns to prefer composite tools even while still learning to get
    # the answer right.
    if composite_names:
        r_comp = reward_composite_ratio(student_seq, composite_names)
    else:
        r_comp = 0.0

    pen = format_penalty(trajectory)

    total = (
        weights.answer * r_ans
        + weights.tool_seq * r_tools
        + weights.plan_sim * r_plan
        + weights.composite * r_comp
        - weights.format * pen
    )
    breakdown = {
        "r_answer":    r_ans,
        "r_tool_seq":  r_tools,
        "r_plan_sim":  r_plan,
        "r_composite": r_comp,
        "penalty":     pen,
        "total":       total,
    }
    return total, breakdown


# ---------------------------------------------------------------------------
# Group-relative advantages (the "GR" in GRPO)
# ---------------------------------------------------------------------------
def group_relative_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    """Standardise *rewards* within a group: ``(r - mean) / (std + eps)``.

    A group is the set of G samples drawn from the same prompt.
    """
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(var)
    return [(r - mean) / (std + eps) for r in rewards]