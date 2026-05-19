"""Multi-turn rollout for the Student model.

A *rollout* generates an entire ReAct trace for one problem, executing tool
calls against a frozen :class:`ToolLibrary` between turns.  The output is a
single chat-formatted token sequence plus a per-token ``assistant_mask`` that
the GRPO loss will use to mask out system/user/observation tokens.

This module uses HuggingFace ``model.generate()`` directly so that nothing
about weight syncing across vLLM is required.  The student is small (1.5B),
so HF generation is fast enough for typical group sizes (G=4–8).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from model.student import STUDENT_SYSTEM_PROMPT
from tool.tool_library import ToolLibrary

log = logging.getLogger(__name__)

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)
_THINK_RE_CLOSED = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_RE_OPEN = re.compile(r"<think>.*", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove Qwen3-style <think>...</think> blocks (closed or trailing-open)."""
    text = _THINK_RE_CLOSED.sub("", text)
    text = _THINK_RE_OPEN.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Trajectory dataclass
# ---------------------------------------------------------------------------
@dataclass
class Trajectory:
    """A single rollout result.

    Token-level fields are 1-D Python lists (not tensors) to keep this
    object cheap to copy across processes.
    """
    input_ids: list[int]
    assistant_mask: list[int]   # 1 for assistant tokens, 0 otherwise
    messages: list[dict[str, str]]
    thoughts: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    n_format_errors: int = 0
    n_tool_errors: int = 0
    hit_max_steps: bool = False

    def to_reward_dict(self) -> dict[str, Any]:
        return {
            "thoughts":        self.thoughts,
            "tool_calls":      self.tool_calls,
            "answer":          self.answer,
            "n_format_errors": self.n_format_errors,
            "n_tool_errors":   self.n_tool_errors,
            "hit_max_steps":   self.hit_max_steps,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_tool_call(text: str) -> tuple[dict[str, Any] | None, bool]:
    """Return (parsed_call, had_format_error)."""
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None, False
    try:
        return json.loads(m.group(1)), False
    except json.JSONDecodeError:
        return None, True


def _strip_thought(response: str) -> str:
    """The 'thought' is everything *outside* the tool_call block."""
    return _TOOL_CALL_RE.sub("", response).strip()


def _build_initial_messages(
    problem: str,
    tool_descriptions: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system",
         "content": STUDENT_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)},
        {"role": "user", "content": problem},
    ]


def _render_and_tokenize_for_generation(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
) -> torch.Tensor:
    # ``enable_thinking=False`` disables Qwen3 thinking mode; on chat templates
    # that don't reference the variable it is silently ignored.
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(
        text, return_tensors="pt", add_special_tokens=False,
    ).input_ids[0]


def _build_assistant_mask(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    assistant_texts: list[str],
) -> tuple[list[int], list[int]]:
    """Render the full conversation and build a token-level assistant mask.

    Returns (input_ids, mask) as Python lists.
    """
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    enc = tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    mask = [0] * len(input_ids)

    # Walk forward through the rendered string, finding each assistant
    # message in order.  We keep a cursor so two identical assistant turns
    # don't both match the first occurrence.
    cursor = 0
    for content in assistant_texts:
        if not content:
            continue
        idx = full_text.find(content, cursor)
        if idx < 0:
            continue
        char_end = idx + len(content)
        for tok_i, (s, e) in enumerate(offsets):
            if s >= idx and e <= char_end:
                mask[tok_i] = 1
        cursor = char_end
    return input_ids, mask


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout_one(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    tool_library: ToolLibrary,
    problem: str,
    tool_descriptions: str,
    max_steps: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: torch.device | str = "cuda",
) -> Trajectory:
    """Roll out one full multi-turn ReAct trace for *problem*."""
    messages = _build_initial_messages(problem, tool_descriptions)
    assistant_texts: list[str] = []
    thoughts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    n_format_errors = 0
    n_tool_errors = 0
    answer = ""
    hit_max_steps = False

    for step_idx in range(max_steps):
        prompt_ids = _render_and_tokenize_for_generation(tokenizer, messages).to(device)
        out = model.generate(
            prompt_ids.unsqueeze(0),
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]
        new_token_ids = out[prompt_ids.shape[0]:]
        response = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        response = _strip_think(response)

        assistant_texts.append(response)
        messages.append({"role": "assistant", "content": response})

        thought = _strip_thought(response)
        if thought:
            thoughts.append(thought)

        tool_call, fmt_err = _parse_tool_call(response)
        if fmt_err:
            n_format_errors += 1

        if tool_call is None:
            # No tool call -> treat as final answer
            answer = response
            break

        # Execute the tool
        name = tool_call.get("name", "")
        args = tool_call.get("arguments", {}) or {}
        tool_calls.append(tool_call)
        if name not in tool_library:
            observation = f"Error: tool '{name}' not found."
            n_tool_errors += 1
        else:
            try:
                result = tool_library.execute(name, **args)
                observation = str(result)
            except Exception as e:
                observation = f"Error executing '{name}': {e}"
                n_tool_errors += 1

        messages.append(
            {"role": "user", "content": f"Tool result: {observation}"}
        )
    else:
        # Loop completed without break -> max_steps hit
        hit_max_steps = True
        # Force one final generation as the answer
        prompt_ids = _render_and_tokenize_for_generation(tokenizer, messages).to(device)
        out = model.generate(
            prompt_ids.unsqueeze(0),
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]
        new_token_ids = out[prompt_ids.shape[0]:]
        final = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        final = _strip_think(final)
        assistant_texts.append(final)
        messages.append({"role": "assistant", "content": final})
        answer = final

    input_ids, mask = _build_assistant_mask(tokenizer, messages, assistant_texts)

    return Trajectory(
        input_ids=input_ids,
        assistant_mask=mask,
        messages=messages,
        thoughts=thoughts,
        tool_calls=tool_calls,
        answer=answer,
        n_format_errors=n_format_errors,
        n_tool_errors=n_tool_errors,
        hit_max_steps=hit_max_steps,
    )


# ---------------------------------------------------------------------------
# Group rollout (G samples per problem) — batched
# ---------------------------------------------------------------------------
def _left_pad_batch(
    seqs: list[torch.Tensor],
    pad_id: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad a list of 1-D token tensors into a [B, L] batch + attention mask."""
    max_len = max(s.shape[0] for s in seqs)
    batch = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        batch[i, max_len - s.shape[0]:] = s
        attn[i, max_len - s.shape[0]:] = 1
    return batch, attn


@torch.no_grad()
def rollout_batch(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    tool_library: ToolLibrary,
    problems: list[str],
    tool_descriptions_list: list[str],
    max_steps: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: torch.device | str = "cuda",
) -> list[Trajectory]:
    """Run batched, lockstep ReAct rollouts for a list of (problem, tool_desc)
    pairs.

    All trajectories run in parallel: at each inner step the still-active
    sequences are padded into one batch, ``model.generate`` is called once,
    and each output is parsed/executed independently.  Sequences that emit a
    non-tool-call response (or hit ``max_steps``) drop out of the active set;
    the loop ends when all are done.

    This is the underlying primitive used by both ``rollout_group`` (G samples
    of one prompt) and the validation loop (B distinct prompts in one shot).
    """
    if len(problems) != len(tool_descriptions_list):
        raise ValueError(
            f"problems and tool_descriptions_list must have the same length, "
            f"got {len(problems)} and {len(tool_descriptions_list)}"
        )
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Per-trajectory mutable state
    states: list[dict[str, Any]] = [
        {
            "messages": _build_initial_messages(problems[i], tool_descriptions_list[i]),
            "assistant_texts": [],
            "thoughts": [],
            "tool_calls": [],
            "n_format_errors": 0,
            "n_tool_errors": 0,
            "answer": "",
            "hit_max_steps": False,
            "done": False,
        }
        for i in range(len(problems))
    ]

    def _generate_for(indices: list[int]) -> list[str]:
        """Run one batched generate over the given trajectory indices and
        return the decoded responses (post-`_strip_think`)."""
        prompt_ids_list = [
            _render_and_tokenize_for_generation(tokenizer, states[i]["messages"]).to(device)
            for i in indices
        ]
        batch_input, attn_mask = _left_pad_batch(prompt_ids_list, pad_id, device)
        out = model.generate(
            batch_input,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_len = batch_input.shape[1]
        responses: list[str] = []
        for j in range(len(indices)):
            new_token_ids = out[j, prompt_len:]
            text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
            responses.append(_strip_think(text))
        return responses

    for _step_idx in range(max_steps):
        active = [i for i, s in enumerate(states) if not s["done"]]
        if not active:
            break

        responses = _generate_for(active)

        for resp_j, idx in enumerate(active):
            response = responses[resp_j]
            s = states[idx]
            s["assistant_texts"].append(response)
            s["messages"].append({"role": "assistant", "content": response})

            thought = _strip_thought(response)
            if thought:
                s["thoughts"].append(thought)

            tool_call, fmt_err = _parse_tool_call(response)
            if fmt_err:
                s["n_format_errors"] += 1

            if tool_call is None:
                # No tool call -> treat as final answer
                s["answer"] = response
                s["done"] = True
                continue

            name = tool_call.get("name", "")
            args = tool_call.get("arguments", {}) or {}
            s["tool_calls"].append(tool_call)
            if name not in tool_library:
                observation = f"Error: tool '{name}' not found."
                s["n_tool_errors"] += 1
            else:
                try:
                    result = tool_library.execute(name, **args)
                    observation = str(result)
                except Exception as e:
                    observation = f"Error executing '{name}': {e}"
                    s["n_tool_errors"] += 1

            s["messages"].append(
                {"role": "user", "content": f"Tool result: {observation}"}
            )

    # Anything still active hit max_steps -> force one final generation
    not_done = [i for i, s in enumerate(states) if not s["done"]]
    if not_done:
        finals = _generate_for(not_done)
        for resp_j, idx in enumerate(not_done):
            final = finals[resp_j]
            s = states[idx]
            s["assistant_texts"].append(final)
            s["messages"].append({"role": "assistant", "content": final})
            s["answer"] = final
            s["hit_max_steps"] = True
            s["done"] = True

    trajectories: list[Trajectory] = []
    for s in states:
        input_ids, mask = _build_assistant_mask(
            tokenizer, s["messages"], s["assistant_texts"]
        )
        trajectories.append(
            Trajectory(
                input_ids=input_ids,
                assistant_mask=mask,
                messages=s["messages"],
                thoughts=s["thoughts"],
                tool_calls=s["tool_calls"],
                answer=s["answer"],
                n_format_errors=s["n_format_errors"],
                n_tool_errors=s["n_tool_errors"],
                hit_max_steps=s["hit_max_steps"],
            )
        )
    return trajectories


def rollout_group(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    tool_library: ToolLibrary,
    problem: str,
    tool_descriptions: str,
    group_size: int,
    **kwargs: Any,
) -> list[Trajectory]:
    """Generate *group_size* independent trajectories for the same problem,
    batching all G sequences through ``model.generate`` at every inner step.

    Thin wrapper around :func:`rollout_batch` that tiles the same prompt
    ``group_size`` times so the G samples form one GRPO group whose advantages
    are computed relative to each other.
    """
    return rollout_batch(
        model=model,
        tokenizer=tokenizer,
        tool_library=tool_library,
        problems=[problem] * group_size,
        tool_descriptions_list=[tool_descriptions] * group_size,
        **kwargs,
    )