"""Reformat Teacher CodeAgent trajectories into Student-style ReAct traces.

A teacher trajectory is a list of ``code_action``/``observation`` pairs from
``smolagents.CodeAgent``.  The Student model expects a chat-format multi-turn
conversation where each assistant turn ends in ``<tool_call>{...}</tool_call>``
and each tool result is fed back as a ``user`` turn.

We use the teacher LLM itself (one batched call per training set) to do the
rewriting, since aligning Python code blocks to extracted-tool calls is too
brittle to do with pure regexes.
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from typing import Any

from vllm import SamplingParams

from model.student import STUDENT_SYSTEM_PROMPT

log = logging.getLogger(__name__)


REWRITE_PROMPT = textwrap.dedent("""\
    You are converting a step-by-step CodeAgent solution into a ReAct-style
    trace that a smaller student model will imitate.  The student can only
    call ONE tool per step using a <tool_call> JSON block.

    ## Problem
    {problem}

    ## Available tools (the student may only call these by name)
    {tool_descriptions}

    ## Original CodeAgent steps
    {steps_text}

    ## Final answer
    {final_answer}

    ## Instructions
    Rewrite the solution as a JSON object with the following schema:

    {{
      "plan": [
        "<one-sentence sub-step 1>",
        "<one-sentence sub-step 2>",
        ...
      ],
      "turns": [
        {{
          "thought": "<one or two sentences explaining what to do next>",
          "tool_name": "<one of the tools above>",
          "arguments": {{ "<arg>": <value>, ... }},
          "observation": "<the value the tool returns on this input>"
        }},
        ...
      ],
      "final_text": "<one or two sentences stating the final answer, ending with '#### <number>'>"
    }}

    Rules:
    - Use ONLY tool names from the list above.
    - Each turn corresponds to exactly one tool call.
    - The number of turns should match the number of distinct sub-steps in the plan.
    - Arguments must be JSON-serialisable Python literals.
    - Output ONLY the JSON object, no surrounding prose, no markdown fences.
""")


def _format_tool_descriptions(tool_names: list[str], tool_library) -> str:
    """Render a tool list the way the student's system prompt does."""
    if not tool_names:
        return "(none)"
    lines: list[str] = []
    for name in tool_names:
        if name not in tool_library:
            continue
        meta = tool_library.get_tool(name).metadata
        sig = meta.signature
        desc = meta.description.summary or "No description"
        lines.append(
            f"- **{name}**({sig.input_type}) -> {sig.output_type}: {desc}"
        )
    return "\n".join(lines) if lines else "(none)"


def _format_steps_text(steps: list[dict[str, Any]]) -> str:
    """Render the raw CodeAgent steps for the rewrite prompt."""
    blocks: list[str] = []
    for i, step in enumerate(steps, 1):
        code = step.get("code", "").strip()
        obs = str(step.get("observation", "")).strip()
        if not code and not obs:
            continue
        block = f"Step {i}:\n```python\n{code}\n```"
        if obs:
            block += f"\nObservation: {obs}"
        blocks.append(block)
    return "\n\n".join(blocks)


def _build_rewrite_messages(
    problem: str,
    steps: list[dict[str, Any]],
    tool_names: list[str],
    tool_library,
    final_answer: str,
) -> list[dict[str, str]]:
    prompt = REWRITE_PROMPT.format(
        problem=problem,
        tool_descriptions=_format_tool_descriptions(tool_names, tool_library),
        steps_text=_format_steps_text(steps),
        final_answer=final_answer,
    )
    return [{"role": "user", "content": prompt}]


def _strip_think_tags(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text


def _coerce_json_like(text: str) -> str:
    """Apply cheap fixups for common LLM JSON mistakes.

    Handles trailing commas, Python literals (``None``/``True``/``False``),
    and single-quoted strings/keys. Only used as a fallback after strict
    ``json.loads`` has already failed.
    """
    # Trailing commas before } or ]
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    # Python literals -> JSON literals (only when they look like bare tokens)
    text = re.sub(r"(?<![\"\w])None(?![\"\w])", "null", text)
    text = re.sub(r"(?<![\"\w])True(?![\"\w])", "true", text)
    text = re.sub(r"(?<![\"\w])False(?![\"\w])", "false", text)
    return text


# strict=False lets raw control chars (newlines, tabs) sit inside string
# values — LLMs routinely emit multi-line observations/thoughts without
# escaping them as \n, and the default strict=True would reject those.
_JSON_DECODER = json.JSONDecoder(strict=False)


def _extract_first_json_object(
    text: str,
) -> tuple[dict[str, Any] | None, json.JSONDecodeError | None]:
    """Scan ``text`` for the first balanced ``{...}`` that parses as JSON.

    Uses ``raw_decode`` from each ``{`` candidate so trailing prose after
    the object is tolerated, and only the outermost balanced object is
    returned. Returns ``(obj, None)`` on success, or ``(None, last_err)``
    with the most recent ``JSONDecodeError`` so the caller can diagnose.
    """
    last_err: json.JSONDecodeError | None = None
    # Try each '{' as a potential start — earliest is usually right, but the
    # model may emit a stray brace inside a think/explanation preamble.
    for match in re.finditer(r"\{", text):
        start = match.start()
        try:
            obj, _ = _JSON_DECODER.raw_decode(text[start:])
        except json.JSONDecodeError as e:
            last_err = e
            continue
        if isinstance(obj, dict):
            return obj, None
    return None, last_err


def _parse_rewrite_response(
    response: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Best-effort parse of the rewriter's JSON output.

    Returns ``(obj, None)`` on success, or ``(None, diagnostic)`` where
    ``diagnostic`` is a short human-readable reason for the failure.
    """
    original = response
    response = _strip_think_tags(response).strip()
    # Strip ```json fences if present (anywhere, not just at the very ends)
    response = re.sub(r"```(?:json)?\s*", "", response)
    response = response.replace("```", "").strip()

    # Strategy 1: raw_decode starting at each '{'
    obj, last_err = _extract_first_json_object(response)

    # Strategy 2: outermost {...} + cheap JSON fixups
    if obj is None:
        start = response.find("{")
        end = response.rfind("}")
        if start >= 0 and end > start:
            candidate = _coerce_json_like(response[start : end + 1])
            try:
                obj = json.loads(candidate, strict=False)
            except json.JSONDecodeError as e:
                last_err = e
                obj = None

    # Strategy 3: apply fixups then try raw_decode again (handles truncation
    # after a trailing comma or similar)
    if obj is None:
        fixed = _coerce_json_like(response)
        obj, err3 = _extract_first_json_object(fixed)
        if obj is None and err3 is not None:
            last_err = err3

    if not isinstance(obj, dict):
        if last_err is not None:
            # Show ±80 chars around the failure offset so the caller can see
            # the exact token that broke the parse.
            pos = getattr(last_err, "pos", 0) or 0
            lo, hi = max(0, pos - 80), pos + 80
            around = response[lo:hi].replace("\n", "\\n")
            diag = f"{last_err.msg} at pos {pos}: {around!r}"
        else:
            snippet = original[:200].replace("\n", "\\n")
            diag = f"no '{{' found; head={snippet!r}"
        log.debug("[format_traces] could not locate JSON object; %s", diag)
        return None, diag
    if "turns" not in obj or not isinstance(obj["turns"], list):
        diag = f"parsed object missing 'turns' list; keys={list(obj.keys())}"
        log.debug("[format_traces] %s", diag)
        return None, diag
    return obj, None


def _rendered_chat_text(
    problem: str,
    tool_names: list[str],
    tool_library,
    parsed: dict[str, Any],
    tokenizer,
) -> tuple[str, list[tuple[int, int]]]:
    """Materialise the parsed rewrite as a single chat-formatted string and
    return the (start, end) character spans of each assistant turn so the
    SFT loss can be masked to assistant tokens only.
    """
    tool_desc = _format_tool_descriptions(tool_names, tool_library)
    system = STUDENT_SYSTEM_PROMPT.format(tool_descriptions=tool_desc)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": problem},
    ]
    assistant_texts: list[str] = []

    for turn in parsed["turns"]:
        thought = str(turn.get("thought", "")).strip()
        tool_name = turn.get("tool_name", "")
        arguments = turn.get("arguments", {})
        observation = str(turn.get("observation", "")).strip()
        try:
            args_json = json.dumps(arguments)
        except (TypeError, ValueError):
            args_json = "{}"
        assistant_text = (
            f"{thought}\n"
            f'<tool_call>{{"name": "{tool_name}", "arguments": {args_json}}}'
            f"</tool_call>"
        )
        assistant_texts.append(assistant_text)
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append(
            {"role": "user", "content": f"Tool result: {observation}"}
        )

    final_text = str(parsed.get("final_text", "")).strip()
    if final_text:
        assistant_texts.append(final_text)
        messages.append({"role": "assistant", "content": final_text})

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    # Locate each assistant turn's character span by simple substring search.
    spans: list[tuple[int, int]] = []
    cursor = 0
    for text in assistant_texts:
        idx = rendered.find(text, cursor)
        if idx < 0:
            # Chat template may have escaped/transformed the text — skip span
            continue
        spans.append((idx, idx + len(text)))
        cursor = idx + len(text)
    return rendered, spans


def format_solutions_batch(
    problems: list[str],
    solutions: list[dict[str, Any]],
    teacher,
    tokenizer,
) -> list[dict[str, Any] | None]:
    """Convert a batch of teacher solutions into student-format records.

    Each record contains:
        problem, gold-style final_answer, teacher_plan, teacher_tool_seq,
        teacher_text (chat-formatted string), assistant_char_spans.

    Returns ``None`` for any solution that could not be parsed.
    """
    library = teacher.tool_library

    # Build per-solution prompts (skip empty ones)
    prompts: list[str] = []
    indices: list[int] = []
    for i, sol in enumerate(solutions):
        steps = sol.get("steps") or []
        tool_names = sol.get("tool_names") or []
        if not steps or not tool_names:
            continue
        msgs = _build_rewrite_messages(
            problem=problems[i],
            steps=steps,
            tool_names=tool_names,
            tool_library=library,
            final_answer=str(sol.get("answer", "")),
        )
        # Disable Qwen3 thinking mode here: the rewriter is a deterministic
        # format-transformation task, and enabling <think> makes the model
        # burn the token budget on reasoning, frequently truncating before
        # the JSON is emitted. Tokenizers that don't know the kwarg ignore it.
        try:
            chat_prompt = teacher._llm._tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            chat_prompt = teacher._llm._tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        prompts.append(chat_prompt)
        indices.append(i)

    results: list[dict[str, Any] | None] = [None] * len(solutions)
    if not prompts:
        return results

    # 4096 gives enough headroom for long GSM8K rewrites; 2048 was truncating
    # the JSON body for multi-step solutions.
    params = SamplingParams(max_tokens=4096, temperature=0.0)
    outputs = teacher._llm._llm.generate(prompts, params)

    for idx, out in zip(indices, outputs):
        sol = solutions[idx]
        completion = out.outputs[0]
        raw_text = completion.text
        # vLLM reports finish_reason = "length" when the budget ran out and
        # "stop" when the model emitted EOS on its own. Critical signal for
        # distinguishing "prompt too long" from "model went off-schema".
        finish_reason = getattr(completion, "finish_reason", None)
        prompt_tok_len = len(getattr(out, "prompt_token_ids", []) or [])
        output_tok_len = len(getattr(completion, "token_ids", []) or [])
        parsed, diag = _parse_rewrite_response(raw_text)
        if parsed is None:
            snippet = raw_text[:200].replace("\n", "\\n")
            log.warning(
                "[format_traces] Failed to parse rewrite for index %d "
                "(finish=%s, prompt_toks=%d, output_toks=%d, text_len=%d, "
                "reason=%s, head=%r)",
                idx, finish_reason, prompt_tok_len, output_tok_len,
                len(raw_text), diag, snippet,
            )
            continue
        try:
            rendered, spans = _rendered_chat_text(
                problem=problems[idx],
                tool_names=sol.get("tool_names") or [],
                tool_library=library,
                parsed=parsed,
                tokenizer=tokenizer,
            )
        except Exception as e:
            log.warning("[format_traces] Render failed for index %d: %s", idx, e)
            continue

        results[idx] = {
            "problem": problems[idx],
            "gold_answer": sol.get("answer"),
            "teacher_plan": parsed.get("plan", []),
            "teacher_tool_seq": [
                t.get("tool_name") for t in parsed["turns"] if t.get("tool_name")
            ],
            "teacher_text": rendered,
            "assistant_char_spans": spans,
            "tool_names_in_prompt": sol.get("tool_names") or [],
        }
    return results