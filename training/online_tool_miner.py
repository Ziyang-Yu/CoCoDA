"""Online tool distillation for Stage 3 co-evolution (Algorithm 1, lines 18–23).

When a student rollout succeeds (R_i >= rho) the trainer calls this module
to derive a candidate composite tool ``t+`` from the trajectory's tool-call
sequence.  The candidate goes through the Algorithm 3 insertion check
(duplicate lookup, dependency validation, compile) and is returned as a
serialised ``ToolEntry`` dict that the trainer broadcasts to every rank.

This miner purposefully avoids spinning up a second LLM inside the GRPO
loop — the teacher-based extraction path is already available in
:mod:`model.teacher` for the offline warm-start.  Online we synthesise a
minimal composite by stringing the student's primitive calls together; this
is enough to grow the library during training without blowing up wall-clock
or memory.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from tool.tool import Tool
from tool.tool_library import ToolLibrary, ToolType
from tool.tool_metadata import (
    ToolDescription,
    ToolExample,
    ToolMetadata,
    ToolSignature,
    ToolSpecification,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Teacher-backed miner: reuse the Stage-1 Teacher vLLM process to distil
# a new tool from a successful GRPO trajectory (Algorithm 1 L19, faithful
# to the paper — higher-quality tools than the heuristic chain below).
# ---------------------------------------------------------------------------
def _trajectory_to_pseudo_code(trajectory: dict) -> str:
    """Serialise a rollout's thoughts + tool calls into a Python-ish snippet
    that the teacher's TOOL_EXTRACTION_PROMPT can consume.

    The teacher was trained to read problem + solution-code pairs, so we
    present the trajectory in that shape even though it was produced by the
    student's ReAct loop rather than smolagents.
    """
    thoughts = trajectory.get("thoughts") or []
    tool_calls = trajectory.get("tool_calls") or []
    lines: list[str] = []
    step = 0
    for th, call in zip(thoughts, tool_calls):
        if th:
            for t_line in str(th).strip().splitlines():
                lines.append(f"# {t_line}")
        name = call.get("name") or call.get("tool") or "unknown_tool"
        args = call.get("arguments") or call.get("args") or {}
        try:
            args_repr = ", ".join(
                f"{k}={v!r}" for k, v in args.items()
            ) if isinstance(args, dict) else repr(args)
        except Exception:
            args_repr = str(args)
        lines.append(f"step_{step} = {name}({args_repr})")
        step += 1
    answer = trajectory.get("answer")
    if answer is not None:
        lines.append(f"answer = {answer!r}")
    return "\n".join(lines)


def mine_with_teacher(
    teacher: "Any",
    trajectory: dict,
    problem: str,
    library: ToolLibrary,
) -> list[dict]:
    """Run the Stage-1 teacher's TOOL_EXTRACTION_PROMPT on a successful
    trajectory and return a list of serialised :class:`ToolEntry` dicts
    ready for cross-rank broadcast.

    The teacher instance is the *same* object used during warm-start
    (:class:`model.teacher.Teacher`); its vLLM process is reused after
    warm-start has finished.  Each extracted tool is duplicate-checked
    against the current library via :meth:`ToolLibrary.insert_tool`
    (Algorithm 3) before being serialised.
    """
    code = _trajectory_to_pseudo_code(trajectory)
    if not code.strip():
        return []

    # Point the teacher at the live library so duplicate checks see every
    # tool — including ones mined earlier in this GRPO run.
    prev_lib = teacher.tool_library
    teacher.tool_library = library
    entries: list[dict] = []
    try:
        extracted = teacher.extract_tool(problem, code)
        for et in extracted:
            try:
                name = teacher.add_tool_to_library(et)
            except Exception as exc:
                log.warning("Teacher miner skip: %s", exc)
                continue
            if name is None or name not in library:
                continue
            entries.append({
                "name": name,
                "entry": library.get_entry(name).to_dict(),
            })
    finally:
        teacher.tool_library = prev_lib
    return entries

def mine_with_teacher_batch(
    teacher: "Any",
    trajectories: list[dict],
    problems: list[str],
    library: ToolLibrary,
) -> list[list[dict]]:
    """Batched variant of :func:`mine_with_teacher`.

    Collapses N sequential TOOL_EXTRACTION_PROMPT calls into **one** vLLM
    ``generate()`` via :meth:`Teacher.extract_tools_batch`.  Returns a list
    parallel to *trajectories*; entry *i* is the list of
    ``{name, entry}`` dicts mined from ``trajectories[i]`` (possibly empty).

    Note
    ----
    In the sequential path, sample *j*'s extraction prompt sees tools added
    from samples *0..j-1* within the same step.  The batched version renders
    every prompt against the pre-batch library snapshot, so within-step
    cross-sample awareness is lost — but across outer steps (where most
    library growth actually happens) behaviour is unchanged.  Duplicate
    inserts are still caught by :meth:`Teacher.add_tool_to_library`.
    """
    if len(trajectories) != len(problems):
        raise ValueError(
            f"trajectories ({len(trajectories)}) and problems "
            f"({len(problems)}) must align"
        )
    out: list[list[dict]] = [[] for _ in trajectories]
    if not trajectories:
        return out

    # Build pseudo-codes, dropping empty ones but keeping index alignment.
    kept_indices: list[int] = []
    kept_problems: list[str] = []
    kept_codes: list[str] = []
    for i, (traj, prob) in enumerate(zip(trajectories, problems)):
        code = _trajectory_to_pseudo_code(traj)
        if code.strip():
            kept_indices.append(i)
            kept_problems.append(prob)
            kept_codes.append(code)
    if not kept_codes:
        return out

    prev_lib = teacher.tool_library
    teacher.tool_library = library
    try:
        batch_extracted = teacher.extract_tools_batch(kept_problems, kept_codes)
        for slot_idx, extracted in zip(kept_indices, batch_extracted):
            for et in extracted:
                try:
                    name = teacher.add_tool_to_library(et)
                except Exception as exc:
                    log.warning("Teacher miner skip: %s", exc)
                    continue
                if name is None or name not in library:
                    continue
                out[slot_idx].append({
                    "name": name,
                    "entry": library.get_entry(name).to_dict(),
                })
    finally:
        teacher.tool_library = prev_lib
    return out


_SAFE_NAME_RE = re.compile(r"[^0-9a-zA-Z_]")


def _slugify(parts: list[str]) -> str:
    base = "_".join(_SAFE_NAME_RE.sub("_", p) for p in parts if p)
    base = re.sub(r"_+", "_", base).strip("_")
    return base or "composite"


def _pick_primitive_sequence(
    tool_calls: list[dict[str, Any]], library: ToolLibrary
) -> list[str]:
    """Return the ordered primitive-tool names used in this rollout.

    Non-primitive calls (composites from prior iterations) are skipped so
    that the synthesised composite bottoms out at primitives, matching
    Algorithm 3's L3/L4 "materialise leaves" step.
    """
    seq: list[str] = []
    for call in tool_calls:
        name = call.get("name") or call.get("tool")
        if not name or name not in library:
            continue
        if library.get_type(name) is ToolType.PRIMITIVE:
            seq.append(name)
    return seq


def mine_composite_from_trajectory(
    trajectory: dict[str, Any],
    library: ToolLibrary,
    problem: str = "",
) -> dict[str, Any] | None:
    """Build a candidate composite :class:`ToolEntry` dict from *trajectory*.

    Returns ``None`` if the trajectory has fewer than two distinct primitive
    calls (nothing new to compose) or if the resulting name collides with
    an existing library entry (Algorithm 3 duplicate check).
    """
    calls = trajectory.get("tool_calls") or []
    seq = _pick_primitive_sequence(calls, library)
    if len(seq) < 2:
        return None

    unique = list(dict.fromkeys(seq))
    if len(unique) < 2:
        return None

    name = "co_" + _slugify(unique)
    if name in library:
        return None

    first = library.get_tool(unique[0]).metadata.signature
    last = library.get_tool(unique[-1]).metadata.signature

    # Algorithm 3 duplicate check against the hierarchical index I.
    dupes = library.hier_retrieve_candidate(
        name=name,
        input_type=first.input_type or "Any",
        output_type=last.output_type or "Any",
        summary=f"Composite of {' -> '.join(unique)}",
        tags=["online", "composite"],
        domain="auto",
        shortlist=4,
    )
    for cand in dupes:
        if library.get_dependencies(cand) == unique:
            return None

    input_type = first.input_type or "Any"
    output_type = last.output_type or "Any"

    body_lines = [f"def {name}(x):"]
    body_lines.append(f'    """Composite mined from a successful rollout."""')
    body_lines.append("    out = x")
    for dep in unique:
        body_lines.append(f"    out = {dep}(out)")
    body_lines.append("    return out")
    code = "\n".join(body_lines)

    signature = ToolSignature(
        name=name,
        input_type=input_type,
        output_type=output_type,
        dependencies=unique,
    )
    description = ToolDescription(
        summary=f"Composite of {' -> '.join(unique)} discovered during GRPO.",
        tags=["online", "composite"],
        domain="auto",
    )
    specification = ToolSpecification(complexity="O(len(chain))")
    examples: list[ToolExample] = []
    if problem:
        examples.append(
            ToolExample(
                input="<input>",
                output="<output>",
                explanation=f"Applied to: {problem[:80]}",
            )
        )

    tool_obj = Tool(
        metadata=ToolMetadata(
            signature=signature,
            description=description,
            specification=specification,
            examples=examples,
        ),
        code=code,
    )

    try:
        tool_obj.compile()
    except Exception as exc:
        log.warning("Online miner failed to compile %s: %s", name, exc)
        return None

    return {"tool": tool_obj.to_dict(), "tool_type": ToolType.COMPOSITE.value}
