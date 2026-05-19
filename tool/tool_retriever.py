"""4-step hierarchical tool retrieval pipeline.

Given a problem and a large ToolLibrary, the retriever selects the most
relevant tools through progressive filtering:

    Step 0 -- Task decomposition: planner LLM breaks the problem into typed
              sub-goals (description + input_type -> output_type).
    Step 1 -- L1 type filtering: match sub-goal types against tool signatures
              to prune the candidate set quickly.
    Step 2 -- L2 description scan (Round 1 prompt): LLM ranks candidates by
              their one-line NL summaries, shortlisting ~12 tools.
    Step 3 -- L3-L4 deep inspection (Round 2 prompt): LLM inspects full specs
              and examples for the shortlisted tools and makes the final
              selection.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from tool.tool_library import ToolLibrary


# ---------------------------------------------------------------------------
# Sub-goal representation
# ---------------------------------------------------------------------------
@dataclass
class SubGoal:
    """A single typed sub-goal produced by the planner."""

    description: str
    input_type: str   # e.g. "List[float]"
    output_type: str  # e.g. "float"


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
DECOMPOSE_PROMPT = """\
You are a task planner. Given a problem, decompose it into sequential sub-goals.
For each sub-goal, specify:
- description: what this step does
- input_type: Python type of the input (e.g. List[float], str, int)
- output_type: Python type of the output

Problem: {problem}

Respond with ONLY a JSON array, no other text:
[{{"description": "...", "input_type": "...", "output_type": "..."}}]"""

DESCRIPTION_SCAN_PROMPT = """\
You are selecting tools to solve a problem. Below are the sub-goals and \
candidate tools (name: description).

Sub-goals:
{sub_goals_text}

Candidate tools:
{tools_text}

Select the most relevant tools for these sub-goals (up to {max_shortlist} total).
Respond with ONLY a JSON array of tool names, no other text:
["tool1", "tool2", ...]"""

DEEP_INSPECT_PROMPT = """\
You are making the final tool selection. Below are candidate tools with full \
specifications and examples.

Problem: {problem}

{tools_detail_text}

Select the tools most useful for solving this problem (up to {max_final}).
Respond with ONLY a JSON array of tool names, no other text:
["tool1", "tool2", ...]"""


# ---------------------------------------------------------------------------
# Type compatibility
# ---------------------------------------------------------------------------
_TYPE_ALIASES: dict[str, str] = {
    "integer": "int",
    "string": "str",
    "boolean": "bool",
    "number": "float",
    "double": "float",
    "list": "List",
    "dict": "Dict",
    "tuple": "Tuple",
    "set": "Set",
    "none": "None",
}

_NUMERIC_PROMOTIONS = {"int", "float"}


def _normalize_type(t: str) -> str:
    """Normalize a type string for comparison."""
    t = t.strip()
    # Map common aliases
    low = t.lower()
    if low in _TYPE_ALIASES:
        return _TYPE_ALIASES[low]
    return t


def _parse_generic(t: str) -> tuple[str, list[str]]:
    """Parse ``Outer[Inner1, Inner2]`` into ``('Outer', ['Inner1', 'Inner2'])``.

    Returns ``(t, [])`` for non-generic types.
    """
    t = t.strip()
    bracket = t.find("[")
    if bracket == -1:
        return (t, [])
    outer = t[:bracket].strip()
    inner_str = t[bracket + 1 : -1].strip() if t.endswith("]") else t[bracket + 1 :].strip()
    # Split on top-level commas only (respect nested brackets)
    parts: list[str] = []
    depth, start = 0, 0
    for i, ch in enumerate(inner_str):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(inner_str[start:i].strip())
            start = i + 1
    parts.append(inner_str[start:].strip())
    return (outer, parts)


def types_compatible(query_type: str, tool_type: str) -> bool:
    """Check whether *query_type* is compatible with *tool_type*.

    Rules:
    - ``Any`` matches everything.
    - Bare types match exactly (with alias normalization).
    - Numeric types ``int`` and ``float`` are mutually compatible.
    - Generic types must match on outer name and recurse on parameters.
    """
    q = _normalize_type(query_type)
    t = _normalize_type(tool_type)

    if q.lower() == "any" or t.lower() == "any":
        return True
    if q == t:
        return True

    q_outer, q_params = _parse_generic(q)
    t_outer, t_params = _parse_generic(t)

    q_outer_n = _normalize_type(q_outer)
    t_outer_n = _normalize_type(t_outer)

    # Non-generic bare types
    if not q_params and not t_params:
        # Numeric promotion
        if q_outer_n in _NUMERIC_PROMOTIONS and t_outer_n in _NUMERIC_PROMOTIONS:
            return True
        return q_outer_n.lower() == t_outer_n.lower()

    # Outer container must match
    if q_outer_n.lower() != t_outer_n.lower():
        return False

    # If one side has no parameters, accept (e.g. List vs List[int])
    if not q_params or not t_params:
        return True

    # Recurse on parameters
    if len(q_params) != len(t_params):
        return False
    return all(types_compatible(qp, tp) for qp, tp in zip(q_params, t_params))


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------
def _extract_json_array(text: str) -> list[Any] | None:
    """Try to extract a JSON array from LLM output."""
    # Try direct parse first
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find [...] in the text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# ToolRetriever
# ---------------------------------------------------------------------------
class ToolRetriever:
    """Hierarchical 4-step tool retrieval pipeline.

    Parameters
    ----------
    tool_library:
        The :class:`ToolLibrary` to search.
    generate_fn:
        A callable ``(messages: list[dict]) -> str`` that runs LLM inference.
        Typically wraps ``Student._generate``.
    max_type_matches:
        Cap on type-matched candidates per sub-goal (Step 1).
    max_shortlist:
        Maximum tools to keep after description scan (Step 2).
    max_final:
        Maximum tools to select after deep inspection (Step 3).
    """

    def __init__(
        self,
        tool_library: ToolLibrary,
        generate_fn: Callable[[list[dict[str, str]]], str],
        max_type_matches: int = 500,
        max_shortlist: int = 12,
        max_final: int = 8,
        generate_batch_fn: Callable[
            [list[list[dict[str, str]]]], list[str]
        ] | None = None,
    ) -> None:
        self.tool_library = tool_library
        self.generate_fn = generate_fn
        self.generate_batch_fn = generate_batch_fn
        self.max_type_matches = max_type_matches
        self.max_shortlist = max_shortlist
        self.max_final = max_final

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def retrieve(self, problem: str) -> str:
        """Run the full 4-step pipeline and return formatted tool descriptions.

        Falls back to returning all tools (capped) if any step fails.
        """
        if len(self.tool_library) == 0:
            return "(no tools available)"

        # If the library is small enough, skip the pipeline
        if len(self.tool_library) <= self.max_final:
            return self._format_tools_full(list(self.tool_library.tool_names))

        # Step 0: Task decomposition
        sub_goals = self._step0_decompose(problem)

        # Step 1: L1 type filtering
        type_matches = self._step1_type_filter(sub_goals)
        all_candidates = set()
        for names in type_matches.values():
            all_candidates.update(names)

        # If type filtering returns too few, include all tools
        if not all_candidates:
            all_candidates = set(self.tool_library.tool_names)

        # Step 2: L2 description scan (Round 1)
        shortlisted = self._step2_description_scan(
            sub_goals, all_candidates, problem
        )

        # If shortlisting failed, use type-matched candidates directly
        if not shortlisted:
            shortlisted = list(all_candidates)[: self.max_shortlist]

        # Step 3: L3-L4 deep inspection (Round 2)
        final = self._step3_deep_inspect(shortlisted, problem)

        # If deep inspection failed, use shortlisted directly
        if not final:
            final = shortlisted[: self.max_final]

        # Algorithm 2 line 9 – materialise leaves: a composite's primitive
        # dependencies should also be in the candidate set so the student
        # has the full call chain.
        final = self._materialise_leaves(final)
        return self._format_tools_full(final)

    def hier_retrieve(
        self,
        query: str,
        *,
        shortlist: int | None = None,
        k: int | None = None,
    ) -> list[str]:
        """Formal Algorithm 2 signature ``HIERRETRIEVE(q, I, G, k)``.

        ``I`` and ``G`` are implicit in ``self.tool_library`` (they are kept
        in sync by :meth:`ToolLibrary.insert_tool`).  Returns the selected
        tool names (not formatted descriptions); callers that need the
        student-prompt format should use :meth:`retrieve` or
        :meth:`_format_tools_full` directly.
        """
        if shortlist is not None:
            prev_shortlist, self.max_shortlist = self.max_shortlist, shortlist
        if k is not None:
            prev_k, self.max_final = self.max_final, k
        try:
            if len(self.tool_library) == 0:
                return []
            if len(self.tool_library) <= self.max_final:
                return list(self.tool_library.tool_names)
            sub_goals = self._step0_decompose(query)
            type_matches = self._step1_type_filter(sub_goals)
            candidates: set[str] = set()
            for names in type_matches.values():
                candidates.update(names)
            if not candidates:
                candidates = set(self.tool_library.tool_names)
            shortlisted = self._step2_description_scan(sub_goals, candidates, query)
            if not shortlisted:
                shortlisted = list(candidates)[: self.max_shortlist]
            final = self._step3_deep_inspect(shortlisted, query)
            if not final:
                final = shortlisted[: self.max_final]
            return self._materialise_leaves(final)
        finally:
            if shortlist is not None:
                self.max_shortlist = prev_shortlist
            if k is not None:
                self.max_final = prev_k

    def _materialise_leaves(self, names: list[str]) -> list[str]:
        """For each composite in *names*, append its direct dependencies so
        the final candidate set includes the full primitive chain.
        """
        out: list[str] = list(names)
        seen = set(out)
        for name in list(names):
            for dep in self.tool_library.get_dependencies(name):
                if dep not in seen:
                    out.append(dep)
                    seen.add(dep)
        return out

    # ------------------------------------------------------------------
    # Batched entry point
    # ------------------------------------------------------------------
    def retrieve_batch(self, problems: list[str]) -> list[str]:
        """Run the 4-step pipeline for multiple problems using batched LLM calls.

        Returns a list of formatted tool description strings (one per problem).
        Requires ``generate_batch_fn`` to be set; falls back to sequential
        :meth:`retrieve` otherwise.
        """
        if self.generate_batch_fn is None:
            return [self.retrieve(p) for p in problems]

        n = len(problems)

        if len(self.tool_library) == 0:
            return ["(no tools available)"] * n

        # If the library is small enough, skip the pipeline for all
        if len(self.tool_library) <= self.max_final:
            desc = self._format_tools_full(list(self.tool_library.tool_names))
            return [desc] * n

        # --- Step 0: Batched task decomposition ---
        decompose_messages = [
            [{"role": "user", "content": DECOMPOSE_PROMPT.format(problem=p)}]
            for p in problems
        ]
        decompose_responses = self.generate_batch_fn(decompose_messages)

        all_sub_goals: list[list[SubGoal]] = []
        for i, response in enumerate(decompose_responses):
            parsed = _extract_json_array(response)
            sub_goals: list[SubGoal] = []
            if parsed:
                for item in parsed:
                    if isinstance(item, dict):
                        sub_goals.append(SubGoal(
                            description=item.get("description", ""),
                            input_type=item.get("input_type", "Any"),
                            output_type=item.get("output_type", "Any"),
                        ))
            if not sub_goals:
                sub_goals = [SubGoal(
                    description=problems[i],
                    input_type="Any",
                    output_type="Any",
                )]
            all_sub_goals.append(sub_goals)

        # --- Step 1: Type filtering (no LLM, per-problem) ---
        all_candidates: list[set[str]] = []
        for sub_goals in all_sub_goals:
            type_matches = self._step1_type_filter(sub_goals)
            candidates: set[str] = set()
            for names in type_matches.values():
                candidates.update(names)
            if not candidates:
                candidates = set(self.tool_library.tool_names)
            all_candidates.append(candidates)

        # --- Step 2: Batched description scan ---
        scan_messages: list[list[dict[str, str]]] = []
        for sub_goals, candidates in zip(all_sub_goals, all_candidates):
            sg_lines = [
                f"{i}. {sg.description}: {sg.input_type} -> {sg.output_type}"
                for i, sg in enumerate(sub_goals, 1)
            ]
            tool_lines = []
            for name in sorted(candidates):
                tool = self.tool_library.get_tool(name)
                summary = tool.metadata.description.summary or "No description"
                tool_lines.append(f"  {name}: {summary}")

            prompt = DESCRIPTION_SCAN_PROMPT.format(
                sub_goals_text="\n".join(sg_lines),
                tools_text="\n".join(tool_lines),
                max_shortlist=self.max_shortlist,
            )
            scan_messages.append([{"role": "user", "content": prompt}])

        scan_responses = self.generate_batch_fn(scan_messages)

        all_shortlisted: list[list[str]] = []
        for i, response in enumerate(scan_responses):
            parsed = _extract_json_array(response)
            shortlisted: list[str] = []
            if parsed:
                shortlisted = [
                    nm for nm in parsed
                    if isinstance(nm, str) and nm in self.tool_library
                ][:self.max_shortlist]
            if not shortlisted:
                shortlisted = list(all_candidates[i])[:self.max_shortlist]
            all_shortlisted.append(shortlisted)

        # --- Step 3: Batched deep inspection ---
        inspect_messages: list[list[dict[str, str]]] = []
        for shortlisted, problem in zip(all_shortlisted, problems):
            detail_lines = []
            for name in shortlisted:
                tool = self.tool_library.get_tool(name)
                meta = tool.metadata
                sig = meta.signature
                desc = meta.description
                spec = meta.specification

                block = [f"**{name}**"]
                block.append(
                    f"  Signature: ({sig.input_type}) -> {sig.output_type}"
                )
                block.append(f"  Description: {desc.summary}")
                if spec.preconditions:
                    block.append(f"  Pre: {'; '.join(spec.preconditions)}")
                if spec.postconditions:
                    block.append(f"  Post: {'; '.join(spec.postconditions)}")
                if spec.input_description:
                    block.append(f"  Input: {spec.input_description}")
                if spec.output_description:
                    block.append(f"  Output: {spec.output_description}")
                if spec.complexity:
                    block.append(f"  Complexity: {spec.complexity}")
                for ex in meta.examples[:3]:
                    ex_str = f"  Example: {ex.input} -> {ex.output}"
                    if ex.explanation:
                        ex_str += f"  ({ex.explanation})"
                    block.append(ex_str)
                detail_lines.append("\n".join(block))

            prompt = DEEP_INSPECT_PROMPT.format(
                problem=problem,
                tools_detail_text="\n\n".join(detail_lines),
                max_final=self.max_final,
            )
            inspect_messages.append([{"role": "user", "content": prompt}])

        inspect_responses = self.generate_batch_fn(inspect_messages)

        results: list[str] = []
        for i, response in enumerate(inspect_responses):
            parsed = _extract_json_array(response)
            final: list[str] = []
            if parsed:
                final = [
                    nm for nm in parsed
                    if isinstance(nm, str) and nm in self.tool_library
                ][:self.max_final]
            if not final:
                final = all_shortlisted[i][:self.max_final]
            results.append(self._format_tools_full(final))

        return results

    # ------------------------------------------------------------------
    # Step 0: Task decomposition
    # ------------------------------------------------------------------
    def _step0_decompose(self, problem: str) -> list[SubGoal]:
        """Use LLM to decompose the problem into typed sub-goals."""
        prompt = DECOMPOSE_PROMPT.format(problem=problem)
        messages = [{"role": "user", "content": prompt}]
        response = self.generate_fn(messages)

        parsed = _extract_json_array(response)
        if parsed:
            sub_goals = []
            for item in parsed:
                if isinstance(item, dict):
                    sub_goals.append(SubGoal(
                        description=item.get("description", ""),
                        input_type=item.get("input_type", "Any"),
                        output_type=item.get("output_type", "Any"),
                    ))
            if sub_goals:
                return sub_goals

        # Fallback: single catch-all sub-goal
        return [SubGoal(description=problem, input_type="Any", output_type="Any")]

    # ------------------------------------------------------------------
    # Step 1: L1 type filtering
    # ------------------------------------------------------------------
    def _step1_type_filter(
        self, sub_goals: list[SubGoal]
    ) -> dict[str, set[str]]:
        """Filter tools by type compatibility with each sub-goal.

        Uses the L1 inverted index on the library (``by_input`` /
        ``by_output``) to prune the candidate set before falling back to a
        full scan.  This keeps Algorithm 2 Step 1 linear in the matching
        types rather than in ``|V|``.
        """
        result: dict[str, set[str]] = {}
        index = self.tool_library.index

        for sg in sub_goals:
            # Fast path: exact I/O lookup via the index
            by_io = index.by_io.get((sg.input_type, sg.output_type))
            if by_io:
                result[sg.description] = set(list(by_io)[: self.max_type_matches])
                continue

            # Next best: union of type-compatible pools from L1 index keys
            candidates: set[str] = set()
            for in_t, names in index.by_input.items():
                if types_compatible(sg.input_type, in_t):
                    candidates.update(names)
                    if len(candidates) >= self.max_type_matches:
                        break
            if len(candidates) < self.max_type_matches:
                for out_t, names in index.by_output.items():
                    if types_compatible(sg.output_type, out_t):
                        candidates.update(names)
                        if len(candidates) >= self.max_type_matches:
                            break

            # Final fallback: full scan (rare; triggered for generic types).
            if not candidates:
                for name in self.tool_library.tool_names:
                    tool = self.tool_library.get_tool(name)
                    sig = tool.metadata.signature
                    if (
                        types_compatible(sg.input_type, sig.input_type)
                        and types_compatible(sg.output_type, sig.output_type)
                    ):
                        candidates.add(name)
                        if len(candidates) >= self.max_type_matches:
                            break

            result[sg.description] = set(list(candidates)[: self.max_type_matches])

        return result

    # ------------------------------------------------------------------
    # Step 2: L2 description scan (Round 1 prompt)
    # ------------------------------------------------------------------
    def _step2_description_scan(
        self,
        sub_goals: list[SubGoal],
        candidates: set[str],
        problem: str,
    ) -> list[str]:
        """LLM ranks candidates by their NL descriptions."""
        # Build sub-goals text
        sg_lines = []
        for i, sg in enumerate(sub_goals, 1):
            sg_lines.append(
                f"{i}. {sg.description}: {sg.input_type} -> {sg.output_type}"
            )
        sub_goals_text = "\n".join(sg_lines)

        # Build tools text (name + one-line summary, ~15 tokens each)
        tool_lines = []
        for name in sorted(candidates):
            tool = self.tool_library.get_tool(name)
            summary = tool.metadata.description.summary or "No description"
            tool_lines.append(f"  {name}: {summary}")
        tools_text = "\n".join(tool_lines)

        prompt = DESCRIPTION_SCAN_PROMPT.format(
            sub_goals_text=sub_goals_text,
            tools_text=tools_text,
            max_shortlist=self.max_shortlist,
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.generate_fn(messages)

        parsed = _extract_json_array(response)
        if parsed:
            # Filter to only valid tool names
            valid = [
                n for n in parsed
                if isinstance(n, str) and n in self.tool_library
            ]
            if valid:
                return valid[: self.max_shortlist]

        return []

    # ------------------------------------------------------------------
    # Step 3: L3-L4 deep inspection (Round 2 prompt)
    # ------------------------------------------------------------------
    def _step3_deep_inspect(
        self, shortlisted: list[str], problem: str
    ) -> list[str]:
        """LLM inspects full specs and examples for final selection."""
        # Build detailed tool descriptions
        detail_lines = []
        for name in shortlisted:
            tool = self.tool_library.get_tool(name)
            meta = tool.metadata
            sig = meta.signature
            desc = meta.description
            spec = meta.specification

            block = [f"**{name}**"]
            block.append(f"  Signature: ({sig.input_type}) -> {sig.output_type}")
            block.append(f"  Description: {desc.summary}")

            if spec.preconditions:
                block.append(f"  Pre: {'; '.join(spec.preconditions)}")
            if spec.postconditions:
                block.append(f"  Post: {'; '.join(spec.postconditions)}")
            if spec.input_description:
                block.append(f"  Input: {spec.input_description}")
            if spec.output_description:
                block.append(f"  Output: {spec.output_description}")
            if spec.complexity:
                block.append(f"  Complexity: {spec.complexity}")

            # L4: Examples
            for ex in meta.examples[:3]:  # limit to 3 examples per tool
                ex_str = f"  Example: {ex.input} -> {ex.output}"
                if ex.explanation:
                    ex_str += f"  ({ex.explanation})"
                block.append(ex_str)

            detail_lines.append("\n".join(block))

        tools_detail_text = "\n\n".join(detail_lines)

        prompt = DEEP_INSPECT_PROMPT.format(
            problem=problem,
            tools_detail_text=tools_detail_text,
            max_final=self.max_final,
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.generate_fn(messages)

        parsed = _extract_json_array(response)
        if parsed:
            valid = [
                n for n in parsed
                if isinstance(n, str) and n in self.tool_library
            ]
            if valid:
                return valid[: self.max_final]

        return []

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    def _format_tools_full(self, names: list[str]) -> str:
        """Format selected tools with L1 signature + L2 summary for the
        student's system prompt."""
        if not names:
            return "(no tools available)"

        lines: list[str] = []
        for name in names:
            tool = self.tool_library.get_tool(name)
            meta = tool.metadata
            sig = meta.signature
            desc = meta.description.summary or "No description"
            spec = meta.specification

            line = f"- **{name}**({sig.input_type}) -> {sig.output_type}: {desc}"
            if spec.input_description:
                line += f"\n  Input: {spec.input_description}"
            if spec.output_description:
                line += f"\n  Output: {spec.output_description}"
            lines.append(line)
        return "\n".join(lines)
