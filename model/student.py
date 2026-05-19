"""Student model – a smaller LLM that learns to solve problems by leveraging
skills (tools) from the ToolLibrary.

The student can operate in two modes:
1. **Direct mode** – solve problems using its own reasoning.
2. **Tool-augmented mode** – solve problems with access to skills from the
   ToolLibrary, calling them as tools during a ReAct-style loop.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from vllm import LLM, SamplingParams
from tool.tool_library import ToolLibrary
from tool.tool_retriever import ToolRetriever


@dataclass
class StudentConfig:
    """Configuration for the Student model."""

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: int = 2048
    max_steps: int = 10
    temperature: float = 0.0
    gpu_memory_utilization: float = 0.9
    max_tools: int = 50
    # Hierarchical retrieval settings
    use_hierarchical_retrieval: bool = True
    retrieval_max_type_matches: int = 500
    retrieval_max_shortlist: int = 12
    retrieval_max_final: int = 8


STUDENT_SYSTEM_PROMPT = (
    "You are a problem-solving assistant. Solve the given problem step by step.\n"
    "You have access to the following tools:\n"
    "{tool_descriptions}\n\n"
    "To use a tool, output a tool call in the following format:\n"
    '<tool_call>{{"name": "<tool_name>", "arguments": {{"arg1": value1}}}}</tool_call>\n\n'
    "After receiving the tool result, continue reasoning.\n"
    "When you have the final answer, output it clearly.\n"
)

STUDENT_DIRECT_PROMPT = (
    "You are a problem-solving assistant. "
    "Solve the given problem step by step and provide the final answer.\n"
)


class Student:
    """Student model that solves problems, optionally using tools from a
    :class:`ToolLibrary`.

    The model is loaded lazily on first use to avoid unnecessary GPU memory
    allocation.

    Parameters
    ----------
    config:
        A :class:`StudentConfig` controlling the model, generation params, etc.
    tool_library:
        An optional :class:`ToolLibrary` providing skills the student can call.
    """

    def __init__(
        self,
        config: StudentConfig | None = None,
        tool_library: ToolLibrary | None = None,
    ) -> None:
        self.config = config or StudentConfig()
        self.tool_library = tool_library
        self._model = None
        self._tokenizer = None
        self._retriever: ToolRetriever | None = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Load the student model and tokenizer via vLLM."""
        if self._model is not None:
            return

        self._model = LLM(
            model=self.config.model_name,
            dtype="auto",
            trust_remote_code=True,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
        )
        self._tokenizer = self._model.get_tokenizer()
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    # ------------------------------------------------------------------
    # Retriever initialisation
    # ------------------------------------------------------------------
    def _ensure_retriever(self) -> ToolRetriever | None:
        """Lazily create the :class:`ToolRetriever` (requires model loaded)."""
        if self._retriever is not None:
            return self._retriever
        if (
            not self.tool_library
            or len(self.tool_library) == 0
            or not self.config.use_hierarchical_retrieval
        ):
            return None
        self._load_model()
        self._retriever = ToolRetriever(
            tool_library=self.tool_library,
            generate_fn=self._generate,
            generate_batch_fn=self._generate_batch,
            max_type_matches=self.config.retrieval_max_type_matches,
            max_shortlist=self.config.retrieval_max_shortlist,
            max_final=self.config.retrieval_max_final,
        )
        return self._retriever

    # ------------------------------------------------------------------
    # Tool description builder
    # ------------------------------------------------------------------
    def _build_tool_descriptions(self, problem: str | None = None) -> str:
        """Build a formatted description of available tools for the prompt.

        When hierarchical retrieval is enabled and a *problem* is provided,
        runs the 4-step retrieval pipeline (decompose -> type filter ->
        description scan -> deep inspect).  Otherwise falls back to a simple
        recency-based listing.
        """
        if not self.tool_library or len(self.tool_library) == 0:
            return "(no tools available)"

        # Try hierarchical retrieval
        if problem and self.config.use_hierarchical_retrieval:
            retriever = self._ensure_retriever()
            if retriever:
                return retriever.retrieve(problem)

        # Fallback: naive recency-based listing
        names = list(self.tool_library.tool_names)
        if len(names) > self.config.max_tools:
            names = names[-self.config.max_tools:]

        lines: list[str] = []
        for name in names:
            tool = self.tool_library.get_tool(name)
            meta = tool.metadata
            sig = meta.signature
            desc = meta.description.summary or "No description"
            lines.append(
                f"- **{name}**({sig.input_type}) -> {sig.output_type}: {desc}"
            )
        return "\n".join(lines)

    def _build_tool_descriptions_batch(
        self, problems: list[str]
    ) -> list[str]:
        """Batched version of :meth:`_build_tool_descriptions`.

        Uses :meth:`ToolRetriever.retrieve_batch` so all retrieval LLM calls
        are batched across problems.
        """
        if not self.tool_library or len(self.tool_library) == 0:
            return ["(no tools available)"] * len(problems)

        if self.config.use_hierarchical_retrieval:
            retriever = self._ensure_retriever()
            if retriever:
                return retriever.retrieve_batch(problems)

        # Fallback: same naive listing for every problem
        desc = self._build_tool_descriptions(problem=None)
        return [desc] * len(problems)

    # ------------------------------------------------------------------
    # Chat prompt construction
    # ------------------------------------------------------------------
    def _build_messages(
        self, problem: str, *, use_tools: bool
    ) -> list[dict[str, str]]:
        """Build initial chat messages for the student."""
        if use_tools and self.tool_library and len(self.tool_library) > 0:
            system = STUDENT_SYSTEM_PROMPT.format(
                tool_descriptions=self._build_tool_descriptions(problem=problem)
            )
        else:
            system = STUDENT_DIRECT_PROMPT

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": problem},
        ]

    def _build_messages_batch(
        self, problems: list[str], *, use_tools: bool
    ) -> list[list[dict[str, str]]]:
        """Build initial chat messages for a batch of problems."""
        if use_tools and self.tool_library and len(self.tool_library) > 0:
            tool_descs = self._build_tool_descriptions_batch(problems)
            return [
                [
                    {"role": "system", "content": STUDENT_SYSTEM_PROMPT.format(
                        tool_descriptions=td
                    )},
                    {"role": "user", "content": p},
                ]
                for td, p in zip(tool_descs, problems)
            ]
        else:
            return [
                [
                    {"role": "system", "content": STUDENT_DIRECT_PROMPT},
                    {"role": "user", "content": p},
                ]
                for p in problems
            ]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _generate(self, messages: list[dict[str, str]]) -> str:
        """Run a single generation turn and return the assistant text."""
        self._load_model()
        assert self._model is not None and self._tokenizer is not None

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        sampling_params = SamplingParams(
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        outputs = self._model.generate([text], sampling_params)
        return outputs[0].outputs[0].text.strip()

    def _generate_batch(
        self, messages_list: list[list[dict[str, str]]]
    ) -> list[str]:
        """Run a batched generation and return assistant texts."""
        self._load_model()
        assert self._model is not None and self._tokenizer is not None

        prompts = [
            self._tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in messages_list
        ]
        sampling_params = SamplingParams(
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        outputs = self._model.generate(prompts, sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]

    # ------------------------------------------------------------------
    # Tool-call parsing & execution
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_tool_call(text: str) -> dict[str, Any] | None:
        """Extract the first ``<tool_call>...</tool_call>`` from *text*."""
        match = re.search(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL
        )
        if not match:
            return None
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None

    def _execute_tool_call(self, call: dict[str, Any]) -> str:
        """Execute a parsed tool call against the ToolLibrary."""
        name = call.get("name", "")
        args = call.get("arguments", {})

        if not self.tool_library or name not in self.tool_library:
            return f"Error: tool '{name}' not found."

        try:
            result = self.tool_library.execute(name, **args)
            return str(result)
        except Exception as e:
            return f"Error executing '{name}': {e}"

    # ------------------------------------------------------------------
    # Solve: direct mode (no tools)
    # ------------------------------------------------------------------
    def solve_direct(self, problem: str) -> dict[str, Any]:
        """Solve *problem* without using any tools.

        Returns a dict with ``answer`` and ``response`` keys.
        """
        messages = self._build_messages(problem, use_tools=False)
        response = self._generate(messages)
        return {"answer": response, "response": response}

    # ------------------------------------------------------------------
    # Solve: tool-augmented ReAct loop
    # ------------------------------------------------------------------
    def solve_with_tools(self, problem: str) -> dict[str, Any]:
        """Solve *problem* using a ReAct-style loop with ToolLibrary skills.

        The student generates text, and if it contains a ``<tool_call>`` block,
        the tool is executed and the observation is fed back as context. This
        continues until the student produces a final answer or ``max_steps``
        is reached.

        Returns a dict with:
        - ``answer``: final generated text.
        - ``steps``: list of step dicts (generation, tool_call, observation).
        """
        messages = self._build_messages(problem, use_tools=True)
        steps: list[dict[str, Any]] = []

        for step_idx in range(self.config.max_steps):
            response = self._generate(messages)

            step: dict[str, Any] = {"step": step_idx + 1, "generation": response}

            tool_call = self._parse_tool_call(response)
            if tool_call is None:
                # No tool call – treat as final answer
                steps.append(step)
                return {"answer": response, "steps": steps}

            # Execute the tool and feed observation back
            observation = self._execute_tool_call(tool_call)
            step["tool_call"] = tool_call
            step["observation"] = observation
            steps.append(step)

            # Append assistant + observation to messages for next turn
            messages.append({"role": "assistant", "content": response})
            messages.append(
                {"role": "user", "content": f"Tool result: {observation}"}
            )

        # Max steps reached – return last generation
        final = self._generate(messages)
        steps.append({"step": self.config.max_steps + 1, "generation": final})
        return {"answer": final, "steps": steps}

    # ------------------------------------------------------------------
    # Unified solve interface
    # ------------------------------------------------------------------
    def solve(self, problem: str, *, use_tools: bool = True) -> dict[str, Any]:
        """Solve *problem*.

        Parameters
        ----------
        use_tools:
            If *True* and a :class:`ToolLibrary` is available, run the
            tool-augmented ReAct loop. Otherwise solve directly.
        """
        if use_tools and self.tool_library and len(self.tool_library) > 0:
            return self.solve_with_tools(problem)
        return self.solve_direct(problem)

    # ------------------------------------------------------------------
    # Batched solve: direct mode
    # ------------------------------------------------------------------
    def solve_direct_batch(self, problems: list[str]) -> list[dict[str, Any]]:
        """Solve *problems* without tools in a single batched vLLM call."""
        messages_list = self._build_messages_batch(problems, use_tools=False)
        responses = self._generate_batch(messages_list)
        return [{"answer": r, "response": r} for r in responses]

    # ------------------------------------------------------------------
    # Batched solve: tool-augmented ReAct loop
    # ------------------------------------------------------------------
    def solve_with_tools_batch(
        self, problems: list[str]
    ) -> list[dict[str, Any]]:
        """Solve *problems* using a batched ReAct loop with tool calls.

        All active problems share a single ``vLLM.generate()`` call per step.
        Problems that finish (no tool call or max steps) are removed from the
        active set.
        """
        n = len(problems)
        # Build initial messages per problem (retrieval is batched)
        all_messages = self._build_messages_batch(problems, use_tools=True)
        all_steps: list[list[dict[str, Any]]] = [[] for _ in range(n)]
        results: list[dict[str, Any] | None] = [None] * n
        active_indices = list(range(n))

        for step_idx in range(self.config.max_steps):
            if not active_indices:
                break

            # Batched generation for all active problems
            active_messages = [all_messages[i] for i in active_indices]
            responses = self._generate_batch(active_messages)

            next_active: list[int] = []
            for idx, response in zip(active_indices, responses):
                step: dict[str, Any] = {
                    "step": step_idx + 1,
                    "generation": response,
                }

                tool_call = self._parse_tool_call(response)
                if tool_call is None:
                    # No tool call — final answer
                    all_steps[idx].append(step)
                    results[idx] = {
                        "answer": response,
                        "steps": all_steps[idx],
                    }
                    continue

                # Execute tool and feed observation back
                observation = self._execute_tool_call(tool_call)
                step["tool_call"] = tool_call
                step["observation"] = observation
                all_steps[idx].append(step)

                all_messages[idx].append(
                    {"role": "assistant", "content": response}
                )
                all_messages[idx].append(
                    {"role": "user", "content": f"Tool result: {observation}"}
                )
                next_active.append(idx)

            active_indices = next_active

        # Handle problems that hit max_steps
        if active_indices:
            final_messages = [all_messages[i] for i in active_indices]
            finals = self._generate_batch(final_messages)
            for idx, final in zip(active_indices, finals):
                all_steps[idx].append({
                    "step": self.config.max_steps + 1,
                    "generation": final,
                })
                results[idx] = {"answer": final, "steps": all_steps[idx]}

        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Unified batched solve
    # ------------------------------------------------------------------
    def solve_batch(
        self,
        problems: list[str],
        *,
        use_tools: bool = True,
    ) -> list[dict[str, Any]]:
        """Solve a batch of problems using batched vLLM calls."""
        if use_tools and self.tool_library and len(self.tool_library) > 0:
            return self.solve_with_tools_batch(problems)
        return self.solve_direct_batch(problems)
