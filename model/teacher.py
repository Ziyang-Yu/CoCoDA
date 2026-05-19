"""Teacher model – uses smolagents CodeAgent to solve problems step-by-step,
then summarises the generated code into reusable tools and inserts them into
the ToolLibrary.

Supports both single-problem solving (via smolagents CodeAgent) and
**batched** solving (via :class:`BatchCodeAgent`) where multiple problems
are fed to vLLM in a single ``generate()`` call per reasoning step.
"""

from __future__ import annotations

import logging
import re
import textwrap
from typing import Any

from smolagents import ChatMessage, CodeAgent, tool
from smolagents.agents import BatchCodeAgent
from vllm import LLM, SamplingParams

from tool.tool import Tool
from tool.tool_library import ToolLibrary
from tool.tool_metadata import (
    ToolDescription,
    ToolExample,
    ToolMetadata,
    ToolSignature,
    ToolSpecification,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default system prompt that instructs the teacher to work step-by-step
# ---------------------------------------------------------------------------
TEACHER_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert programmer and problem solver.
    When given a problem you MUST:
    1. Break the problem into clear sub-steps.
    2. Implement each sub-step as a self-contained Python code block.
    3. After solving the problem, print the final answer.

    Think step by step. Write clean, well-commented code.
""")

# Prompt used to ask the teacher LLM to summarise code into a reusable tool
TOOL_EXTRACTION_PROMPT = textwrap.dedent("""\
    You are a code analyst. Given the following problem and its step-by-step
    solution code, extract **all distinct reusable Python functions** that
    generalise the solution patterns.

    ## Problem
    {problem}

    ## Solution code
    ```python
    {code}
    ```

    ## Existing primitive tools in library
    {library_primitives}

    ## Instructions
    1. Identify every distinct reusable pattern in the solution code.
       This may be one function or several (e.g. a helper utility AND
       a core algorithm). Only create multiple functions when they are
       genuinely independent; do NOT split a single algorithm into
       artificial pieces.
    2. For EACH function:
       - Use a clear, descriptive snake_case name.
       - Include a docstring explaining what it does, its parameters,
         and return value.
       - Accept generic inputs (not hard-coded to this specific problem).
       - Return the result (not print it).
    3. **Classify** each function as either:
       - **primitive**: a self-contained atomic function that does NOT
         call any existing primitive tools from the library above.
       - **composite**: a higher-level function that internally calls
         one or more existing primitive tools listed above. If a function
         composes existing primitives, it MUST be marked as composite and
         list those primitives in the dependencies field.
    4. After EACH function definition, immediately provide its metadata
       as a Python comment block:
       ```
       # TOOL_META
       # name: <function_name>
       # summary: <one-line description>
       # tool_type: <primitive OR composite>
       # dependencies: <comma-separated names of called primitive tools, or empty>
       # domain: <e.g. math, string, graph, dp, ...>
       # tags: <comma-separated tags>
       # input_type: <type annotation string>
       # output_type: <type annotation string>
       # complexity: <e.g. O(n), O(n log n), ...>
       ```
    5. After the TOOL_META block, provide 1-3 usage examples as a
       TOOL_EXAMPLES block. Each example has an input, expected output,
       and a brief explanation. Use the problem above to derive at least
       one concrete example. Format:
       ```
       # TOOL_EXAMPLES
       # example: <input_expression> -> <expected_output> | <brief explanation>
       # example: <input_expression> -> <expected_output> | <brief explanation>
       ```
       The input_expression should be a valid Python call, and the
       expected_output should be the repr of the return value.

    Return ONLY the function definitions, their TOOL_META comment
    blocks, and TOOL_EXAMPLES blocks. Separate multiple
    function–metadata pairs with a blank line.
""")


class VLLMModel:
    """Lightweight vLLM wrapper compatible with smolagents ``CodeAgent``.

    Loads a local model via vLLM offline inference and exposes the
    ``__call__`` interface that ``CodeAgent`` expects (returns a
    ``ChatMessage``).
    """

    def __init__(
        self,
        model_id: str,
        gpu_memory_utilization: float = 0.9,
        enable_sleep_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        self.model_id = model_id
        self._sleep_supported = enable_sleep_mode
        # When data_parallel_size > 1, vLLM requires external_launcher backend
        dp_size = kwargs.get("data_parallel_size", 1)
        if dp_size > 1:
            kwargs.setdefault(
                "distributed_executor_backend", "external_launcher",
            )
        # Disable the flashinfer allreduce+rmsnorm fusion pass. vLLM auto-enables
        # it at TP>1 on Hopper/Blackwell when flashinfer is installed, which
        # triggers a JIT compile of trtllm_allreduce_fusion.cuh. The header
        # shipped with the pinned flashinfer build fails to compile against the
        # installed CUDA toolchain (`std::optional` not visible in device code,
        # `AllReduceFusionPattern` undefined). vLLM otherwise just falls back
        # to native NCCL allreduce, so disabling the pass is the no-op fix.
        kwargs.setdefault(
            "compilation_config",
            {"pass_config": {"fuse_allreduce_rms": False}},
        )
        # Explicitly set max_model_len so long rewrite prompts (full CodeAgent
        # trajectories + tool lists + schema boilerplate) don't get clamped by
        # vLLM's auto-sized KV cache. Without this, vLLM picks ~4-8k based on
        # available memory, and format_traces.py silently truncated outputs
        # mid-JSON because prompt_len + max_tokens exceeded the window.
        # Caller can override via kwargs if they need a different value.
        kwargs.setdefault("max_model_len", 16384)
        if enable_sleep_mode:
            kwargs.setdefault("enable_sleep_mode", True)
        self._llm = LLM(
            model=model_id,
            dtype="auto",
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            **kwargs,
        )
        self._tokenizer = self._llm.get_tokenizer()

    def sleep(self, level: int = 2) -> None:
        """Offload weights (level=2) or just free KV cache (level=1) to CPU.

        Requires ``enable_sleep_mode=True`` at construction.  A no-op if the
        underlying engine wasn't built with sleep support.
        """
        if not self._sleep_supported:
            return
        self._llm.sleep(level=level)

    def wake_up(self) -> None:
        """Restore weights and KV cache allocation."""
        if not self._sleep_supported:
            return
        self._llm.wake_up()

    def generate(
        self,
        messages: list[dict[str, str]],
        stop_sequences: list[str] | None = None,
        grammar: str | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        # Normalise messages: smolagents may pass ChatMessage objects
        # and/or content as a list (OpenAI multimodal format).
        normalised = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                # ChatMessage or similar object with attributes
                role = getattr(msg, "role", "user")
                content = getattr(msg, "content", "")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            normalised.append({"role": role, "content": content})

        prompt = self._tokenizer.apply_chat_template(
            normalised, tokenize=False, add_generation_prompt=True
        )
        params = SamplingParams(
            max_tokens=4096,
            temperature=0.0,
            stop=stop_sequences or [],
        )
        outputs = self._llm.generate([prompt], params)
        content = outputs[0].outputs[0].text
        # Strip Qwen3-style <think>...</think> blocks so that smolagents'
        # CodeAgent parser can find the <code>...</code> tags it expects.
        # First remove properly closed blocks, then remove any unclosed
        # trailing <think> block (the model sometimes omits </think>).
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
        content = re.sub(r"<think>.*", "", content, flags=re.DOTALL)
        return ChatMessage(role="assistant", content=content)


class Teacher:
    """Teacher model that uses a smolagents ``CodeAgent`` to solve problems
    step-by-step and then distil the solution into reusable tools stored in a
    :class:`ToolLibrary`.

    Parameters
    ----------
    model_id:
        The local model identifier for vLLM
        (e.g. ``"Qwen/Qwen2.5-7B-Instruct"``).
    tool_library:
        An optional pre-existing :class:`ToolLibrary` to insert tools into.
        If *None*, a fresh empty library is created.
    additional_tools:
        Extra smolagents tools to make available to the ``CodeAgent``.
    max_steps:
        Maximum reasoning steps for the ``CodeAgent``.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        tool_library: ToolLibrary | None = None,
        additional_tools: list[Any] | None = None,
        max_steps: int = 20,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        enable_sleep_mode: bool = False,
    ) -> None:
        self.model_id = model_id
        self.tool_library = tool_library or ToolLibrary()
        self.max_steps = max_steps

        # Build the underlying smolagents CodeAgent with vLLM
        self._llm = VLLMModel(
            model_id=model_id,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            enable_sleep_mode=enable_sleep_mode,
        )
        self._agent = CodeAgent(
            tools=additional_tools or [],
            model=self._llm,
            max_steps=max_steps,
            additional_authorized_imports=["math", "itertools", "collections",
                                           "functools", "re", "heapq",
                                           "bisect", "string", "operator"],
        )

    def sleep(self, level: int = 2) -> None:
        """Offload weights (level=2) / free KV cache (level=1) to CPU."""
        self._llm.sleep(level=level)

    def wake_up(self) -> None:
        """Restore weights and KV cache allocation."""
        self._llm.wake_up()

    # ------------------------------------------------------------------
    # Core: solve a problem step-by-step
    # ------------------------------------------------------------------
    def solve(self, problem: str) -> dict[str, Any]:
        """Solve *problem* using the CodeAgent.

        Returns a dict with keys:
        - ``answer``: the final answer produced by the agent.
        - ``steps``: list of logged step dicts from the agent run.
        - ``code``: concatenated Python code from all code steps.
        """
        result = self._agent.run(
            TEACHER_SYSTEM_PROMPT + "\n\nProblem:\n" + problem,
            reset=True,
        )

        # Collect code snippets from agent memory steps
        code_snippets: list[str] = []
        steps: list[dict[str, Any]] = []
        for step_log in self._agent.memory.steps:
            step_info: dict[str, Any] = {}
            if hasattr(step_log, "tool_calls"):
                step_info["tool_calls"] = str(step_log.tool_calls)

            # CodeAgent stores generated code as code_action in ActionStep
            code = getattr(step_log, "code_action", None)
            if code:
                code_snippets.append(code)
                step_info["code"] = code

            observation = getattr(step_log, "observations", None)
            if observation:
                step_info["observation"] = observation

            if step_info:
                steps.append(step_info)

        return {
            "answer": result,
            "steps": steps,
            "code": "\n\n".join(code_snippets),
        }

    # ------------------------------------------------------------------
    # Tool extraction: summarise code -> reusable function + metadata
    # ------------------------------------------------------------------
    def _format_library_primitives(self) -> str:
        """Return a summary of existing primitive tools for the extraction prompt."""
        prims = self.tool_library.primitive_names
        if not prims:
            return "(none yet)"
        lines = []
        for name in sorted(prims):
            entry = self.tool_library.get_entry(name)
            summary = entry.tool.metadata.description.summary or ""
            lines.append(f"- {name}: {summary}")
        return "\n".join(lines)

    def extract_tool(self, problem: str, code: str) -> list[dict[str, Any]]:
        """Ask the LLM to distil *code* into reusable tools.

        Returns a list of dicts, each with ``function_code`` and ``metadata``
        keys. The list may be empty if extraction fails entirely.
        """
        prompt = TOOL_EXTRACTION_PROMPT.format(
            problem=problem,
            code=code,
            library_primitives=self._format_library_primitives(),
        )

        # Use the raw LLM (not the agent) for a single-turn generation
        messages = [{"role": "user", "content": prompt}]
        raw_response = self._llm.generate(messages)

        # Handle both string and ChatMessage responses
        response_text = raw_response
        if hasattr(raw_response, "content"):
            response_text = raw_response.content

        return self._parse_tool_response(response_text)

    def _parse_tool_response(self, response: str) -> list[dict[str, Any]]:
        """Parse the LLM response into one or more extracted tool dicts.

        The response may contain multiple function + ``# TOOL_META`` pairs.
        Each pair is parsed independently; malformed pairs are silently
        skipped.
        """
        parts = response.split("# TOOL_META")
        if len(parts) < 2:
            return []

        tools: list[dict[str, Any]] = []
        # parts[0] is code before the first TOOL_META; parts[i] (i>=1) is
        # the meta block for the i-th tool, possibly followed by the next
        # tool's code.  We pair code[i] with meta from parts[i+1].
        for i in range(1, len(parts)):
            meta_and_next = parts[i]

            # --- extract metadata from the beginning of this part ----------
            meta_lines: list[str] = []
            remaining_lines: list[str] = []
            in_meta = True
            for line in meta_and_next.splitlines():
                stripped = line.strip()
                if in_meta:
                    # Meta lines are comment lines (possibly blank)
                    if stripped == "" or stripped.startswith("#"):
                        meta_lines.append(line)
                    else:
                        in_meta = False
                        remaining_lines.append(line)
                else:
                    remaining_lines.append(line)

            meta: dict[str, str] = {}
            for line in meta_lines:
                line = line.strip().lstrip("#").strip()
                if ":" in line and not line.startswith(("TOOL_META", "TOOL_EXAMPLES", "example:")):
                    key, _, value = line.partition(":")
                    meta[key.strip()] = value.strip()

            # --- extract function code for this tool ----------------------
            # The code sits at the end of the *previous* part.  For the first
            # tool (i==1) that is parts[0]; for subsequent tools it is the
            # remaining_lines of the previous iteration that we append to a
            # running buffer.  To keep things simple we combine the previous
            # part's tail with any remaining lines from the prior meta section.
            if i == 1:
                function_code = parts[0].strip()
            else:
                # Code for tool i is whatever appeared after the (i-1)-th
                # meta block – we saved that in `_prev_remaining`.
                function_code = _prev_remaining.strip()  # noqa: F821 (set in prior iteration)

            # Save remaining lines for the *next* iteration
            _prev_remaining = "\n".join(remaining_lines)  # noqa: F841

            # Clean up markdown fences
            function_code = re.sub(r"^```python\s*\n?", "", function_code)
            function_code = re.sub(r"\n?```\s*$", "", function_code)
            function_code = function_code.strip()

            if not function_code:
                continue

            if "name" not in meta:
                match = re.search(r"def\s+(\w+)\s*\(", function_code)
                if match:
                    meta["name"] = match.group(1)
                else:
                    continue

            # --- extract examples from TOOL_EXAMPLES block ----------------
            examples: list[dict[str, str]] = []
            for line in meta_lines:
                line_text = line.strip().lstrip("#").strip()
                if line_text.startswith("example:"):
                    ex_body = line_text[len("example:"):].strip()
                    # Format: input_expr -> expected_output | explanation
                    if "->" in ex_body:
                        input_part, _, rest = ex_body.partition("->")
                        input_part = input_part.strip()
                        if "|" in rest:
                            output_part, _, explanation = rest.partition("|")
                        else:
                            output_part = rest
                            explanation = ""
                        examples.append({
                            "input": input_part.strip(),
                            "output": output_part.strip(),
                            "explanation": explanation.strip(),
                        })

            tools.append({
                "function_code": function_code,
                "metadata": meta,
                "examples": examples,
            })

        return tools

    # ------------------------------------------------------------------
    # Insert an extracted tool into the ToolLibrary
    # ------------------------------------------------------------------
    def _build_tool_from_extracted(self, extracted: dict[str, Any]) -> Tool:
        """Convert an extracted tool dict into a :class:`Tool`."""
        meta = extracted["metadata"]
        name = meta["name"]

        # Parse dependencies from LLM-provided metadata
        raw_deps = meta.get("dependencies", "")
        dependencies = [
            d.strip() for d in raw_deps.split(",") if d.strip()
        ] if raw_deps else []

        signature = ToolSignature(
            name=name,
            input_type=meta.get("input_type", "Any"),
            output_type=meta.get("output_type", "Any"),
            dependencies=dependencies,
        )
        description = ToolDescription(
            summary=meta.get("summary", ""),
            tags=[t.strip() for t in meta.get("tags", "").split(",") if t.strip()],
            domain=meta.get("domain", ""),
        )
        specification = ToolSpecification(
            complexity=meta.get("complexity", ""),
        )
        examples = [
            ToolExample(
                input=ex["input"],
                output=ex["output"],
                explanation=ex.get("explanation", ""),
            )
            for ex in extracted.get("examples", [])
        ]
        tool_metadata = ToolMetadata(
            signature=signature,
            description=description,
            specification=specification,
            examples=examples,
        )

        return Tool(metadata=tool_metadata, code=extracted["function_code"])

    def add_tool_to_library(self, extracted: dict[str, Any]) -> str:
        """Build a :class:`Tool` from an extracted dict and add it to the library.

        Classifies the tool as composite or primitive based on the LLM's
        ``tool_type`` metadata. A tool is added as composite if the LLM
        labelled it "composite" AND all its declared dependencies exist as
        primitives in the library; otherwise it falls back to primitive.

        Returns the name of the newly added tool.
        """
        tool_obj = self._build_tool_from_extracted(extracted)
        name = tool_obj.metadata.signature.name

        # Verify the code compiles before adding
        tool_obj.compile()

        meta = extracted["metadata"]
        llm_type = meta.get("tool_type", "primitive").strip().lower()
        deps = tool_obj.metadata.signature.dependencies

        if llm_type == "composite" and deps:
            # Validate that every dependency is a registered primitive
            missing = [d for d in deps if d not in self.tool_library or
                       self.tool_library.get_type(d).value != "primitive"]
            if missing:
                log.warning(
                    "Composite tool '%s' has unresolved dependencies %s; "
                    "falling back to primitive.",
                    name, missing,
                )
                tool_obj.metadata.signature.dependencies = []
                self.tool_library.add_primitive(tool_obj)
            else:
                self.tool_library.add_composite(tool_obj)
        else:
            # Clear any stray dependencies for a primitive
            tool_obj.metadata.signature.dependencies = []
            self.tool_library.add_primitive(tool_obj)

        return name

    # ------------------------------------------------------------------
    # End-to-end: solve + extract + store
    # ------------------------------------------------------------------
    def solve_and_learn(self, problem: str) -> dict[str, Any]:
        """Solve a problem, extract reusable tools, and store them.

        Returns a dict with:
        - ``answer``: the solution answer.
        - ``tool_names``: list of names of successfully stored tools.
        - ``tool_name``: first stored tool name (or *None*) – kept for
          backward compatibility.
        - ``steps``: the agent reasoning steps.
        - ``code``: the concatenated code.
        - ``tools``: list of raw extracted tool dicts.
        - ``tool``: first extracted tool dict (or *None*) – kept for
          backward compatibility.
        """
        solution = self.solve(problem)

        extracted_tools: list[dict[str, Any]] = []
        tool_names: list[str] = []
        if solution["code"]:
            extracted_tools = self.extract_tool(problem, solution["code"])
            for et in extracted_tools:
                try:
                    tool_names.append(self.add_tool_to_library(et))
                except Exception:
                    # Tool failed to compile or register – skip it
                    pass

        return {
            "answer": solution["answer"],
            "tool_names": tool_names,
            "tool_name": tool_names[0] if tool_names else None,
            "steps": solution["steps"],
            "code": solution["code"],
            "tools": extracted_tools,
            "tool": extracted_tools[0] if extracted_tools else None,
        }

    def solve_batch(
        self,
        problems: list[str],
        *,
        learn: bool = True,
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        """Solve a batch of problems, optionally extracting tools.

        Parameters
        ----------
        problems:
            List of problem descriptions.
        learn:
            If *True* (default), run :meth:`solve_and_learn` which extracts
            and stores tools. Otherwise just solve.
        show_progress:
            Print progress to stdout.
        """
        results: list[dict[str, Any]] = []
        for i, problem in enumerate(problems):
            if show_progress:
                print(f"[Teacher] Solving {i + 1}/{len(problems)}...")
            if learn:
                results.append(self.solve_and_learn(problem))
            else:
                results.append(self.solve(problem))
        return results

    # ------------------------------------------------------------------
    # Batched solving via BatchCodeAgent
    # ------------------------------------------------------------------
    def _ensure_batch_agent(self) -> BatchCodeAgent:
        """Lazily create the :class:`BatchCodeAgent`."""
        if not hasattr(self, "_batch_agent"):
            self._batch_agent = BatchCodeAgent(
                llm=self._llm._llm,
                tokenizer=self._llm._tokenizer,
                additional_authorized_imports=[
                    "math", "itertools", "collections",
                    "functools", "re", "heapq",
                    "bisect", "string", "operator",
                ],
                max_steps=self.max_steps,
            )
        return self._batch_agent

    def solve_batch_parallel(
        self,
        problems: list[str],
    ) -> list[dict[str, Any]]:
        """Solve *problems* in parallel using batched vLLM calls.

        Each problem goes through the same multi-step Thought→Code→Observe
        loop as :meth:`solve`, but all active problems share a single
        ``vllm.LLM.generate()`` call per step.

        Returns a list of dicts with keys ``answer``, ``code``, ``steps``.
        """
        agent = self._ensure_batch_agent()
        tasks = [
            TEACHER_SYSTEM_PROMPT + "\n\nProblem:\n" + p
            for p in problems
        ]
        return agent.run_batch(tasks)

    def extract_tools_batch(
        self,
        problems: list[str],
        codes: list[str],
    ) -> list[list[dict[str, Any]]]:
        """Extract tools from multiple (problem, code) pairs in one
        batched vLLM call.

        Returns a list (one per problem) of extracted tool lists.
        """
        library_primitives = self._format_library_primitives()
        prompts_text = [
            TOOL_EXTRACTION_PROMPT.format(
                problem=p, code=c, library_primitives=library_primitives,
            )
            for p, c in zip(problems, codes)
        ]
        if not prompts_text:
            return []

        # Build chat-format prompts and batch generate
        raw_prompts = []
        for text in prompts_text:
            messages = [{"role": "user", "content": text}]
            raw_prompts.append(
                self._llm._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            )

        params = SamplingParams(max_tokens=4096, temperature=0.0)
        outputs = self._llm._llm.generate(raw_prompts, params)

        all_tools: list[list[dict[str, Any]]] = []
        for out in outputs:
            response_text = out.outputs[0].text
            # Strip Qwen3-style think tags
            response_text = re.sub(r"<think>.*?</think>\s*", "", response_text, flags=re.DOTALL)
            response_text = re.sub(r"<think>.*", "", response_text, flags=re.DOTALL)
            all_tools.append(self._parse_tool_response(response_text))
        return all_tools

    def solve_and_learn_batch(
        self,
        problems: list[str],
    ) -> list[dict[str, Any]]:
        """Batched version of :meth:`solve_and_learn`.

        1. Solve all *problems* in parallel via :meth:`solve_batch_parallel`.
        2. Extract tools from all solutions in one batched call.
        3. Insert valid tools into the :class:`ToolLibrary`.

        Returns a list of result dicts (same shape as :meth:`solve_and_learn`).
        """
        # Phase 1: batched solve
        solutions = self.solve_batch_parallel(problems)

        # Phase 2: batched tool extraction
        # Collect (problem, code) pairs where code is non-empty
        extract_problems: list[str] = []
        extract_codes: list[str] = []
        extract_indices: list[int] = []
        for i, sol in enumerate(solutions):
            if sol["code"]:
                extract_problems.append(problems[i])
                extract_codes.append(sol["code"])
                extract_indices.append(i)

        all_tools: list[list[dict[str, Any]]] = [[] for _ in range(len(problems))]
        if extract_problems:
            batch_tools = self.extract_tools_batch(extract_problems, extract_codes)
            for i, tools in zip(extract_indices, batch_tools):
                all_tools[i] = tools

        # Phase 3: insert tools into library
        results: list[dict[str, Any]] = []
        for i, sol in enumerate(solutions):
            tool_names: list[str] = []
            for et in all_tools[i]:
                try:
                    tool_names.append(self.add_tool_to_library(et))
                except Exception:
                    pass

            results.append({
                "answer": sol["answer"],
                "tool_names": tool_names,
                "tool_name": tool_names[0] if tool_names else None,
                "steps": sol["steps"],
                "code": sol["code"],
                "tools": all_tools[i],
                "tool": all_tools[i][0] if all_tools[i] else None,
            })

        return results
