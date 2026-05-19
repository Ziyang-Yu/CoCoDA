"""vLLM-backed rollout engine for the GRPO trainer.

This module wraps an in-process :class:`vllm.LLM` instance per DDP rank using
``distributed_executor_backend="external_launcher"`` so that:

  * each torchrun rank owns its own vLLM engine, bound to its local GPU;
  * the trainer pushes updated policy weights into the engine after every
    optimizer step via ``collective_rpc`` → ``model_runner.model.load_weights``
    (no IPC, no NCCL side-channel — vLLM lives in the same Python process and
    just copies the in-memory tensors);
  * the rollout API matches :func:`training.rollout.rollout_batch` exactly so
    the trainer can swap one for the other based on a CLI flag.

The multi-turn ReAct loop is identical in shape to the HF version: at every
inner step the still-active sequences are batched into one ``llm.generate``
call, parsed, and either tool-executed or marked as finished.  The big
difference is generation throughput — vLLM's continuous batching plus
prefix caching deliver several-× the tokens/s of HF ``model.generate`` for
the same hardware.

CUDA graphs are disabled (``enforce_eager=True``) because RLHF weight updates
re-write parameter storage and CUDA-graph captures can race with the new
weights.  We trade ~30% rollout throughput for safety; re-enabling CUDA
graphs is possible but needs a wake-up/sleep cycle around every weight sync,
which is out of scope here.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from tool.tool_library import ToolLibrary
from training.rollout import (
    Trajectory,
    _build_assistant_mask,
    _build_initial_messages,
    _parse_tool_call,
    _strip_think,
    _strip_thought,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker-side callable invoked through LLM.collective_rpc
# ---------------------------------------------------------------------------
def _worker_load_weights(
    worker_self: Any,
    weights: list[tuple[str, torch.Tensor]],
) -> None:
    """Run **inside the vLLM worker** to push (name, tensor) pairs into the
    inner model.

    Because we use ``external_launcher``, the "worker" is in the same Python
    process as the trainer; ``collective_rpc`` just calls this function with
    the worker as the first arg, no serialization.  Tensors are passed by
    reference, so the only real cost is whatever ``model.load_weights`` does
    internally to slot HF parameter names into vLLM's fused layout.
    """
    # WorkerWrapperBase delegates unknown attrs to the wrapped GPUWorker via
    # __getattr__, so worker_self.model_runner == worker_self.worker.model_runner.
    model = worker_self.model_runner.model
    model.load_weights(weights)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class VLLMRolloutEngine:
    """Per-rank in-process vLLM engine + multi-turn ReAct rollout loop."""

    def __init__(
        self,
        *,
        model_path: str,
        tokenizer: PreTrainedTokenizerBase,
        gpu_memory_utilization: float = 0.30,
        dtype: str = "bfloat16",
        max_model_len: int = 8192,
        seed: int = 0,
        enable_sleep_mode: bool = False,
    ) -> None:
        # Lazy import so the rest of the trainer still loads even if vLLM
        # isn't installed.  By the time we hit this constructor we know the
        # user opted in.
        from vllm import LLM, SamplingParams  # noqa: F401  (SamplingParams used below)

        log.info(
            "Initialising vLLM rollout engine: model=%s gpu_mem_util=%.2f "
            "max_model_len=%d dtype=%s sleep_mode=%s",
            model_path, gpu_memory_utilization, max_model_len, dtype,
            enable_sleep_mode,
        )
        self._sleep_supported = enable_sleep_mode
        self._llm = LLM(
            model=model_path,
            dtype=dtype,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=1,
            distributed_executor_backend="external_launcher",
            enforce_eager=True,        # see module docstring
            disable_log_stats=True,
            seed=seed,
            enable_sleep_mode=enable_sleep_mode,
        )
        # Always use the trainer's tokenizer for chat-template rendering and
        # mask building so token ids match the policy's training-time view.
        self._tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------
    def sync_weights(self, policy_module: torch.nn.Module) -> None:
        """Push the current policy weights into the vLLM engine.

        Iterates ``named_parameters()`` (not ``state_dict()``) to avoid
        constructing a redundant dict; the tensors are detached views of the
        live policy parameters and ``model.load_weights`` does the HF-to-vLLM
        layout remap (e.g. fusing q/k/v into qkv_proj).

        Should be called once per outer GRPO step, *before* rolling out.
        """
        weights = [
            (name, param.detach())
            for name, param in policy_module.named_parameters()
        ]
        self._llm.collective_rpc(
            _worker_load_weights,
            args=(weights,),
        )

    # ------------------------------------------------------------------
    # Sleep / wake — swap weights off GPU so the teacher miner can fit
    # ------------------------------------------------------------------
    def sleep(self, level: int = 2) -> None:
        """Offload weights (level=2) / free KV cache (level=1) to CPU.

        No-op unless the engine was built with ``enable_sleep_mode=True``.
        """
        if not self._sleep_supported:
            return
        self._llm.sleep(level=level)

    def wake_up(self) -> None:
        """Restore weights and KV cache allocation."""
        if not self._sleep_supported:
            return
        self._llm.wake_up()

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    def rollout_batch(
        self,
        *,
        tool_library: ToolLibrary,
        problems: list[str],
        tool_descriptions_list: list[str],
        max_steps: int = 8,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> list[Trajectory]:
        """Drop-in replacement for :func:`training.rollout.rollout_batch` that
        uses vLLM for generation.

        Same lockstep semantics: at every inner step the still-active
        sequences are batched into one ``llm.generate`` call.  Sequences that
        emit a non-tool-call response (or hit ``max_steps``) drop out of the
        active set; the loop ends when all are done.

        The Trajectory's ``input_ids`` / ``assistant_mask`` are built with the
        trainer's HF tokenizer at the end (same code path as the HF rollout)
        so downstream PPO logprob computation sees identical token streams.
        """
        from vllm import SamplingParams

        if len(problems) != len(tool_descriptions_list):
            raise ValueError(
                f"problems and tool_descriptions_list must have the same length, "
                f"got {len(problems)} and {len(tool_descriptions_list)}"
            )

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

        # Two SamplingParams: one with </tool_call> as a stop string for the
        # ReAct inner steps (so we don't waste tokens after the tool call
        # closes), one without for the forced final-answer turn after
        # max_steps (where we want the unrestricted answer).
        common = dict(
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p if temperature > 0 else 1.0,
            max_tokens=max_new_tokens,
            n=1,
        )
        sp_step = SamplingParams(
            **common,
            stop=["</tool_call>"],
            include_stop_str_in_output=True,
        )
        sp_final = SamplingParams(**common)

        def _render(idx: int) -> str:
            return self._tokenizer.apply_chat_template(
                states[idx]["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

        def _generate_for(indices: list[int], sp: "SamplingParams") -> list[str]:
            prompts = [_render(i) for i in indices]
            outputs = self._llm.generate(prompts, sp, use_tqdm=False)
            return [_strip_think(o.outputs[0].text.strip()) for o in outputs]

        for _step_idx in range(max_steps):
            active = [i for i, s in enumerate(states) if not s["done"]]
            if not active:
                break
            responses = _generate_for(active, sp_step)

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

        # Anything still active hit max_steps -> force one final unrestricted
        # generation as the answer.
        not_done = [i for i, s in enumerate(states) if not s["done"]]
        if not_done:
            finals = _generate_for(not_done, sp_final)
            for resp_j, idx in enumerate(not_done):
                final = finals[resp_j]
                s = states[idx]
                s["assistant_texts"].append(final)
                s["messages"].append({"role": "assistant", "content": final})
                s["answer"] = final
                s["hit_max_steps"] = True
                s["done"] = True

        # Build Trajectories using the trainer's HF tokenizer so token ids /
        # assistant masks line up exactly with what the PPO update expects.
        trajectories: list[Trajectory] = []
        for s in states:
            input_ids, mask = _build_assistant_mask(
                self._tokenizer, s["messages"], s["assistant_texts"]
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

    # ------------------------------------------------------------------
    # Convenience: G samples of one prompt
    # ------------------------------------------------------------------
    def rollout_group(
        self,
        *,
        tool_library: ToolLibrary,
        problem: str,
        tool_descriptions: str,
        group_size: int,
        **kwargs: Any,
    ) -> list[Trajectory]:
        """Drop-in replacement for :func:`training.rollout.rollout_group`.

        Tiles the same (problem, tool_desc) ``group_size`` times so the G
        samples form one GRPO group.  vLLM's prefix caching makes the shared
        prefix essentially free.
        """
        return self.rollout_batch(
            tool_library=tool_library,
            problems=[problem] * group_size,
            tool_descriptions_list=[tool_descriptions] * group_size,
            **kwargs,
        )
