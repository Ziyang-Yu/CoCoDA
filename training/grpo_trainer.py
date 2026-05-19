"""GRPO trainer for the Student model.

Implements the *Group Relative Policy Optimization* loop from DeepSeek-Math
on top of plain HuggingFace + torch (no TRL dependency).

For each prompt we sample G trajectories with :func:`training.rollout.rollout_group`,
score them with :func:`training.rewards.combined_reward`, and update the policy
with the standard PPO-clipped surrogate using the group-relative advantage as
the per-trajectory baseline.  A KL term against a frozen reference model
prevents the policy from drifting away from the SFT warm-start.

Run:
    python -m training.grpo_trainer \\
        --traces      outputs/grpo_run/teacher_traces.jsonl \\
        --library     outputs/grpo_run/tool_library.json \\
        --val         outputs/grpo_run/val.jsonl \\
        --policy-init outputs/grpo_run/checkpoints/sft \\
        --output-dir  outputs/grpo_run/checkpoints/grpo
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

from main import is_correct
from model.student import STUDENT_SYSTEM_PROMPT
from tool.tool_library import ToolLibrary
from tool.tool_retriever import ToolRetriever
from training.data import read_jsonl
from training.online_tool_miner import (
    mine_composite_from_trajectory,
    mine_with_teacher_batch,
)
from training.rewards import (
    RewardWeights,
    combined_reward,
    group_relative_advantages,
)
from training.rollout import Trajectory, rollout_batch, rollout_group, rollout_one

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_tool_descriptions(tool_names: list[str], library: ToolLibrary) -> str:
    """Render the system-prompt tool list for a fixed (frozen) shortlist.

    The trainer pre-computes a shortlist per training example so that the
    student sees the same tool descriptions during rollout, logprob
    re-evaluation, and val-time eval.
    """
    if not tool_names:
        return "(no tools available)"
    lines: list[str] = []
    for name in tool_names:
        if name not in library:
            continue
        meta = library.get_tool(name).metadata
        sig = meta.signature
        desc = meta.description.summary or "No description"
        lines.append(
            f"- **{name}**({sig.input_type}) -> {sig.output_type}: {desc}"
        )
    return "\n".join(lines) if lines else "(no tools available)"


def _logprobs_for_trajectory(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Per-token logprobs of *input_ids* under *model*, restricted to the
    positions where *mask* == 1 (assistant tokens).

    Returns a flat tensor of length ``mask[1:].sum()``.

    Kept for clarity / single-sample call sites; the hot path uses
    :func:`_batched_assistant_logprobs` instead.
    """
    logits = model(input_ids.unsqueeze(0)).logits[0]   # [seq, vocab]
    log_probs = F.log_softmax(logits[:-1], dim=-1)     # predict token t+1
    targets = input_ids[1:]
    chosen = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [seq-1]
    return chosen[mask[1:].bool()]


def _batched_assistant_logprobs(
    model: torch.nn.Module,
    items: list[tuple[torch.Tensor, torch.Tensor]],
    pad_id: int,
    device: torch.device,
    mini_batch_size: int,
) -> list[torch.Tensor]:
    """Per-token assistant logprobs for a list of (input_ids, mask) pairs,
    computed in mini-batches with right-padding.

    Each mini-batch runs ONE forward through *model* instead of one forward
    per item.  Returns a list of 1-D tensors aligned with *items*; entry *i*
    has length ``items[i][1][1:].sum()``.

    Right-padding is used so that the position IDs of the *real* tokens match
    what they'd be in an unbatched forward (decoder-only models with rotary
    embeddings depend on this).  An ``attention_mask`` is passed so the model
    doesn't attend to pad positions.

    Gradients flow through the slicing, so this helper is used by both the
    no_grad caching path in :meth:`GRPOTrainer._rollout_batch` and the
    gradient-bearing PPO update in :meth:`GRPOTrainer._ppo_update`.  Callers
    are responsible for wrapping the call in ``torch.no_grad`` when
    appropriate.
    """
    out: list[torch.Tensor] = [torch.empty(0)] * len(items)
    for start in range(0, len(items), mini_batch_size):
        chunk = items[start : start + mini_batch_size]
        max_len = max(ids.shape[0] for ids, _ in chunk)
        B = len(chunk)
        batch_ids = torch.full(
            (B, max_len), pad_id, dtype=torch.long, device=device
        )
        attn = torch.zeros((B, max_len), dtype=torch.long, device=device)
        for j, (ids, _m) in enumerate(chunk):
            L = ids.shape[0]
            batch_ids[j, :L] = ids
            attn[j, :L] = 1
        logits = model(batch_ids, attention_mask=attn).logits  # [B, L, V]
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        targets = batch_ids[:, 1:]
        chosen = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
        for j, (ids, mask) in enumerate(chunk):
            L = ids.shape[0]
            row = chosen[j, : L - 1]
            m = mask[1:].bool()
            out[start + j] = row[m]
    return out


def _kl_estimate(
    logp_pol: torch.Tensor, logp_ref: torch.Tensor
) -> torch.Tensor:
    """K3 estimator (Schulman): unbiased and non-negative."""
    logr = logp_ref - logp_pol
    return torch.exp(logr) - logr - 1.0


# ---------------------------------------------------------------------------
# Training-example representation
# ---------------------------------------------------------------------------
@dataclass
class TrainExample:
    problem: str
    gold_answer: str
    teacher_plan: list[str]
    teacher_tool_seq: list[str]
    tool_names_in_prompt: list[str]
    tool_descriptions: str   # cached, rendered once


def _load_train_examples(
    traces_path: str | Path, library: ToolLibrary
) -> list[TrainExample]:
    out: list[TrainExample] = []
    for rec in read_jsonl(traces_path):
        names = rec.get("tool_names_in_prompt") or []
        if not names:
            continue
        out.append(
            TrainExample(
                problem=rec["problem"],
                gold_answer=rec["gold_answer"],
                teacher_plan=rec.get("teacher_plan") or [],
                teacher_tool_seq=rec.get("teacher_tool_seq") or [],
                tool_names_in_prompt=names,
                tool_descriptions=_build_tool_descriptions(names, library),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Rolled-out sample with cached old/ref logprobs
# ---------------------------------------------------------------------------
@dataclass
class RolloutSample:
    trajectory: Trajectory
    advantage: float
    reward_breakdown: dict[str, float]
    old_logprobs: torch.Tensor   # detached, computed once per outer step
    ref_logprobs: torch.Tensor   # detached
    input_ids: torch.Tensor
    mask: torch.Tensor


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def _init_distributed() -> tuple[bool, int, int, int]:
    """Initialise the NCCL process group from torchrun env vars.

    Returns ``(is_distributed, rank, world_size, local_rank)``.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local)
        return True, rank, world, local
    return False, 0, 1, 0


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------
class GRPOTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.distributed, self.rank, self.world_size, self.local_rank = _init_distributed()
        self.is_main = (self.rank == 0)
        if self.distributed:
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Silence non-main ranks so the SLURM .out is readable
        if not self.is_main:
            logging.getLogger().setLevel(logging.WARNING)

        if self.is_main:
            log.info(
                "Distributed: %s  rank=%d/%d  device=%s",
                self.distributed, self.rank, self.world_size, self.device,
            )
            log.info("Loading tokenizer + policy from %s", args.policy_init)

        self.tokenizer = AutoTokenizer.from_pretrained(args.policy_init)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if args.bf16 else torch.float32
        policy_raw = AutoModelForCausalLM.from_pretrained(
            args.policy_init, torch_dtype=dtype,
        ).to(self.device)
        # Non-reentrant checkpointing is required for DDP: the default reentrant
        # mode replays forward during backward, which fires DDP's autograd hooks
        # twice and trips "marked ready twice" errors.
        if args.grad_ckpt:
            policy_raw.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if hasattr(policy_raw, "config"):
            policy_raw.config.use_cache = False
        policy_raw.train()

        if self.distributed:
            # static_graph=True is needed because _ppo_update accumulates several
            # forward passes into a single backward; without it DDP would mark
            # the same parameter ready multiple times per iteration and crash.
            self.policy = DDP(
                policy_raw,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
            self.policy_module = policy_raw  # for rollout / save / no-grad work
        else:
            self.policy = policy_raw
            self.policy_module = policy_raw

        if self.is_main:
            log.info("Loading reference model from %s", args.policy_init)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            args.policy_init, torch_dtype=dtype,
        ).to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        if self.is_main:
            log.info("Loading tool library from %s", args.library)
        self.library = ToolLibrary.load(args.library)
        # Library is mutable during Stage 3 co-evolution; rank 0 mines new
        # composites from successful rollouts and broadcasts them.
        self.library_version = 0

        if self.is_main:
            log.info("Loading training examples from %s", args.traces)
        self.train_examples = _load_train_examples(args.traces, self.library)
        if self.is_main:
            log.info("Loaded %d training examples", len(self.train_examples))

        if args.val:
            self.val_examples: list[dict[str, str]] = list(read_jsonl(args.val))
            if self.is_main:
                log.info("Loaded %d val examples", len(self.val_examples))
        else:
            self.val_examples = []

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )

        self.weights = RewardWeights(
            answer=args.w_answer,
            tool_seq=args.w_tool_seq,
            plan_sim=args.w_plan_sim,
            composite=args.w_composite,
            format=args.w_format,
        )
        self.composite_names = set(self.library.composite_names)

        self.output_dir = Path(args.output_dir)
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "logs.jsonl").touch(exist_ok=True)
        if self.distributed:
            dist.barrier()

        # Optional LLM-backed HIERRETRIEVE.  Wired lazily so index/hybrid
        # modes never pay the cost of constructing a retriever.
        self.llm_retriever: ToolRetriever | None = None
        self._retrieve_outer: int = 0

        # Optional teacher-backed online miner (rank 0 only).  Reuses the
        # Stage-1 Teacher vLLM after warm-start; materialised lazily the
        # first time a successful trajectory needs distillation.
        self._teacher_miner: Any = None

        # Optional vLLM rollout engine.  Created AFTER the policy + ref model
        # are loaded so vLLM measures the *remaining* HBM correctly.  Each
        # DDP rank gets its own in-process engine bound to its local GPU via
        # ``distributed_executor_backend="external_launcher"``.
        self.vllm_engine = None
        if args.use_vllm_rollout:
            from training.vllm_rollout import VLLMRolloutEngine
            if self.is_main:
                log.info(
                    "Creating vLLM rollout engine (gpu_mem_util=%.2f, "
                    "max_model_len=%d) on every rank",
                    args.vllm_gpu_mem_util, args.vllm_max_model_len,
                )
            self.vllm_engine = VLLMRolloutEngine(
                model_path=args.policy_init,
                tokenizer=self.tokenizer,
                gpu_memory_utilization=args.vllm_gpu_mem_util,
                dtype="bfloat16" if args.bf16 else "float32",
                max_model_len=args.vllm_max_model_len,
                seed=args.seed + self.rank,
                # Enable sleep so rank 0 can swap this engine out of GPU
                # memory while the 32B teacher miner runs (Algorithm 1 L19).
                enable_sleep_mode=args.online_miner == "teacher"
                    and args.enable_online_insert,
            )
            if self.distributed:
                dist.barrier()

    # --------------------------------------------------------------
    # Generation-mode toggle
    # --------------------------------------------------------------
    @contextmanager
    def _generation_mode(self):
        """Temporarily disable gradient checkpointing and enable the KV cache
        on the policy module.

        The trainer permanently turns gradient checkpointing on (and disables
        ``use_cache``) so the *training* forward/backward fits in memory and
        plays nicely with DDP.  But those same settings are catastrophic for
        rollout: ``use_cache=False`` makes ``model.generate`` recompute the
        full prefix at every new token, and gradient checkpointing forces
        recomputation of the forward graph that we don't need under no_grad.
        We toggle them off for the duration of any generation block and
        restore them afterwards so training behaviour is unchanged.
        """
        if self.args.grad_ckpt:
            self.policy_module.gradient_checkpointing_disable()
        prev_use_cache = getattr(self.policy_module.config, "use_cache", False)
        self.policy_module.config.use_cache = True
        try:
            yield
        finally:
            if self.args.grad_ckpt:
                self.policy_module.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            self.policy_module.config.use_cache = prev_use_cache

    # --------------------------------------------------------------
    # One training step (one batch of prompts)
    # --------------------------------------------------------------
    def _retrieve_generate_fn(self, messages: list[dict[str, str]]) -> str:
        """Generate-fn for the LLM-backed retriever.

        Routes through the vLLM engine if available (cheap, matches rollout
        path); otherwise falls back to HF ``model.generate`` in no_grad
        under ``_generation_mode``.
        """
        if self.vllm_engine is not None and hasattr(
            self.vllm_engine, "generate_chat"
        ):
            return self.vllm_engine.generate_chat(
                messages, max_tokens=512, temperature=0.0,
            )
        # HF fallback: short, greedy, no_grad
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad(), self._generation_mode():
            out = self.policy_module.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _ensure_llm_retriever(self) -> ToolRetriever:
        """Lazily instantiate the LLM-backed Algorithm 2 retriever bound to
        the current library (which already carries the DAG G and index I).
        """
        if self.llm_retriever is None or self.llm_retriever.tool_library is not self.library:
            self.llm_retriever = ToolRetriever(
                tool_library=self.library,
                generate_fn=self._retrieve_generate_fn,
                max_shortlist=max(self.args.retrieve_k * 2, 12),
                max_final=self.args.retrieve_k,
            )
        return self.llm_retriever

    def _refresh_tool_descriptions(self, batch: list[TrainExample]) -> None:
        """Re-run HIERRETRIEVE(q, I, G, k) per example against the current
        library so newly inserted tools become visible in the student prompt.

        Mode is controlled by ``--retrieve-mode``:
          * ``index`` – cheap, LLM-free scoring (index-only Algorithm 2).
          * ``llm``   – full Algorithm 2 with LLM-backed decompose / scan /
            deep-inspect; accurate but expensive.
          * ``hybrid`` – LLM path every ``--retrieve-llm-every`` outer steps;
            index-only in between.
        """
        mode = self.args.retrieve_mode
        k = self.args.retrieve_k
        use_llm = mode == "llm" or (
            mode == "hybrid"
            and self.args.retrieve_llm_every > 0
            and (self._retrieve_outer % self.args.retrieve_llm_every == 0)
        )
        self._retrieve_outer += 1

        retriever = self._ensure_llm_retriever() if use_llm else None
        for ex in batch:
            try:
                if retriever is not None:
                    names = retriever.hier_retrieve(ex.problem, k=k)
                else:
                    names = self.library.retrieve_for_query(ex.problem, k=k)
            except Exception as exc:
                log.warning(
                    "HIERRETRIEVE failed (%s); falling back to index-only",
                    exc,
                )
                names = self.library.retrieve_for_query(ex.problem, k=k)
            if names:
                ex.tool_names_in_prompt = names
                ex.tool_descriptions = _build_tool_descriptions(names, self.library)

    def _rollout_batch(self, batch: list[TrainExample]) -> list[RolloutSample]:
        """Roll out *batch* (this rank's local shard) and return scored samples.

        Uses ``self.policy_module`` (the unwrapped HF model) so DDP autograd
        hooks aren't triggered during inference / no_grad logprob caching.
        """
        if self.args.retrieve_per_rollout:
            self._refresh_tool_descriptions(batch)

        samples: list[RolloutSample] = []
        G = self.args.group_size

        if self.vllm_engine is not None:
            # vLLM path: sync weights once, then run ALL local prompts × G in
            # one batched rollout (vLLM's continuous batching keeps the GPU
            # saturated and prefix caching deduplicates the shared prefixes
            # of each group).
            with torch.no_grad():
                self.vllm_engine.sync_weights(self.policy_module)
            flat_problems = [ex.problem for ex in batch for _ in range(G)]
            flat_descs = [ex.tool_descriptions for ex in batch for _ in range(G)]
            flat_trajs = self.vllm_engine.rollout_batch(
                tool_library=self.library,
                problems=flat_problems,
                tool_descriptions_list=flat_descs,
                max_steps=self.args.rollout_max_steps,
                max_new_tokens=self.args.rollout_max_new_tokens,
                temperature=self.args.rollout_temperature,
                top_p=self.args.rollout_top_p,
            )
            # Reshape [B*G] -> [B][G] so the reward / advantage code below
            # is identical to the HF path.
            all_trajs: list[list[Trajectory]] = [
                flat_trajs[i * G : (i + 1) * G] for i in range(len(batch))
            ]
        else:
            # HF path: roll out every prompt in this rank's local shard with
            # the KV cache enabled and gradient checkpointing off (see
            # ``_generation_mode``).
            with self._generation_mode():
                self.policy_module.eval()
                all_trajs = []
                for ex in batch:
                    trajs = rollout_group(
                        model=self.policy_module,
                        tokenizer=self.tokenizer,
                        tool_library=self.library,
                        problem=ex.problem,
                        tool_descriptions=ex.tool_descriptions,
                        group_size=G,
                        max_steps=self.args.rollout_max_steps,
                        max_new_tokens=self.args.rollout_max_new_tokens,
                        temperature=self.args.rollout_temperature,
                        top_p=self.args.rollout_top_p,
                        device=self.device,
                    )
                    all_trajs.append(trajs)
                self.policy_module.train()

        # Collect (trajectory, advantage, reward_breakdown, ids, mask) for
        # every trajectory that has at least one assistant token, then run
        # the policy + ref forwards in mini-batches instead of one-per-sample.
        items: list[tuple[torch.Tensor, torch.Tensor]] = []
        metas: list[tuple[Trajectory, float, dict[str, float]]] = []
        for ex, trajs in zip(batch, all_trajs):
            rewards: list[float] = []
            breakdowns: list[dict[str, float]] = []
            for t in trajs:
                r, br = combined_reward(
                    trajectory=t.to_reward_dict(),
                    gold_answer=ex.gold_answer,
                    teacher_tool_seq=ex.teacher_tool_seq,
                    teacher_plan=ex.teacher_plan,
                    composite_names=self.composite_names,
                    weights=self.weights,
                )
                rewards.append(r)
                breakdowns.append(br)

            # Group-relative advantages
            advs = group_relative_advantages(rewards)

            for t, adv, br in zip(trajs, advs, breakdowns):
                ids = torch.tensor(t.input_ids, dtype=torch.long, device=self.device)
                mask = torch.tensor(t.assistant_mask, dtype=torch.long, device=self.device)
                if mask.sum() == 0:
                    continue   # nothing to train on
                items.append((ids, mask))
                metas.append((t, adv, br))

        if items:
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            with torch.no_grad():
                old_lps = _batched_assistant_logprobs(
                    self.policy_module, items, pad_id, self.device,
                    self.args.logp_batch_size,
                )
                ref_lps = _batched_assistant_logprobs(
                    self.ref_model, items, pad_id, self.device,
                    self.args.logp_batch_size,
                )
            for (t, adv, br), (ids, mask), old_lp, ref_lp in zip(
                metas, items, old_lps, ref_lps,
            ):
                samples.append(RolloutSample(
                    trajectory=t,
                    advantage=adv,
                    reward_breakdown=br,
                    old_logprobs=old_lp.detach(),
                    ref_logprobs=ref_lp.detach(),
                    input_ids=ids,
                    mask=mask,
                ))
        return samples

    def _ppo_update(self, samples: list[RolloutSample]) -> dict[str, float]:
        """One PPO update step.

        DDP requires every rank to perform the *same number* of backward calls
        in the same order; we therefore accumulate the per-trajectory losses
        into a single tensor on each rank and call ``backward()`` exactly once
        per ppo_epoch.  If a rank ended up with zero local samples (e.g. all
        of its rollouts had empty assistant masks) we emit a dummy zero-loss
        touching one parameter so the all-reduce still goes through.
        """
        eps = self.args.clip_eps
        beta_kl = self.args.kl_coef

        running = {
            "loss": 0.0, "policy_loss": 0.0, "kl": 0.0,
            "ratio_mean": 0.0, "n_tokens": 0,
        }

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        for _ in range(self.args.ppo_epochs):
            random.shuffle(samples)
            self.optimizer.zero_grad(set_to_none=True)

            total_loss = torch.tensor(0.0, device=self.device)
            local_tokens = 0

            # One batched forward through DDP per mini-batch instead of one
            # forward per trajectory.  Gradients flow through the per-row
            # slicing inside ``_batched_assistant_logprobs``, so the per-sample
            # PPO surrogate below sees the same logprobs it would have seen
            # under the unbatched code path.
            if samples:
                items = [(s.input_ids, s.mask) for s in samples]
                new_lps = _batched_assistant_logprobs(
                    self.policy, items, pad_id, self.device,
                    self.args.logp_batch_size,
                )
            else:
                new_lps = []

            for s, new_lp in zip(samples, new_lps):
                ratio = torch.exp(new_lp - s.old_logprobs)
                adv = torch.full_like(new_lp, s.advantage)

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
                policy_loss = -torch.min(surr1, surr2)

                kl = _kl_estimate(new_lp, s.ref_logprobs)
                token_loss = policy_loss + beta_kl * kl
                total_loss = total_loss + token_loss.sum()
                local_tokens += int(new_lp.numel())

                running["policy_loss"] += float(policy_loss.sum().detach())
                running["kl"]          += float(kl.sum().detach())
                running["ratio_mean"]  += float(ratio.mean().detach()) * new_lp.numel()
                running["n_tokens"]    += int(new_lp.numel())

            if local_tokens == 0:
                # Keep DDP in sync even if this rank has nothing to learn from.
                first_param = next(self.policy.parameters())
                total_loss = first_param.sum() * 0.0
                norm_loss = total_loss
            else:
                norm_loss = total_loss / local_tokens

            running["loss"] += float(norm_loss.detach()) * max(1, local_tokens)
            norm_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        # Aggregate metrics across ranks for logging
        if self.distributed:
            t = torch.tensor(
                [running["loss"], running["policy_loss"], running["kl"],
                 running["ratio_mean"], running["n_tokens"]],
                dtype=torch.float64, device=self.device,
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            (running["loss"], running["policy_loss"], running["kl"],
             running["ratio_mean"], running["n_tokens"]) = t.tolist()

        n = max(1, running["n_tokens"])
        return {
            "loss":        running["loss"] / n,
            "policy_loss": running["policy_loss"] / n,
            "kl":          running["kl"] / n,
            "ratio_mean":  running["ratio_mean"] / n,
        }

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, examples: list[dict[str, str]], n: int | None = None) -> float:
        if not examples:
            return 0.0
        if n is not None:
            examples = examples[:n]

        # Shard val across ranks: rank r evaluates examples[r::world_size]
        local = examples[self.rank :: self.world_size] if self.distributed else examples

        # For val we use the *full* library shortlist of every training tool
        # so that we measure how well the student generalises across the same
        # frozen tool set it trained on.
        all_tool_names = list(self.library.tool_names)
        tool_desc = _build_tool_descriptions(all_tool_names, self.library)

        # Chunk the per-rank val shard into batches and run them through the
        # batched lockstep rollout.  Distinct prompts run in parallel; each
        # batched generate keeps the GPU busy instead of one example at a time.
        eval_bs = max(1, self.args.eval_batch_size)
        correct = 0

        if self.vllm_engine is not None:
            # vLLM path: weights were just synced by the most recent
            # _rollout_batch (eval is called right after a training step), so
            # the engine already holds the current policy.  Re-sync defensively
            # in case the optimizer ran since then.
            with torch.no_grad():
                self.vllm_engine.sync_weights(self.policy_module)
            for start in range(0, len(local), eval_bs):
                chunk = local[start : start + eval_bs]
                trajs = self.vllm_engine.rollout_batch(
                    tool_library=self.library,
                    problems=[ex["question"] for ex in chunk],
                    tool_descriptions_list=[tool_desc] * len(chunk),
                    max_steps=self.args.rollout_max_steps,
                    max_new_tokens=self.args.rollout_max_new_tokens,
                    temperature=0.0,    # greedy at eval
                )
                for ex, traj in zip(chunk, trajs):
                    if is_correct(traj.answer, ex["answer"]):
                        correct += 1
        else:
            with self._generation_mode():
                self.policy_module.eval()
                for start in range(0, len(local), eval_bs):
                    chunk = local[start : start + eval_bs]
                    trajs = rollout_batch(
                        model=self.policy_module,
                        tokenizer=self.tokenizer,
                        tool_library=self.library,
                        problems=[ex["question"] for ex in chunk],
                        tool_descriptions_list=[tool_desc] * len(chunk),
                        max_steps=self.args.rollout_max_steps,
                        max_new_tokens=self.args.rollout_max_new_tokens,
                        temperature=0.0,    # greedy at eval
                        device=self.device,
                    )
                    for ex, traj in zip(chunk, trajs):
                        if is_correct(traj.answer, ex["answer"]):
                            correct += 1
                self.policy_module.train()

        if self.distributed:
            t = torch.tensor([correct, len(local)], dtype=torch.float64, device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            global_correct, global_total = t.tolist()
            return global_correct / max(1.0, global_total)
        return correct / max(1, len(local))

    # --------------------------------------------------------------
    # Main loop
    # --------------------------------------------------------------
    def _sample_global_batch(self, outer: int) -> list[TrainExample]:
        """All ranks pick the same global batch via a shared seed, then each
        rank slices its own contiguous chunk so prompts aren't duplicated.
        """
        rng = random.Random(self.args.seed + outer)
        global_size = min(self.args.prompts_per_step, len(self.train_examples))
        global_batch = rng.sample(self.train_examples, k=global_size)
        if not self.distributed:
            return global_batch
        # Round-robin slice; if global_size doesn't divide world_size some
        # ranks just get one extra prompt.
        return global_batch[self.rank :: self.world_size]

    def _ensure_teacher_miner(self) -> Any:
        """Rank-0 helper: lazily import + instantiate the Stage-1 Teacher so
        its vLLM process can be reused by the online miner.  The Teacher is
        materialised on demand (first successful rollout) so runs that never
        hit a successful trajectory don't pay the startup cost.
        """
        if not self.is_main:
            return None
        if self._teacher_miner is not None:
            return self._teacher_miner
        from model.teacher import Teacher
        log.info(
            "Instantiating teacher miner on rank 0 (model=%s, gpu_mem_util=%.2f)",
            self.args.teacher_miner_model,
            self.args.teacher_miner_gpu_mem_util,
        )
        self._teacher_miner = Teacher(
            model_id=self.args.teacher_miner_model,
            tool_library=self.library,
            gpu_memory_utilization=self.args.teacher_miner_gpu_mem_util,
            enable_sleep_mode=True,
        )
        return self._teacher_miner

    def _co_evolve_library(
        self, samples: list[RolloutSample], local_batch: list[TrainExample],
    ) -> int:
        """Stage 3 of Algorithm 1: for each successful rollout (R_i >= rho),
        mine a composite tool on rank 0, broadcast the serialized entry to
        every rank, and insert into ``self.library`` via Algorithm 3.

        Returns the number of new tools inserted this step (across all ranks).
        """
        if not self.args.enable_online_insert:
            return 0

        rho = self.args.insert_threshold
        require_correct = self.args.insert_require_correct
        miner_kind = self.args.online_miner
        # Pair each sample with its source problem text via trajectory messages.
        local_entries: list[tuple[str, dict[str, Any]]] = []
        if self.is_main:
            # First pass: gate by reward threshold + Algorithm 1 L17 R(T*)==1
            # correctness, then collect the (trajectory, problem) pairs that
            # survive.  This lets the teacher path below run ONE batched
            # extract_tools_batch() instead of N sequential 32B prefills.
            filtered_samples: list[RolloutSample] = []
            filtered_problems: list[str] = []
            for s in samples:
                if s.reward_breakdown.get("total", 0.0) < rho:
                    continue
                if (
                    require_correct
                    and s.reward_breakdown.get("r_answer", 0.0) < 1.0 - 1e-6
                ):
                    continue
                problem = ""
                for msg in s.trajectory.messages:
                    if msg.get("role") == "user":
                        problem = msg.get("content", "")
                        break
                filtered_samples.append(s)
                filtered_problems.append(problem)

            use_teacher = miner_kind == "teacher" and bool(filtered_samples)
            if use_teacher:
                # Swap: put the rollout engine to sleep so the 32B miner fits
                # on rank 0's GPU, then wake (or instantiate) the teacher.
                if self.vllm_engine is not None:
                    self.vllm_engine.sleep(level=2)
                teacher = self._ensure_teacher_miner()
                # After its first construction the teacher is awake; on later
                # calls it was put to sleep at the end of the previous mine.
                teacher.wake_up()

                # Teacher path: one batched TOOL_EXTRACTION_PROMPT call over
                # every successful rollout.  add_tool_to_library (inside the
                # helper) still runs sequentially — that part is pure CPU
                # (compile + duplicate check), so keeping it in a loop is
                # fine; the expensive piece was the 32B forward pass.
                try:
                    per_sample = mine_with_teacher_batch(
                        teacher,
                        [s.trajectory.to_reward_dict() for s in filtered_samples],
                        filtered_problems,
                        self.library,
                    )
                except Exception as exc:
                    log.warning("Teacher miner (batched) failed: %s", exc)
                    per_sample = [[] for _ in filtered_samples]
                for harvested in per_sample:
                    for item in harvested:
                        local_entries.append((item["name"], item["entry"]))

                # Reverse swap: teacher back to CPU, rollout engine back up.
                teacher.sleep(level=2)
                if self.vllm_engine is not None:
                    self.vllm_engine.wake_up()
            elif miner_kind != "teacher":
                for s, problem in zip(filtered_samples, filtered_problems):
                    entry = mine_composite_from_trajectory(
                        s.trajectory.to_reward_dict(),
                        self.library,
                        problem=problem,
                    )
                    if entry is None:
                        continue
                    name = entry["tool"]["metadata"]["signature"]["name"]
                    if name in self.library:
                        continue
                    local_entries.append((name, entry))

        # Broadcast list of (name, entry_dict) from rank 0 to everyone else.
        if self.distributed:
            payload = [local_entries] if self.is_main else [None]
            dist.broadcast_object_list(payload, src=0)
            entries_to_apply = payload[0] or []
        else:
            entries_to_apply = local_entries

        applied = 0
        for name, entry in entries_to_apply:
            already_present = name in self.library
            try:
                self.library.add_entry_dict(name, entry)
            except Exception as exc:
                log.warning("Failed to insert mined tool %s: %s", name, exc)
                continue
            # Count "newly-visible-on-this-rank" insertions.  With the teacher
            # miner rank 0 has already inserted via add_tool_to_library, so we
            # treat the broadcast list itself as the authoritative tally to
            # keep the snapshot / refresh path in sync across ranks.
            if not already_present:
                applied += 1
        # If the broadcast list was non-empty but rank 0 had pre-inserted
        # everything, still trigger downstream refresh so snapshot + logs fire.
        if entries_to_apply and applied == 0 and self.is_main:
            applied = len(entries_to_apply)

        if applied:
            # Refresh cached state that depends on library contents.
            self.composite_names = set(self.library.composite_names)
            self.library_version += 1
            # Surface the updated library to every training example so the
            # next rollout sees the new tools even without retrieve-per-rollout.
            if not self.args.retrieve_per_rollout:
                for ex in self.train_examples:
                    ex.tool_descriptions = _build_tool_descriptions(
                        ex.tool_names_in_prompt, self.library,
                    )
            if self.is_main:
                snap = self.output_dir / f"library_v{self.library_version}.json"
                self.library.save(snap)
                log.info(
                    "[online-insert] +%d tools (total=%d) -> %s",
                    applied, len(self.library), snap,
                )
        return applied

    def train(self) -> None:
        log_path = self.output_dir / "logs.jsonl"
        best_val = -1.0

        for outer in range(self.args.num_outer_steps):
            local_batch = self._sample_global_batch(outer)
            global_size = min(self.args.prompts_per_step, len(self.train_examples))
            if self.is_main:
                log.info(
                    "[step %d] rolling out %d global prompts (%d local) × G=%d ...",
                    outer, global_size, len(local_batch), self.args.group_size,
                )
            samples = self._rollout_batch(local_batch)

            # Stage 3 online co-evolution: mine + insert new tools BEFORE the
            # PPO update so next step's rollouts can use the updated library.
            n_inserted = self._co_evolve_library(samples, local_batch)

            # _ppo_update is collective; every rank must call it even if it
            # has zero local samples, otherwise DDP all-reduces will hang.
            metrics = self._ppo_update(samples)
            metrics["n_inserted"] = float(n_inserted)

            # Reward stats — gather across ranks for accurate logging
            local_total = float(sum(s.reward_breakdown["total"]    for s in samples))
            local_ans   = float(sum(s.reward_breakdown["r_answer"] for s in samples))
            local_n     = float(len(samples))
            if self.distributed:
                stats = torch.tensor(
                    [local_total, local_ans, local_n],
                    dtype=torch.float64, device=self.device,
                )
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                global_total, global_ans, global_n = stats.tolist()
            else:
                global_total, global_ans, global_n = local_total, local_ans, local_n

            denom = max(1.0, global_n)
            mean_reward = global_total / denom
            mean_answer = global_ans / denom

            if self.is_main:
                log.info(
                    "[step %d] reward=%.3f answer=%.2f loss=%.4f kl=%.4f",
                    outer, mean_reward, mean_answer,
                    metrics.get("loss", 0.0), metrics.get("kl", 0.0),
                )
                with log_path.open("a") as f:
                    f.write(json.dumps({
                        "step": outer,
                        "mean_reward": mean_reward,
                        "mean_answer": mean_answer,
                        **metrics,
                    }) + "\n")

            if (outer + 1) % self.args.eval_every == 0 and self.val_examples:
                eval_n = self.args.eval_n if self.args.eval_n > 0 else None
                val_acc = self.evaluate(self.val_examples, n=eval_n)
                if self.is_main:
                    log.info("[step %d] val_acc=%.3f", outer, val_acc)
                    with log_path.open("a") as f:
                        f.write(json.dumps({"step": outer, "val_acc": val_acc}) + "\n")
                    if val_acc > best_val:
                        best_val = val_acc
                        self._save("best")
                if self.distributed:
                    dist.barrier()

            if (outer + 1) % self.args.save_every == 0:
                if self.is_main:
                    self._save(f"step_{outer + 1}")
                if self.distributed:
                    dist.barrier()

        if self.is_main:
            self._save("final")
        if self.distributed:
            dist.barrier()
            dist.destroy_process_group()

    def _save(self, tag: str) -> None:
        # Only invoked on rank 0; saves the unwrapped HF model.
        path = self.output_dir / tag
        path.mkdir(parents=True, exist_ok=True)
        self.policy_module.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        log.info("Checkpoint saved -> %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Data / models
    p.add_argument("--traces", required=True)
    p.add_argument("--library", required=True)
    p.add_argument("--val", default=None)
    p.add_argument("--policy-init", required=True,
                   help="Path to SFT-warm-started checkpoint (or base model).")
    p.add_argument("--output-dir", required=True)

    # GRPO knobs
    p.add_argument("--num-outer-steps", type=int, default=200)
    p.add_argument("--prompts-per-step", type=int, default=4)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--ppo-epochs", type=int, default=1)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--kl-coef", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    # Rollout knobs
    p.add_argument("--rollout-max-steps", type=int, default=8)
    p.add_argument("--rollout-max-new-tokens", type=int, default=512)
    p.add_argument("--rollout-temperature", type=float, default=1.0)
    p.add_argument("--rollout-top-p", type=float, default=0.95)

    # Reward weights
    p.add_argument("--w-answer",   type=float, default=1.0)
    p.add_argument("--w-tool-seq", type=float, default=0.3)
    p.add_argument("--w-plan-sim", type=float, default=0.3)
    p.add_argument("--w-composite", type=float, default=0.2)
    p.add_argument("--w-format",   type=float, default=0.2)

    # Eval / checkpointing
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-n", type=int, default=0,
                   help="Number of val examples per evaluation "
                        "(0 = use all of them).")
    p.add_argument("--eval-batch-size", type=int, default=16,
                   help="Number of val problems rolled out in parallel "
                        "via rollout_batch.")
    p.add_argument("--save-every", type=int, default=50)

    # Mini-batch size for the batched assistant-logprob forwards used by
    # both the old/ref caching path and the PPO update.  Larger = fewer
    # forward passes but more padding waste; 8 is a good default for a
    # 1.7B model on H200.
    p.add_argument("--logp-batch-size", type=int, default=32)
    p.add_argument("--grad-ckpt", action="store_true", default=False,
                   help="Enable gradient checkpointing on the policy. Off by "
                        "default: with a 0.6B-1B bf16 student the activation "
                        "memory is small enough that checkpointing just costs "
                        "wall-clock (~2× slower PPO step).")

    # vLLM rollout (opt-in).  When enabled, each DDP rank co-locates an
    # in-process vLLM engine on its local GPU and runs all generation through
    # it.  Weights are pushed into the engine once per outer step via
    # collective_rpc + load_weights.  The HF generate path remains the
    # default fallback.
    p.add_argument("--use-vllm-rollout", action="store_true",
                   help="Use vLLM for rollout generation instead of HF "
                        "model.generate.")
    p.add_argument("--vllm-gpu-mem-util", type=float, default=0.30,
                   help="Fraction of GPU HBM vLLM may use (relative to total). "
                        "Leave headroom for the training model + Adam states "
                        "+ ref model.")
    p.add_argument("--vllm-max-model-len", type=int, default=8192,
                   help="vLLM max_model_len. Must be >= the longest "
                        "system+user+assistant trace your rollouts produce.")

    # Stage 3 online co-evolution (Algorithm 1, lines 18–23)
    p.add_argument("--enable-online-insert", action="store_true",
                   help="Mine new composite tools from successful rollouts "
                        "and insert them into the live library each step.")
    p.add_argument("--insert-threshold", type=float, default=0.8,
                   help="Reward threshold rho; trajectories with total reward "
                        ">= rho are candidates for tool distillation.")
    p.add_argument("--retrieve-per-rollout", action="store_true",
                   help="Call HIERRETRIEVE(q, I, G, k) per training example "
                        "before each rollout so newly inserted tools show up "
                        "in the student's prompt.")
    p.add_argument("--retrieve-k", type=int, default=8,
                   help="k for HIERRETRIEVE — number of tools materialised "
                        "into the student's prompt per query.")
    p.add_argument("--retrieve-mode", choices=["index", "llm", "hybrid"],
                   default="index",
                   help="HIERRETRIEVE backend: 'index' (LLM-free, cheap), "
                        "'llm' (full Algorithm 2 per query, accurate but "
                        "slow), or 'hybrid' (LLM every N steps, index "
                        "otherwise).")
    p.add_argument("--retrieve-llm-every", type=int, default=5,
                   help="In --retrieve-mode=hybrid, call the LLM retriever "
                        "every N outer steps; index-only on the rest.")
    p.add_argument("--online-miner", choices=["heuristic", "teacher"],
                   default="heuristic",
                   help="Online tool miner backend.  'heuristic' chains "
                        "primitives observed in the rollout into a composite "
                        "(no extra LLM).  'teacher' reuses the Stage-1 Teacher "
                        "vLLM process via TOOL_EXTRACTION_PROMPT — paper-"
                        "faithful, higher-quality, but pays the cost of "
                        "keeping the teacher model resident on rank 0.")
    p.add_argument("--teacher-miner-model",
                   default="Qwen/Qwen2.5-7B-Instruct",
                   help="Model id for the rank-0 teacher miner when "
                        "--online-miner=teacher.")
    p.add_argument("--teacher-miner-gpu-mem-util", type=float, default=0.25,
                   help="Fraction of rank-0 HBM the teacher miner's vLLM may "
                        "use.  Keep low — the policy + ref model already "
                        "occupy most of the GPU.")
    p.add_argument("--insert-require-correct", action="store_true",
                   help="Require r_answer == 1 (not just total >= rho) before "
                        "a successful rollout is distilled into a new tool. "
                        "Matches Algorithm 1's R(T*)==1 gate.")

    # Misc
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    trainer = GRPOTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()