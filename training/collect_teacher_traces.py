"""Phase 1 — collect teacher trajectories.

Runs the Teacher (smolagents CodeAgent + extracted-tool library) on a slice of
the training set and writes one JSON-line record per *successfully solved*
problem to ``teacher_traces.jsonl``.

Each record is the input format the SFT warm-start and the GRPO plan-alignment
reward both consume:

    {
      "problem": ...,
      "gold_answer": ...,
      "teacher_plan": [...],
      "teacher_tool_seq": [...],
      "teacher_text": "<full chat-formatted student-style trace>",
      "assistant_char_spans": [[start, end], ...],
      "tool_names_in_prompt": [...]
    }

Run:
    python -m training.collect_teacher_traces \\
        --teacher-model Qwen/Qwen2.5-7B-Instruct \\
        --student-model Qwen/Qwen2.5-1.5B-Instruct \\
        --train-limit 500 \\
        --output-dir outputs/grpo_run/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from transformers import AutoTokenizer

from main import is_correct
from model.teacher import Teacher
from tool.tool_library import ToolLibrary
from training.data import load_gsm8k_split, write_jsonl
from training.format_traces import format_solutions_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--student-model", default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="Used only for its tokenizer (chat template).")
    p.add_argument("--library-path", default=None,
                   help="Optional pre-existing ToolLibrary JSON to seed.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-limit", type=int, default=None,
                   help="Cap teacher solving to first N train problems "
                        "(default: use the full GSM8K train split).")
    p.add_argument("--val-limit", type=int, default=None,
                   help="Cap held-out val to first N problems "
                        "(default: full ~10%% of GSM8K train).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--teacher-max-steps", type=int, default=20)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_gsm8k_split(
        train_limit=args.train_limit,
        val_limit=args.val_limit,
    )
    log.info("train=%d, val=%d", len(splits["train"]), len(splits["val"]))

    library = (
        ToolLibrary.load(args.library_path) if args.library_path else ToolLibrary()
    )
    teacher = Teacher(
        model_id=args.teacher_model,
        tool_library=library,
        max_steps=args.teacher_max_steps,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    student_tok = AutoTokenizer.from_pretrained(args.student_model)
    if student_tok.pad_token is None:
        student_tok.pad_token = student_tok.eos_token

    # ---- Solve in batches and immediately format ----
    out_records: list[dict] = []
    n_total = 0
    n_correct = 0
    n_kept = 0

    for batch_start in range(0, len(splits["train"]), args.batch_size):
        batch = splits["train"][batch_start : batch_start + args.batch_size]
        questions = [b["question"] for b in batch]
        golds = [b["answer"] for b in batch]

        log.info(
            "[%d/%d] Solving batch of %d ...",
            batch_start + len(batch), len(splits["train"]), len(batch),
        )
        try:
            sols = teacher.solve_and_learn_batch(questions)
        except Exception as e:
            log.warning("Batch failed: %s", e)
            continue

        # Filter to solutions that (a) actually solved the problem and (b)
        # produced at least one extracted tool — both are required for the
        # plan-alignment reward to be meaningful.
        keep_questions: list[str] = []
        keep_solutions: list[dict] = []
        keep_golds: list[str] = []
        for q, sol, gold in zip(questions, sols, golds):
            n_total += 1
            if not is_correct(str(sol.get("answer", "")), gold):
                continue
            n_correct += 1
            if not sol.get("tool_names"):
                continue
            keep_questions.append(q)
            keep_solutions.append(sol)
            keep_golds.append(gold)

        if not keep_questions:
            continue

        formatted = format_solutions_batch(
            problems=keep_questions,
            solutions=keep_solutions,
            teacher=teacher,
            tokenizer=student_tok,
        )

        for q, gold, rec in zip(keep_questions, keep_golds, formatted):
            if rec is None:
                continue
            rec["gold_answer"] = gold  # use the GSM8K-formatted gold
            out_records.append(rec)
            n_kept += 1

        log.info(
            "running stats: total=%d, teacher_correct=%d, kept=%d, lib_size=%d",
            n_total, n_correct, n_kept, len(library),
        )

    # ---- Save ----
    out_path = output_dir / "teacher_traces.jsonl"
    write_jsonl(out_path, out_records)
    log.info("Wrote %d traces -> %s", len(out_records), out_path)

    # Snapshot the library so the student sees the same tools at train + eval
    lib_path = output_dir / "tool_library.json"
    library.save(lib_path)
    log.info("Saved frozen library -> %s", lib_path)

    # Also dump the val split so the GRPO trainer uses an identical slice
    val_path = output_dir / "val.jsonl"
    write_jsonl(val_path, splits["val"])
    log.info("Wrote %d val examples -> %s", len(splits["val"]), val_path)


if __name__ == "__main__":
    main()