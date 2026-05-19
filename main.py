"""Co-Evolve-Skills: Train / Validate / Test pipeline on GSM8K.

Training  – The Teacher solves training problems, extracts reusable tools,
            and populates the shared ToolLibrary.
Validation – The Student solves validation problems using the ToolLibrary
             (tool-augmented) and without tools (direct) to measure accuracy
             and decide whether to continue training.
Test      – Final evaluation of the Student on the held-out test set.

Usage examples:
    # Full pipeline (train -> val -> test)
    python main.py

    # Skip training, load an existing library and evaluate
    python main.py --skip-train --library-path checkpoints/tool_library.json

    # Only run the test phase on a saved library
    python main.py --skip-train --skip-val --library-path checkpoints/tool_library.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset

from model.teacher import Teacher
from model.student import Student, StudentConfig
from tool.tool_library import ToolLibrary

# ---------------------------------------------------------------------------
# Data-parallel helpers (set by torchrun when DP > 1)
# ---------------------------------------------------------------------------
def _get_dp_rank() -> int:
    return int(os.environ.get("RANK", 0))

def _get_dp_world() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))

def _is_main_rank() -> bool:
    return _get_dp_rank() == 0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Answer extraction (reused from eval_qwen3_gsm8k.py)
# ---------------------------------------------------------------------------
ANS_RE = re.compile(r"####\s*(\-?[\d,\.]+)")


def extract_answer_hf(text: str) -> str | float | int:
    """Extract the answer after the ``####`` marker (standard GSM8K format)."""
    m = ANS_RE.search(text)
    if m:
        num_str = m.group(1).strip().replace(",", "")
        try:
            return int(num_str) if "." not in num_str else float(num_str)
        except ValueError:
            pass
    return "[invalid]"


def extract_answer(text: str) -> str | float | int:
    """Extract answer: prefer ``####`` format, fallback to last number."""
    ans = extract_answer_hf(text)
    if ans != "[invalid]":
        return ans
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        num_str = numbers[-1]
        try:
            return int(num_str) if "." not in num_str else float(num_str)
        except ValueError:
            pass
    return "[invalid]"


def is_correct(model_output: str, gold_answer: str) -> bool:
    """Check whether *model_output* matches the *gold_answer*."""
    gold = extract_answer_hf(gold_answer)
    if gold == "[invalid]":
        return False
    pred = extract_answer(model_output)
    if pred == "[invalid]":
        return False
    try:
        return float(pred) == float(gold)
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def load_gsm8k(val_ratio: float = 0.1, seed: int = 42) -> dict:
    """Load GSM8K and split train into train/val.

    Returns a dict with keys ``"train"``, ``"val"``, ``"test"``, each a list
    of dicts with ``"question"`` and ``"answer"`` keys.
    """
    log.info("Loading GSM8K dataset ...")
    ds = load_dataset("gsm8k", "main")
    train_ds = ds["train"].shuffle(seed=seed)
    test_ds = ds["test"]

    n_val = max(1, int(len(train_ds) * val_ratio))
    val_split = train_ds.select(range(n_val))
    train_split = train_ds.select(range(n_val, len(train_ds)))

    log.info(
        "Dataset sizes — train: %d, val: %d, test: %d",
        len(train_split), len(val_split), len(test_ds),
    )
    return {
        "train": [{"question": r["question"], "answer": r["answer"]} for r in train_split],
        "val":   [{"question": r["question"], "answer": r["answer"]} for r in val_split],
        "test":  [{"question": r["question"], "answer": r["answer"]} for r in test_ds],
    }


# ---------------------------------------------------------------------------
# Training phase
# ---------------------------------------------------------------------------
def train(
    teacher: Teacher,
    train_data: list[dict],
    checkpoint_dir: Path,
    *,
    save_every: int = 50,
    batch_size: int = 1,
) -> ToolLibrary:
    """Teacher solves training problems and extracts tools into the ToolLibrary.

    The library is checkpointed every *save_every* problems.

    When *batch_size* > 1, problems are solved in parallel using batched
    vLLM calls (see :meth:`Teacher.solve_and_learn_batch`).
    """
    library = teacher.tool_library
    results: list[dict[str, Any]] = []
    n = len(train_data)
    correct, total = 0, 0

    log.info("=== Training phase: %d problems (batch_size=%d) ===", n, batch_size)

    if batch_size <= 1:
        # ---------- Original sequential loop ----------
        for i, sample in enumerate(train_data):
            question = sample["question"]
            gold = sample["answer"]

            log.info("[Train %d/%d] Solving ...", i + 1, n)
            t0 = time.time()
            try:
                out = teacher.solve_and_learn(question)
            except Exception as e:
                log.warning("[Train %d/%d] Teacher failed: %s", i + 1, n, e)
                results.append({"index": i, "correct": False, "error": str(e)})
                total += 1
                continue
            elapsed = time.time() - t0

            pred_correct = is_correct(str(out["answer"]), gold)
            correct += int(pred_correct)
            total += 1

            results.append({
                "index": i,
                "correct": pred_correct,
                "tool_names": out.get("tool_names", []),
                "tool_name": out.get("tool_name"),
                "elapsed": round(elapsed, 2),
            })
            log.info(
                "[Train %d/%d] correct=%s  tools=%s  time=%.1fs  acc=%.2f%%",
                i + 1, n, pred_correct, out.get("tool_names", []),
                elapsed, 100 * correct / total,
            )

            # Checkpoint periodically
            if (i + 1) % save_every == 0:
                _save_checkpoint(library, results, checkpoint_dir, tag=f"train_{i+1}")
    else:
        # ---------- Batched parallel loop ----------
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch = train_data[batch_start:batch_end]
            questions = [s["question"] for s in batch]
            golds = [s["answer"] for s in batch]

            log.info(
                "[Train %d–%d/%d] Solving batch of %d ...",
                batch_start + 1, batch_end, n, len(batch),
            )
            t0 = time.time()
            try:
                batch_out = teacher.solve_and_learn_batch(questions)
            except Exception as e:
                log.warning(
                    "[Train %d–%d/%d] Batch failed: %s",
                    batch_start + 1, batch_end, n, e,
                )
                for j in range(len(batch)):
                    results.append({
                        "index": batch_start + j, "correct": False, "error": str(e),
                    })
                    total += 1
                continue
            elapsed = time.time() - t0

            for j, (out, gold) in enumerate(zip(batch_out, golds)):
                global_idx = batch_start + j
                pred_correct = is_correct(str(out["answer"]), gold)
                correct += int(pred_correct)
                total += 1
                results.append({
                    "index": global_idx,
                    "correct": pred_correct,
                    "tool_names": out.get("tool_names", []),
                    "tool_name": out.get("tool_name"),
                    "elapsed": round(elapsed / len(batch), 2),
                })

            log.info(
                "[Train %d–%d/%d] batch_time=%.1fs  acc=%.2f%%  tools_in_lib=%d",
                batch_start + 1, batch_end, n, elapsed,
                100 * correct / total if total else 0, len(library),
            )

            # Checkpoint periodically
            if batch_end % save_every < batch_size or batch_end == n:
                _save_checkpoint(library, results, checkpoint_dir, tag=f"train_{batch_end}")

    _save_checkpoint(library, results, checkpoint_dir, tag="train_final")
    log.info(
        "Training complete — accuracy: %.2f%% (%d/%d), tools in library: %d",
        100 * correct / total if total else 0, correct, total, len(library),
    )
    return library


# ---------------------------------------------------------------------------
# Evaluation (shared by validation and test)
# ---------------------------------------------------------------------------
def evaluate(
    student: Student,
    data: list[dict],
    *,
    phase: str = "eval",
    use_tools: bool = True,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Evaluate the Student on *data*.

    When *batch_size* > 1, problems are solved in parallel using batched
    vLLM calls via :meth:`Student.solve_batch`.

    Returns a dict with ``accuracy``, ``correct``, ``total``, and per-example
    ``results``.
    """
    results: list[dict[str, Any]] = []
    correct, total = 0, 0
    n = len(data)

    log.info(
        "=== %s phase: %d problems (use_tools=%s, batch_size=%d) ===",
        phase, n, use_tools, batch_size,
    )

    if batch_size <= 1:
        # ---------- Sequential loop ----------
        for i, sample in enumerate(data):
            question = sample["question"]
            gold = sample["answer"]

            t0 = time.time()
            try:
                out = student.solve(question, use_tools=use_tools)
            except Exception as e:
                log.warning("[%s %d/%d] Student failed: %s", phase, i + 1, n, e)
                results.append({"index": i, "correct": False, "error": str(e)})
                total += 1
                continue
            elapsed = time.time() - t0

            answer_text = str(out.get("answer", ""))
            pred_correct = is_correct(answer_text, gold)
            correct += int(pred_correct)
            total += 1

            results.append({
                "index": i,
                "correct": pred_correct,
                "prediction": answer_text[:500],
                "elapsed": round(elapsed, 2),
            })

            if (i + 1) % 50 == 0 or (i + 1) == n:
                log.info(
                    "[%s %d/%d] running_acc=%.2f%%",
                    phase, i + 1, n, 100 * correct / total,
                )
    else:
        # ---------- Batched loop ----------
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch = data[batch_start:batch_end]
            questions = [s["question"] for s in batch]
            golds = [s["answer"] for s in batch]

            t0 = time.time()
            try:
                batch_out = student.solve_batch(
                    questions, use_tools=use_tools
                )
            except Exception as e:
                log.warning(
                    "[%s %d–%d/%d] Batch failed: %s",
                    phase, batch_start + 1, batch_end, n, e,
                )
                for j in range(len(batch)):
                    results.append({
                        "index": batch_start + j,
                        "correct": False,
                        "error": str(e),
                    })
                    total += 1
                continue
            elapsed = time.time() - t0

            for j, (out, gold) in enumerate(zip(batch_out, golds)):
                global_idx = batch_start + j
                answer_text = str(out.get("answer", ""))
                pred_correct = is_correct(answer_text, gold)
                correct += int(pred_correct)
                total += 1
                results.append({
                    "index": global_idx,
                    "correct": pred_correct,
                    "prediction": answer_text[:500],
                    "elapsed": round(elapsed / len(batch), 2),
                })

            log.info(
                "[%s %d–%d/%d] batch_time=%.1fs  running_acc=%.2f%%",
                phase, batch_start + 1, batch_end, n, elapsed,
                100 * correct / total if total else 0,
            )

    accuracy = correct / total if total else 0.0
    log.info(
        "%s complete — accuracy: %.2f%% (%d/%d)",
        phase, 100 * accuracy, correct, total,
    )
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _save_checkpoint(
    library: ToolLibrary,
    results: list[dict],
    checkpoint_dir: Path,
    tag: str,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    lib_path = checkpoint_dir / "tool_library.json"
    library.save(lib_path)
    res_path = checkpoint_dir / f"results_{tag}.json"
    res_path.write_text(json.dumps(results, indent=2, default=str))
    log.info("Checkpoint saved: %s  (%d tools)", lib_path, len(library))


def save_eval_results(
    metrics: dict[str, Any],
    output_dir: Path,
    filename: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(json.dumps(metrics, indent=2, default=str))
    log.info("Eval results saved: %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Co-Evolve-Skills: GSM8K train / val / test pipeline",
    )

    # Model configuration
    p.add_argument(
        "--teacher-model", type=str, default="openai/gpt-4o",
        help="LiteLLM model id for the Teacher (default: openai/gpt-4o)",
    )
    p.add_argument(
        "--student-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model id for the Student (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    p.add_argument(
        "--teacher-max-steps", type=int, default=20,
        help="Max reasoning steps for the Teacher CodeAgent",
    )
    p.add_argument(
        "--student-max-steps", type=int, default=10,
        help="Max ReAct loop steps for the Student",
    )
    p.add_argument(
        "--student-max-tokens", type=int, default=2048,
        help="Max new tokens per Student generation turn",
    )
    p.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Number of GPUs for vLLM tensor parallelism (default: 1)",
    )
    p.add_argument(
        "--data-parallel-size", type=int, default=1,
        help="Number of vLLM data-parallel replicas (default: 1). "
             "Total GPUs = tensor_parallel_size * data_parallel_size.",
    )

    # Data
    p.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="Fraction of train split to use as validation (default: 0.1)",
    )
    p.add_argument(
        "--train-limit", type=int, default=None,
        help="Limit training to first N examples (for debugging)",
    )
    p.add_argument(
        "--val-limit", type=int, default=None,
        help="Limit validation to first N examples",
    )
    p.add_argument(
        "--test-limit", type=int, default=None,
        help="Limit test to first N examples",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # Phases
    p.add_argument(
        "--skip-train", action="store_true",
        help="Skip training; load library from --library-path instead",
    )
    p.add_argument(
        "--skip-val", action="store_true",
        help="Skip the validation phase",
    )
    p.add_argument(
        "--skip-test", action="store_true",
        help="Skip the test phase",
    )

    # Checkpointing / output
    p.add_argument(
        "--library-path", type=str, default=None,
        help="Path to a saved ToolLibrary JSON (used with --skip-train)",
    )
    p.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Directory for checkpoints and results (default: outputs)",
    )
    p.add_argument(
        "--save-every", type=int, default=50,
        help="Checkpoint the library every N training problems",
    )
    p.add_argument(
        "--batch-size", type=int, default=1,
        help="Number of problems to solve in parallel via batched vLLM calls "
             "(default: 1 = sequential, same as before)",
    )

    # Retrieval configuration
    p.add_argument(
        "--no-hierarchical-retrieval", action="store_true",
        help="Disable 4-step hierarchical tool retrieval (use naive recency-based method)",
    )
    p.add_argument(
        "--retrieval-max-shortlist", type=int, default=12,
        help="Max tools after L2 description scan (default: 12)",
    )
    p.add_argument(
        "--retrieval-max-final", type=int, default=8,
        help="Max tools after L3-L4 deep inspection (default: 8)",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    dp_rank = _get_dp_rank()
    dp_world = _get_dp_world()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_id
    checkpoint_dir = output_dir / "checkpoints"

    if _is_main_rank():
        log.info("Run id: %s", run_id)
        log.info("Output dir: %s", output_dir)
        if dp_world > 1:
            log.info("Data parallelism: %d ranks", dp_world)

    # ---- Load data --------------------------------------------------------
    splits = load_gsm8k(val_ratio=args.val_ratio, seed=args.seed)
    if args.train_limit:
        splits["train"] = splits["train"][: args.train_limit]
    if args.val_limit:
        splits["val"] = splits["val"][: args.val_limit]
    if args.test_limit:
        splits["test"] = splits["test"][: args.test_limit]

    # ---- Build / load ToolLibrary ----------------------------------------
    if args.skip_train and args.library_path:
        if _is_main_rank():
            log.info("Loading ToolLibrary from %s", args.library_path)
        library = ToolLibrary.load(args.library_path)
    else:
        library = ToolLibrary()

    # ---- Training ---------------------------------------------------------
    if not args.skip_train:
        teacher = Teacher(
            model_id=args.teacher_model,
            tool_library=library,
            max_steps=args.teacher_max_steps,
            tensor_parallel_size=args.tensor_parallel_size,
            data_parallel_size=args.data_parallel_size,
        )

        # Shard training data across DP ranks
        train_data = splits["train"]
        if dp_world > 1:
            train_data = train_data[dp_rank::dp_world]
            if _is_main_rank():
                log.info(
                    "Rank %d/%d: processing %d/%d training problems",
                    dp_rank, dp_world, len(train_data), len(splits["train"]),
                )

        library = train(
            teacher,
            train_data,
            checkpoint_dir,
            save_every=args.save_every,
            batch_size=args.batch_size,
        )

        if _is_main_rank():
            log.info("ToolLibrary after training:\n%s", library.describe())

            # Save trained library with teacher name and dataset name
            teacher_tag = args.teacher_model.replace("/", "_")
            dataset_tag = "gsm8k"
            lib_save_path = output_dir / f"library_{teacher_tag}_{dataset_tag}.json"
            library.save(lib_save_path)
            log.info("Saved trained library to %s", lib_save_path)

        # Free teacher model to reclaim GPU memory for the student
        del teacher
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()

    # Non-main DP ranks exit after training — only rank 0 does eval/test
    if not _is_main_rank():
        return

    # ---- Validation -------------------------------------------------------
    if not args.skip_val:
        student_cfg = StudentConfig(
            model_name=args.student_model,
            max_new_tokens=args.student_max_tokens,
            max_steps=args.student_max_steps,
            use_hierarchical_retrieval=not args.no_hierarchical_retrieval,
            retrieval_max_shortlist=args.retrieval_max_shortlist,
            retrieval_max_final=args.retrieval_max_final,
        )
        student = Student(config=student_cfg, tool_library=library)

        # Evaluate with tools
        val_tools = evaluate(
            student, splits["val"], phase="Validation (tools)", use_tools=True,
            batch_size=args.batch_size,
        )
        save_eval_results(val_tools, output_dir, "val_with_tools.json")

        log.info(
            "Validation summary — with_tools: %.2f%%",
            100 * val_tools["accuracy"],
        )

    # ---- Test -------------------------------------------------------------
    if not args.skip_test:
        # Reuse or create the student
        if args.skip_val or "student" not in dir():
            student_cfg = StudentConfig(
                model_name=args.student_model,
                max_new_tokens=args.student_max_tokens,
                max_steps=args.student_max_steps,
                use_hierarchical_retrieval=not args.no_hierarchical_retrieval,
                retrieval_max_shortlist=args.retrieval_max_shortlist,
                retrieval_max_final=args.retrieval_max_final,
            )
            student = Student(config=student_cfg, tool_library=library)

        test_tools = evaluate(
            student, splits["test"], phase="Test (tools)", use_tools=True,
            batch_size=args.batch_size,
        )
        save_eval_results(test_tools, output_dir, "test_with_tools.json")

        log.info(
            "Test summary — with_tools: %.2f%%",
            100 * test_tools["accuracy"],
        )

    # ---- Final summary ----------------------------------------------------
    log.info("Done. All outputs saved to: %s", output_dir)


if __name__ == "__main__":
    main()
