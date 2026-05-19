"""SFT warm-start of the Student on teacher traces.

Trains the student model with cross-entropy loss on assistant tokens only,
using the ``teacher_text`` field of ``teacher_traces.jsonl``.

Run:
    python -m training.sft_warmstart \\
        --traces outputs/grpo_run/teacher_traces.jsonl \\
        --student-model Qwen/Qwen2.5-1.5B-Instruct \\
        --output-dir outputs/grpo_run/checkpoints/sft \\
        --epochs 2 --batch-size 4 --lr 5e-6
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from training.data import read_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset — turns each trace into (input_ids, labels) with non-assistant
# tokens labelled -100 (ignored by the loss).
# ---------------------------------------------------------------------------
class TeacherTraceDataset(Dataset):
    def __init__(
        self,
        traces_path: str | Path,
        tokenizer,
        max_length: int = 2048,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = [r for r in read_jsonl(traces_path) if r.get("teacher_text")]
        log.info("Loaded %d SFT records from %s", len(self.records), traces_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        text = rec["teacher_text"]
        spans = rec.get("assistant_char_spans") or []

        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]

        labels = [-100] * len(input_ids)
        for span in spans:
            start_c, end_c = int(span[0]), int(span[1])
            for tok_i, (s, e) in enumerate(offsets):
                if s >= start_c and e <= end_c and s < e:
                    labels[tok_i] = input_ids[tok_i]

        return {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels":         torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class PadCollator:
    """Right-pad a batch of variable-length (input_ids, labels) tensors."""
    pad_token_id: int

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(b["input_ids"].size(0) for b in batch)
        out = {
            "input_ids":      [],
            "attention_mask": [],
            "labels":         [],
        }
        for b in batch:
            pad_len = max_len - b["input_ids"].size(0)
            out["input_ids"].append(
                torch.cat([b["input_ids"],
                           torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            )
            out["attention_mask"].append(
                torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
            )
            out["labels"].append(
                torch.cat([b["labels"],
                           torch.full((pad_len,), -100, dtype=torch.long)])
            )
        return {k: torch.stack(v) for k, v in out.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", required=True, help="teacher_traces.jsonl path")
    p.add_argument("--student-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--bf16", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )

    dataset = TeacherTraceDataset(args.traces, tokenizer, max_length=args.max_length)
    collator = PadCollator(pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=args.bf16,
        report_to=[],
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    log.info("SFT warm-start saved to %s", args.output_dir)


if __name__ == "__main__":
    main()