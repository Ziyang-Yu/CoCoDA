"""Dataset loading and JSONL helpers shared across the GRPO pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Write *records* as JSON-lines, returning the number of lines written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, default=str))
            f.write("\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Iterate over JSON-line records in *path*."""
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_gsm8k_split(
    val_ratio: float = 0.1,
    seed: int = 42,
    train_limit: int | None = None,
    val_limit: int | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Re-export of :func:`main.load_gsm8k` so the training package does
    not need to import the project-root ``main.py`` directly."""
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main")
    train_ds = ds["train"].shuffle(seed=seed)
    n_val = max(1, int(len(train_ds) * val_ratio))
    val_split = train_ds.select(range(n_val))
    train_split = train_ds.select(range(n_val, len(train_ds)))

    train = [
        {"question": r["question"], "answer": r["answer"]} for r in train_split
    ]
    val = [
        {"question": r["question"], "answer": r["answer"]} for r in val_split
    ]
    if train_limit:
        train = train[:train_limit]
    if val_limit:
        val = val[:val_limit]
    return {"train": train, "val": val}
