# CoCoDA: Co-evolving Compositional DAG for Tool-Augmented Agents

This repository contains the source code for the NeurIPS 2026 submission
*CoCoDA: Co-evolving Compositional DAG for Tool-Augmented Agents*.

CoCoDA represents the tool library as a **compositional code DAG** whose nodes
are primitive or composite tools and whose edges encode invocation
dependencies. The same DAG simultaneously (i) bounds context cost via **Typed
DAG Retrieval** — a four-stage cascade over typed signatures (L1),
descriptions (L2), pre/post-condition specifications (L3) and worked examples
(L4) — and (ii) trains the planner via a graph-aware GRPO objective that
credits composites by their primitive expansion size `flat(·)`.

## Repository layout

```
.
├── main.py                       # End-to-end pipeline (train / val / test on GSM8K)
├── model/
│   ├── teacher.py                # Teacher CodeAgent: solves problems and proposes composites
│   └── student.py                # Student planner (vLLM-backed ReAct loop)
├── tool/
│   ├── tool.py                   # Tool object: body + executor
│   ├── tool_metadata.py          # 4-level record: L1 signature / L2 desc / L3 spec / L4 examples
│   ├── tool_library.py           # Compositional DAG, INSERTTOOL, acyclicity / dedup checks
│   └── tool_retriever.py         # TYPEDDAGRETRIEVE (Algorithm 2)
├── training/
│   ├── collect_teacher_traces.py # Stage 1: experience-based tool distillation
│   ├── format_traces.py          # Convert trajectories to SFT data
│   ├── sft_warmstart.py          # Stage 2: cold-start SFT on warm library
│   ├── rollout.py                # Group rollout for GRPO
│   ├── vllm_rollout.py           # Batched vLLM rollout backend
│   ├── rewards.py                # R = R_res + λ R_comp (graph-aware reward)
│   ├── online_tool_miner.py      # Stage 3 (d): teacher abstractor + INSERTTOOL
│   ├── grpo_trainer.py           # Stage 3: coupled GRPO + library update
│   └── data.py                   # Dataset loaders (GSM8K / MATH / WTQ / FinQA / EvalPlus / MBPP)
```

## Installation

The code targets Python 3.10+ on Linux with CUDA-capable GPUs (the main results
are reported on 4× NVIDIA H200 80 GB).

```bash
conda create -n cocoda python=3.10 -y
conda activate cocoda
pip install -r requirements.txt
```

The training stack uses PyTorch 2.5, DeepSpeed ZeRO-3 (full FT) or PEFT 0.13
(LoRA), and vLLM 0.6.3 for rollouts. The teacher is invoked via LiteLLM and
expects credentials for whichever backend you point it at (e.g.
`OPENAI_API_KEY` for `openai/gpt-4o`, or a local OpenAI-compatible server for
Qwen3-32B).

## Quick start

### End-to-end pipeline on GSM8K

```bash
python main.py \
    --teacher-model openai/gpt-4o \
    --student-model Qwen/Qwen2.5-1.5B-Instruct \
    --output-dir outputs/gsm8k \
    --tensor-parallel-size 1
```

This runs all three stages described in Algorithm 1:

1. **Stage 1 — Experience-based tool distillation.** The teacher solves the
   training set, successful trajectories are abstracted into composites, and
   each candidate is committed via `INSERTTOOL` after acyclicity / spec
   validation.
2. **Stage 2 — Cold-start SFT.** Teacher demonstrations against the warm
   library are used to fine-tune the student.
3. **Stage 3 — Online co-evolution.** GRPO updates the student under
   `R = R_res + λ R_comp`; successful per-query rollouts are folded back into
   the DAG, so the next query sees an updated library.

Useful flags:

| Flag | Description |
| --- | --- |
| `--skip-train --library-path PATH` | Load a previously trained library and only evaluate. |
| `--skip-val` / `--skip-test` | Disable the validation / test phase. |
| `--no-hierarchical-retrieval` | Replace Typed DAG Retrieval with the flat-recency baseline. |
| `--retrieval-max-shortlist`, `--retrieval-max-final` | Control the L2 / L4 survivor sizes (`k2` in the paper). |
| `--batch-size` | Number of queries solved in parallel per vLLM call. |
| `--data-parallel-size` | Replicas spawned by `torchrun` (set with the env vars `RANK`, `WORLD_SIZE`). |

### Running the GRPO stage standalone

```bash
python -m training.grpo_trainer \
    --traces      outputs/gsm8k/teacher_traces.jsonl \
    --library     outputs/gsm8k/tool_library.json \
    --val         outputs/gsm8k/val.jsonl \
    --policy-init outputs/gsm8k/checkpoints/sft \
    --output-dir  outputs/gsm8k/checkpoints/grpo
```

Default hyperparameters match Appendix E of the paper: group size `G = 8`,
clip range `ϵ = 0.2`, KL coefficient `0.01`, compositional weight `λ = 0.20`,
and success threshold `ρ = 0.8`.

## Datasets

CoCoDA is evaluated on six public benchmarks with deterministic verifiers
(see Appendix B / G of the paper):

| Dataset | Task | Verifier |
| --- | --- | --- |
| GSM8K | grade-school math | numeric exact match |
| MATH | competition math | SymPy equivalence |
| WikiTableQuestions | table QA | SQL + normalised match |
| FinQA | financial table QA | numeric program (rel. err. < 1e-3) |
| EvalPlus (HumanEval+) | code generation | extended unit tests (pass@1) |
| MBPP | code generation | unit tests (pass@1) |

All splits are loaded via `datasets.load_dataset`; no manual download is
required.

## Reproducing the main results

The table below summarises wall-clock cost on 4× H200 (Appendix I); see
Table 1 for the full accuracy numbers and Figure 3 for the cost-vs-library
sweep against flat retrieval and RAPTOR-style text-hierarchical RAG.

| Student | Stage 1+2 (h) | Stage 3 / epoch (h) | Total GPU-hours |
| --- | --- | --- | --- |
| 0.6 B | 1.2 | 2.1 | 35 |
| 1.7 B | 1.8 | 3.4 | 55 |
| 4 B   | 2.9 | 5.6 | 87 |
| 8 B   | 3.4 | 6.8 (LoRA) | 98 |

All main-table numbers are averaged over three seeds (`{13, 42, 2026}`).

## Outputs

A run produces:

```
outputs/<run_id>/
├── checkpoints/
│   ├── tool_library.json              # serialised DAG (V, E, I)
│   ├── sft/                           # cold-start SFT weights
│   └── grpo/                          # co-evolved policy
├── teacher_traces.jsonl               # successful trajectories
├── val.jsonl, test.jsonl              # per-query predictions and rewards
└── metrics.json                       # accuracy / latency / library stats
```

`tool_library.json` can be re-loaded with `--skip-train --library-path …` for
inference-only runs or for swapping retrieval substrates in the cost-vs-size
ablation.

## License

Code released under the MIT license; vendored components retain their
upstream licenses. The pretrained Qwen3 weights are governed by their
respective model cards.
