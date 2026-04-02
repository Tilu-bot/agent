#!/usr/bin/env python3
"""Fine-tune a small quantization-friendly model on run data exported from this system.

Usage
-----
    python scripts/fine_tune.py \\
        --data training_data.jsonl \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --output ./ft-model \\
        --epochs 3

The input JSONL file is produced by the API endpoint::

    GET /api/runs/export/training-data?format=alpaca&min_confidence=70

Recommended base models (small, quantization-friendly)
-------------------------------------------------------
* Qwen/Qwen2.5-0.5B-Instruct   (~0.5 B params, ternary-weight friendly)
* Qwen/Qwen2.5-1.5B-Instruct   (~1.5 B params)
* microsoft/phi-3-mini-4k-instruct (~3.8 B params, fits in 4 GB RAM at Q4)
* microsoft/bitnet_b1_58-3B     (native 1.58-bit / ternary weights)

Requirements
------------
    pip install transformers trl datasets peft bitsandbytes

GGUF export (after training)
-----------------------------
    bash scripts/export_gguf.sh ./ft-model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a small LLM on Agentic run data."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to training JSONL file (Alpaca format).",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--output",
        default="./ft-model",
        help="Directory to save the fine-tuned model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length in tokens.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (set 0 to disable LoRA and do full fine-tune).",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization during training (uses more VRAM).",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def alpaca_to_prompt(example: dict) -> str:
    """Convert an Alpaca-format example to a single training string."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")
    if inp:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{output}"
        )
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{output}"
    )


def main() -> None:
    args = parse_args()

    # ── Lazy imports so the script fails early with a clear message ────────────
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType as PeftTaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer
    except ImportError as e:
        print(
            f"Missing dependency: {e}\n"
            "Install with:\n"
            "  pip install transformers trl datasets peft bitsandbytes",
            file=sys.stderr,
        )
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # ── Load training data ─────────────────────────────────────────────────────
    raw = load_jsonl(str(data_path))
    if not raw:
        print("No training examples found in the JSONL file.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(raw)} training examples from {data_path}")

    prompts = [alpaca_to_prompt(ex) for ex in raw]
    dataset = Dataset.from_dict({"text": prompts})

    # ── Model + tokenizer ──────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")

    bnb_config = None
    if not args.no_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── LoRA ──────────────────────────────────────────────────────────────────
    if args.lora_r > 0:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            bias="none",
            task_type=PeftTaskType.CAUSAL_LM,
            target_modules="all-linear",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ── Training ───────────────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 16 // args.batch_size),
        learning_rate=args.lr,
        fp16=not args.no_4bit,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
    )

    print("Starting training…")
    trainer.train()

    print(f"Saving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("Done. Next step: run scripts/export_gguf.sh to convert for llama.cpp/Ollama.")


if __name__ == "__main__":
    main()
