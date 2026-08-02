from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import parse_dapo_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render exactly the prompt consumed by Meta-RLVR."
    )
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.parquet.is_file():
        raise FileNotFoundError(args.parquet)
    if args.row < 0:
        raise ValueError("row must be non-negative.")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    dataset = load_dataset(
        "parquet",
        data_files=str(args.parquet),
        split="train",
    )
    if args.row >= len(dataset):
        raise IndexError(
            f"row {args.row} is outside a dataset with {len(dataset)} rows."
        )
    problem = parse_dapo_row(dataset[args.row])
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
        raise ValueError("Tokenizer does not define a non-empty chat_template.")

    rendered = tokenizer.apply_chat_template(
        problem.conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_attention_mask=False,
    )

    print("=== DATASET ROW ===")
    print(f"row: {args.row}")
    print(f"uid: {problem.uid}")
    print(f"data_source: {problem.data_source}")
    print(f"ground_truth: {problem.ground_truth}")
    print("messages:")
    print(
        json.dumps(
            problem.conversation,
            ensure_ascii=False,
            indent=2,
        )
    )
    print("\n=== TOKENIZER CHAT TEMPLATE ===")
    print(tokenizer.chat_template)
    print("\n=== RENDERED TRAINING PROMPT ===")
    print(rendered)
    print("\n=== TOKENIZATION ===")
    print(f"num_prompt_tokens: {len(encoded['input_ids'])}")
    print(f"bos_token: {tokenizer.bos_token!r} ({tokenizer.bos_token_id})")
    print(f"eos_token: {tokenizer.eos_token!r} ({tokenizer.eos_token_id})")
    print(f"pad_token: {tokenizer.pad_token!r} ({tokenizer.pad_token_id})")


if __name__ == "__main__":
    main()
