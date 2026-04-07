#!/usr/bin/env python3
"""
Collect latent reasoning traces, ablation results, and early decodes over GSM8K.

Produces a JSONL file where each line is a complete trace for one question,
including per-step token predictions, ablation outputs, and progressive
early-decode outputs. All data generation is task-agnostic.

Usage:
    python collect_traces.py                         # full test set
    python collect_traces.py -n 50                   # first 50
    python collect_traces.py --skip-ablation         # traces + early decode only
    python collect_traces.py --split train -n 1000   # training set
"""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset

from codi_model import (
    load_model, generate_with_traces, generate_with_ablation,
    generate_early_decode, extract_number, NUM_LATENT,
)


def trace_to_dict(trace_result):
    """Serialize a TraceResult to a JSON-serializable dict."""
    steps = []
    for s in trace_result.steps:
        steps.append({
            "step": s.step,
            "top_k_ids": s.top_k_ids,
            "top_k_tokens": s.top_k_tokens,
            "top_k_probs": s.top_k_probs,
            "entropy": s.entropy,
            "norm": s.norm,
            "sparsity": s.sparsity,
            "cosine_to_prev": s.cosine_to_prev,
        })
    return {
        "output": trace_result.output,
        "no_reasoning_output": trace_result.no_reasoning_output,
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect CoDi latent traces")
    parser.add_argument("-n", "--num-examples", type=int, default=None)
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path (default: traces_{split}.jsonl)")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip ablation experiments (faster)")
    parser.add_argument("--skip-early-decode", action="store_true",
                        help="Skip early decode experiments")
    parser.add_argument("--ablation-types", nargs="+",
                        default=["zero", "skip", "noise"],
                        help="Ablation types to run")
    args = parser.parse_args()

    output_path = Path(args.output or f"traces_{args.split}.jsonl")

    # Load dataset
    print(f"Loading GSM8K {args.split} set...")
    dataset = load_dataset("gsm8k", "main")
    data = dataset[args.split]
    if args.num_examples:
        data = data.select(range(min(args.num_examples, len(data))))
    print(f"Processing {len(data)} questions")

    # Load model
    model, prj, tokenizer, bot_id, eot_id = load_model()

    t0 = time.time()
    with open(output_path, "w") as f:
        for i, example in enumerate(data):
            question = example["question"].strip()
            answer_str = example["answer"]

            # Parse gold answer
            if "####" in answer_str:
                gold = answer_str.split("####")[-1].strip().replace(",", "")
            else:
                gold = answer_str.strip()
            try:
                gold_num = float(gold)
            except ValueError:
                gold_num = None

            record = {
                "index": i,
                "question": question,
                "gold_answer": gold_num,
            }

            # 1. Full trace with token predictions
            trace = generate_with_traces(
                model, prj, tokenizer, bot_id, eot_id, question,
                collect_no_reasoning=True, top_k=10,
            )
            record.update(trace_to_dict(trace))

            # Store hidden states separately (as file references, not in JSON)
            # The JSONL has everything except raw tensors.

            # 2. Ablation experiments
            if not args.skip_ablation:
                ablations = {}
                for abl_type in args.ablation_types:
                    for step_idx in range(NUM_LATENT):
                        key = f"{abl_type}_step{step_idx}"
                        abl_output = generate_with_ablation(
                            model, prj, tokenizer, bot_id, eot_id, question,
                            ablation_step=step_idx, ablation_type=abl_type,
                        )
                        ablations[key] = abl_output
                record["ablations"] = ablations

            # 3. Early decode at each step
            if not args.skip_early_decode:
                early = generate_early_decode(
                    model, prj, tokenizer, bot_id, eot_id, question,
                )
                record["early_decodes"] = [
                    {"after_step": step, "output": text}
                    for step, text in early
                ]

            f.write(json.dumps(record) + "\n")

            if (i + 1) % 10 == 0 or (i + 1) == len(data):
                elapsed = time.time() - t0
                qps = (i + 1) / elapsed
                print(f"  [{i+1}/{len(data)}]  {qps:.2f} q/s  {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"\nDone. {len(data)} traces saved to {output_path} in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
