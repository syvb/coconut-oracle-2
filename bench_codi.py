#!/usr/bin/env python3
"""
Benchmark CoDi on GSM8K test set, comparing reasoning vs no-reasoning modes.

Usage:
    python bench_codi.py                    # run both modes on full test set
    python bench_codi.py -n 100             # first 100 examples
    python bench_codi.py --mode reasoning   # only reasoning mode
    python bench_codi.py --mode none        # only no-reasoning mode
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset

from codi_model import load_model, generate, extract_number, NUM_LATENT


def parse_gsm8k_answer(answer_str):
    """Extract the numeric answer after #### in GSM8K format."""
    if "####" in answer_str:
        ans = answer_str.split("####")[-1].strip()
    else:
        ans = answer_str.strip()
    ans = ans.replace(",", "")
    try:
        return float(ans)
    except ValueError:
        return float("inf")


def run_benchmark(model, prj, tokenizer, bot_id, eot_id, questions, gold_answers,
                  num_latent, label):
    correct = 0
    total = len(questions)
    results = []
    t0 = time.time()

    for i, (q, gold) in enumerate(zip(questions, gold_answers)):
        response = generate(model, prj, tokenizer, bot_id, eot_id, q,
                            greedy=True, num_latent=num_latent)
        pred = extract_number(response)
        is_correct = (pred == gold)
        correct += is_correct

        results.append({
            "index": i,
            "question": q,
            "gold": gold,
            "predicted": pred,
            "response": response,
            "correct": is_correct,
        })

        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            acc = correct / (i + 1) * 100
            qps = (i + 1) / elapsed
            print(f"  [{label}] {i+1}/{total}  acc={acc:.1f}%  "
                  f"({correct}/{i+1})  {qps:.1f} q/s  {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    acc = correct / total * 100
    return acc, results, elapsed


def main():
    parser = argparse.ArgumentParser(description="Benchmark CoDi on GSM8K")
    parser.add_argument("-n", "--num-examples", type=int, default=None,
                        help="Number of test examples (default: all 1319)")
    parser.add_argument("--mode", choices=["both", "reasoning", "none"],
                        default="both", help="Which mode(s) to benchmark")
    parser.add_argument("--output", type=str, default=None,
                        help="Save detailed results to JSON file")
    args = parser.parse_args()

    # Load dataset
    print("Loading GSM8K test set...")
    dataset = load_dataset("gsm8k", "main")
    test_set = dataset["test"]

    if args.num_examples:
        test_set = test_set.select(range(min(args.num_examples, len(test_set))))

    questions = [ex["question"].strip() for ex in test_set]
    gold_answers = [parse_gsm8k_answer(ex["answer"]) for ex in test_set]
    print(f"Loaded {len(questions)} questions")

    # Load model
    model, prj, tokenizer, bot_id, eot_id = load_model()

    all_results = {}

    # Run benchmarks
    if args.mode in ("both", "reasoning"):
        print(f"\n{'='*60}")
        print(f"  REASONING MODE ({NUM_LATENT} latent steps)")
        print(f"{'='*60}")
        acc_r, results_r, time_r = run_benchmark(
            model, prj, tokenizer, bot_id, eot_id,
            questions, gold_answers,
            num_latent=NUM_LATENT, label="reasoning",
        )
        all_results["reasoning"] = {
            "accuracy": acc_r,
            "num_latent": NUM_LATENT,
            "time_seconds": time_r,
            "num_examples": len(questions),
            "details": results_r,
        }

    if args.mode in ("both", "none"):
        print(f"\n{'='*60}")
        print(f"  NO-REASONING MODE (0 latent steps)")
        print(f"{'='*60}")
        acc_n, results_n, time_n = run_benchmark(
            model, prj, tokenizer, bot_id, eot_id,
            questions, gold_answers,
            num_latent=0, label="no-reasoning",
        )
        all_results["no_reasoning"] = {
            "accuracy": acc_n,
            "num_latent": 0,
            "time_seconds": time_n,
            "num_examples": len(questions),
            "details": results_n,
        }

    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTS  ({len(questions)} questions)")
    print(f"{'='*60}")
    for mode, data in all_results.items():
        qps = data["num_examples"] / data["time_seconds"]
        print(f"  {mode:15s}  acc={data['accuracy']:.1f}%  "
              f"latent={data['num_latent']}  "
              f"{qps:.1f} q/s  {data['time_seconds']:.0f}s")

    if "reasoning" in all_results and "no_reasoning" in all_results:
        diff = all_results["reasoning"]["accuracy"] - all_results["no_reasoning"]["accuracy"]
        print(f"\n  Reasoning advantage: {diff:+.1f}%")

        # Show examples where reasoning helped / hurt
        r_details = {r["index"]: r for r in all_results["reasoning"]["details"]}
        n_details = {r["index"]: r for r in all_results["no_reasoning"]["details"]}
        helped = [i for i in r_details if r_details[i]["correct"] and not n_details[i]["correct"]]
        hurt = [i for i in r_details if not r_details[i]["correct"] and n_details[i]["correct"]]
        print(f"  Reasoning helped: {len(helped)} questions")
        print(f"  Reasoning hurt:   {len(hurt)} questions")

    # Save results
    if args.output:
        # Strip the heavy details for a smaller file unless requested
        out_path = Path(args.output)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
