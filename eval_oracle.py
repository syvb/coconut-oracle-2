#!/usr/bin/env python3
"""
Evaluate the Latent Oracle against ground-truth data from traces.

Metrics:
1. Ablation prediction - can oracle predict what happens when a step is removed?
2. Early decode prediction - can oracle predict partial outputs?
3. Consistency - do oracle's "critical step" claims match actual ablation data?
4. Contrastive accuracy - can oracle predict whether reasoning helps?

Usage:
    python eval_oracle.py --traces traces_test.jsonl --oracle-dir oracle_model/best
    python eval_oracle.py --traces traces_test.jsonl --oracle-dir oracle_model/best --num 100
"""

import argparse
import json
import re
from pathlib import Path

import torch

from oracle_inference import load_oracle, oracle_answer
from generate_oracle_data import make_oracle_input
from codi_model import NUM_LATENT


def extract_step_number(text):
    """Extract a step number from oracle response text."""
    # Match "encoding" or "pre-reasoning" as step -1
    if re.search(r"(?i)\bencoding\b|pre-reasoning|before any", text):
        return -1
    m = re.search(r"[Ss]tep\s+(\d+)", text)
    return int(m.group(1)) if m else None


def eval_ablation_prediction(records, oracle_model, oracle_tok, max_records=None):
    """Test if oracle can predict which step is most critical."""
    correct = 0
    total = 0

    for record in records[:max_records]:
        if "ablations" not in record:
            continue

        # Ground truth: which step causes largest change when zeroed
        max_change = -1
        gt_critical = 0
        original = record["output"].strip()
        for step_idx in range(NUM_LATENT):
            key = f"zero_step{step_idx}"
            if key not in record["ablations"]:
                continue
            ablated = record["ablations"][key].strip()
            change = sum(1 for a, b in zip(original, ablated) if a != b) + \
                     abs(len(original) - len(ablated))
            if change > max_change:
                max_change = change
                gt_critical = step_idx

        # Ask oracle
        oracle_input = make_oracle_input(
            record["question"], record["steps"], record["output"],
            "Which latent step is most critical for the final output?"
        )
        answer = oracle_answer(oracle_model, oracle_tok, oracle_input)
        predicted_step = extract_step_number(answer)

        if predicted_step == gt_critical:
            correct += 1
        total += 1

    acc = correct / max(total, 1) * 100
    print(f"  Ablation prediction (critical step): {correct}/{total} = {acc:.1f}%")
    return {"correct": correct, "total": total, "accuracy": acc}


def eval_early_decode_convergence(records, oracle_model, oracle_tok, max_records=None):
    """Test if oracle can predict when the output converges."""
    correct = 0
    total = 0

    for record in records[:max_records]:
        if "early_decodes" not in record:
            continue

        # Ground truth: first step where output matches final
        final = record["output"].strip()
        gt_convergence = None
        for ed in record["early_decodes"]:
            if ed["output"].strip() == final:
                gt_convergence = ed["after_step"]
                break

        if gt_convergence is None:
            continue

        # Ask oracle
        oracle_input = make_oracle_input(
            record["question"], record["steps"], record["output"],
            "At which step does the model first produce its final output?"
        )
        answer = oracle_answer(oracle_model, oracle_tok, oracle_input)
        predicted = extract_step_number(answer)

        # Allow +/- 1 step tolerance
        if predicted is not None and abs(predicted - gt_convergence) <= 1:
            correct += 1
        total += 1

    acc = correct / max(total, 1) * 100
    print(f"  Early decode convergence: {correct}/{total} = {acc:.1f}% (±1 step)")
    return {"correct": correct, "total": total, "accuracy": acc}


def eval_contrastive(records, oracle_model, oracle_tok, max_records=None):
    """Test if oracle can predict whether reasoning changes the output."""
    correct = 0
    total = 0

    for record in records[:max_records]:
        if record.get("no_reasoning_output") is None:
            continue

        gt_changed = record["output"].strip() != record["no_reasoning_output"].strip()

        oracle_input = make_oracle_input(
            record["question"], record["steps"], record["output"],
            "Does latent reasoning change the output for this input?"
        )
        answer = oracle_answer(oracle_model, oracle_tok, oracle_input).lower()
        predicted_yes = "yes" in answer[:20]

        if predicted_yes == gt_changed:
            correct += 1
        total += 1

    acc = correct / max(total, 1) * 100
    print(f"  Contrastive (reasoning changes output?): {correct}/{total} = {acc:.1f}%")
    return {"correct": correct, "total": total, "accuracy": acc}


def eval_redundancy_consistency(records, oracle_model, oracle_tok, max_records=None):
    """Test if oracle's redundancy claims match ablation data."""
    consistent = 0
    total = 0

    for record in records[:max_records]:
        if "ablations" not in record:
            continue

        # Ground truth: which steps are redundant (zero ablation = no change)
        gt_redundant = set()
        original = record["output"].strip()
        for step_idx in range(NUM_LATENT):
            key = f"zero_step{step_idx}"
            if key in record["ablations"] and record["ablations"][key].strip() == original:
                gt_redundant.add(step_idx)

        if not gt_redundant:
            continue

        oracle_input = make_oracle_input(
            record["question"], record["steps"], record["output"],
            "Are any latent steps redundant?"
        )
        answer = oracle_answer(oracle_model, oracle_tok, oracle_input)

        # Extract mentioned step numbers
        predicted_redundant = set(int(m) for m in re.findall(r"\b(\d)\b", answer)
                                  if int(m) < NUM_LATENT)

        if predicted_redundant:
            # Check overlap
            overlap = len(predicted_redundant & gt_redundant)
            precision = overlap / len(predicted_redundant) if predicted_redundant else 0
            if precision > 0.5:
                consistent += 1
        total += 1

    acc = consistent / max(total, 1) * 100
    print(f"  Redundancy consistency: {consistent}/{total} = {acc:.1f}%")
    return {"consistent": consistent, "total": total, "accuracy": acc}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Latent Oracle")
    parser.add_argument("--traces", required=True, help="Trace JSONL file")
    parser.add_argument("--oracle-dir", default="oracle_model/best")
    parser.add_argument("--oracle-base", default=None)
    parser.add_argument("--num", type=int, default=None,
                        help="Max records to evaluate (default: all)")
    parser.add_argument("--output", default=None, help="Save results JSON")
    parser.add_argument("--no-token-info", action="store_true",
                        help="Strip token info from trace inputs (must match training)")
    args = parser.parse_args()

    if args.no_token_info:
        import generate_oracle_data
        generate_oracle_data._no_token_info = True
        print("Token info stripped from trace inputs (--no-token-info)")

    # Load oracle
    oracle_model, oracle_tok = load_oracle(args.oracle_dir, args.oracle_base)

    # Load traces
    print(f"Loading traces from {args.traces}...")
    records = []
    with open(args.traces) as f:
        for line in f:
            records.append(json.loads(line))
    n = args.num or len(records)
    records = records[:n]
    print(f"Evaluating on {len(records)} records\n")

    results = {}

    print("Running evaluations:")
    results["ablation"] = eval_ablation_prediction(records, oracle_model, oracle_tok)
    results["convergence"] = eval_early_decode_convergence(records, oracle_model, oracle_tok)
    results["contrastive"] = eval_contrastive(records, oracle_model, oracle_tok)
    results["redundancy"] = eval_redundancy_consistency(records, oracle_model, oracle_tok)

    print(f"\nOverall:")
    for name, r in results.items():
        print(f"  {name:25s}: {r['accuracy']:.1f}%")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
